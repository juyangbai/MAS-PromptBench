"""Single-agent ReAct topology specialized for BFCL (Berkeley Function Calling)."""

# Config
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, List, Optional

from huggingface_hub import hf_hub_download
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

# Shared telemetry.
_TOPO_ROOT = str(Path(__file__).resolve().parents[3])
if _TOPO_ROOT not in sys.path:
    sys.path.insert(0, _TOPO_ROOT)
from topologies.telemetry import langchain_telemetry, normalize  # noqa: E402
from topologies.output_contracts import append_output_contract_from_path  # noqa: E402
import keyword
from pydantic import Field, create_model

from bfcl_eval.constants.enums import Language
from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING, ModelConfig
from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker


VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://lai:8001/v1")
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3.5-9B")


def _register_model_with_bfcl(model_id: str) -> None:
    """Tell bfcl-eval how to handle function names for `model_id`.

    `ast_checker` → `convert_func_name` looks up the model in
    MODEL_CONFIG_MAPPING to decide whether to rewrite '.' → '_' in function
    names (a workaround for FC-tuned models that can't emit dots during
    inference). Qwen3.5-9B handles dots fine — same as the registered
    `qwen3-8b`/`qwen3-14b` entries — but our model name isn't in the
    registry, so without this we hit KeyError on any instance with a dotted
    function name (e.g. `math.factorial`).

    Only `underscore_to_dot` is consulted during scoring; the handler is
    never instantiated, so we reuse a sibling Qwen3 entry's reference.
    """
    if model_id in MODEL_CONFIG_MAPPING:
        return
    template = MODEL_CONFIG_MAPPING["qwen3-8b"]
    MODEL_CONFIG_MAPPING[model_id] = ModelConfig(
        model_name=model_id,
        display_name=model_id,
        url=template.url,
        org=template.org,
        license=template.license,
        model_handler=template.model_handler,
        is_fc_model=True,
        underscore_to_dot=False,
    )


_register_model_with_bfcl(MODEL_ID)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PROMPT_PATH = _REPO_ROOT / "configs" / "prompts" / "single" / "bfcl" / "solver.txt"

HF_DATASET = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"

# Phase 1: AST-scoreable single-turn subsets.
AST_CATEGORIES = ("simple", "multiple", "parallel", "parallel_multiple")


# System Prompt
def _load_system_prompt() -> str:
    if _PROMPT_PATH.exists():
        return append_output_contract_from_path(_PROMPT_PATH.read_text(), __file__, _PROMPT_PATH.stem)
    # Fallback if the prompt file hasn't been generated yet.
    return append_output_contract_from_path(
        (
        "You are a function-calling assistant. You are given one or more function "
        "schemas and a user request. Decide which function(s) need to be called to "
        "satisfy the request and emit the call(s) with all required parameters set "
        "to values implied by the request.\n\n"
        "Rules:\n"
        "- Only call functions that appear in the provided schemas.\n"
        "- Set every required parameter; include optional parameters only when the "
        "  request clearly implies a value.\n"
        "- For parallel requests, emit all needed calls in a single response.\n"
        "- Do not reply in plain text if a function call is required."
        ),
        __file__,
        _PROMPT_PATH.stem,
    )


SYSTEM_PROMPT = _load_system_prompt()


# Schema → Tool Conversion
# BFCL uses "dict" for the outer object type and a few aliases not in standard
# JSON schema. Map them to Python types used by pydantic.create_model.
_PRIMITIVE_TYPE_MAP = {
    "integer": int,
    "string": str,
    "float": float,
    "number": float,
    "boolean": bool,
    "dict": dict,
    "any": Any,
}


def _py_type_of(prop: dict) -> Any:
    """Map a BFCL JSON-schema property fragment to a Python type annotation."""
    t = (prop or {}).get("type", "any")
    if t in ("array", "tuple"):
        items = prop.get("items") or {}
        item_type = _py_type_of(items) if items else Any
        return List[item_type]
    return _PRIMITIVE_TYPE_MAP.get(t, Any)


def _sanitize_field_name(name: str) -> str:
    """Return a pydantic-safe attribute name for `name`.

    Pydantic v2 rejects fields whose attribute name starts with `_` (reserved
    for private attrs) or collides with a Python keyword. We strip the leading
    underscores and suffix `_` to any resulting keyword; the original name is
    preserved via Field(alias=...) so JSON schema + tool_calls args stay
    aligned to the BFCL schema.
    """
    safe = name.lstrip("_") or "field"
    if keyword.iskeyword(safe):
        safe = safe + "_"
    return safe


def schema_to_tool(schema: dict) -> StructuredTool:
    """Convert a single BFCL function schema into a dummy LangChain tool.

    The tool body returns an empty string — BFCL scores on whether the model
    emitted the correct call, not on any tool output.
    """
    params = schema.get("parameters") or {}
    properties = params.get("properties") or {}
    required = set(params.get("required") or [])

    fields: dict[str, Any] = {}
    for name, prop in properties.items():
        py_type = _py_type_of(prop)
        desc = (prop or {}).get("description", "")
        safe_name = _sanitize_field_name(name)
        alias_kw = {"alias": name} if safe_name != name else {}
        if name in required:
            fields[safe_name] = (py_type, Field(..., description=desc, **alias_kw))
        else:
            fields[safe_name] = (Optional[py_type], Field(None, description=desc, **alias_kw))

    # Sanitize the pydantic model name (pydantic rejects non-identifier chars).
    safe_name = re.sub(r"\W+", "_", schema["name"]) + "Args"
    args_model = create_model(safe_name, **fields) if fields else None

    return StructuredTool.from_function(
        func=lambda **_: "",
        name=schema["name"],
        description=schema.get("description", ""),
        args_schema=args_model,
    )


# Agent
def build_agent(tools: list[StructuredTool]):
    llm = ChatOpenAI(
        model=MODEL_ID,
        base_url=VLLM_BASE_URL,
        api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        # House default: light stochastic sampling + fixed seed gives per-seed
        # reproducibility while breaking greedy-decoding degenerate loops.
        temperature=0.2,
        top_p=0.9,
        seed=0,
        # Per-LLM-call token cap. BFCL scoring only looks at the first
        # tool-call emission; subsequent react turns are noise. Capping
        # prevents runaway generation on turns where the model sees empty
        # tool results and would otherwise stream until context overflow.
        max_tokens=1024,
        extra_body={
            "repetition_penalty": 1.05,
            # BFCL is single-turn function calling; long thinking makes Qwen3
            # drift from the tool-call API and emit text-form calls that
            # vLLM's qwen3_xml parser can't recognize. Disable thinking here.
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    return create_react_agent(model=llm, tools=tools, prompt=SYSTEM_PROMPT)


# Output Parsing
def extract_first_tool_calls(messages: list) -> list[dict]:
    """Return tool_calls from the earliest AIMessage that has any."""
    for msg in messages:
        tcs = getattr(msg, "tool_calls", None) or []
        if tcs:
            return tcs
    return []


def to_canonical(tool_calls: list[dict]) -> list[dict]:
    """LangChain tool_calls → BFCL canonical form.

    LangChain: [{"name": "fn", "args": {...}, "id": "..."}]
    BFCL:      [{"fn": {"arg": value, ...}}, ...]
    """
    return [{tc["name"]: dict(tc.get("args") or {})} for tc in tool_calls]


# Scoring
def score_one(
    function_schemas: list[dict],
    model_output: list[dict],
    ground_truth: list[dict],
    category: str,
) -> dict:
    """Delegate to bfcl-eval's AST checker.

    `ast_checker` branches on `category`: parallel* → order-free parallel match,
    *multiple* → pick-one, otherwise → simple single-call match.
    """
    return ast_checker(
        function_schemas,
        model_output,
        ground_truth,
        Language.PYTHON,
        category,
        MODEL_ID,
    )


# Orchestration
def solve(instance: dict) -> dict:
    """Run the react agent once; capture the first tool-call commitment."""
    tools = [schema_to_tool(s) for s in instance["function"]]
    agent = build_agent(tools)
    messages = instance["question"][0]  # nested list: [[msgs...]]

    start = time.time()
    result = agent.invoke(
        {"messages": messages},
        config={"recursion_limit": 25},
    )
    elapsed = time.time() - start

    tool_calls = extract_first_tool_calls(result["messages"])
    return {
        "messages": result["messages"],
        "tool_calls": tool_calls,
        "model_output": to_canonical(tool_calls),
        "solve_s": elapsed,
    }


# Dataset Loader
def load_instances(
    category: str,
    limit: int | None = None,
    offset: int = 0,
    only: list[str] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Load (rows, ground_truths) for one BFCL subset.

    Each file is JSONL; ground_truth is aligned by instance id.
    """
    main = Path(
        hf_hub_download(HF_DATASET, f"BFCL_v3_{category}.json", repo_type="dataset")
    )
    ans = Path(
        hf_hub_download(
            HF_DATASET,
            f"possible_answer/BFCL_v3_{category}.json",
            repo_type="dataset",
        )
    )

    rows = [json.loads(line) for line in main.read_text().splitlines() if line.strip()]
    gt_by_id = {
        json.loads(line)["id"]: json.loads(line)
        for line in ans.read_text().splitlines()
        if line.strip()
    }

    if only:
        want = set(only)
        rows = [r for r in rows if r["id"] in want]
    rows = rows[offset:]
    if limit is not None:
        rows = rows[:limit]

    missing = [r["id"] for r in rows if r["id"] not in gt_by_id]
    if missing:
        raise RuntimeError(f"no ground truth for ids: {missing[:5]}")

    return rows, [gt_by_id[r["id"]] for r in rows]


# Batch Runner
def _format_trace(messages: list) -> str:
    lines: list[str] = []
    for msg in messages:
        kind = getattr(msg, "type", "unknown")
        if kind == "human":
            lines.append("=== HUMAN ===")
            lines.append(str(msg.content))
        elif kind == "system":
            lines.append("=== SYSTEM ===")
            lines.append(str(msg.content))
        elif kind == "ai":
            lines.append("=== AI ===")
            for tc in getattr(msg, "tool_calls", None) or []:
                lines.append(
                    f"[tool_call] {tc['name']}({json.dumps(tc.get('args') or {})})"
                )
            content = getattr(msg, "content", "")
            if content:
                lines.append(str(content))
        elif kind == "tool":
            lines.append("=== TOOL ===")
            lines.append(str(msg.content))
        lines.append("")
    return "\n".join(lines)


def run_one(
    instance: dict,
    ground_truth: dict,
    category: str,
    out_dir: Path,
) -> dict:
    """Solve one instance and score its first tool-call commitment.

    Writes the message trace to out_dir/traces/<id>.txt. Returns a summary
    dict (never raises to the caller).
    """
    iid = instance["id"]
    summary: dict = {"id": iid, "category": category}

    try:
        solution = solve(instance)
    except Exception as e:
        summary["error"] = f"{type(e).__name__}: {e}"
        summary["stage"] = "solve"
        return summary

    summary["solve_s"] = round(solution["solve_s"], 1)
    summary["tool_calls"] = len(solution["tool_calls"])
    summary["model_output"] = solution["model_output"]
    summary.update(normalize(langchain_telemetry(solution.get("messages") or [])))

    trace_path = out_dir / "traces" / f"{iid}.txt"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text(_format_trace(solution["messages"]))

    try:
        checker_result = score_one(
            instance["function"],
            solution["model_output"],
            ground_truth["ground_truth"],
            category,
        )
    except Exception as e:
        summary["valid"] = False
        summary["error"] = f"{type(e).__name__}: {e}"
        summary["stage"] = "score"
        return summary

    summary["valid"] = bool(checker_result.get("valid"))
    summary["error_type"] = checker_result.get("error_type")
    if not summary["valid"]:
        summary["score_error"] = (checker_result.get("error") or [])[:3]
    return summary


def run_batch(
    category: str,
    limit: int | None = None,
    offset: int = 0,
    only: list[str] | None = None,
    out_dir: Path | None = None,
    verbose: bool = True,
) -> dict:
    """Iterate one AST subset; write predictions.jsonl + results.jsonl.

    Returns an aggregate summary with {n, valid, valid_rate, per_category}.
    """
    out_dir = out_dir or (_REPO_ROOT / "results" / "bfcl")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, gts = load_instances(category, limit, offset, only)
    if verbose:
        print(f"loaded {len(rows)} instance(s) from {HF_DATASET} / {category}")

    preds_path = out_dir / "predictions.jsonl"
    results_path = out_dir / "results.jsonl"

    valid = 0
    with preds_path.open("a") as fp, results_path.open("a") as fr:
        for i, (row, gt) in enumerate(zip(rows, gts), start=1):
            if verbose:
                print(f"\n[{i}/{len(rows)}] {row['id']}  ({category})")
            summary = run_one(row, gt, category, out_dir)
            if summary.get("valid"):
                valid += 1

            fp.write(json.dumps({
                "id": row["id"],
                "category": category,
                "model_output": summary.get("model_output"),
                "model_name_or_path": MODEL_ID,
            }) + "\n")
            fr.write(json.dumps(summary) + "\n")
            fp.flush()
            fr.flush()
            if verbose:
                print(f"  -> {json.dumps(summary)}")

    if verbose:
        print(
            f"\ndone: valid {valid}/{len(rows)}"
            f"\n      predictions -> {preds_path}"
            f"\n      results     -> {results_path}"
        )
    return {
        "n": len(rows),
        "valid": valid,
        "valid_rate": (valid / len(rows)) if rows else 0.0,
        "category": category,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the single-topology BFCL agent on one of the AST subsets.",
        epilog=(
            "Examples:\n"
            "  python topologies/single/bfcl/langgraph_bfcl.py --category simple --limit 5\n"
            "  python topologies/single/bfcl/langgraph_bfcl.py --only simple_0 --only simple_1\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--category",
        default="simple",
        choices=list(AST_CATEGORIES),
        help="BFCL subset to evaluate (default: simple).",
    )
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument(
        "--only",
        action="append",
        default=None,
        metavar="INSTANCE_ID",
        help="Restrict to these instance ids (repeatable).",
    )
    parser.add_argument(
        "--out-dir",
        default=str(_REPO_ROOT / "results" / "bfcl"),
        help="Where to write predictions.jsonl / results.jsonl / traces/.",
    )
    args = parser.parse_args()

    run_batch(
        category=args.category,
        limit=args.limit if not args.only else None,
        offset=args.offset,
        only=args.only,
        out_dir=Path(args.out_dir),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
