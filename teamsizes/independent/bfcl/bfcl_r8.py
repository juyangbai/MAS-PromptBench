"""Independent topology specialized for BFCL (Berkeley Function Calling)."""

# Config
from __future__ import annotations

import argparse
import asyncio
import json
import operator
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

from teamsizes.output_contracts import append_output_contract_from_path
from typing import Annotated, Any, List, Optional

from huggingface_hub import hf_hub_download
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from langgraph.constants import END, START
from langgraph.graph.state import StateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.types import Send

# Shared telemetry.
_TOPO_ROOT = str(Path(__file__).resolve().parents[3])
if _TOPO_ROOT not in sys.path:
    sys.path.insert(0, _TOPO_ROOT)
from topologies.telemetry import langchain_ensemble_telemetry, normalize  # noqa: E402
import keyword
from pydantic import Field, create_model
from typing_extensions import TypedDict

from bfcl_eval.constants.enums import Language
from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING, ModelConfig
from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker


VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://lai:8001/v1")
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3.5-9B")

# Number of parallel replicas. Seeds are 0 .. N_AGENTS-1.
N_AGENTS = int(os.environ.get("INDEPENDENT_N_AGENTS", "8"))

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PROMPT_PATH = (
    _REPO_ROOT / "configs" / "prompts" / "independent" / "bfcl" / "caller.txt"
)
SYSTEM_PROMPT = append_output_contract_from_path(_PROMPT_PATH.read_text().strip(), __file__, _PROMPT_PATH.stem)

HF_DATASET = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"

# Phase 1: AST-scoreable single-turn subsets (same scope as single/bfcl).
AST_CATEGORIES = ("simple", "multiple", "parallel", "parallel_multiple")


# Model registration in bfcl-eval
def _register_model_with_bfcl(model_id: str) -> None:
    """Tell bfcl-eval how to handle function names for `model_id`.

    `ast_checker` -> `convert_func_name` looks up the model in
    MODEL_CONFIG_MAPPING to decide whether to rewrite '.' -> '_' in
    function names. Qwen3.5-9B handles dots fine (same as the registered
    qwen3-8b/-14b entries), but our model name isn't in the registry, so
    without this we hit KeyError on any instance with a dotted name
    (e.g. math.factorial).
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


# Schema -> Tool conversion
# BFCL uses "dict" for the outer object type and a few aliases not in
# standard JSON schema. Map them to Python types for pydantic.create_model.
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
    t = (prop or {}).get("type", "any")
    if t in ("array", "tuple"):
        items = prop.get("items") or {}
        item_type = _py_type_of(items) if items else Any
        return List[item_type]
    return _PRIMITIVE_TYPE_MAP.get(t, Any)


def _sanitize_field_name(name: str) -> str:
    """Return a pydantic-safe attribute name for `name`.

    Pydantic v2 rejects fields whose attribute name starts with `_` (reserved
    for private attrs) or collides with a Python keyword. Strip the leading
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

    safe_name = re.sub(r"\W+", "_", schema["name"]) + "Args"
    args_model = create_model(safe_name, **fields) if fields else None

    return StructuredTool.from_function(
        func=lambda **_: "",
        name=schema["name"],
        description=schema.get("description", ""),
        args_schema=args_model,
    )


# Agent
def _build_one_agent(tools: list[StructuredTool], seed: int):
    """Build one replica's react agent with the given dynamic tools + a seed."""
    llm = ChatOpenAI(
        model=MODEL_ID,
        base_url=VLLM_BASE_URL,
        api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        # House default; greedy (temp=0) would collapse all N replicas.
        temperature=0.2,
        top_p=0.9,
        seed=seed,
        # Bounded per-turn output; matches single/bfcl for runaway-generation
        # protection on the turn where the model sees empty tool results.
        max_tokens=2048,
        extra_body={
            "repetition_penalty": 1.05,
            # BFCL is single-turn function calling; long thinking makes Qwen3
            # drift from the tool-call API and emit text-form calls that
            # vLLM's qwen3_xml parser can't recognize.
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
    """LangChain tool_calls -> BFCL canonical: [{fn: {arg: val, ...}}, ...]."""
    return [{tc["name"]: dict(tc.get("args") or {})} for tc in tool_calls]


# Scoring
def score_one(
    function_schemas: list[dict],
    model_output: list[dict],
    ground_truth: list[dict],
    category: str,
) -> dict:
    """Delegate to bfcl-eval's AST checker."""
    return ast_checker(
        function_schemas,
        model_output,
        ground_truth,
        Language.PYTHON,
        category,
        MODEL_ID,
    )


# Aggregation
def _canonical_key(model_output: list[dict]) -> str:
    """Stable bucket key for canonical-form majority vote.

    Two replicas share a bucket iff their calls are byte-equal after:
      - sorting args within each call by key
      - sorting the list of calls by a deterministic (func_name, args_json)
        pair so parallel subsets ignore call-emission order

    JSON-serialized so the key is hashable.
    """
    normalized = []
    for call in model_output:
        normalized_call = {}
        for fn, args in call.items():
            normalized_call[fn] = dict(sorted((args or {}).items()))
        normalized.append(normalized_call)
    # Sort calls: parallel subsets are order-invariant by spec.
    normalized.sort(key=lambda c: json.dumps(c, sort_keys=True))
    return json.dumps(normalized, sort_keys=True)


def majority_vote(answers: list[dict]) -> dict | None:
    """Group replicas by canonical-form equality; return a representative
    from the largest bucket.

    Selection order:
      1. Largest bucket of byte-equal canonical outputs wins.
      2. Ties: first occurrence (lowest agent_id).
      3. None iff no replica produced a non-empty call.
    """
    valid = [a for a in answers if a.get("model_output")]
    if not valid:
        return None

    buckets: dict[str, list[dict]] = {}
    for a in valid:
        key = _canonical_key(a["model_output"])
        buckets.setdefault(key, []).append(a)

    # Winning bucket: largest count, then earliest first-occurrence.
    best_key = None
    best_count = -1
    best_first_id = 10**9
    for key, bucket in buckets.items():
        first_id = min(a["agent_id"] for a in bucket)
        if (len(bucket), -first_id) > (best_count, -best_first_id):
            best_count = len(bucket)
            best_first_id = first_id
            best_key = key
    # Return the replica with the lowest agent_id in the winning bucket.
    return min(buckets[best_key], key=lambda a: a["agent_id"])


# Graph
class State(TypedDict):
    instance: dict
    prompt: list  # BFCL "question" is a list-of-list of message dicts
    answers: Annotated[list[dict], operator.add]


class AgentInput(TypedDict):
    agent_id: int
    seed: int
    function_schemas: list[dict]
    prompt: list


async def _run_replica(inp: AgentInput) -> dict:
    """Run one replica's react agent and return its canonicalized call."""
    tools = [schema_to_tool(s) for s in inp["function_schemas"]]
    agent = _build_one_agent(tools, seed=inp["seed"])

    start = time.time()
    try:
        result = await agent.ainvoke(
            {"messages": inp["prompt"]},
            config={"recursion_limit": 25},
        )
    except Exception as e:
        return {"answers": [{
            "agent_id": inp["agent_id"],
            "seed": inp["seed"],
            "tool_calls": [],
            "model_output": [],
            "error": f"{type(e).__name__}: {e}",
            "solve_s": round(time.time() - start, 2),
            "messages": [],
        }]}
    elapsed = time.time() - start

    tool_calls = extract_first_tool_calls(result["messages"])
    return {"answers": [{
        "agent_id": inp["agent_id"],
        "seed": inp["seed"],
        "tool_calls": tool_calls,
        "model_output": to_canonical(tool_calls),
        "messages": result["messages"],
        "solve_s": round(elapsed, 2),
    }]}


def _fan_out(state: State) -> list[Send]:
    function_schemas = state["instance"]["function"]
    prompt = state["prompt"]
    return [
        Send(
            f"agent_{i}",
            {
                "agent_id": i,
                "seed": i,
                "function_schemas": function_schemas,
                "prompt": prompt,
            },
        )
        for i in range(N_AGENTS)
    ]


def build_graph() -> StateGraph:
    graph = StateGraph(State)
    for i in range(N_AGENTS):
        graph.add_node(f"agent_{i}", _run_replica)
    graph.add_conditional_edges(START, _fan_out)
    graph.add_edge([f"agent_{i}" for i in range(N_AGENTS)], END)
    return graph


# Orchestration
def solve(instance: dict) -> dict:
    """Run the ensemble on one BFCL instance.

    Returns:
        {
            "model_output":    winning canonical call(s) (list of dicts) or [],
            "winner":          agent_id of the selected replica,
            "buckets":         list of [canonical_key_json, count] pairs,
            "per_agent":       list of {agent_id, seed, tool_calls, model_output,
                                        messages, solve_s, error?},
        }
    """
    compiled = build_graph().compile()
    prompt = instance["question"][0]  # BFCL nested: [[msgs...]]
    result = asyncio.run(
        compiled.ainvoke({
            "instance": instance,
            "prompt": prompt,
            "answers": [],
        })
    )
    per_agent = sorted(result["answers"], key=lambda a: a["agent_id"])
    winner = majority_vote(per_agent)

    valid = [a for a in per_agent if a.get("model_output")]
    buckets: dict[str, int] = {}
    for a in valid:
        key = _canonical_key(a["model_output"])
        buckets[key] = buckets.get(key, 0) + 1
    bucket_report = sorted(buckets.items(), key=lambda kv: -kv[1])

    return {
        "model_output": (winner or {}).get("model_output") or [],
        "winner": (winner or {}).get("agent_id"),
        "buckets": bucket_report,
        "per_agent": per_agent,
    }


# Dataset
def load_instances(
    category: str,
    limit: int | None = None,
    offset: int = 0,
    only: list[str] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Load (rows, ground_truths) for one BFCL subset."""
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


# Batch runner
def _format_per_agent(out: dict) -> str:
    """One-line summary per replica for the trace file."""
    lines: list[str] = [
        f"winner: agent_{out.get('winner')}",
        f"buckets: {out.get('buckets')}",
        "",
    ]
    for a in out.get("per_agent") or []:
        calls = a.get("model_output") or []
        call_str = json.dumps(calls, sort_keys=True) if calls else "(no call)"
        err = f"  err={a['error']!r}" if a.get("error") else ""
        lines.append(
            f"agent_{a['agent_id']} seed={a.get('seed')} "
            f"solve_s={a.get('solve_s')}s  {call_str}{err}"
        )
    return "\n".join(lines)


def run_one(
    instance: dict,
    ground_truth: dict,
    category: str,
    out_dir: Path,
) -> dict:
    """Solve one instance with the ensemble and score the majority-vote
    winner. Writes trace to out_dir/traces/<id>.txt."""
    iid = instance["id"]
    summary: dict = {"id": iid, "category": category, "n_agents": N_AGENTS}

    try:
        out = solve(instance)
    except Exception as e:
        summary["error"] = f"{type(e).__name__}: {e}"
        summary["stage"] = "solve"
        return summary

    summary["winner"] = out.get("winner")
    summary["model_output"] = out.get("model_output") or []
    summary["buckets"] = out.get("buckets") or []
    summary["tool_calls"] = len(out.get("model_output") or [])
    summary.update(normalize(langchain_ensemble_telemetry(out.get("per_agent") or [])))

    trace_path = out_dir / "traces" / f"{iid}.txt"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text(_format_per_agent(out))

    try:
        checker_result = score_one(
            instance["function"],
            summary["model_output"],
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
    """Iterate one AST subset; write predictions.jsonl + results.jsonl."""
    _default_root = Path(__file__).resolve().parents[3]
    out_dir = out_dir or (_default_root / "results" / "bfcl_independent_r8")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, gts = load_instances(category, limit, offset, only)
    if verbose:
        print(f"loaded {len(rows)} instance(s) from {HF_DATASET} / {category}  "
              f"(N={N_AGENTS})")

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
                print(f"  -> {json.dumps({k: v for k, v in summary.items() if k != 'buckets'})}")

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


# CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Independent-topology BFCL ensemble runner."
    )
    parser.add_argument(
        "--category", default="simple", choices=list(AST_CATEGORIES),
        help="BFCL subset (default: simple).",
    )
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument(
        "--only", action="append", default=None, metavar="INSTANCE_ID",
        help="Restrict to these instance ids (repeatable).",
    )
    _default_out = str(
        Path(__file__).resolve().parents[3] / "results" / "bfcl_independent_r8"
    )
    parser.add_argument(
        "--out-dir", default=_default_out,
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
