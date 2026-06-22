"""Sequential topology specialized for BFCL, implemented in LangGraph."""

# Config
from __future__ import annotations

import argparse
import json
import operator
import os
import re
import sys
import time  # noqa: F401 — used by run_one
from pathlib import Path

from teamsizes.output_contracts import append_output_contract_from_path
from typing import Annotated

from typing_extensions import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

# Shared telemetry.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_TOPO_ROOT = str(_REPO_ROOT)
if _TOPO_ROOT not in sys.path:
    sys.path.insert(0, _TOPO_ROOT)
from topologies.telemetry import langchain_telemetry, normalize  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402

from bfcl_eval.constants.enums import Language  # noqa: E402
from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING, ModelConfig  # noqa: E402
from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker  # noqa: E402


VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://n12:8000/v1")
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3.5-9B")

_PROMPTS_DIR = _REPO_ROOT / "configs" / "prompts" / "sequential" / "bfcl"

HF_DATASET = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
AST_CATEGORIES = ("simple", "multiple", "parallel", "parallel_multiple")


def _load_prompt(role: str) -> str:
    return append_output_contract_from_path((_PROMPTS_DIR / f"{role}.txt").read_text().strip(), __file__, role)


# Model registration in bfcl-eval
def _register_model_with_bfcl(model_id: str) -> None:
    """Tell bfcl-eval how to handle function names for `model_id`.

    `ast_checker` -> `convert_func_name` looks up the model in
    MODEL_CONFIG_MAPPING to decide whether to rewrite '.' -> '_' in
    function names. Qwen3.5-9B handles dots fine (same as the
    registered qwen3-8b/-14b entries), but our model name isn't in
    the registry, so without this we hit KeyError on any instance
    with a dotted name (e.g. `math.factorial`).
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


# LLM
def _build_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=MODEL_ID,
        base_url=VLLM_BASE_URL,
        api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        temperature=0.2,
        top_p=0.9,
        seed=0,
        # Bounded per-turn output; BFCL calls are short.
        max_tokens=2048,
        extra_body={
            "repetition_penalty": 1.05,
            # Same rationale as single/bfcl: long thinking makes Qwen3
            # drift from the canonical output format.
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )


# Per-stage task descriptions (same as CrewAI Task.description)
_TASK_DESCRIPTIONS = {
    "analyzer": (
        "Parse the user request below to extract the intent "
        "(what the user wants done), entities (names, numbers, "
        "dates), and any implicit constraints. Do NOT reference "
        "the schemas yet.\n\n"
        "USER REQUEST:\n{user_request}"
    ),
    "inspector": (
        "Given the Analyzer's summary and the function schema(s) "
        "below, map the intent onto a specific function and "
        "propose argument values. Name the required parameters "
        "and the types they must have.\n\n"
        "USER REQUEST:\n{user_request}\n\n"
        "SCHEMAS:\n{schemas_text}"
    ),
    "caller": (
        "Emit the call in BFCL canonical JSON. Follow the "
        "Inspector's plan exactly. Output a SINGLE fenced ```json "
        "block containing a list of dicts with one key per dict "
        "where the key is the function name and the value is the "
        "arg dict. For a single call it's a one-element list. For "
        "parallel calls it's a multi-element list. Example:\n"
        "```json\n[{{\"calculate_triangle_area\": {{\"base\": 10, \"height\": 5}}}}]\n```\n\n"
        "USER REQUEST:\n{user_request}\n\n"
        "SCHEMAS:\n{schemas_text}"
    ),
    "verifier": (
        "Validate the Caller's canonical JSON against the "
        "schema(s). Check: (a) each function name appears in the "
        "schemas, (b) every required parameter is present, (c) "
        "each argument's value has the schema's declared type "
        "(coerce only where the schema permits). If correct, "
        "re-emit the SAME JSON. If an error is found, emit a "
        "CORRECTED canonical JSON. Output a SINGLE fenced ```json "
        "block as the FINAL answer.\n\n"
        "USER REQUEST:\n{user_request}\n\n"
        "SCHEMAS:\n{schemas_text}"
    ),
    'intent_extractor': (
        'Parse the user request to extract intent (one sentence) and the entities/values referenced. Do NOT inspect schemas yet.\n\nUSER REQUEST:\n{user_request}\n\nSCHEMAS:\n{schemas_text}'
    ),
    'schema_indexer': (
        'Index the available schemas: numbered list of <fn> -> <one-line purpose>. Flag the schema(s) that match the intent.\n\nUSER REQUEST:\n{user_request}\n\nSCHEMAS:\n{schemas_text}'
    ),
    'arg_planner': (
        'Given the chosen function and entities, plan the exact argument values. Output: per-arg spec with types.\n\nUSER REQUEST:\n{user_request}\n\nSCHEMAS:\n{schemas_text}'
    ),
    'type_validator': (
        "Validate the caller's emitted JSON against the schema. If correct, repeat unchanged; if not, emit a corrected fenced ```json``` block as the final answer.\n\nUSER REQUEST:\n{user_request}\n\nSCHEMAS:\n{schemas_text}"
    ),
}


# StateGraph scaffolding (sequential 4-stage pipeline)
def _merge_dict(a: dict | None, b: dict | None) -> dict:
    out = dict(a or {})
    out.update(b or {})
    return out


class SequentialState(TypedDict, total=False):
    inputs: dict
    by_stage: Annotated[dict, _merge_dict]
    messages: Annotated[list, operator.add]


def _format_user(
    template: str, inputs: dict, by_stage: dict, prior_roles: list[str]
) -> str:
    body = template.format(**inputs)
    for r in prior_roles:
        body += f"\n\n--- PRIOR STAGE: {r} ---\n{by_stage.get(r, '')}"
    return body


def _make_tool_node(role, sys_prompt, tools, llm, template, prior_roles):
    from langgraph.prebuilt import create_react_agent

    agent = create_react_agent(model=llm, tools=tools, prompt=sys_prompt)

    def node(state: SequentialState) -> dict:
        user = _format_user(
            template, state["inputs"], state.get("by_stage") or {}, prior_roles
        )
        res = agent.invoke(
            {"messages": [("user", user)]},
            config={"recursion_limit": 50},
        )
        raw = next(
            (
                m.content
                for m in reversed(res["messages"])
                if getattr(m, "type", None) == "ai" and getattr(m, "content", "")
            ),
            "",
        )
        ai_msgs = [m for m in res["messages"] if getattr(m, "type", None) == "ai"]
        return {"by_stage": {role: raw}, "messages": ai_msgs}

    return node


def _make_plain_node(role, sys_prompt, llm, template, prior_roles):
    def node(state: SequentialState) -> dict:
        user = _format_user(
            template, state["inputs"], state.get("by_stage") or {}, prior_roles
        )
        ai = llm.invoke(
            [SystemMessage(content=sys_prompt), HumanMessage(content=user)]
        )
        return {"by_stage": {role: ai.content or ""}, "messages": [ai]}

    return node


def _build_graph(llm: ChatOpenAI):
    """Build the 4-stage analyzer -> inspector -> caller -> verifier pipeline."""
    stages = [
        (
            'intent_extractor',
            _load_prompt('intent_extractor'),
            [],
            _TASK_DESCRIPTIONS['intent_extractor'],
        ),
        (
            'analyzer',
            _load_prompt('analyzer'),
            [],
            _TASK_DESCRIPTIONS['analyzer'],
        ),
        (
            'schema_indexer',
            _load_prompt('schema_indexer'),
            [],
            _TASK_DESCRIPTIONS['schema_indexer'],
        ),
        (
            'inspector',
            _load_prompt('inspector'),
            [],
            _TASK_DESCRIPTIONS['inspector'],
        ),
        (
            'arg_planner',
            _load_prompt('arg_planner'),
            [],
            _TASK_DESCRIPTIONS['arg_planner'],
        ),
        (
            'caller',
            _load_prompt('caller'),
            [],
            _TASK_DESCRIPTIONS['caller'],
        ),
        (
            'type_validator',
            _load_prompt('type_validator'),
            [],
            _TASK_DESCRIPTIONS['type_validator'],
        ),
        (
            'verifier',
            _load_prompt('verifier'),
            [],
            _TASK_DESCRIPTIONS['verifier'],
        ),
    
    ]

    graph = StateGraph(SequentialState)
    prior: list[str] = []
    for role, sys_p, tools, tmpl in stages:
        node_fn = (
            _make_tool_node(role, sys_p, tools, llm, tmpl, list(prior))
            if tools
            else _make_plain_node(role, sys_p, llm, tmpl, list(prior))
        )
        graph.add_node(role, node_fn)
        prior.append(role)

    graph.add_edge(START, stages[0][0])
    for a, b in zip(stages, stages[1:]):
        graph.add_edge(a[0], b[0])
    graph.add_edge(stages[-1][0], END)

    return graph.compile(), [s[0] for s in stages]


# Output Parsing
_FENCED_RE = re.compile(r"```(?:\w*)\s*([\s\S]*?)\s*```")


def extract_canonical(text: str) -> list[dict] | None:
    """Extract the last fenced JSON list-of-dicts from `text`.

    Returns None if no fenced JSON parses to a non-empty list-of-dicts.
    """
    candidates: list[str] = []
    for m in _FENCED_RE.finditer(text):
        candidates.append(m.group(1))
    # Prefer the LAST fenced block (the verifier's final emission).
    for cand in reversed(candidates):
        try:
            parsed = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, list) and parsed and all(isinstance(x, dict) for x in parsed):
            return parsed
    return None


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


# Orchestration
def _escape_braces(s: str) -> str:
    """Escape `{` / `}` in a value so .format() treats them as literal
    text. Raw JSON of schemas contains braces that would otherwise be
    parsed as format placeholders."""
    return s.replace("{", "{{").replace("}", "}}")


def _flatten_user_request(question: list) -> str:
    """BFCL stores `question` as [[msgs...]]; for single-turn subsets we
    take the concatenated user-turn contents."""
    if not question:
        return ""
    turns = question[0] if isinstance(question[0], list) else question
    parts = []
    for msg in turns:
        if isinstance(msg, dict):
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"[{role}] {content}")
        else:
            parts.append(str(msg))
    return "\n".join(parts)


def solve(instance: dict) -> dict:
    """Run the 4-stage sequential graph on one BFCL instance.

    Returns:
        {
            "model_output": canonical call list [{fn: {arg: val}}] or [],
            "by_stage":     {analyzer, inspector, caller, verifier}
                            -> each stage's text output,
            "raw":          verifier's final text,
            "telemetry":    normalized 5-key token/call counts,
        }
    """
    llm = _build_llm()
    compiled, roles = _build_graph(llm)

    user_request = _flatten_user_request(instance["question"])
    schemas_text = json.dumps(instance["function"], indent=2)

    result = compiled.invoke(
        {
            "inputs": {
                "user_request": _escape_braces(user_request),
                "schemas_text": _escape_braces(schemas_text),
            },
            "by_stage": {},
            "messages": [],
        }
    )

    stages_out = result.get("by_stage") or {}
    final = stages_out.get(roles[-1], "")

    # Prefer the verifier's output, fall back to the caller's.
    model_output = extract_canonical(stages_out.get("verifier", "") or final)
    if model_output is None:
        model_output = extract_canonical(stages_out.get("caller", ""))

    return {
        "model_output": model_output or [],
        "by_stage": stages_out,
        "raw": final,
        "telemetry": normalize(langchain_telemetry(result.get("messages") or [])),
    }


# Batch runner
def run_one(
    instance: dict,
    ground_truth: dict,
    category: str,
    out_dir: Path,
) -> dict:
    """Solve one BFCL instance via the 4-stage graph and score the final
    canonical output. Writes trace to out_dir/traces/<id>.txt."""
    iid = instance["id"]
    summary: dict = {"id": iid, "category": category}

    try:
        out = solve(instance)
    except Exception as e:
        summary["error"] = f"{type(e).__name__}: {e}"
        summary["stage"] = "solve"
        return summary

    summary["model_output"] = out.get("model_output") or []
    summary["tool_calls"] = len(summary["model_output"])
    summary.update(out.get("telemetry") or {})

    trace_path = out_dir / "traces" / f"{iid}.txt"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("w") as f:
        for stage, content in (out.get("by_stage") or {}).items():
            f.write(f"=== {stage.upper()} ===\n{content}\n\n")

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
    out_dir = out_dir or (_default_root / "results" / "bfcl_sequential_r8")
    out_dir = Path(out_dir).resolve()
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


# CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sequential-topology BFCL runner (LangGraph 4-stage)."
    )
    parser.add_argument(
        "--category", default="simple", choices=list(AST_CATEGORIES),
        help="BFCL subset (default: simple).",
    )
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument(
        "--only", action="append", default=None, metavar="INSTANCE_ID",
    )
    _default_out = str(
        Path(__file__).resolve().parents[3] / "results" / "bfcl_sequential_r8"
    )
    parser.add_argument("--out-dir", default=_default_out)
    args = parser.parse_args()

    run_batch(
        category=args.category,
        limit=args.limit if not args.only else None,
        offset=args.offset,
        only=args.only,
        out_dir=Path(args.out_dir),
    )
