"""Centralized topology specialized for BFCL, LangGraph."""

# Config
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

from teamsizes.output_contracts import append_output_contract_from_path
from typing import Annotated, Optional

from typing_extensions import TypedDict

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, create_react_agent

_REPO_ROOT = Path(__file__).resolve().parents[3]

# Shared telemetry.
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

_PROMPTS_DIR = _REPO_ROOT / "configs" / "prompts" / "centralized" / "bfcl"

HF_DATASET = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
AST_CATEGORIES = ("simple", "multiple", "parallel", "parallel_multiple")

# Same cap as AutoGen sibling's MaxMessageTermination(24).
MAX_TURNS = 24


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
        # BFCL calls are short; bounded output is fine.
        max_tokens=2048,
        extra_body={
            "repetition_penalty": 1.05,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )


# Delegation tools (routing markers)
# The manager "calls" these to hand the floor to a specific worker. The
# body echoes the instructions, producing a ToolMessage the worker can
# read as context. The router after `manager_tools` inspects the name
# to route to the right worker node. BFCL has no real agent tools — the
# manager's whole tool set is just these three markers.
@tool("delegate_to_inspector_worker")
def delegate_to_inspector_worker(instructions: str) -> str:
    """Hand the next turn to the inspector_worker. Use this to have the
    inspector read the provided schema(s) and produce a structured argument
    plan (which schema to call, which arguments are required, where each
    value comes from in the user request).

    Args:
        instructions: what you want the inspector to analyze this turn.
    """
    return instructions


@tool("delegate_to_caller_worker")
def delegate_to_caller_worker(instructions: str) -> str:
    """Hand the next turn to the caller_worker. Use this once the inspector's
    arg plan is ready; the caller composes the canonical JSON call (a list
    of dicts of the form [{"<fn_name>": {<args>}}]).

    Args:
        instructions: what call shape you want composed this turn.
    """
    return instructions


@tool("delegate_to_validator_worker")
def delegate_to_validator_worker(instructions: str) -> str:
    """Hand the next turn to the validator_worker. Use this once a candidate
    canonical call exists; the validator checks name existence, required
    params, and type validity against the provided schemas.

    Args:
        instructions: what the validator should check this turn.
    """
    return instructions


DELEGATION_TOOLS = [
    delegate_to_inspector_worker,
    delegate_to_caller_worker,
    delegate_to_validator_worker,
]
DELEGATION_NAMES = {t.name for t in DELEGATION_TOOLS}

# BFCL's manager has no real tools; its whole tool surface is the three
# delegation markers.
MANAGER_TOOLS = DELEGATION_TOOLS


# Manager prompt suffix
_MANAGER_TERMINATE_NUDGE = (
    "\n\nWhen you emit the final fenced ```json``` block containing "
    "the canonical call list, immediately follow it with the literal "
    "string TERMINATE on its own line so the group-chat knows to stop."
    "\n\nDelegation: when you want a specific worker to act, call the "
    "matching delegate_to_<worker> tool with clear instructions "
    "(instead of merely addressing them in free-form text). The three "
    "workers are: inspector_worker, caller_worker, validator_worker."
)


# State + nodes
class CentralizedState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    turn_count: int


def _tag_source(msg: BaseMessage, source: str) -> None:
    """Attach an AutoGen-style `source` field via additional_kwargs."""
    try:
        kw = dict(getattr(msg, "additional_kwargs", None) or {})
        kw["source"] = source
        msg.additional_kwargs = kw
    except Exception:
        pass


def _manager_system() -> str:
    return _load_prompt("manager") + _MANAGER_TERMINATE_NUDGE


def _manager_node(state: CentralizedState) -> dict:
    llm = _build_llm().bind_tools(MANAGER_TOOLS)
    sys_msg = SystemMessage(content=_manager_system())
    ai = llm.invoke([sys_msg] + state["messages"])
    # AutoGen messages carry a `.source` name; we mimic that on the
    # AIMessage via additional_kwargs for trace rendering parity.
    _tag_source(ai, "manager")
    return {"messages": [ai], "turn_count": int(state.get("turn_count", 0)) + 1}


_manager_tool_node = ToolNode(MANAGER_TOOLS)


def _route_from_manager(state: CentralizedState) -> str:
    msgs = state["messages"]
    if not msgs:
        return "manager"
    last = msgs[-1]
    if int(state.get("turn_count", 0)) >= MAX_TURNS:
        return END
    if isinstance(last, AIMessage):
        content = last.content or ""
        if isinstance(content, str) and "TERMINATE" in content:
            return END
        if getattr(last, "tool_calls", None):
            return "manager_tools"
    # No tool call, no TERMINATE — loop back and let the manager try again.
    return "manager"


def _route_from_manager_tools(state: CentralizedState) -> str:
    # Find the most recent AIMessage with tool_calls; its tool_calls tell
    # us which delegation was requested. (BFCL has no non-delegation tools,
    # so any tool_call here is necessarily a delegation.)
    for m in reversed(state["messages"]):
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            for tc in m.tool_calls:
                name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
                if name in DELEGATION_NAMES:
                    return name.removeprefix("delegate_to_")
            return "manager"
    return "manager"


def _make_worker_node(name: str, tools: list, llm: ChatOpenAI):
    sys_prompt = _load_prompt(name)
    agent = create_react_agent(model=llm, tools=tools, prompt=sys_prompt)

    def node(state: CentralizedState) -> dict:
        # create_react_agent returns {"messages": [full history incl. input]},
        # so we splice out only the new messages it appended.
        prior = list(state["messages"])
        result = agent.invoke(
            {"messages": prior},
            config={"recursion_limit": 30},
        )
        full = result["messages"]
        new_msgs = full[len(prior):]
        # Tag worker outputs with source for trace rendering.
        for m in new_msgs:
            if isinstance(m, AIMessage):
                _tag_source(m, name)
        # Each new AIMessage counts as one turn; tool-executions don't.
        n_turns = sum(1 for m in new_msgs if isinstance(m, AIMessage))
        return {"messages": new_msgs, "turn_count": int(state.get("turn_count", 0)) + n_turns}

    return node


def _build_graph(llm: Optional[ChatOpenAI] = None):
    if llm is None:
        llm = _build_llm()

    graph = StateGraph(CentralizedState)
    graph.add_node("manager", _manager_node)
    graph.add_node("manager_tools", _manager_tool_node)

    worker_specs = [
        ("inspector_worker", []),
        ("caller_worker", []),
        ("validator_worker", []),
    ]
    for name, tools in worker_specs:
        graph.add_node(name, _make_worker_node(name, tools, llm))

    graph.add_edge(START, "manager")
    graph.add_conditional_edges(
        "manager",
        _route_from_manager,
        {
            "manager_tools": "manager_tools",
            "manager": "manager",
            END: END,
        },
    )
    graph.add_conditional_edges(
        "manager_tools",
        _route_from_manager_tools,
        {
            "inspector_worker": "inspector_worker",
            "caller_worker": "caller_worker",
            "validator_worker": "validator_worker",
            "manager": "manager",
        },
    )
    for name, _ in worker_specs:
        graph.add_edge(name, "manager")

    return graph.compile(), ["manager"] + [n for n, _ in worker_specs]


# Prompt scaffolding
def format_task(user_request: str, schemas_text: str) -> str:
    """Build the single task string sent to the group chat.

    The user request + schemas both go into the initial `user` message
    so every agent sees them from the start of the dialogue.
    """
    return (
        "USER REQUEST:\n"
        f"{user_request}\n\n"
        "SCHEMAS:\n"
        f"{schemas_text}\n\n"
        "Emit the final canonical call list as a SINGLE fenced ```json "
        "block. Canonical form is a list of dicts; each dict has exactly "
        "ONE key equal to the ACTUAL function name from one of the "
        "schemas above, and the value is the arguments dict. "
        "Do NOT use the literal string 'fn_name' as the key, and do NOT "
        "use the shape {\"fn_name\": \"<name>\", \"args\": {...}}. "
        "Example for a schema named calculate_area: "
        "[{\"calculate_area\": {\"width\": 5, \"height\": 3}}]. "
        "For parallel calls, emit multiple such dicts in the list."
    )


# Output parsing
_FENCED_RE = re.compile(r"```(?:\w*)\s*([\s\S]*?)\s*```")

# Common non-canonical shapes the model emits instead of {name: args}:
#   {"fn_name": "<real>", "args": {...}}           (model misreads the template)
#   {"name": "<real>", "arguments": {...}}         (OpenAI tool-call shape)
#   {"function_name": "<real>", "arguments": {...}}
_NAME_ARGS_PAIRS = (
    ("fn_name", "args"),
    ("name", "arguments"),
    ("function_name", "arguments"),
    ("function", "arguments"),
)


def _normalize_call(d: dict) -> dict:
    """Normalize the common wrong shapes back to canonical {name: args}.

    Leaves a dict alone if it's already canonical (single string-keyed arg dict).
    """
    for name_key, args_key in _NAME_ARGS_PAIRS:
        if (
            name_key in d
            and args_key in d
            and isinstance(d[name_key], str)
            and isinstance(d[args_key], dict)
        ):
            return {d[name_key]: d[args_key]}
    return d


def extract_canonical(text: str) -> list[dict] | None:
    """Extract the last fenced JSON list-of-dicts from `text`, normalizing
    common non-canonical shapes emitted by chat models.

    Returns None if no fenced JSON parses to a non-empty list-of-dicts.
    Strips trailing TERMINATE so the fence regex doesn't misalign.
    """
    text = re.sub(r"\bTERMINATE\b", "", text)
    candidates: list[str] = [m.group(1) for m in _FENCED_RE.finditer(text)]
    for cand in reversed(candidates):
        try:
            parsed = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, list) and parsed and all(isinstance(x, dict) for x in parsed):
            return [_normalize_call(d) for d in parsed]
    return None


# Scoring
def score_one(
    function_schemas: list[dict],
    model_output: list[dict],
    ground_truth: list[dict],
    category: str,
) -> dict:
    """Delegate to bfcl-eval's AST checker (aligned to other topologies)."""
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


def _communications_source(m: BaseMessage) -> str:
    kw = getattr(m, "additional_kwargs", None) or {}
    src = kw.get("source")
    if src:
        return src
    t = getattr(m, "type", None)
    return {"human": "user", "ai": "assistant", "tool": "tool"}.get(t, t or "?")


def _communications_to_record(m: BaseMessage) -> dict:
    content = getattr(m, "content", "") or ""
    if not isinstance(content, str):
        content = str(content)
    return {"source": _communications_source(m), "content": content}


def solve(instance: dict) -> dict:
    """Run the centralized team on one BFCL instance.

    Returns:
        {
            "model_output": canonical call list [{fn: {arg: val}}] or [],
            "raw":          manager's last message content,
            "messages":     list of {source, content} from every turn,
            "telemetry":    normalized 5-key token/call counts,
        }
    """
    compiled, _ = _build_graph()
    user_request = _flatten_user_request(instance["question"])
    schemas_text = json.dumps(instance["function"], indent=2)
    task = format_task(user_request, schemas_text)

    result = compiled.invoke(
        {"messages": [HumanMessage(content=task)], "turn_count": 0},
        config={"recursion_limit": MAX_TURNS * 4},
    )
    msgs = result.get("messages") or []
    rendered = [_communications_to_record(m) for m in msgs]

    manager_msgs = [r for r in rendered if r["source"] == "manager"]
    final = manager_msgs[-1]["content"] if manager_msgs else ""
    # Prefer manager's final fenced JSON; fall back to caller_worker's.
    model_output = extract_canonical(final)
    if model_output is None:
        caller_msgs = [r for r in rendered if r["source"] == "caller_worker"]
        if caller_msgs:
            model_output = extract_canonical(caller_msgs[-1]["content"])
    if model_output is None:
        # Final fallback: scan all messages in reverse.
        for r in reversed(rendered):
            mo = extract_canonical(r["content"] or "")
            if mo is not None:
                model_output = mo
                break
    return {
        "model_output": model_output or [],
        "raw": final,
        "messages": rendered,
        "telemetry": normalize(langchain_telemetry(msgs)),
    }


# Batch runner
def run_one(
    instance: dict,
    ground_truth: dict,
    category: str,
    out_dir: Path,
) -> dict:
    """Solve one BFCL instance via the manager-worker team and score the
    canonical output. Writes group-chat trace to out_dir/traces/<id>.txt."""
    iid = instance["id"]
    summary: dict = {"id": iid, "category": category}

    t0 = time.time()
    try:
        out = solve(instance)
    except Exception as e:
        summary["error"] = f"{type(e).__name__}: {e}"
        summary["stage"] = "solve"
        summary["solve_s"] = round(time.time() - t0, 1)
        return summary
    summary["solve_s"] = round(time.time() - t0, 1)

    summary["model_output"] = out.get("model_output") or []
    summary["n_messages"] = len(out.get("messages") or [])
    summary["tool_calls"] = len(summary["model_output"])
    summary.update(out.get("telemetry") or {})

    trace_path = out_dir / "traces" / f"{iid}.txt"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("w") as f:
        for m in out.get("messages") or []:
            src = m.get("source", "?")
            content = m.get("content", "")
            f.write(f"=== {str(src).upper()} ===\n{content}\n\n")

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
    out_dir = out_dir or (_default_root / "results" / "bfcl_centralized_langgraph")
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
        description="Centralized-topology BFCL runner (LangGraph manager/worker)."
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
        Path(__file__).resolve().parents[3] / "results" / "bfcl_centralized_langgraph"
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
