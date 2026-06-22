"""Centralized topology specialized for GPQA-Diamond, LangGraph."""

# Config
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path

from topologies.output_contracts import append_output_contract_from_path
from typing import Annotated, Any, Optional

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

# Shared telemetry.
_REPO_ROOT = Path(__file__).resolve().parents[4]
_TOPO_ROOT = str(_REPO_ROOT)
if _TOPO_ROOT not in sys.path:
    sys.path.insert(0, _TOPO_ROOT)
from topologies.telemetry import langchain_telemetry, normalize  # noqa: E402


VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://n12:8000/v1")
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3.5-9B")

_PROMPTS_DIR = _REPO_ROOT / "configs" / "prompts" / "centralized" / "gpqa"

# Same cap as AutoGen sibling's MaxMessageTermination(16).
MAX_TURNS = 16

# Per-row wall-clock cap (kept for parity with AutoGen sibling; not
# currently enforced as a hard deadline in LangGraph — MAX_TURNS bounds
# the loop).
PER_ROW_TIMEOUT_S = 120


def _load_prompt(role: str) -> str:
    return append_output_contract_from_path((_PROMPTS_DIR / f"{role}.txt").read_text().strip(), __file__, role)


# Tools
@tool
def calculator(expression: str) -> str:
    """Evaluate a numeric Python expression (arithmetic + math functions).

    Supports +, -, *, /, **, parentheses, and math functions (sqrt, log,
    log10, log2, exp, sin, cos, tan, asin, acos, atan, floor, ceil, pow,
    pi, e). Example: calculator("(4/3) * pi * 2**3")
    """
    import math

    allowed = {
        k: getattr(math, k)
        for k in (
            "sqrt", "log", "log10", "log2", "exp",
            "sin", "cos", "tan", "asin", "acos", "atan",
            "floor", "ceil", "pow", "pi", "e",
        )
    }
    allowed["__builtins__"] = {}
    try:
        return str(eval(expression, allowed))
    except Exception as e:
        return f"ERROR: {e}"


# Delegation tools (routing markers)
# The manager "calls" these to hand the floor to a specific worker. The
# body echoes the instructions, producing a ToolMessage the worker can
# read as context. The router after `manager_tools` inspects the name
# to route to the right worker node.
@tool("delegate_to_analyzer_worker")
def delegate_to_analyzer_worker(instructions: str) -> str:
    """Hand the next turn to the analyzer_worker. Use this when you need
    the analyzer to identify scientific principles and derive each of
    the four options.

    Args:
        instructions: what you want the analyzer to analyze this turn.
    """
    return instructions


@tool("delegate_to_solver_worker")
def delegate_to_solver_worker(instructions: str) -> str:
    """Hand the next turn to the solver_worker. Use this when you want
    the solver to pick one letter + rationale given the analyzer's
    output.

    Args:
        instructions: what you want the solver to decide this turn.
    """
    return instructions


@tool("delegate_to_verifier_worker")
def delegate_to_verifier_worker(instructions: str) -> str:
    """Hand the next turn to the verifier_worker. Use this when you want
    the verifier to sanity-check the solver's letter against the
    analyzer's output.

    Args:
        instructions: what the verifier should verify this turn.
    """
    return instructions


DELEGATION_TOOLS = [
    delegate_to_analyzer_worker,
    delegate_to_solver_worker,
    delegate_to_verifier_worker,
]
DELEGATION_NAMES = {t.name for t in DELEGATION_TOOLS}

MANAGER_TOOLS = [calculator] + DELEGATION_TOOLS


# LLM
def _build_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=MODEL_ID,
        base_url=VLLM_BASE_URL,
        api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        temperature=0.2,
        top_p=0.9,
        seed=0,
        max_tokens=4096,
        extra_body={
            "repetition_penalty": 1.05,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )


# Manager system nudge
_MANAGER_TERMINATE_NUDGE = (
    "\n\nWhen you emit the final 'Answer: X' line, immediately follow it "
    "with the literal string TERMINATE on its own line so the group-chat "
    "knows to stop.\n\n"
    "Delegation: when you want a specific worker to act, call the "
    "matching delegate_to_<worker> tool with clear instructions "
    "(instead of merely addressing them in free-form text). The three "
    "workers are: analyzer_worker, solver_worker, verifier_worker."
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
    # us whether any delegation was requested.
    for m in reversed(state["messages"]):
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            for tc in m.tool_calls:
                name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
                if name in DELEGATION_NAMES:
                    return name.removeprefix("delegate_to_")
            # Only real-tool calls were made; loop back to manager.
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
        for m in new_msgs:
            if isinstance(m, AIMessage):
                _tag_source(m, name)
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
        ("analyzer_worker", [calculator]),
        ("solver_worker", [calculator]),
        ("verifier_worker", [calculator]),
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
            "analyzer_worker": "analyzer_worker",
            "solver_worker": "solver_worker",
            "verifier_worker": "verifier_worker",
            "manager": "manager",
        },
    )
    for name, _ in worker_specs:
        graph.add_edge(name, "manager")

    return graph.compile(), ["manager"] + [n for n, _ in worker_specs]


# Output parsing
_LETTERS = ["A", "B", "C", "D"]

# Strip markdown `**bold**` / backticks before regex matching — the 9B
# frequently emits `"**Answer:** B"` and the bare regexes miss the
# match without this. Same fix as single/sequential/independent gpqa.
_MARKDOWN_STRIP_RE = re.compile(r"[*_`]+")
# Primary: "Answer: X" / "Final answer: X"
_ANSWER_RE = re.compile(
    r"\b(?:final\s+)?answer\b\s*[:\s]*\(?([A-D])\)?",
    re.IGNORECASE,
)
# Fallback: "Option X" / "choice X" / "Option: X" / "correct option is C"
_OPTION_RE = re.compile(
    r"\b(?:option|choice)\b\s*(?:is)?\s*[:\s]*\(?([A-D])\)?",
    re.IGNORECASE,
)
# Fallback: bare A-D on its own line near the end.
_BARE_LETTER_RE = re.compile(
    r"(?:^|\n)\s*\(?([A-D])\)?\s*(?:[.\n]|$)",
    re.MULTILINE,
)


def extract_answer(text: str) -> str | None:
    """Return the MCQ letter from the manager's final output.

    Matches the cascade + markdown stripping used by single/sequential/
    independent/decentralized gpqa so resolve rates are byte-comparable.
    """
    cleaned = _MARKDOWN_STRIP_RE.sub("", text)
    for pattern in (_ANSWER_RE, _OPTION_RE, _BARE_LETTER_RE):
        matches = pattern.findall(cleaned)
        if matches:
            return matches[-1].upper()
    return None


# Format helpers
def format_mcq(question: str, choices: list[str]) -> str:
    """Build the user-facing MCQ string (4 choices, labeled A-D)."""
    assert len(choices) == 4, "GPQA expects exactly 4 choices."
    body = "\n".join(f"{_LETTERS[i]}) {choices[i]}" for i in range(4))
    return f"{question}\n\n{body}"


# Orchestration
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


def solve(question: str, choices: list[str]) -> dict:
    """Run the centralized team on one GPQA-style MCQ.

    Returns:
        {
            "answer":    final letter A/B/C/D or None,
            "raw":       manager's last message content,
            "messages":  list of {source, content} from every turn,
            "telemetry": normalized 5-key token/call counts,
        }
    """
    compiled, _ = _build_graph()
    mcq = format_mcq(question, choices)
    result = compiled.invoke(
        {"messages": [HumanMessage(content=mcq)], "turn_count": 0},
        config={"recursion_limit": MAX_TURNS * 4},
    )
    msgs = result.get("messages") or []
    rendered = [_communications_to_record(m) for m in msgs]

    manager_msgs = [r for r in rendered if r["source"] == "manager"]
    final = manager_msgs[-1]["content"] if manager_msgs else ""
    answer = extract_answer(final)
    if answer is None:
        for r in reversed(rendered):
            a = extract_answer(r["content"] or "")
            if a is not None:
                answer = a
                break
    return {
        "answer": answer,
        "raw": final,
        "messages": rendered,
        "telemetry": normalize(langchain_telemetry(msgs)),
    }


# Dataset loader (aligned to single/independent/sequential gpqa)
_HF_DATASET = "Idavidrein/gpqa"
_HF_CONFIG = "gpqa_diamond"
_HF_SPLIT = "train"


def _stable_row_id(row: dict, fallback_idx: int) -> str:
    """Stable id for a GPQA row — md5 hash of question text. Matches
    single/independent/sequential gpqa so per-row comparisons line up
    on the same id across topologies."""
    q = (row.get("Question") or "").strip()
    if q:
        return "gpqa_" + hashlib.md5(q.encode("utf-8")).hexdigest()[:10]
    return f"gpqa_idx_{fallback_idx}"


def load_instances(
    limit: int | None = None,
    offset: int = 0,
    only: list[str] | None = None,
    shuffle_seed: int = 0,
) -> list[dict]:
    """Load GPQA-Diamond rows with 4 choices shuffled DETERMINISTICALLY
    per row (`Random(f"{shuffle_seed}|{row_id}")`). Aligned to
    the other gpqa topologies so `correct_letter` matches for each row
    id across topologies at the same `shuffle_seed`.
    """
    from datasets import load_dataset

    ds = load_dataset(_HF_DATASET, _HF_CONFIG)[_HF_SPLIT]
    rows: list[dict] = []
    for i, row in enumerate(ds):
        rid = _stable_row_id(row, i)
        if only is not None and rid not in set(only):
            continue
        correct = (row.get("Correct Answer") or "").strip()
        incorrects = [(row.get(f"Incorrect Answer {k}") or "").strip() for k in (1, 2, 3)]
        if not correct or any(not x for x in incorrects):
            continue
        four = [correct, *incorrects]
        rng = random.Random(f"{shuffle_seed}|{rid}")
        indices = list(range(4))
        rng.shuffle(indices)
        shuffled = [four[j] for j in indices]
        correct_slot = indices.index(0)
        rows.append({
            "id": rid,
            "question": (row.get("Question") or "").strip(),
            "choices": shuffled,
            "correct_letter": _LETTERS[correct_slot],
            "raw": dict(row),
        })
    rows = rows[offset:]
    if limit is not None:
        rows = rows[:limit]
    return rows


# Batch eval
def run_batch(
    instances: list[dict],
    out_path: Path | None = None,
    verbose: bool = True,
) -> dict:
    """Run the 4-agent centralized team on every instance, compare
    manager-emitted letter vs gold, return aggregate summary +
    optionally write per-instance predictions to JSONL.

    Per-instance record shape:
        {id, question, choices, correct_letter,
         predicted_letter, correct,
         raw, n_messages, latency_s, error}
    """
    per_instance: list[dict] = []
    n = len(instances)
    n_correct = 0
    n_extracted = 0
    start = time.time()

    out_f = None
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_f = open(out_path, "w")

    try:
        for i, inst in enumerate(instances):
            t0 = time.time()
            try:
                out = solve(inst["question"], inst["choices"])
                error = None
            except Exception as e:
                out = {"answer": None, "raw": "", "messages": [], "telemetry": {}}
                error = f"{type(e).__name__}: {e}"
            latency_s = time.time() - t0

            pred = out["answer"]
            gold = inst["correct_letter"]
            is_correct = pred is not None and pred == gold
            if pred is not None:
                n_extracted += 1
            if is_correct:
                n_correct += 1

            rec: dict[str, Any] = {
                "id": inst["id"],
                "question": inst["question"],
                "choices": inst["choices"],
                "correct_letter": gold,
                "predicted_letter": pred,
                "correct": is_correct,
                "raw": out.get("raw") or "",
                "n_messages": len(out.get("messages") or []),
                "latency_s": round(latency_s, 2),
                **(out.get("telemetry") or {}),
                "error": error,
            }
            per_instance.append(rec)
            if out_f is not None:
                out_f.write(json.dumps(rec) + "\n")
                out_f.flush()

            if verbose:
                running_acc = n_correct / (i + 1)
                mark = "+" if is_correct else ("?" if pred is None else "-")
                print(
                    f"[{i + 1:>3}/{n}] {inst['id']} {mark}  "
                    f"pred={pred or '-'}  gold={gold}  "
                    f"msgs={rec['n_messages']}  "
                    f"acc={running_acc:.3f}  lat={latency_s:.1f}s",
                    flush=True,
                )
    finally:
        if out_f is not None:
            out_f.close()

    elapsed = time.time() - start
    summary = {
        "n": n,
        "n_extracted": n_extracted,
        "n_correct": n_correct,
        "accuracy": (n_correct / n) if n else 0.0,
        "extracted_acc": (n_correct / n_extracted) if n_extracted else 0.0,
        "total_s": round(elapsed, 1),
        "per_instance": per_instance,
    }
    if verbose:
        print(
            f"\n=== centralized/GPQA-Diamond batch complete ===\n"
            f"  n={summary['n']}  n_extracted={summary['n_extracted']}  "
            f"n_correct={summary['n_correct']}\n"
            f"  accuracy={summary['accuracy']:.3f}  "
            f"extracted_acc={summary['extracted_acc']:.3f}  "
            f"total_s={summary['total_s']}\n"
        )
    return summary


# Demo
def _canned_demo() -> None:
    question = (
        "A circular wire loop of radius R carries a steady current I. "
        "What is the magnitude of the magnetic field at the geometric center "
        "of the loop? (mu_0 is the vacuum permeability.)"
    )
    choices = [
        "mu_0 * I / (2 * R)",
        "mu_0 * I / (4 * pi * R)",
        "mu_0 * I / R",
        "mu_0 * I / (pi * R)",
    ]
    out = solve(question, choices)
    print(f"\n=== Extracted answer: {out['answer']}  (expected: A) ===")
    print(f"=== {len(out['messages'])} messages across the group chat ===")
    for m in out["messages"]:
        snippet = m["content"][:400].replace("\n", " ")
        print(f"  [{m['source']}] {snippet}")


# CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Centralized-topology GPQA-Diamond runner (LangGraph)."
    )
    parser.add_argument(
        "--batch", action="store_true",
        help="Run the real GPQA-Diamond eval (else: one canned MCQ demo).",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument(
        "--shuffle-seed", type=int, default=0,
        help="Per-row choice-shuffling seed (default 0 — matches "
             "single/independent/sequential gpqa for cross-topology parity).",
    )
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()

    if not args.batch:
        _canned_demo()
        sys.exit(0)

    print(
        f"loading GPQA-Diamond from {_HF_DATASET} [{_HF_CONFIG}/{_HF_SPLIT}] "
        f"(base_url={VLLM_BASE_URL}) ..."
    )
    instances = load_instances(
        limit=args.limit, offset=args.offset,
        shuffle_seed=args.shuffle_seed, only=args.only,
    )
    if not instances:
        print("no instances loaded (check --limit/--offset/--only)", file=sys.stderr)
        sys.exit(1)
    print(f"  loaded {len(instances)} instance(s)")
    _default_out = str(
        Path(__file__).resolve().parents[4]
        / "results" / "gpqa_centralized_langgraph" / "predictions.jsonl"
    )
    out_path = Path(args.out) if args.out else Path(_default_out)
    run_batch(instances, out_path=out_path)
    if out_path:
        print(f"  predictions written to {out_path}")
