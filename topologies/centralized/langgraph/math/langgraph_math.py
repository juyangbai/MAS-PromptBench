"""Centralized topology specialized for competition MATH, LangGraph."""

# Config
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from topologies.output_contracts import append_output_contract_from_path
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

# Shared telemetry.
_REPO_ROOT = Path(__file__).resolve().parents[4]
_TOPO_ROOT = str(_REPO_ROOT)
if _TOPO_ROOT not in sys.path:
    sys.path.insert(0, _TOPO_ROOT)
from topologies.telemetry import langchain_telemetry, normalize  # noqa: E402


VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://n12:8000/v1")
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3.5-9B")

_PROMPTS_DIR = _REPO_ROOT / "configs" / "prompts" / "centralized" / "math"

# Same cap as AutoGen sibling's MaxMessageTermination(18).
MAX_TURNS = 18


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


WORKER_TOOLS = [calculator]


# Delegation tools (routing markers)
# The manager "calls" these to hand the floor to a specific worker. The
# body echoes the instructions, producing a ToolMessage the worker can
# read as context. The router after `manager_tools` inspects the name
# to route to the right worker node.
@tool("delegate_to_decomposer_worker")
def delegate_to_decomposer_worker(instructions: str) -> str:
    """Hand the next turn to the decomposer_worker. Use this when you need
    the problem broken into an ordered list of arithmetic sub-steps.

    Args:
        instructions: what you want the decomposer to plan this turn.
    """
    return instructions


@tool("delegate_to_computation_worker")
def delegate_to_computation_worker(instructions: str) -> str:
    """Hand the next turn to the computation_worker. Use this when a
    specific arithmetic sub-expression needs to be evaluated.

    Args:
        instructions: what the computation worker should compute this turn.
    """
    return instructions


@tool("delegate_to_verifier_worker")
def delegate_to_verifier_worker(instructions: str) -> str:
    """Hand the next turn to the verifier_worker. Use this when a
    candidate answer exists and needs to be re-derived via an
    alternative path to cross-check.

    Args:
        instructions: what the verifier should independently re-derive.
    """
    return instructions


DELEGATION_TOOLS = [
    delegate_to_decomposer_worker,
    delegate_to_computation_worker,
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


# Team
_MANAGER_TERMINATE_NUDGE = (
    "\n\nWhen you emit the final \\boxed{...} answer, immediately follow "
    "it with the literal string TERMINATE on its own line so the group-"
    "chat knows to stop.\n\n"
    "Delegation: when you want a specific worker to act, call the "
    "matching delegate_to_<worker> tool with clear instructions "
    "(instead of merely addressing them in free-form text). The three "
    "workers are: decomposer_worker, computation_worker, verifier_worker."
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
        ("decomposer_worker", WORKER_TOOLS),
        ("computation_worker", WORKER_TOOLS),
        ("verifier_worker", WORKER_TOOLS),
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
            "decomposer_worker": "decomposer_worker",
            "computation_worker": "computation_worker",
            "verifier_worker": "verifier_worker",
            "manager": "manager",
        },
    )
    for name, _ in worker_specs:
        graph.add_edge(name, "manager")

    return graph.compile(), ["manager"] + [n for n, _ in worker_specs]


# Output parsing
def extract_boxed(text: str) -> str | None:
    """Return the inner content of the LAST \boxed{...} in the text.

    Handles nested braces (e.g., \boxed{\frac{1}{2}}) via brace
    counting. Aligned to single/math's extractor.
    """
    marker = r"\boxed{"
    idx = text.rfind(marker)
    if idx < 0:
        return None
    start = idx + len(marker) - 1
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : i]
    return None  # unbalanced


def extract_answer(text: str) -> str | None:
    return extract_boxed(text)


# Scoring (aligned port of Hendrycks MATH is_equiv)
# Source: https://github.com/hendrycks/math/blob/main/modeling/math_equivalence.py
# Do not modify — this is the community-standard MATH scorer used by
# lm-evaluation-harness and published MATH results.


def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string


def _remove_right_units(string):
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def _strip_string(string):
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = _remove_right_units(string)
    string = string.replace("\\%", "")
    string = string.replace("\\%", "")
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]
    string = _fix_sqrt(string)
    string = string.replace(" ", "")
    string = _fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    string = _fix_a_slash_b(string)
    return string


def is_equiv(str1, str2, verbose: bool = False) -> bool:
    """Hendrycks MATH equivalence (official)."""
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False
    try:
        ss1 = _strip_string(str1)
        ss2 = _strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except Exception:
        return str1 == str2


def exact_match_score(pred: str, gold: str) -> float:
    return float(is_equiv(pred, gold))


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


def solve(problem: str) -> dict:
    """Run the centralized team on one MATH problem.

    Returns:
        {
            "answer":    inner content of the LAST \\boxed{...} or None,
            "raw":       manager's last message content,
            "messages":  list of {source, content} from every turn,
            "telemetry": normalized 5-key token/call counts,
        }
    """
    compiled, _ = _build_graph()
    result = compiled.invoke(
        {"messages": [HumanMessage(content=problem)], "turn_count": 0},
        config={"recursion_limit": MAX_TURNS * 4},
    )
    msgs = result.get("messages") or []
    rendered = [_communications_to_record(m) for m in msgs]

    manager_msgs = [r for r in rendered if r["source"] == "manager"]
    final = manager_msgs[-1]["content"] if manager_msgs else ""
    answer = extract_answer(final)
    if answer is None:
        # Fallback: scan messages in reverse for a boxed answer.
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


# Dataset loader
# qwedsacf/competition_math filtered to Precalculus / Level 5 (312 rows).
# Gold is extracted from the LAST \\boxed{...} in the `solution` column.
# IDs are stable MD5 of problem text for cross-topology parity.
_HF_DATASET = "qwedsacf/competition_math"
_HF_SPLIT = "train"
_SUBJECT = "Precalculus"
_LEVEL = "Level 5"


def load_instances(
    limit: int | None = None,
    offset: int = 0,
    only: list[str] | None = None,
) -> list[dict]:
    """Load the Precalculus / Level-5 subset of qwedsacf/competition_math."""
    import hashlib

    from datasets import load_dataset

    ds = load_dataset(_HF_DATASET)[_HF_SPLIT]
    rows: list[dict] = []
    for row in ds:
        if row.get("type") != _SUBJECT:
            continue
        if row.get("level") != _LEVEL:
            continue
        problem = (row.get("problem") or "").strip()
        solution = (row.get("solution") or "").strip()
        if not problem or not solution:
            continue
        gold = extract_boxed(solution)
        if gold is None:
            continue
        rid = "math_" + hashlib.md5(problem.encode("utf-8")).hexdigest()[:10]
        if only is not None and rid not in set(only):
            continue
        rows.append({
            "id": rid,
            "problem": problem,
            "answer": gold,
            "subject": row.get("type"),
            "level": row.get("level"),
            "raw": {
                "problem": problem,
                "solution": solution,
                "type": row.get("type"),
                "level": row.get("level"),
            },
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
    """Run `solve()` on every problem, compare boxed answer vs gold via
    Hendrycks `is_equiv`, return aggregate summary + optionally write
    per-instance predictions to JSONL.
    """
    per_instance: list[dict] = []
    n = len(instances)
    em_sum = 0.0
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
                out = solve(inst["problem"])
                error = None
            except Exception as e:
                out = {"answer": None, "raw": "", "messages": []}
                error = f"{type(e).__name__}: {e}"
            latency_s = time.time() - t0

            pred = out["answer"]
            gold = inst["answer"]
            if pred is not None:
                n_extracted += 1
                em = exact_match_score(pred, gold)
            else:
                em = 0.0
            em_sum += em

            rec_out = {
                "id": inst["id"],
                "problem": inst["problem"],
                "gold_answer": gold,
                "predicted_answer": pred,
                "em": em,
                "subject": inst.get("subject"),
                "level": inst.get("level"),
                "raw": out.get("raw") or "",
                "n_messages": len(out.get("messages") or []),
                "latency_s": round(latency_s, 2),
                **(out.get("telemetry") or {}),
                "error": error,
            }
            per_instance.append(rec_out)
            if out_f is not None:
                out_f.write(json.dumps(rec_out) + "\n")
                out_f.flush()

            if verbose:
                running_em = em_sum / (i + 1)
                mark = "OK" if em == 1.0 else ("?" if pred is None else "X")
                pred_disp = (pred or "-")[:30]
                gold_disp = (gold or "-")[:30]
                print(
                    f"[{i + 1:>3}/{n}] {inst['id'][:30]:<30} {mark}  "
                    f"em={em:.0f}  pred={pred_disp!r} gold={gold_disp!r}  "
                    f"EM={running_em:.3f} lat={latency_s:.1f}s",
                    flush=True,
                )
    finally:
        if out_f is not None:
            out_f.close()

    elapsed = time.time() - start
    summary = {
        "n": n,
        "n_extracted": n_extracted,
        "em_sum": em_sum,
        "em": (em_sum / n) if n else 0.0,
        "extracted_em": (em_sum / n_extracted) if n_extracted else 0.0,
        "total_s": round(elapsed, 1),
        "per_instance": per_instance,
    }
    if verbose:
        print(
            f"\n=== centralized/MATH batch complete ===\n"
            f"  n={summary['n']}  n_extracted={summary['n_extracted']}\n"
            f"  EM={summary['em']:.3f}  (on extracted only: {summary['extracted_em']:.3f})\n"
            f"  total_s={summary['total_s']}\n"
        )
    return summary


# Demo
def _canned_demo() -> None:
    problem = (
        "Compute the value of $\\frac{7!}{5!}$. "
        "Put your final answer inside \\boxed{}."
    )
    expected = "42"
    out = solve(problem)
    print(f"\n=== Extracted boxed answer: {out['answer']!r}  (expected: {expected!r}) ===")
    if out["answer"] is not None:
        em = exact_match_score(out["answer"], expected)
        print(f"=== EM (Hendrycks is_equiv): {em:.2f} ===")
    print(f"=== {len(out['messages'])} messages across the group chat ===")
    for m in out["messages"]:
        snippet = m["content"][:300].replace("\n", " ")
        print(f"  [{m['source']}] {snippet}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Centralized-topology MATH runner (LangGraph manager/worker)."
    )
    parser.add_argument(
        "--batch", action="store_true",
        help="Run the real MATH eval (else: one canned demo).",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()

    if not args.batch:
        _canned_demo()
        sys.exit(0)

    print(f"loading MATH from {_HF_DATASET} [{_HF_SPLIT}] ...")
    instances = load_instances(
        limit=args.limit, offset=args.offset, only=args.only,
    )
    if not instances:
        print("no instances loaded (check --limit/--offset/--only)", file=sys.stderr)
        sys.exit(1)
    print(f"  loaded {len(instances)} instance(s)")
    out_path = Path(args.out) if args.out else None
    if out_path is None:
        out_path = _REPO_ROOT / "results" / "math_centralized_langgraph" / "predictions.jsonl"
    run_batch(instances, out_path=out_path)
    print(f"  predictions written to {out_path}")
