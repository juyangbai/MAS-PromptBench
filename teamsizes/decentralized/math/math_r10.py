"""Decentralized debate topology specialized for competition MATH, LangGraph."""

# Config
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import re
import signal
import sys
import time
from pathlib import Path

from teamsizes.output_contracts import append_output_contract_from_path
from typing import Optional

from typing_extensions import TypedDict

import langchain_core.tools
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
)
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent

# Shared telemetry.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_TOPO_ROOT = str(_REPO_ROOT)
if _TOPO_ROOT not in sys.path:
    sys.path.insert(0, _TOPO_ROOT)
from topologies.telemetry import langchain_telemetry, normalize  # noqa: E402


# Stall safeguards
# Per-row wall-clock cap using SIGALRM (main-thread only — batch runner
# iterates rows on main thread, so this is safe). On symbolic MATH
# problems (matrices, trig) the calculator tool returns ERROR for
# non-numeric expressions and peers loop retrying variants.
PER_ROW_TIMEOUT_S = 120
_MAX_TOOL_LOOPS = 4  # was 6
# create_react_agent uses its own recursion_limit; give it enough
# headroom to cover the same tool-loop budget (roughly 3x max_tool_loops
# to account for alternating AI/Tool messages).
_RECURSION_LIMIT = _MAX_TOOL_LOOPS * 3


class _RowTimeout(Exception):
    """Raised when a single row exceeds PER_ROW_TIMEOUT_S."""


def _row_timeout_handler(signum, frame):
    raise _RowTimeout(f"row exceeded {PER_ROW_TIMEOUT_S}s")


@contextlib.contextmanager
def _row_timeout_guard(seconds: int):
    """Install SIGALRM for `seconds`; uninstall on exit regardless of
    outcome so timeouts in one row don't bleed into the next.

    Python's `signal` module only works on the main thread — when the
    batch runner executes us under a ThreadPoolExecutor worker, installing
    SIGALRM raises `ValueError: signal only works in main thread`. Skip
    the guard when not on the main thread; concurrency is the caller's
    responsibility there.
    """
    import threading
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    old = signal.signal(signal.SIGALRM, _row_timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://n12:8000/v1")
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3.5-9B")

N_AGENTS = int(os.environ.get("DECENTRALIZED_N_AGENTS", "10"))
N_ROUNDS = int(os.environ.get("DECENTRALIZED_N_ROUNDS", "2"))

_PROMPTS_DIR = _REPO_ROOT / "configs" / "prompts" / "decentralized" / "math"


def _load_prompt(role: str) -> str:
    return append_output_contract_from_path((_PROMPTS_DIR / f"{role}.txt").read_text().strip(), __file__, role)


SYSTEM_PROMPT = _load_prompt("debater")


# Tool (LangChain @tool)
@langchain_core.tools.tool
def calculator(expression: str) -> str:
    """Evaluate a numeric Python expression (arithmetic + math functions).

    Supports +, -, *, /, **, parentheses, and math functions (sqrt, log, log10,
    log2, exp, sin, cos, tan, asin, acos, atan, floor, ceil, pow, pi, e).
    Example: calculator("(4/3) * pi * 2**3")
    """
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


TOOLS = [calculator]


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


def _build_agent():
    """One react agent, reused across all peers + rounds. Each peer keeps
    its own message history; the agent is stateless."""
    return create_react_agent(model=_build_llm(), tools=TOOLS, prompt=SYSTEM_PROMPT)


# Peer injection (aligned template to openai sibling)
def _peer_injection(others_final: list[BaseMessage], problem: str) -> HumanMessage:
    body = ["These are the final solutions from other peer agents in the previous round:"]
    for i, m in enumerate(others_final):
        content = getattr(m, "content", "") or ""
        if not isinstance(content, str):
            content = str(content)
        body.append(f"\nPeer {i + 1}:\n```\n{content}\n```")
    body.append(
        "\nCompare their derivations and final boxed answers against your own. "
        "Revise your answer ONLY if a peer catches an error in your work or "
        "presents concretely stronger reasoning. Re-emit your final answer "
        "inside \\boxed{...} at the end.\n\nOriginal problem:\n" + problem
    )
    return HumanMessage(content="\n".join(body))


# State + round node
class DebateState(TypedDict, total=False):
    contexts: list[list[BaseMessage]]       # per-peer message histories (sans system)
    round_finals: list[list[BaseMessage]]   # per-round final AIMessages (one per peer)
    round: int
    problem: str


def _last_ai(msgs: list[BaseMessage]) -> BaseMessage | None:
    """Return the last AIMessage with non-empty content (no tool_calls)."""
    for m in reversed(msgs):
        if isinstance(m, AIMessage) and (m.content or "") and not getattr(m, "tool_calls", None):
            return m
    # Fallback: any AIMessage
    for m in reversed(msgs):
        if isinstance(m, AIMessage):
            return m
    return None


def _round_node(state: DebateState) -> dict:
    agent = _build_agent()
    r = int(state.get("round", 0))
    problem = state["problem"]
    contexts = [list(c) for c in state["contexts"]]
    prev_finals = state.get("round_finals") or []

    this_round_finals: list[BaseMessage] = []

    for i in range(len(contexts)):
        ctx = contexts[i]
        if r > 0 and prev_finals:
            others = [prev_finals[-1][j] for j in range(len(contexts)) if j != i]
            ctx = ctx + [_peer_injection(others, problem)]

        result = agent.invoke(
            {"messages": ctx},
            config={"recursion_limit": _RECURSION_LIMIT},
        )
        # result["messages"] includes the input ctx + all new AI/Tool messages.
        contexts[i] = result["messages"]

        final = _last_ai(contexts[i]) or AIMessage(content="")
        this_round_finals.append(final)

    return {
        "contexts": contexts,
        "round_finals": prev_finals + [this_round_finals],
        "round": r + 1,
    }


def _route(state: DebateState) -> str:
    return END if int(state.get("round", 0)) >= N_ROUNDS else "round"


def _build_graph():
    g = StateGraph(DebateState)
    g.add_node("round", _round_node)
    g.add_edge(START, "round")
    g.add_conditional_edges("round", _route, {"round": "round", END: END})
    return g.compile()


# Output parsing
def extract_boxed(text: str) -> str | None:
    """Return the inner content of the LAST \boxed{...} in the text.
    Handles nested braces via brace counting. Aligned to
    single/math's extractor."""
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
                return text[start + 1:i]
    return None


def extract_answer(text: str) -> str | None:
    return extract_boxed(text)


# Scoring (aligned port of Hendrycks MATH is_equiv)
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
                a, b = substr[0], substr[1]
                if b != "{":
                    if len(substr) > 2:
                        new_str += "{" + a + "}{" + b + "}" + substr[2:]
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        new_str += "{" + a + "}" + b + substr[2:]
                    else:
                        new_str += "{" + a + "}" + b
    return new_str


def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        return "\\frac{" + str(a) + "}{" + str(b) + "}"
    except:
        return string


def _remove_right_units(string):
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
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


# Aggregation (best-of-N via equivalence-bucketing majority)
def best_of_n(answers: list[str | None]) -> str | None:
    """Majority over Hendrycks-equivalence buckets — aligned to
    independent/math's aggregator. Tie-break by lowest index (the first
    bucket that reaches max length wins, which by construction contains
    the lowest-index answer)."""
    valid = [a for a in answers if a]
    if not valid:
        return None
    buckets: list[list[str]] = []
    for a in valid:
        for b in buckets:
            if is_equiv(a, b[0]):
                b.append(a)
                break
        else:
            buckets.append([a])
    best = max(buckets, key=len)
    return best[0]


# Back-compat alias matching openai sibling naming.
equiv_majority = best_of_n


# Orchestration
def _init_contexts(n: int, problem: str) -> list[list[BaseMessage]]:
    """Each peer's initial history = [HumanMessage(problem)]. System prompt
    is injected by create_react_agent via its `prompt=` arg, not embedded
    here."""
    return [[HumanMessage(content=problem)] for _ in range(n)]


def solve(problem: str) -> dict:
    compiled = _build_graph()
    init_state: DebateState = {
        "contexts": _init_contexts(N_AGENTS, problem),
        "round_finals": [],
        "round": 0,
        "problem": problem,
    }
    with _row_timeout_guard(PER_ROW_TIMEOUT_S):
        result = compiled.invoke(init_state)
    contexts = result.get("contexts") or []

    per_peer = []
    answers: list[str | None] = []
    for i, ctx in enumerate(contexts):
        final_msg = _last_ai(ctx)
        final = getattr(final_msg, "content", "") or "" if final_msg else ""
        if not isinstance(final, str):
            final = str(final)
        ans = extract_answer(final)
        per_peer.append({"peer": i, "answer": ans, "raw": final})
        if ans is not None:
            answers.append(ans)

    # Aggregate telemetry across all peers' full histories.
    flat_msgs: list[BaseMessage] = []
    for ctx in contexts:
        flat_msgs.extend(ctx)

    return {
        "answer": best_of_n(answers),
        "per_peer": per_peer,
        "all_contexts": contexts,
        "telemetry": normalize(langchain_telemetry(flat_msgs)),
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
    """Run `solve()` on every problem, compare debate boxed answer vs gold
    via Hendrycks `is_equiv`, return aggregate summary + optionally write
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
                out = {"answer": None, "per_peer": [], "all_contexts": [], "telemetry": {}}
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

            compact_per_peer = [
                {"peer": p["peer"], "answer": p["answer"],
                 "raw": (p.get("raw") or "")[:2000]}
                for p in out.get("per_peer") or []
            ]
            rec_out = {
                "id": inst["id"],
                "problem": inst["problem"],
                "gold_answer": gold,
                "predicted_answer": pred,
                "em": em,
                "subject": inst.get("subject"),
                "level": inst.get("level"),
                "per_peer": compact_per_peer,
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
                mark = "Y" if em == 1.0 else ("?" if pred is None else "N")
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
            f"\n=== decentralized/MATH-500 batch complete "
            f"(N={N_AGENTS}, R={N_ROUNDS}) ===\n"
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
    print(f"\n=== Debate: N={N_AGENTS} peers x R={N_ROUNDS} rounds ===")
    for p in out["per_peer"]:
        ans = p["answer"] if p["answer"] is not None else "(none)"
        print(f"  peer {p['peer']}: boxed={ans!r}")
    print(f"\n=== Majority-vote final answer: {out['answer']!r}  (expected: {expected!r}) ===")
    if out["answer"] is not None:
        em = exact_match_score(out["answer"], expected)
        print(f"=== EM (Hendrycks is_equiv): {em:.2f} ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Decentralized-topology MATH runner (LangGraph debate)."
    )
    parser.add_argument(
        "--batch", action="store_true",
        help="Run the real MATH-500 eval (else: one canned demo).",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()

    if not args.batch:
        _canned_demo()
        sys.exit(0)

    print(
        f"loading MATH-500 from {_HF_DATASET} [{_HF_SPLIT}] "
        f"(N={N_AGENTS}, R={N_ROUNDS}) ..."
    )
    instances = load_instances(
        limit=args.limit, offset=args.offset, only=args.only,
    )
    if not instances:
        print("no instances loaded (check --limit/--offset/--only)", file=sys.stderr)
        sys.exit(1)
    print(f"  loaded {len(instances)} instance(s)")
    out_path = Path(args.out) if args.out else None
    run_batch(instances, out_path=out_path)
    if out_path:
        print(f"  predictions written to {out_path}")
