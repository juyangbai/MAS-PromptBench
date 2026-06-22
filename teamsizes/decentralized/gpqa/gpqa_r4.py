"""Decentralized debate topology specialized for GPQA-Diamond, LangGraph."""

# Config
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import re
import signal
import sys
import time
from collections import Counter
from pathlib import Path

from teamsizes.output_contracts import append_output_contract_from_path
from typing import Optional

from typing_extensions import TypedDict

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
)
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent

_REPO_ROOT = Path(__file__).resolve().parents[3]

# Shared telemetry.
_TOPO_ROOT = str(_REPO_ROOT)
if _TOPO_ROOT not in sys.path:
    sys.path.insert(0, _TOPO_ROOT)
from topologies.telemetry import langchain_telemetry, normalize  # noqa: E402


# Stall safeguards
# Per-row wall-clock cap using SIGALRM (main-thread only — batch runner
# iterates rows on main thread, so this is safe). Without this, 4 peers
# × 2 rounds × N tool loops can stall a single row for many minutes when
# one LLM call hangs or the model loops on calculator ERROR messages.
PER_ROW_TIMEOUT_S = 120
_MAX_TOOL_LOOPS = 4

# create_react_agent uses its own recursion_limit; give it enough headroom
# to cover the tool-loop budget (roughly 3x max_tool_loops to account for
# alternating AI/Tool messages).
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

N_AGENTS = int(os.environ.get("DECENTRALIZED_N_AGENTS", "4"))
N_ROUNDS = int(os.environ.get("DECENTRALIZED_N_ROUNDS", "2"))

_PROMPTS_DIR = _REPO_ROOT / "configs" / "prompts" / "decentralized" / "gpqa"


def _load_prompt(role: str) -> str:
    return append_output_contract_from_path((_PROMPTS_DIR / f"{role}.txt").read_text().strip(), __file__, role)


SYSTEM_PROMPT = _load_prompt("debater")


# Tool (LangChain @tool)
@tool
def calculator(expression: str) -> str:
    """Evaluate a numeric Python expression (arithmetic + math functions
    like sqrt, log, sin, pi, e). Example: calculator("(4/3) * pi * 2**3")
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
def _peer_injection(others_final: list[BaseMessage], question: str) -> HumanMessage:
    """Build the 'peers said X, Y, Z — revise if warranted' user message.

    others_final: list of the OTHER peers' final AIMessages from the
    previous round (content strings only, tool-call turns already resolved).
    """
    body = ["These are the final responses from other peer agents in the previous round:"]
    for i, m in enumerate(others_final):
        content = getattr(m, "content", "") or ""
        if not isinstance(content, str):
            content = str(content)
        body.append(f"\nPeer {i + 1}:\n```\n{content}\n```")
    body.append(
        "\nCompare their reasoning against your own. Revise your answer ONLY "
        "if a peer's reasoning concretely outweighs yours. Re-emit a single "
        "`Answer: <letter>` line at the end.\n\n"
        "Original question:\n" + question
    )
    return HumanMessage(content="\n".join(body))


# State + round node
class DebateState(TypedDict, total=False):
    contexts: list[list[BaseMessage]]       # per-peer message histories (sans system)
    round_finals: list[list[BaseMessage]]   # per-round final AIMessages (one per peer)
    round: int
    mcq: str


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
    mcq = state["mcq"]
    contexts = [list(c) for c in state["contexts"]]
    prev_finals = state.get("round_finals") or []

    this_round_finals: list[BaseMessage] = []

    for i in range(len(contexts)):
        ctx = contexts[i]
        if r > 0 and prev_finals:
            others = [prev_finals[-1][j] for j in range(len(contexts)) if j != i]
            ctx = ctx + [_peer_injection(others, mcq)]

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


# Output parsing (aligned to openai sibling)
_LETTERS = ["A", "B", "C", "D"]

# Strip markdown `**bold**` / backticks before matching — the 9B
# frequently emits `**Answer:** B` which otherwise breaks the regexes.
_MARKDOWN_STRIP_RE = re.compile(r"[*_`]+")
_ANSWER_RE = re.compile(
    r"\b(?:final\s+)?answer\b\s*[:\s]*\(?([A-D])\)?",
    re.IGNORECASE,
)
_OPTION_RE = re.compile(
    r"\b(?:option|choice)\b\s*(?:is)?\s*[:\s]*\(?([A-D])\)?",
    re.IGNORECASE,
)
_BARE_LETTER_RE = re.compile(
    r"(?:^|\n)\s*\(?([A-D])\)?\s*(?:[.\n]|$)",
    re.MULTILINE,
)


def extract_answer(text: str) -> str | None:
    """Return the MCQ letter from a peer's final output.

    Matches the cascade + markdown stripping used by single/independent/
    sequential/centralized gpqa so extracted letters are comparable
    across topologies.
    """
    cleaned = _MARKDOWN_STRIP_RE.sub("", text)
    for pattern in (_ANSWER_RE, _OPTION_RE, _BARE_LETTER_RE):
        matches = pattern.findall(cleaned)
        if matches:
            return matches[-1].upper()
    return None


# Aggregation (best-of-N: letter-majority vote, tie-break lowest index)
def best_of_n(per_peer_letters: list[Optional[str]]) -> Optional[str]:
    """Pick the most-common letter; tie-break by lowest peer index that
    voted for a top-count letter."""
    valid_with_idx = [
        (i, l) for i, l in enumerate(per_peer_letters) if l in _LETTERS
    ]
    if not valid_with_idx:
        return None
    counts = Counter(l for _, l in valid_with_idx)
    max_count = max(counts.values())
    # Walk peers in ascending index order; return first whose letter is a top.
    for i, l in valid_with_idx:
        if counts[l] == max_count:
            return l
    return None


# Format helpers
def format_mcq(question: str, choices: list[str]) -> str:
    """Build the user-facing MCQ string (4 choices, labeled A-D)."""
    assert len(choices) == 4, "GPQA expects exactly 4 choices."
    body = "\n".join(f"{_LETTERS[i]}) {choices[i]}" for i in range(4))
    return f"{question}\n\n{body}"


# Orchestration
def _init_contexts(n: int, mcq: str) -> list[list[BaseMessage]]:
    """Each peer's initial history = [HumanMessage(mcq)]. System prompt is
    injected by create_react_agent via its `prompt=` arg, not embedded here.
    """
    return [[HumanMessage(content=mcq)] for _ in range(n)]


def solve(question: str, choices: list[str]) -> dict:
    """Run the N-peer × R-round debate on one GPQA-style MCQ.

    Returns:
        {
            "answer":       best-of-N letter (A/B/C/D) or None,
            "per_peer":     [{peer, letter, raw}] — per-peer final
                            assistant messages + extracted letters,
            "all_contexts": raw LangChain message contexts (one per peer),
            "telemetry":    normalized 5-key token/call counts,
        }
    """
    compiled = _build_graph()
    mcq = format_mcq(question, choices)
    init_state: DebateState = {
        "contexts": _init_contexts(N_AGENTS, mcq),
        "round_finals": [],
        "round": 0,
        "mcq": mcq,
    }
    with _row_timeout_guard(PER_ROW_TIMEOUT_S):
        result = compiled.invoke(init_state)
    contexts = result.get("contexts") or []

    per_peer = []
    letters: list[Optional[str]] = []
    for i, ctx in enumerate(contexts):
        final_msg = _last_ai(ctx)
        final = getattr(final_msg, "content", "") or "" if final_msg else ""
        if not isinstance(final, str):
            final = str(final)
        letter = extract_answer(final)
        per_peer.append({"peer": i, "letter": letter, "raw": final})
        letters.append(letter)

    # Aggregate telemetry across all peers' full histories.
    flat_msgs: list[BaseMessage] = []
    for ctx in contexts:
        flat_msgs.extend(ctx)

    return {
        "answer": best_of_n(letters),
        "per_peer": per_peer,
        "all_contexts": contexts,
        "telemetry": normalize(langchain_telemetry(flat_msgs)),
    }


# Dataset loader (aligned to other gpqa topologies)
_HF_DATASET = "Idavidrein/gpqa"
_HF_CONFIG = "gpqa_diamond"
_HF_SPLIT = "train"


def _stable_row_id(row: dict, fallback_idx: int) -> str:
    """Stable id for a GPQA row — md5 hash of question text. Matches
    the other gpqa topologies so per-row diffs line up on the same id."""
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
    """Run the N-peer × R-round debate on every instance, compare the
    best-of-N letter vs gold, return aggregate summary + optionally
    write per-instance predictions to JSONL.

    Per-instance record shape:
        {id, question, choices, correct_letter,
         predicted_letter, correct,
         per_peer: [{peer, letter, raw_tail}] — one entry per debater,
         latency_s, error}
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
                out = {"answer": None, "per_peer": [], "all_contexts": []}
                error = f"{type(e).__name__}: {e}"
            latency_s = time.time() - t0

            pred = out["answer"]
            gold = inst["correct_letter"]
            is_correct = pred is not None and pred == gold
            if pred is not None:
                n_extracted += 1
            if is_correct:
                n_correct += 1

            # Per-peer record: keep letter + short tail of raw (last 300 chars)
            # so the JSONL stays manageable. Full contexts omitted.
            compact_per_peer = [
                {
                    "peer": p["peer"], "letter": p["letter"],
                    "raw_tail": (p["raw"] or "")[-300:],
                }
                for p in out.get("per_peer") or []
            ]
            rec = {
                "id": inst["id"],
                "question": inst["question"],
                "choices": inst["choices"],
                "correct_letter": gold,
                "predicted_letter": pred,
                "correct": is_correct,
                "per_peer": compact_per_peer,
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
                mark = "y" if is_correct else ("?" if pred is None else "n")
                peer_letters = [p["letter"] or "-" for p in compact_per_peer]
                print(
                    f"[{i + 1:>3}/{n}] {inst['id']} {mark}  "
                    f"pred={pred or '-'}  gold={gold}  "
                    f"peers=[{','.join(peer_letters)}]  "
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
            f"\n=== decentralized/GPQA-Diamond batch complete "
            f"(N={N_AGENTS} peers x R={N_ROUNDS} rounds) ===\n"
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
    print(f"\n=== Debate: N={N_AGENTS} peers x R={N_ROUNDS} rounds ===")
    for p in out["per_peer"]:
        snippet = (p["raw"] or "").strip().splitlines()
        tail = snippet[-3:] if len(snippet) > 3 else snippet
        print(f"  peer {p['peer']}: letter={p['letter']}   tail={' | '.join(tail)[:200]}")
    print(f"\n=== Best-of-N final answer: {out['answer']}  (expected: A) ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Decentralized-topology GPQA-Diamond runner (LangGraph debate)."
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
             "single/independent/sequential/centralized gpqa).",
    )
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()

    if not args.batch:
        _canned_demo()
        sys.exit(0)

    print(
        f"loading GPQA-Diamond from {_HF_DATASET} [{_HF_CONFIG}/{_HF_SPLIT}] "
        f"(base_url={VLLM_BASE_URL}, N={N_AGENTS} peers x R={N_ROUNDS} rounds) ..."
    )
    instances = load_instances(
        limit=args.limit, offset=args.offset,
        shuffle_seed=args.shuffle_seed, only=args.only,
    )
    if not instances:
        print("no instances loaded (check --limit/--offset/--only)", file=sys.stderr)
        sys.exit(1)
    print(f"  loaded {len(instances)} instance(s)")
    _default_out = (
        _REPO_ROOT / "results" / "gpqa_decentralized_langgraph" / "predictions.jsonl"
    )
    out_path = Path(args.out) if args.out else _default_out
    run_batch(instances, out_path=out_path)
    print(f"  predictions written to {out_path}")
