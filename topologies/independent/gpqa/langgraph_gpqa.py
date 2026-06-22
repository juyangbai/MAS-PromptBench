"""Independent topology specialized for GPQA-Diamond."""

# Config
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import operator
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

from topologies.output_contracts import append_output_contract_from_path
from typing import Annotated

from langchain_core.tools import tool
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
from typing_extensions import TypedDict


VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://lai:8001/v1")
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3.5-9B")

# Number of parallel replicas. Seeds are 0 .. N_AGENTS-1.
N_AGENTS = int(os.environ.get("INDEPENDENT_N_AGENTS", "4"))

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PROMPT_PATH = (
    _REPO_ROOT / "configs" / "prompts" / "independent" / "gpqa" / "solver.txt"
)

# Mirror single/gpqa: prompt is checked into git; fail fast if missing.
SYSTEM_PROMPT = append_output_contract_from_path(_PROMPT_PATH.read_text().strip(), __file__, _PROMPT_PATH.stem)


# Tools
@tool
def calculator(expression: str) -> str:
    """Evaluate a numeric Python expression (arithmetic + math functions).

    Supports +, -, *, /, **, parentheses, and math functions (sqrt, log, log10,
    log2, exp, sin, cos, tan, asin, acos, atan, floor, ceil, pow, pi, e).
    Example: calculator("(4/3) * pi * 2**3")
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


# Agent
def format_prompt(question: str, choices: list[str]) -> str:
    """Build the user-facing MCQ prompt. Expects exactly 4 choices."""
    assert len(choices) == 4, "GPQA expects exactly 4 choices."
    letters = ["A", "B", "C", "D"]
    body = "\n".join(f"{letters[i]}) {choices[i]}" for i in range(4))
    return f"{question}\n\n{body}"


def _build_one_agent(seed: int):
    """Build one replica's react agent, seeded differently from its siblings.

    The seed is the ONLY per-replica difference: same model, tools, prompt,
    temperature, top_p, and repetition_penalty. This keeps the ensemble
    reproducible (same seed -> same output) while giving each replica a
    distinct sample from the posterior.
    """
    llm = ChatOpenAI(
        model=MODEL_ID,
        base_url=VLLM_BASE_URL,
        api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        max_tokens=4096,
        # House default sampling. Greedy (temp=0) would collapse all N
        # replicas to the same answer, defeating the ensemble.
        temperature=0.2,
        top_p=0.9,
        seed=seed,
        extra_body={
            "repetition_penalty": 1.05,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    return create_react_agent(model=llm, tools=[calculator], prompt=SYSTEM_PROMPT)


# Output Parsing
import re  # noqa: E402  (kept near the parser, not the top, for readability)

# `_MARKDOWN_STRIP_RE` pre-cleans `**bold**` / `*italic*` wrappers before
# regex matching — Qwen3.5-9B frequently emits `**Answer:** B` or
# `correct option is **A**`, and without stripping the `**` the letter
# regexes miss the match.
_MARKDOWN_STRIP_RE = re.compile(r"[*_`]+")
# Primary: "Answer: A", "Final answer: (B)", "**Answer: C**", etc.
_ANSWER_RE = re.compile(
    r"\b(?:final\s+)?answer\b\s*[:\s]*\(?([A-D])\)?", re.IGNORECASE
)
# Fallback 1: "Option A", "choice B", "correct option is C", "Option: B"
_OPTION_RE = re.compile(
    r"\b(?:option|choice)\b\s*(?:is)?\s*[:\s]*\(?([A-D])\)?", re.IGNORECASE
)
# Fallback 2: a bare "A" / "A)" / "(A)" sitting on its own at end-of-line
_BARE_LETTER_RE = re.compile(
    r"(?:^|\n)\s*\(?([A-D])\)?\s*(?:[.\n]|$)", re.MULTILINE
)


def strip_thinking(text: str) -> str:
    """Cut everything up through the last </think> tag (Qwen3 convention)."""
    index = text.lower().rfind("</think>")
    if index >= 0:
        text = text[index + len("</think>"):]
    return text.strip()


def extract_answer(text: str) -> str | None:
    """Return the MCQ letter from the model's final response.

    Matches the 3-pattern cascade used by single/sequential/centralized/
    decentralized gpqa so extracted letters are comparable. Strips
    markdown emphasis first so `**Answer:** B` works the same as
    `Answer: B`. The LAST match per pattern wins (models revise earlier
    letter mentions during chain-of-thought).
    """
    cleaned = _MARKDOWN_STRIP_RE.sub("", text)
    for pattern in (_ANSWER_RE, _OPTION_RE, _BARE_LETTER_RE):
        matches = pattern.findall(cleaned)
        if matches:
            return matches[-1].upper()
    return None


def majority_vote(answers: list[dict]) -> str | None:
    """Majority vote over per-replica letter answers.

    Ignores replicas whose letter is None (no answer extractable). Ties
    broken by first occurrence in the `answers` list (i.e., lowest agent
    index among tied letters).
    """
    letters = [a["answer"] for a in answers if a.get("answer") is not None]
    if not letters:
        return None
    counts = Counter(letters)
    top_count = max(counts.values())
    # Preserve first-occurrence order among equally-common letters.
    for letter in letters:
        if counts[letter] == top_count:
            return letter
    return None  # unreachable


# Graph
class State(TypedDict):
    """Graph state.

    `answers` uses `operator.add` as its reducer so the N concurrent agent
    nodes can each emit a single-element list that gets concatenated into
    the final aggregate. `prompt` / `question` / `choices` are carried
    through so each agent gets the full MCQ context.
    """

    question: str
    choices: list[str]
    prompt: str
    answers: Annotated[list[dict], operator.add]


class AgentInput(TypedDict):
    """Per-replica input (carried via Send)."""

    agent_id: int
    seed: int
    prompt: str


# Stall safeguards
# Per-row wall-clock cap: if the ensemble's whole fan-out/fan-in takes
# longer than this, asyncio.wait_for raises TimeoutError, the batch
# runner's except-clause logs it, and we move to the next row. Without
# this, one pathological row can chew 15+ min of GPU time.
# recursion_limit was 25 — too lenient for a 4-way parallel ensemble
# where each replica's runaway compounds under vLLM continuous batching.
PER_ROW_TIMEOUT_S = 120
_RECURSION_LIMIT = 15


async def _run_replica(inp: AgentInput) -> dict:
    """Run one replica's react agent and return its extracted answer."""
    agent = _build_one_agent(seed=inp["seed"])
    result = await agent.ainvoke(
        {"messages": [("user", inp["prompt"])]},
        config={"recursion_limit": _RECURSION_LIMIT},
    )
    # Clean Qwen3 thinking blocks out of every AI message.
    for msg in result["messages"]:
        if msg.type == "ai" and isinstance(msg.content, str):
            msg.content = strip_thinking(msg.content)
    final = result["messages"][-1].content
    return {
        "answers": [
            {
                "agent_id": inp["agent_id"],
                "seed": inp["seed"],
                "answer": extract_answer(final),
                "raw": final,
                "messages": result["messages"],
            }
        ]
    }


def _fan_out(state: State) -> list[Send]:
    """Dispatch N parallel replicas, each with a distinct seed."""
    return [
        Send(
            f"agent_{i}",
            {"agent_id": i, "seed": i, "prompt": state["prompt"]},
        )
        for i in range(N_AGENTS)
    ]


def build_graph() -> StateGraph:
    """Fan-out -> N independent react agents -> fan-in.

    No edges between agent nodes: each sees only its own AgentInput slice.
    """
    graph = StateGraph(State)
    for i in range(N_AGENTS):
        graph.add_node(f"agent_{i}", _run_replica)
    graph.add_conditional_edges(START, _fan_out)
    graph.add_edge([f"agent_{i}" for i in range(N_AGENTS)], END)
    return graph


# Orchestration
def solve(question: str, choices: list[str]) -> dict:
    """Run the ensemble on one GPQA-style MCQ.

    Returns:
        {
            "answer":   majority-vote letter (A/B/C/D) or None,
            "per_agent": list of {agent_id, seed, answer, raw, messages},
            "votes":    Counter of letter -> count across replicas,
        }
    """
    compiled = build_graph().compile()
    prompt = format_prompt(question, choices)

    async def _run():
        return await asyncio.wait_for(
            compiled.ainvoke({
                "question": question,
                "choices": choices,
                "prompt": prompt,
                "answers": [],
            }),
            timeout=PER_ROW_TIMEOUT_S,
        )

    result = asyncio.run(_run())
    per_agent = sorted(result["answers"], key=lambda a: a["agent_id"])
    letters = [a["answer"] for a in per_agent if a.get("answer") is not None]
    return {
        "answer": majority_vote(per_agent),
        "per_agent": per_agent,
        "votes": dict(Counter(letters)),
    }


# Dataset loader (aligned to single/gpqa for cross-topology parity)
_LETTERS = ["A", "B", "C", "D"]
_HF_DATASET = "Idavidrein/gpqa"
_HF_CONFIG = "gpqa_diamond"
_HF_SPLIT = "train"


def _stable_row_id(row: dict, fallback_idx: int) -> str:
    """Stable id for a GPQA row — hash of question text (matches
    single/gpqa's id scheme so per-row comparisons across topologies
    line up on the same id)."""
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
    """Load GPQA-Diamond rows and emit instances with 4 choices shuffled
    DETERMINISTICALLY per row (`Random(f"{shuffle_seed}|{row_id}")`). Same
    algorithm as single/gpqa — identical `shuffle_seed` produces identical
    choice orderings, so `correct_letter` is identical across topologies
    for each row id.

    Returns list of dicts:
        {id, question, choices, correct_letter, raw}
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
    _propagate_errors: bool = False,
) -> dict:
    """Run the N-replica ensemble on every instance, compare majority-vote
    letter vs gold, return an aggregate summary + optionally write
    predictions JSONL.

    Per-instance JSONL record shape:
        {
            id, question, choices, correct_letter,
            predicted_letter, correct, votes,
            per_agent: [{agent_id, seed, answer, raw}],
            latency_s, error
        }
    (Omit the full `messages` list per agent to keep the JSONL compact.)
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
                if _propagate_errors:
                    raise
                out = {"answer": None, "per_agent": [], "votes": {}}
                error = f"{type(e).__name__}: {e}"
            latency_s = time.time() - t0

            pred = out["answer"]
            gold = inst["correct_letter"]
            is_correct = pred is not None and pred == gold
            if pred is not None:
                n_extracted += 1
            if is_correct:
                n_correct += 1

            compact_per_agent = [
                {
                    "agent_id": a["agent_id"], "seed": a["seed"],
                    "answer": a["answer"], "raw": a["raw"],
                }
                for a in out["per_agent"]
            ]
            telem = normalize(langchain_ensemble_telemetry(out.get("per_agent") or []))
            rec = {
                "id": inst["id"],
                "question": inst["question"],
                "choices": inst["choices"],
                "correct_letter": gold,
                "predicted_letter": pred,
                "correct": is_correct,
                "votes": out["votes"],
                "per_agent": compact_per_agent,
                "latency_s": round(latency_s, 2),
                **telem,
                "error": error,
            }
            per_instance.append(rec)
            if out_f is not None:
                out_f.write(json.dumps(rec) + "\n")
                out_f.flush()

            if verbose:
                running_acc = n_correct / (i + 1)
                mark = "✓" if is_correct else ("?" if pred is None else "✗")
                votes_str = ",".join(f"{k}={v}" for k, v in sorted((out["votes"] or {}).items()))
                print(
                    f"[{i + 1:>3}/{n}] {inst['id']} {mark}  "
                    f"pred={pred or '-'}  gold={gold}  votes=[{votes_str}]  "
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
            f"\n=== independent/GPQA-Diamond batch complete (N={N_AGENTS}) ===\n"
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
    print(f"\n=== Ensemble ({N_AGENTS} replicas) vote: {out['votes']}")
    print(f"=== Majority-vote answer: {out['answer']}  (expected: A) ===\n")
    for a in out["per_agent"]:
        print(f"--- agent_{a['agent_id']} (seed {a['seed']}) -> {a['answer']!r} ---")

def run_one(instance: dict, out_dir: Path | None = None) -> dict:
    """Single-instance entrypoint for `concurrent_runner.py`.

    Calls `run_batch([instance], _propagate_errors=True)` so any transient
    exception (APIConnectionError, TimeoutError, BadRequestError "Unterminated
    string", etc.) bubbles up to the runner's retry-with-backoff wrapper
    instead of being swallowed into an `error` field on a "successful" row.
    """
    summary = run_batch([instance], out_path=None, verbose=False, _propagate_errors=True)
    return summary["per_instance"][0]



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Independent-topology GPQA-Diamond runner (LangGraph ensemble)."
    )
    parser.add_argument(
        "--batch", action="store_true",
        help="Run the real GPQA-Diamond eval (else: one canned MCQ demo).",
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="Max rows to run (batch mode).")
    parser.add_argument("--offset", type=int, default=0,
                        help="Row offset into the split (batch mode).")
    parser.add_argument("--shuffle-seed", type=int, default=0,
                        help="Seed for per-row choice shuffling (default 0 — "
                             "matches single/gpqa for cross-topology parity).")
    parser.add_argument("--out", type=str, default=None,
                        help="Per-instance JSONL output path (batch mode).")
    parser.add_argument("--only", nargs="*", default=None,
                        help="Restrict to specific instance ids (batch mode).")
    args = parser.parse_args()

    if not args.batch:
        _canned_demo()
        sys.exit(0)

    print(
        f"loading GPQA-Diamond from {_HF_DATASET} [{_HF_CONFIG}/{_HF_SPLIT}] "
        f"(N_AGENTS={N_AGENTS}) ..."
    )
    instances = load_instances(
        limit=args.limit, offset=args.offset,
        shuffle_seed=args.shuffle_seed, only=args.only,
    )
    if not instances:
        print("no instances loaded (check --limit/--offset/--only)", file=sys.stderr)
        sys.exit(1)
    print(f"  loaded {len(instances)} instance(s)")
    out_path = Path(args.out) if args.out else None
    run_batch(instances, out_path=out_path)
    if out_path:
        print(f"  predictions written to {out_path}")
