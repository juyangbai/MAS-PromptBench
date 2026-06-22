"""Centralized topology specialized for GPQA-Diamond, AutoGen."""

# Config
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path

from topologies.output_contracts import append_output_contract_from_path
from typing import Sequence

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage
from autogen_agentchat.teams import SelectorGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Shared telemetry.
_TOPO_ROOT = str(Path(__file__).resolve().parents[4])
if _TOPO_ROOT not in sys.path:
    sys.path.insert(0, _TOPO_ROOT)
from topologies.telemetry import autogen_telemetry, normalize  # noqa: E402


VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://lai:8001/v1")
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3.5-9B")

_REPO_ROOT = Path(__file__).resolve().parents[4]
_PROMPTS_DIR = _REPO_ROOT / "configs" / "prompts" / "centralized" / "gpqa"


def _load_prompt(role: str) -> str:
    return append_output_contract_from_path((_PROMPTS_DIR / f"{role}.txt").read_text().strip(), __file__, role)


# Tools
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


# LLM
def _build_client() -> OpenAIChatCompletionClient:
    """Build an OpenAI-compatible client pointed at our local vLLM.

    vLLM doesn't advertise its own model_info, so we pass the minimum
    `model_info` AutoGen needs (tool/function calling yes; vision no).
    `extra_body` threads Qwen3-specific sampling params (repetition
    penalty, and enable_thinking=False so the model doesn't burn its
    token budget inside a <think> block).
    """
    return OpenAIChatCompletionClient(
        model=MODEL_ID,
        base_url=VLLM_BASE_URL,
        api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        model_info={
            "vision": False,
            "function_calling": True,
            "json_output": True,
            "family": "qwen",
            "structured_output": False,
        },
        temperature=0.2,
        top_p=0.9,
        seed=0,
        max_tokens=2048,
        extra_body={
            "repetition_penalty": 1.05,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )


# Team
_MANAGER_TERMINATE_NUDGE = (
    "\n\nWhen you emit the final 'Answer: X' line, immediately follow it "
    "with the literal string TERMINATE on its own line so the group-chat "
    "knows to stop."
)


def build_team() -> SelectorGroupChat:
    client = _build_client()

    manager = AssistantAgent(
        "manager",
        description="Coordinator that delegates to 3 workers and synthesizes the final letter.",
        model_client=client,
        system_message=_load_prompt("manager") + _MANAGER_TERMINATE_NUDGE,
        tools=[calculator],
    )

    analyzer_worker = AssistantAgent(
        "analyzer_worker",
        description="Analyzes scientific principles and derives each option.",
        model_client=client,
        system_message=_load_prompt("analyzer_worker"),
        tools=[calculator],
    )

    solver_worker = AssistantAgent(
        "solver_worker",
        description="Picks one letter + rationale given the manager's instruction and analysis.",
        model_client=client,
        system_message=_load_prompt("solver_worker"),
        tools=[calculator],
    )

    verifier_worker = AssistantAgent(
        "verifier_worker",
        description="Sanity-checks the solver's letter against the analyzer's output.",
        model_client=client,
        system_message=_load_prompt("verifier_worker"),
        tools=[calculator],
    )

    # Force manager-routing: after any worker speaks, the manager MUST be
    # the next speaker (so workers never chain turns with each other).
    # When the last message is already the manager's, let the SelectorGroupChat's
    # default LLM-based selector pick the next worker (or end).
    def _selector_func(messages: Sequence[BaseAgentEvent | BaseChatMessage]) -> str | None:
        if not messages:
            return manager.name
        if messages[-1].source != manager.name:
            return manager.name
        return None  # let default selector pick

    selector_prompt = (
        "You are coordinating a 4-agent team on a multiple-choice science question.\n"
        "Select the next agent to act.\n\n{roles}\n\n"
        "Conversation so far:\n{history}\n\n"
        "Pick exactly one agent from {participants}."
    )

    termination = TextMentionTermination("TERMINATE") | MaxMessageTermination(_MAX_MESSAGES)

    return SelectorGroupChat(
        [manager, analyzer_worker, solver_worker, verifier_worker],
        model_client=client,
        termination_condition=termination,
        selector_prompt=selector_prompt,
        selector_func=_selector_func,
        allow_repeated_speaker=True,
    )


# Stall safeguards
# Per-row wall-clock cap + tighter MaxMessageTermination. Without these
# the manager/worker loop can spiral on ambiguous rows (manager keeps
# asking workers to re-verify), burning >3 min per row.
PER_ROW_TIMEOUT_S = 120
_MAX_MESSAGES = 16  # was 20


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
async def solve_async(question: str, choices: list[str]) -> dict:
    """Run the centralized team on one GPQA-style MCQ.

    Returns:
        {
            "answer":   final letter A/B/C/D or None,
            "raw":      manager's last message (contains 'Answer: X TERMINATE'),
            "messages": list of {source, content} from all agents' turns,
        }
    """
    team = build_team()
    mcq = format_mcq(question, choices)
    result = await asyncio.wait_for(team.run(task=mcq), timeout=PER_ROW_TIMEOUT_S)
    # `result.messages` is a list of BaseChatMessage; normalize to simple dicts.
    messages = [
        {
            "source": getattr(m, "source", None),
            "content": getattr(m, "content", None)
            if isinstance(getattr(m, "content", None), str)
            else str(getattr(m, "content", "")),
        }
        for m in result.messages
    ]
    # Find the manager's last message — that's where 'Answer: X' lives.
    manager_msgs = [m for m in messages if m["source"] == "manager"]
    final = manager_msgs[-1]["content"] if manager_msgs else ""
    return {
        "answer": extract_answer(final),
        "raw": final,
        "messages": messages,
        "telemetry": normalize(autogen_telemetry(result)),
    }


def solve(question: str, choices: list[str]) -> dict:
    return asyncio.run(solve_async(question, choices))


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
    (Full message transcripts are not persisted — AutoGen group-chat
    transcripts for a single GPQA row can balloon to multi-KB per row.
    The manager's final content is preserved in `raw` for auditing.)
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
                out = {"answer": None, "raw": "", "messages": []}
                error = f"{type(e).__name__}: {e}"
            latency_s = time.time() - t0

            pred = out["answer"]
            gold = inst["correct_letter"]
            is_correct = pred is not None and pred == gold
            if pred is not None:
                n_extracted += 1
            if is_correct:
                n_correct += 1

            rec = {
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
                mark = "✓" if is_correct else ("?" if pred is None else "✗")
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Centralized-topology GPQA-Diamond runner (AutoGen)."
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
    out_path = Path(args.out) if args.out else None
    run_batch(instances, out_path=out_path)
    if out_path:
        print(f"  predictions written to {out_path}")
