"""Decentralized debate topology specialized for GPQA-Diamond, OpenAI SDK."""

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

from topologies.output_contracts import append_output_contract_from_path

from openai import OpenAI

# Shared telemetry.
_TOPO_ROOT = str(Path(__file__).resolve().parents[4])
if _TOPO_ROOT not in sys.path:
    sys.path.insert(0, _TOPO_ROOT)
from topologies.telemetry import (  # noqa: E402
    openai_sdk_telemetry, openai_sdk_accumulate, normalize,
)


# Module-level accumulator: reset at each solve() entry, summed across every
# `client.chat.completions.create` call via the helper in _telemetry.py.
# Accurate token counts require this because `response.usage` lives on each
# per-call response, not on the aggregated contexts list.
_TELEM_ACC: dict = {
    "prompt_tokens": 0, "completion_tokens": 0,
    "total_tokens": 0, "n_llm_calls": 0, "n_tool_calls": 0,
}


def _reset_telem_acc() -> None:
    for k in _TELEM_ACC:
        _TELEM_ACC[k] = 0


# Stall safeguards
# Per-row wall-clock cap using SIGALRM when running on the main thread. The
# concurrent runner may execute run_batch([row]) inside a ThreadPool worker,
# where Python disallows signal handlers, so the guard becomes a no-op there.
# Without this, 4 peers x 2 rounds x N tool loops can stall a single row for
# many minutes when one LLM call hangs or the model loops on calculator ERROR
# messages. max_tool_loops also lowered to shave each per-turn runaway budget.
PER_ROW_TIMEOUT_S = 120
_MAX_TOOL_LOOPS = 4  # was 5


class _RowTimeout(Exception):
    """Raised when a single row exceeds PER_ROW_TIMEOUT_S."""


def _row_timeout_handler(signum, frame):
    raise _RowTimeout(f"row exceeded {PER_ROW_TIMEOUT_S}s")


@contextlib.contextmanager
def _row_timeout_guard(seconds: int):
    """Install SIGALRM for `seconds`; uninstall on exit regardless of
    outcome so timeouts in one row don't bleed into the next."""
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


VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://lai:8001/v1")
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3.5-9B")

N_AGENTS = int(os.environ.get("DECENTRALIZED_N_AGENTS", "4"))
N_ROUNDS = int(os.environ.get("DECENTRALIZED_N_ROUNDS", "2"))

_REPO_ROOT = Path(__file__).resolve().parents[4]
_PROMPTS_DIR = _REPO_ROOT / "configs" / "prompts" / "decentralized" / "gpqa"


def _load_prompt(role: str) -> str:
    return append_output_contract_from_path((_PROMPTS_DIR / f"{role}.txt").read_text().strip(), __file__, role)


SYSTEM_PROMPT = _load_prompt("debater")


# Tool
def calculator(expression: str) -> str:
    """Evaluate a numeric Python expression (arithmetic + math functions)."""
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


_CALCULATOR_SCHEMA = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": (
            "Evaluate a numeric Python expression (arithmetic + math "
            "functions like sqrt, log, sin, pi, e). Example: "
            "calculator(\"(4/3) * pi * 2**3\")"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A numeric Python expression.",
                },
            },
            "required": ["expression"],
        },
    },
}


def _dispatch_tool(name: str, arguments: dict) -> str:
    if name == "calculator":
        return calculator(arguments.get("expression", ""))
    return f"ERROR: unknown tool {name!r}"


# Client / completion
def _build_client() -> OpenAI:
    return OpenAI(base_url=VLLM_BASE_URL, api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"))


def _completion_kwargs() -> dict:
    """Sampling parameters matching the house default used by sequential,
    independent, and centralized (temp=0.2, top_p=0.9, seed=0,
    repetition_penalty=1.05, enable_thinking=False)."""
    return {
        "model": MODEL_ID,
        "temperature": 0.2,
        "top_p": 0.9,
        "seed": 0,
        "max_tokens": 2048,
        "extra_body": {
            "repetition_penalty": 1.05,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    }


def _chat_with_tools(client: OpenAI, messages: list[dict], max_tool_loops: int = _MAX_TOOL_LOOPS) -> dict:
    """One agent turn: call the model, resolve any tool_calls locally, loop
    until the model returns a content-only assistant message. Returns the
    final assistant message dict (which gets appended to the agent's
    context). Also mutates `messages` in-place to carry the tool call +
    tool result turns so the model sees its own scratchpad.
    """
    kwargs = _completion_kwargs()
    kwargs["tools"] = [_CALCULATOR_SCHEMA]
    kwargs["tool_choice"] = "auto"

    for _ in range(max_tool_loops):
        resp = client.chat.completions.create(messages=messages, **kwargs)
        openai_sdk_accumulate(_TELEM_ACC, resp)
        msg = resp.choices[0].message
        dump = msg.model_dump() if hasattr(msg, "model_dump") else dict(msg)
        tool_calls = dump.get("tool_calls") or []
        # Append the assistant turn (tool_call OR content) to conversation.
        messages.append(dump)

        if not tool_calls:
            return dump

        # Resolve each tool call locally + append a tool-role message.
        for tc in tool_calls:
            fn = tc.get("function", {}) if isinstance(tc, dict) else {}
            name = fn.get("name", "")
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            result = _dispatch_tool(name, args)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.get("id"),
                "content": result,
            })

    # Tool-loop budget exhausted; return whatever the last assistant msg was.
    return dump


# Debate loop
def _peer_injection(others_final: list[dict], question: str) -> dict:
    """Build the 'peers said X, Y, Z — revise if warranted' user message.

    others_final: list of the OTHER peers' final assistant messages from the
    previous round (content strings only, tool-call turns already resolved).
    """
    body = ["These are the final responses from other peer agents in the previous round:"]
    for i, m in enumerate(others_final):
        body.append(f"\nPeer {i + 1}:\n```\n{m.get('content') or ''}\n```")
    body.append(
        "\nCompare their reasoning against your own. Revise your answer ONLY "
        "if a peer's reasoning concretely outweighs yours. Re-emit a single "
        "`Answer: <letter>` line at the end.\n\n"
        "Original question:\n" + question
    )
    return {"role": "user", "content": "\n".join(body)}


def run_debate(question: str, choices: list[str]) -> list[list[dict]]:
    """Run N agents × R rounds on one GPQA-style MCQ.

    Returns: list of N contexts. Each context is the full conversation for
    that peer (system + user + interleaved assistant/tool turns across all
    rounds).
    """
    client = _build_client()
    mcq_body = format_mcq(question, choices)

    # Initial context per peer: system + MCQ user-turn.
    contexts: list[list[dict]] = [
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": mcq_body},
        ]
        for _ in range(N_AGENTS)
    ]
    # Track each peer's FINAL assistant message per round (for peer-injection
    # into the next round). Shape: round_finals[r][i] = assistant dict.
    round_finals: list[list[dict]] = []

    for r in range(N_ROUNDS):
        this_round: list[dict] = []
        for i, ctx in enumerate(contexts):
            if r > 0:
                others_final = [round_finals[r - 1][j] for j in range(N_AGENTS) if j != i]
                ctx.append(_peer_injection(others_final, mcq_body))
            final_msg = _chat_with_tools(client, ctx)
            this_round.append(final_msg)
        round_finals.append(this_round)

    return contexts


# Output parsing (aligned to single/independent/sequential/centralized gpqa)
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


def majority_vote(letters: list[str]) -> str | None:
    """Pick the most-common letter, breaking ties by first-occurrence order."""
    valid = [l for l in letters if l in _LETTERS]
    if not valid:
        return None
    counts = Counter(valid)
    max_count = max(counts.values())
    for letter in valid:
        if counts[letter] == max_count:
            return letter
    return None


# Format helpers
def format_mcq(question: str, choices: list[str]) -> str:
    """Build the user-facing MCQ string (4 choices, labeled A-D)."""
    assert len(choices) == 4, "GPQA expects exactly 4 choices."
    body = "\n".join(f"{_LETTERS[i]}) {choices[i]}" for i in range(4))
    return f"{question}\n\n{body}"


# Orchestration
def solve(question: str, choices: list[str]) -> dict:
    """Run the N-peer × R-round debate on one GPQA-style MCQ.

    Returns:
        {
            "answer":         round-R majority letter (A/B/C/D) or None,
            "per_peer":       [{peer, letter, raw}] — per-peer final
                              assistant messages + extracted letters,
            "all_contexts":   raw OpenAI chat contexts (one per peer),
        }
    """
    _reset_telem_acc()
    with _row_timeout_guard(PER_ROW_TIMEOUT_S):
        contexts = run_debate(question, choices)
    per_peer = []
    letters = []
    for i, ctx in enumerate(contexts):
        # Last message in ctx is the peer's final assistant turn for round R-1.
        final = ctx[-1].get("content") or ""
        letter = extract_answer(final)
        per_peer.append({"peer": i, "letter": letter, "raw": final})
        if letter is not None:
            letters.append(letter)
    # Prefer exact per-response usage counts from _TELEM_ACC; fall back to
    # context-walk counts if the accumulator saw nothing (e.g. if the
    # client lib stopped surfacing `response.usage` mid-run).
    telem = dict(_TELEM_ACC)
    if telem["n_llm_calls"] == 0:
        telem = openai_sdk_telemetry(contexts)
    return {
        "answer": majority_vote(letters),
        "per_peer": per_peer,
        "all_contexts": contexts,
        "telemetry": normalize(telem),
    }


# Dataset loader (aligned to other gpqa topologies)
_HF_DATASET = "Idavidrein/gpqa"
_HF_CONFIG = "gpqa_diamond"
_HF_SPLIT = "train"


def _stable_row_id(row: dict, fallback_idx: int) -> str:
    """Stable id for a GPQA row — md5 hash of question text. Matches
    the other 4 gpqa topologies so per-row diffs line up on the same id."""
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
    round-R majority-vote letter vs gold, return aggregate summary +
    optionally write per-instance predictions to JSONL.

    Per-instance record shape:
        {id, question, choices, correct_letter,
         predicted_letter, correct,
         per_peer: [{peer, letter, raw_last}] — one entry per debater,
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
                mark = "✓" if is_correct else ("?" if pred is None else "✗")
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
            f"(N={N_AGENTS} peers × R={N_ROUNDS} rounds) ===\n"
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
    print(f"\n=== Debate: N={N_AGENTS} peers × R={N_ROUNDS} rounds ===")
    for p in out["per_peer"]:
        snippet = (p["raw"] or "").strip().splitlines()
        tail = snippet[-3:] if len(snippet) > 3 else snippet
        print(f"  peer {p['peer']}: letter={p['letter']}   tail={' | '.join(tail)[:200]}")
    print(f"\n=== Majority-vote final answer: {out['answer']}  (expected: A) ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Decentralized-topology GPQA-Diamond runner (OpenAI SDK debate)."
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
        f"(base_url={VLLM_BASE_URL}, N={N_AGENTS} peers × R={N_ROUNDS} rounds) ..."
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
