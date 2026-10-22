"""GPQA-Diamond: data, answer extraction, scoring and batch records.

The benchmark is the gpqa_diamond config of Idavidrein/gpqa (198 questions).
Each row's correct answer and three distractors are shuffled into options A-D
with a per-row seed, so every runner sees the same orderings; a prediction is
the option letter extracted from the model's final text, scored by exact match.
"""

from __future__ import annotations

import hashlib
import random
import re
from collections import Counter
from collections.abc import Callable, Iterable
from functools import partial
from pathlib import Path

from core import batch

DATASET = "gpqa"
HF_DATASET = "Idavidrein/gpqa"
HF_CONFIG = "gpqa_diamond"
HF_SPLIT = "train"  # GPQA-Diamond is a single split
SOURCE = f"GPQA-Diamond from {HF_DATASET} [{HF_CONFIG}/{HF_SPLIT}]"
LETTERS = ("A", "B", "C", "D")

# Appended to the centralized manager prompt.
TERMINATE_NUDGE = (
    "\n\nWhen you emit the final 'Answer: X' line, immediately follow it "
    "with the literal string TERMINATE on its own line so the group-chat "
    "knows to stop."
)

DEMO_QUESTION = (
    "A circular wire loop of radius R carries a steady current I. "
    "What is the magnitude of the magnetic field at the geometric center "
    "of the loop? (mu_0 is the vacuum permeability.)"
)
DEMO_CHOICES = [
    "mu_0 * I / (2 * R)",
    "mu_0 * I / (4 * pi * R)",
    "mu_0 * I / R",
    "mu_0 * I / (pi * R)",
]
DEMO_ANSWER = "A"


# Data
def row_id(row: dict, index: int) -> str:
    """``gpqa_`` plus the first 10 hex digits of the MD5 of the question (the row index if it has none)."""
    question = (row.get("Question") or "").strip()
    if question:
        return "gpqa_" + hashlib.md5(question.encode("utf-8")).hexdigest()[:10]
    return f"gpqa_idx_{index}"


def shuffle_choices(correct: str, incorrect: list[str], seed: str) -> tuple[list[str], str]:
    """The four answers in a ``random.Random(seed)`` order and the letter of the correct one."""
    answers = [correct, *incorrect]
    order = list(range(4))
    random.Random(seed).shuffle(order)
    return [answers[j] for j in order], LETTERS[order.index(0)]


def load_instances(
    limit: int | None = None,
    offset: int = 0,
    only: list[str] | None = None,
    shuffle_seed: int = 0,
) -> list[dict]:
    """Load the rows as ``{id, question, choices, correct_letter, raw}``.

    The choices of row ``id`` are shuffled with the seed ``f"{shuffle_seed}|{id}"``;
    rows missing the correct answer or a distractor are skipped.
    """
    from datasets import load_dataset

    rows: list[dict] = []
    for index, row in enumerate(load_dataset(HF_DATASET, HF_CONFIG)[HF_SPLIT]):
        rid = row_id(row, index)
        if only is not None and rid not in set(only):
            continue
        correct = (row.get("Correct Answer") or "").strip()
        incorrect = [(row.get(f"Incorrect Answer {k}") or "").strip() for k in (1, 2, 3)]
        if not correct or not all(incorrect):
            continue
        choices, letter = shuffle_choices(correct, incorrect, f"{shuffle_seed}|{rid}")
        rows.append(
            {
                "id": rid,
                "question": (row.get("Question") or "").strip(),
                "choices": choices,
                "correct_letter": letter,
                "raw": dict(row),
            }
        )
    rows = rows[offset:]
    return rows if limit is None else rows[:limit]


def add_arguments(parser) -> None:
    """The ``--shuffle-seed`` option of every GPQA runner."""
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=0,
        metavar="SEED",
        help="seed of the per-row choice shuffling (default 0: the orderings every runner uses)",
    )


def format_prompt(question: str, choices: list[str]) -> str:
    """User message: the question, a blank line and the options as ``A) ...`` lines."""
    assert len(choices) == 4, "GPQA expects exactly 4 choices."
    body = "\n".join(f"{LETTERS[i]}) {choices[i]}" for i in range(4))
    return f"{question}\n\n{body}"


def format_agents_prompt(question: str, choices: list[str]) -> str:
    """Task text of the Agents SDK debaters: options as ``A. ...`` lines and a one-letter instruction."""
    assert len(choices) == 4, "GPQA expects exactly 4 choices."
    rendered = [f"{LETTERS[i]}. {choices[i]}" for i in range(4)]
    return f"{question}\n\n" + "\n".join(rendered) + "\n\nReturn the final answer as one letter A, B, C, or D."


def peer_review_prompt(peer_outputs: list[str], question: str) -> str:
    """Debate message showing a peer the other peers' previous final responses."""
    body = ["These are the final responses from other peer agents in the previous round:"]
    body += [f"\nPeer {i + 1}:\n```\n{text}\n```" for i, text in enumerate(peer_outputs)]
    body.append(
        "\nCompare their reasoning against your own. Revise your answer ONLY "
        "if a peer's reasoning concretely outweighs yours. Re-emit a single "
        "`Answer: <letter>` line at the end.\n\n"
        "Original question:\n" + question
    )
    return "\n".join(body)


# Answer extraction: emphasis markers are dropped first ("**Answer:** B"), then
# the patterns are tried in order and the last match of the first one that hits wins.
_MARKDOWN_RE = re.compile(r"[*_`]+")
_ANSWER_PATTERNS = (
    re.compile(r"\b(?:final\s+)?answer\b\s*[:\s]*\(?([A-D])\)?", re.IGNORECASE),  # "Answer: B", "Final answer: (C)"
    re.compile(r"\b(?:option|choice)\b\s*(?:is)?\s*[:\s]*\(?([A-D])\)?", re.IGNORECASE),  # "the correct option is C"
    re.compile(r"(?:^|\n)\s*\(?([A-D])\)?\s*(?:[.\n]|$)", re.MULTILINE),  # a bare letter on its own line
)


def extract_answer(text: str) -> str | None:
    """The option letter of the model's final text, or None."""
    cleaned = _MARKDOWN_RE.sub("", text)
    for pattern in _ANSWER_PATTERNS:
        matches = pattern.findall(cleaned)
        if matches:
            return matches[-1].upper()
    return None


# Aggregation over several agents' letters
def _first_most_common(letters: list[str]) -> str | None:
    """The most common letter; ties go to the one seen first."""
    if not letters:
        return None
    counts = Counter(letters)
    return max(letters, key=counts.__getitem__)


def majority_vote(answers: list[dict]) -> str | None:
    """Majority letter over ``{"answer": ...}`` records (None skipped; ties: lowest index)."""
    return _first_most_common([a["answer"] for a in answers if a.get("answer") is not None])


def best_of_n(letters: Iterable[str | None]) -> str | None:
    """Majority over the valid letters A-D (ties: lowest index)."""
    return _first_most_common([letter for letter in letters if letter in LETTERS])


def votes(letters: Iterable[str | None]) -> dict[str, int]:
    """Count of every extracted letter, in first-seen order."""
    return dict(Counter(letter for letter in letters if letter is not None))


# Batch records
def record(inst: dict, pred: str | None, **fields) -> dict:
    """Per-instance record: id, question, choices, gold and predicted letter, correctness, then ``fields``."""
    gold = inst["correct_letter"]
    return {
        "id": inst["id"],
        "question": inst["question"],
        "choices": inst["choices"],
        "correct_letter": gold,
        "predicted_letter": pred,
        "correct": pred is not None and pred == gold,
        **fields,
    }


def per_peer_tails(per_peer: list[dict]) -> list[dict]:
    """Debate record field: each peer's letter and the last 300 characters of its final text."""
    return [{"peer": p["peer"], "letter": p["letter"], "raw_tail": (p["raw"] or "")[-300:]} for p in per_peer]


def stage_excerpts(by_stage: dict) -> dict:
    """Sequential record field: the first 800 characters of every stage's output."""
    return {role: (text or "")[:800] for role, text in by_stage.items()}


def summarize(per_instance: list[dict]) -> dict:
    """Batch scores: n, extracted and correct predictions, accuracy (overall and on extracted rows)."""
    n = len(per_instance)
    n_extracted = sum(1 for rec in per_instance if rec["predicted_letter"] is not None)
    n_correct = sum(1 for rec in per_instance if rec["correct"])
    return {
        "n": n,
        "n_extracted": n_extracted,
        "n_correct": n_correct,
        "accuracy": (n_correct / n) if n else 0.0,
        "extracted_acc": (n_correct / n_extracted) if n_extracted else 0.0,
    }


def progress_line(index: int, total: int, rec: dict, done: list[dict]) -> str:
    """One verbose progress line after ``rec`` (``done`` holds the records so far)."""
    pred = rec["predicted_letter"]
    accuracy = sum(1 for r in done if r["correct"]) / len(done)
    mark = "✓" if rec["correct"] else ("?" if pred is None else "✗")
    return (
        f"[{index + 1:>3}/{total}] {rec['id']} {mark}  pred={pred or '-'}  gold={rec['correct_letter']}  "
        f"acc={accuracy:.3f}  lat={rec['latency_s']:.1f}s"
    )


def banner(label: str, summary: dict) -> str:
    """End-of-batch report."""
    return (
        f"\n=== {label} batch complete ===\n"
        f"  n={summary['n']}  n_extracted={summary['n_extracted']}  n_correct={summary['n_correct']}\n"
        f"  accuracy={summary['accuracy']:.3f}  extracted_acc={summary['extracted_acc']:.3f}  "
        f"total_s={summary['total_s']}\n"
    )


def run_batch(
    instances: list[dict],
    row: Callable[[int, dict], dict],
    *,
    out_path: Path | None = None,
    verbose: bool = True,
    label: str = "GPQA-Diamond",
) -> dict:
    """:func:`core.batch.run_batch` with the GPQA summary, progress lines and report."""
    return batch.run_batch(
        instances,
        row,
        summarize=summarize,
        out_path=out_path,
        verbose=verbose,
        progress=progress_line,
        banner=partial(banner, label),
    )


def print_demo_answer(answer: str | None) -> None:
    """Print the demo's extracted letter next to :data:`DEMO_ANSWER`."""
    print(f"\n=== Extracted answer: {answer}  (expected: {DEMO_ANSWER}) ===")
