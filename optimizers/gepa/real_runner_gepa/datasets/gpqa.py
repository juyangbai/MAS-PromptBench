"""GPQA-Diamond loader/scorer for real-runner GEPA."""
from __future__ import annotations

import hashlib
import random
import re

import dspy

from real_runner_gepa.datasets.split_utils import train_val_split_excluding_real_eval


HF_DATASET = "Idavidrein/gpqa"
HF_CONFIG = "gpqa_diamond"
HF_SPLIT = "train"
LETTERS = ["A", "B", "C", "D"]


def stable_row_id(row: dict, fallback_idx: int) -> str:
    question = (row.get("Question") or "").strip()
    if question:
        return "gpqa_" + hashlib.md5(question.encode("utf-8")).hexdigest()[:10]
    return f"gpqa_idx_{fallback_idx}"


def format_prompt(question: str, choices: list[str]) -> str:
    body = "\n".join(f"{LETTERS[i]}) {choices[i]}" for i in range(4))
    return f"{question}\n\n{body}"


def load_all(shuffle_seed: int = 0) -> list[dspy.Example]:
    """Load GPQA-Diamond as examples carrying topology-ready instances."""
    from datasets import load_dataset

    ds = load_dataset(HF_DATASET, HF_CONFIG)[HF_SPLIT]
    examples: list[dspy.Example] = []
    for i, row in enumerate(ds):
        rid = stable_row_id(row, i)
        question = (row.get("Question") or "").strip()
        correct = (row.get("Correct Answer") or "").strip()
        incorrects = [(row.get(f"Incorrect Answer {k}") or "").strip() for k in (1, 2, 3)]
        if not question or not correct or any(not item for item in incorrects):
            continue

        four = [correct, *incorrects]
        rng = random.Random(f"{shuffle_seed}|{rid}")
        indices = list(range(4))
        rng.shuffle(indices)
        choices = [four[j] for j in indices]
        gold = LETTERS[indices.index(0)]
        prompt = format_prompt(question, choices)
        instance = {
            "id": rid,
            "question": question,
            "choices": choices,
            "prompt": prompt,
            "correct_letter": gold,
            "raw": dict(row),
        }
        examples.append(
            dspy.Example(
                id=rid,
                question=prompt,
                task_instance=instance,
                answer=gold,
            ).with_inputs("task_instance")
        )
    return examples


def train_val_split(
    examples: list[dspy.Example],
    train_size: int,
    val_size: int,
    seed: int = 0,
    offset: int = 0,
) -> tuple[list[dspy.Example], list[dspy.Example]]:
    return train_val_split_excluding_real_eval("gpqa", examples, train_size, val_size, seed, offset)


def strip_thinking(text: str) -> str:
    index = (text or "").lower().rfind("</think>")
    if index >= 0:
        text = text[index + len("</think>"):]
    return (text or "").strip()


_MARKDOWN_STRIP_RE = re.compile(r"[*_`]+")
_ANSWER_RE = re.compile(r"\b(?:final\s+)?answer\b\s*[:\s]*\(?([A-D])\)?", re.IGNORECASE)
_OPTION_RE = re.compile(r"\b(?:option|choice)\b\s*(?:is)?\s*[:\s]*\(?([A-D])\)?", re.IGNORECASE)
_BARE_LETTER_RE = re.compile(r"(?:^|\n)\s*\(?([A-D])\)?\s*(?:[.\n]|$)", re.MULTILINE)


def extract_letter(text: str | None) -> str | None:
    if not text:
        return None
    cleaned = _MARKDOWN_STRIP_RE.sub("", strip_thinking(text))
    for pattern in (_ANSWER_RE, _OPTION_RE, _BARE_LETTER_RE):
        matches = pattern.findall(cleaned)
        if matches:
            return matches[-1].upper()
    return None


def exact_match_score(pred_letter: str | None, gold_letter: str) -> float:
    if pred_letter is None:
        return 0.0
    return float(pred_letter.strip().upper() == gold_letter.strip().upper())


def _trace_agent_text(pred_trace) -> str:
    if not pred_trace:
        return ""
    parts: list[str] = []
    for _, _, outputs in pred_trace:
        role_trace = getattr(outputs, "agent_trace", None)
        if role_trace:
            parts.append(str(role_trace))
        elif outputs is not None:
            parts.append(str(outputs))
    return "\n".join(parts)


def metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
    """DSPy-compatible metric used by the generic all-dataset GEPA runner."""
    gold = example.answer
    raw = getattr(prediction, "answer", None)
    if raw is None:
        raw = str(prediction)
    pred = extract_letter(raw)
    score = exact_match_score(pred, gold)
    role = pred_name or "program"
    agent_trace = _trace_agent_text(pred_trace) or getattr(prediction, "agent_trace", "")
    if score:
        feedback = (
            f"Correct for role {role}. Extracted {pred!r}, gold {gold!r}. "
            "Keep ending with a clear `Answer: X` or `Final answer: X` line."
        )
    elif pred is None:
        feedback = (
            f"Format failure for role {role}: no A/B/C/D letter was extracted.\n"
            "The scorer checks `Answer: X`, `Final answer: X`, `option X`, "
            "then a bare A-D line. End with a clear final letter.\n"
            f"Question:\n{example.question[:1200]}\n"
            f"Raw output tail:\n{str(raw)[-800:]}\n"
            f"Real-runner trace:\n{agent_trace[:1600]}"
        )
    else:
        feedback = (
            f"Incorrect for role {role}. Extracted {pred!r}, gold {gold!r}.\n"
            "For GPQA, improve by identifying the scientific principle, "
            "eliminating each distractor, checking units/signs/constraints, "
            "and ending with a single final letter.\n"
            f"Question:\n{example.question[:1400]}\n"
            f"Real-runner trace:\n{agent_trace[:1800]}"
        )
    return dspy.Prediction(score=score, feedback=feedback)
