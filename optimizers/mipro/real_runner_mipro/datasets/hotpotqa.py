"""HotpotQA loader/scorer for real-runner MIPRO."""
from __future__ import annotations

import random
import re
import string
from collections import Counter

import dspy

from real_runner_mipro.datasets.split_utils import train_val_split_excluding_real_eval


HF_DATASET = "hotpot_qa"
HF_CONFIG = "distractor"
HF_SPLIT = "validation"


def load_all() -> list[dspy.Example]:
    """Load HotpotQA validation rows as topology-ready examples."""
    from datasets import load_dataset

    ds = load_dataset(HF_DATASET, HF_CONFIG, trust_remote_code=True)[HF_SPLIT]
    examples: list[dspy.Example] = []
    for row in ds:
        rid = row.get("id")
        question = (row.get("question") or "").strip()
        answer = (row.get("answer") or "").strip()
        if not rid or not question or not answer:
            continue
        instance = {
            "id": str(rid),
            "question": question,
            "answer": answer,
            "type": row.get("type"),
            "level": row.get("level"),
            "raw": {k: row.get(k) for k in ("id", "question", "answer", "type", "level")},
        }
        examples.append(
            dspy.Example(
                id=str(rid),
                question=question,
                task_instance=instance,
                answer=answer,
                type=row.get("type"),
                level=row.get("level"),
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
    return train_val_split_excluding_real_eval("hotpotqa", examples, train_size, val_size, seed, offset)


def strip_thinking(text: str) -> str:
    index = (text or "").lower().rfind("</think>")
    if index >= 0:
        text = text[index + len("</think>"):]
    return (text or "").strip()


def normalize_answer(text: str) -> str:
    text = (text or "").lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


_ANSWER_RE = re.compile(
    r"\banswer\b\s*(?:is\s+)?[:\s]+\**\s*(.+?)\s*\**\s*(?:\n|$)",
    re.IGNORECASE,
)


def extract_answer(text: str | None) -> str | None:
    if not text:
        return None
    cleaned = strip_thinking(re.sub(r"\bTERMINATE\b", "", text)).strip()
    matches = _ANSWER_RE.findall(cleaned)
    if matches:
        return matches[-1].strip().rstrip(".,")
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    return lines[-1] if lines else None


def exact_match_score(pred: str | None, gold: str) -> float:
    if pred is None:
        return 0.0
    return float(normalize_answer(pred) == normalize_answer(gold))


def f1_score(pred: str | None, gold: str) -> tuple[float, float, float]:
    if pred is None:
        return 0.0, 0.0, 0.0
    normalized_pred = normalize_answer(pred)
    normalized_gold = normalize_answer(gold)
    zero = (0.0, 0.0, 0.0)
    if normalized_pred in {"yes", "no", "noanswer"} and normalized_pred != normalized_gold:
        return zero
    if normalized_gold in {"yes", "no", "noanswer"} and normalized_pred != normalized_gold:
        return zero
    pred_tokens = normalized_pred.split()
    gold_tokens = normalized_gold.split()
    if not pred_tokens or not gold_tokens:
        return zero
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return zero
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1, precision, recall


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
    pred = extract_answer(raw)
    score = exact_match_score(pred, gold)
    role = pred_name or "program"
    agent_trace = _trace_agent_text(pred_trace) or getattr(prediction, "agent_trace", "")
    if score:
        feedback = (
            f"Correct for role {role}. Extracted {pred!r}, matching gold {gold!r} "
            "after HotpotQA normalization."
        )
    elif pred is None:
        feedback = (
            f"Format failure for role {role}: no short-form answer was extracted.\n"
            "End with `Answer: <short-form>` where the answer is minimal: a name, "
            "year, place, noun phrase, or yes/no.\n"
            f"Question: {example.question[:1000]}\n"
            f"Gold: {gold!r}\n"
            f"Raw output tail:\n{str(raw)[-900:]}\n"
            f"Real-runner trace:\n{agent_trace[:1600]}"
        )
    else:
        f1, precision, recall = f1_score(pred, gold)
        feedback = (
            f"Incorrect for role {role}. Extracted {pred!r}, gold {gold!r}. "
            f"Normalized pred={normalize_answer(pred)!r}, "
            f"gold={normalize_answer(gold)!r}, F1={f1:.3f}, "
            f"P={precision:.3f}, R={recall:.3f}.\n"
            "HotpotQA fixes: search Wikipedia for both hops, disambiguate "
            "entities, use page text before answering, and keep the final "
            "answer short.\n"
            f"Question: {example.question[:1200]}\n"
            f"Real-runner trace:\n{agent_trace[:1800]}"
        )
    return dspy.Prediction(score=score, feedback=feedback)
