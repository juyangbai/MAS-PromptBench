"""MATH loader/scorer for real-runner MIPRO."""
from __future__ import annotations

import hashlib
import importlib
import random
import re
import sys
from pathlib import Path

import dspy

from real_runner_mipro.datasets.split_utils import train_val_split_excluding_real_eval


HF_DATASET = "qwedsacf/competition_math"
HF_SPLIT = "train"
SUBJECT = "Precalculus"
LEVEL = "Level 5"


def extract_boxed(text: str | None) -> str | None:
    """Return the inner content of the last ``\\boxed{...}`` expression."""
    if not text:
        return None
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
                return text[start + 1 : i].strip()
    return None


def strip_thinking(text: str) -> str:
    index = (text or "").lower().rfind("</think>")
    if index >= 0:
        text = text[index + len("</think>"):]
    return (text or "").strip()


def extract_answer(text: str | None) -> str | None:
    """Extract the model's final boxed MATH answer."""
    if not text:
        return None
    cleaned = strip_thinking(re.sub(r"\bTERMINATE\b", "", text)).strip()
    return extract_boxed(cleaned)


def _math_scorer():
    repo_root = Path(__file__).resolve().parents[4]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return importlib.import_module("topologies.single.math.langgraph_math")


def exact_match_score(pred: str | None, gold: str) -> float:
    """Score with the same Hendrycks MATH equivalence used by real runners."""
    if pred is None:
        return 0.0
    return float(_math_scorer().exact_match_score(pred, gold))


def load_all() -> list[dspy.Example]:
    """Load the shared hard MATH slice used by the topology runners."""
    from datasets import load_dataset

    ds = load_dataset(HF_DATASET)[HF_SPLIT]
    examples: list[dspy.Example] = []
    for row in ds:
        if row.get("type") != SUBJECT or row.get("level") != LEVEL:
            continue
        problem = (row.get("problem") or "").strip()
        solution = (row.get("solution") or "").strip()
        if not problem or not solution:
            continue
        gold = extract_boxed(solution)
        if gold is None:
            continue
        rid = "math_" + hashlib.md5(problem.encode("utf-8")).hexdigest()[:10]
        instance = {
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
        }
        examples.append(
            dspy.Example(
                id=rid,
                problem=problem,
                task_instance=instance,
                answer=gold,
                subject=row.get("type"),
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
    return train_val_split_excluding_real_eval("math", examples, train_size, val_size, seed, offset)


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
            f"Correct for role {role}. Extracted boxed answer {pred!r}, "
            f"equivalent to gold {gold!r} under Hendrycks MATH scoring."
        )
    elif pred is None:
        feedback = (
            f"Format failure for role {role}: no final boxed answer was extracted.\n"
            "End with a final line containing exactly one committed answer in "
            "`\\boxed{...}`. The scorer extracts the LAST boxed expression.\n"
            f"Problem: {example.problem[:1200]}\n"
            f"Gold boxed answer: {gold!r}\n"
            f"Raw output tail:\n{str(raw)[-1000:]}\n"
            f"Real-runner trace:\n{agent_trace[:1800]}"
        )
    else:
        feedback = (
            f"Incorrect for role {role}. Extracted {pred!r}, gold {gold!r}; "
            "they are not equivalent under Hendrycks MATH normalization.\n"
            "MATH fixes: verify algebra carefully, use the calculator only for "
            "numeric subexpressions, simplify the final LaTeX, and box only the "
            "final answer.\n"
            f"Problem: {example.problem[:1200]}\n"
            f"Real-runner trace:\n{agent_trace[:1800]}"
        )
    return dspy.Prediction(score=score, feedback=feedback)
