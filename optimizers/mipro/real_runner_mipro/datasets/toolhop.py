"""ToolHop dataset bridge for real-runner MIPRO."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import dspy

from real_runner_mipro.datasets.split_utils import real_eval_ids


REPO_ROOT = Path(__file__).resolve().parents[4]


def load_all() -> list[dspy.Example]:
    """Load ToolHop rows through the shared benchmark loader."""
    from topologies.single.toolhop import langgraph_toolhop as toolhop_common

    rows = toolhop_common.load_instances()
    examples: list[dspy.Example] = []
    for row in rows:
        rid = str(row.get("id"))
        examples.append(
            dspy.Example(
                id=rid,
                question=row.get("question") or "",
                tools=row.get("tools") or {},
                functions=row.get("functions") or [],
                task_instance=row,
                answer=str(row.get("answer", "")),
                raw={"id": rid},
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
    """Shuffle/split while excluding ToolHop's protected 0..99 report IDs."""
    if train_size < 0 or val_size < 0:
        raise ValueError(f"split sizes must be non-negative; got train={train_size} val={val_size}")
    excluded = real_eval_ids("toolhop")
    pool = [example for example in examples if str(example.id) not in excluded]
    rng = random.Random(seed)
    rng.shuffle(pool)
    pool = pool[offset:]
    need = train_size + val_size
    if len(pool) < need:
        raise ValueError(
            f"ToolHop needs {need} examples after excluding {len(excluded)} protected eval IDs "
            f"and offset={offset}; got {len(pool)}"
        )
    train = pool[:train_size]
    val = pool[train_size:need]
    overlap = {str(row.id) for row in train} & {str(row.id) for row in val}
    if overlap:
        raise ValueError(f"ToolHop train/val overlap: {sorted(overlap)[:5]}")
    return train, val


def _prediction_text(prediction: Any) -> str:
    for attr in ("answer", "predicted_answer", "raw"):
        value = getattr(prediction, attr, None)
        if value is not None and str(value).strip():
            return str(value)
    return ""


def _trace_agent_text(pred_trace: Any) -> str:
    if not pred_trace:
        return ""
    parts: list[str] = []
    for _, _, outputs in pred_trace:
        trace = getattr(outputs, "agent_trace", None)
        if trace:
            parts.append(str(trace))
    return "\n".join(parts)


def metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
    """DSPy-compatible ToolHop final-answer metric."""
    from topologies.single.toolhop import langgraph_toolhop as toolhop_common

    pred_text = _prediction_text(prediction)
    gold = str(getattr(example, "answer", ""))
    correct = toolhop_common.score_answer(gold, pred_text, "")
    score = 1.0 if correct else 0.0
    role = pred_name or "program"
    trace_text = _trace_agent_text(pred_trace) or getattr(prediction, "agent_trace", "")
    if correct:
        feedback = f"Correct ToolHop answer for role {role}. Gold answer: {gold}. Predicted answer: {pred_text}."
    else:
        feedback = (
            f"ToolHop failure for role {role}.\n"
            f"Gold answer: {gold}\n"
            f"Predicted answer: {pred_text or '<empty>'}\n"
            f"Real-runner trace:\n{str(trace_text)[:1800]}\n"
            "Actionable fix: call the dynamic tools needed for each hop, preserve intermediate "
            "values exactly, and end with one short final answer wrapped as <answer>...</answer>."
        )
    return dspy.Prediction(score=score, feedback=feedback)
