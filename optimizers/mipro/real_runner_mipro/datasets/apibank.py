"""API-Bank dataset bridge for real-runner MIPRO."""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import dspy

from real_runner_mipro.datasets.split_utils import real_eval_ids


REPO_ROOT = Path(__file__).resolve().parents[4]
BALANCED100_PATH = REPO_ROOT / "benchmarks" / "apibank" / "apibank_eval_ids.json"
LEVEL_ORDER = ("1", "2", "3")


def _format_chat_history(task: dict) -> str:
    from topologies.single.apibank import langgraph_apibank as apibank_common

    return apibank_common._format_chat_history(task.get("chat_history") or [])


def load_all(level: str | int | None = None) -> list[dspy.Example]:
    """Load the combined curated API-Bank manifest as GEPA examples."""
    from topologies.single.apibank import langgraph_apibank as apibank_common

    rows = apibank_common.load_instances(level=level)
    examples: list[dspy.Example] = []
    for row in rows:
        rid = str(row.get("id"))
        level_key = str(row.get("level") or "")
        gold = row.get("gold_api_call") or ""
        examples.append(
            dspy.Example(
                id=rid,
                level=level_key,
                question=_format_chat_history(row),
                gold_api_call=gold,
                ground_truth=row.get("ground_truth") or {},
                task_instance=row,
                answer=gold,
                raw={
                    "id": rid,
                    "level": level_key,
                    "file": row.get("file"),
                    "sample_id": row.get("sample_id"),
                },
            ).with_inputs("task_instance")
        )
    return examples


def _allocation(size: int) -> dict[str, int]:
    if size < 0:
        raise ValueError(f"split size must be non-negative; got {size}")
    base, remainder = divmod(size, len(LEVEL_ORDER))
    return {level: base + (1 if idx < remainder else 0) for idx, level in enumerate(LEVEL_ORDER)}


def _round_robin(by_level: dict[str, list[dspy.Example]]) -> list[dspy.Example]:
    mixed: list[dspy.Example] = []
    positions = {level: 0 for level in LEVEL_ORDER}
    remaining = True
    while remaining:
        remaining = False
        for level in LEVEL_ORDER:
            pos = positions[level]
            rows = by_level.get(level, [])
            if pos < len(rows):
                mixed.append(rows[pos])
                positions[level] = pos + 1
                remaining = True
    return mixed


def level_counts(rows: list[dspy.Example]) -> dict[str, int]:
    return dict(Counter(str(getattr(row, "level", "")) for row in rows))


def train_val_split(
    examples: list[dspy.Example],
    train_size: int,
    val_size: int,
    seed: int = 0,
    offset: int = 0,
) -> tuple[list[dspy.Example], list[dspy.Example]]:
    """Balanced Level-1/2/3 split excluding the protected 100-row eval set."""
    excluded = real_eval_ids("apibank")
    train_need = _allocation(train_size)
    val_need = _allocation(val_size)

    grouped: dict[str, list[dspy.Example]] = {level: [] for level in LEVEL_ORDER}
    for example in examples:
        if str(example.id) in excluded:
            continue
        level = str(getattr(example, "level", ""))
        if level in grouped:
            grouped[level].append(example)

    rng = random.Random(seed)
    for level in LEVEL_ORDER:
        rng.shuffle(grouped[level])
        if offset:
            grouped[level] = grouped[level][offset:]

    train_by_level: dict[str, list[dspy.Example]] = {}
    val_by_level: dict[str, list[dspy.Example]] = {}
    for level in LEVEL_ORDER:
        needed = train_need[level] + val_need[level]
        available = len(grouped[level])
        if available < needed:
            raise ValueError(
                f"API-Bank level {level} needs {needed} examples after excluding "
                f"{len(excluded)} protected eval IDs and offset={offset}; got {available}"
            )
        train_by_level[level] = grouped[level][: train_need[level]]
        val_by_level[level] = grouped[level][train_need[level] : needed]

    train = _round_robin(train_by_level)
    val = _round_robin(val_by_level)
    overlap = {str(row.id) for row in train} & {str(row.id) for row in val}
    if overlap:
        raise ValueError(f"API-Bank train/val overlap: {sorted(overlap)[:5]}")
    return train, val


def _prediction_text(prediction: Any) -> str:
    for attr in ("answer", "predicted_answer", "raw"):
        value = getattr(prediction, attr, None)
        if value:
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
    """DSPy-compatible API-Bank exact API-call metric."""
    from topologies.single.apibank import langgraph_apibank as apibank_common

    pred_text = _prediction_text(prediction)
    result = apibank_common.score_prediction(example.task_instance, pred_text)
    score = 1.0 if result.get("correct") else 0.0
    gold = getattr(example, "gold_api_call", "")
    level = getattr(example, "level", None)
    role = pred_name or "program"
    trace_text = _trace_agent_text(pred_trace) or getattr(prediction, "agent_trace", "")
    if score:
        feedback = (
            f"Correct API-Bank call for role {role} on level {level}. "
            f"Gold call: {gold}. Predicted call: {pred_text}."
        )
    else:
        feedback = (
            f"API-Bank failure for role {role} on level {level}.\n"
            f"Gold call: {gold}\n"
            f"Predicted call: {pred_text or '<empty>'}\n"
            f"Failure stage: {result.get('stage')}\n"
            f"Official scorer error: {result.get('error')}\n"
            f"Predicted API: {result.get('predicted_api_name')} "
            f"params={json.dumps(result.get('predicted_params'), default=str)[:800]}\n"
            f"Real-runner trace:\n{str(trace_text)[:1800]}\n"
            "Actionable fix: emit exactly one bracketed API call, choose the next API required by "
            "the dialogue level, preserve exact API names, and ground every required argument in "
            "dialogue history, prior API results, or ToolSearcher evidence."
        )
    return dspy.Prediction(score=score, feedback=feedback)
