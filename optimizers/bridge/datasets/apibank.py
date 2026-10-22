"""API-Bank dataset bridge for real-runner optimization."""

from __future__ import annotations

import json
import os
from typing import Any

import dspy

from core.paths import BENCHMARKS_DIR

# Full curated pool (train + validation + reported test IDs). The runner's
# default "all" manifest is the 100-row evaluation set, which leaves nothing
# to optimize on once the evaluation IDs are excluded.
POOL_PATH = BENCHMARKS_DIR / "apibank" / "apibank_pool_ids.json"


def _format_chat_history(task: dict) -> str:
    from topologies.single.apibank import langgraph_apibank as apibank_common

    return apibank_common._format_chat_history(task.get("chat_history") or [])


def load_all(level: str | int | None = None) -> list[dspy.Example]:
    """Load the combined curated API-Bank manifest as DSPy examples."""
    from topologies.single.apibank import langgraph_apibank as apibank_common

    if (
        apibank_common.normalize_level(level) == "all"
        and POOL_PATH.is_file()
        and not os.environ.get("APIBANK_CURATED_PATH")
    ):
        os.environ["APIBANK_CURATED_PATH"] = str(POOL_PATH)
        try:
            rows = apibank_common.load_instances(level=level)
        finally:
            os.environ.pop("APIBANK_CURATED_PATH", None)
    else:
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
            f"Correct API-Bank call for role {role} on level {level}. Gold call: {gold}. Predicted call: {pred_text}."
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
