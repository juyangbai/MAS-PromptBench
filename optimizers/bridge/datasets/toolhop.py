"""ToolHop dataset bridge for real-runner optimization."""

from __future__ import annotations

from typing import Any

import dspy


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
    prev_tool_content = str(
        getattr(prediction, "scoring_prev_tool_content", "") or getattr(prediction, "previous_tool_content", "") or ""
    )
    final_text_correct = toolhop_common.score_answer(gold, pred_text, "")
    correct = final_text_correct or toolhop_common.score_answer(gold, pred_text, prev_tool_content)
    score = 1.0 if correct else 0.0
    role = pred_name or "program"
    trace_text = _trace_agent_text(pred_trace) or getattr(prediction, "agent_trace", "")
    if correct:
        route = "final answer" if final_text_correct else "selected agent tool observation"
        feedback = (
            f"Correct ToolHop answer for role {role} via {route}. Gold answer: {gold}. Predicted answer: {pred_text}."
        )
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
