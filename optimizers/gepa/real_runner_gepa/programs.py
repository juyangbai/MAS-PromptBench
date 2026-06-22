"""DSPy bridge whose execution is delegated to real-runner adapters."""
from __future__ import annotations

import json
import os
import re
from typing import Any

import dspy

from real_runner_gepa.protocol import RealRunnerAdapter, validate_adapter
from real_runner_gepa.registry import get_adapter_class


class RealRunnerCall(dspy.Signature):
    """Execute one task through a real runner."""

    task_instance: dict = dspy.InputField(desc="Real runner task instance.")
    tool_calls: list[dict] = dspy.OutputField(desc="Canonical tool calls or final structured output.")
    answer: str = dspy.OutputField(desc="JSON-serialized canonical output.")
    winner: int | None = dspy.OutputField(desc="Winning ensemble member id, when available.")
    vote_summary: str = dspy.OutputField(desc="Compact majority-vote or consensus summary.")
    agent_trace: str = dspy.OutputField(desc="Concise real-runner execution trace for reflection.")


def _safe_attr(role: str) -> str:
    safe = re.sub(r"\W+", "_", role).strip("_")
    return safe or "role"


def _role_signature(role: str) -> type[dspy.Signature]:
    """Create a role-distinct signature so GEPA can match traces cleanly."""
    sig = RealRunnerCall.with_updated_fields(
        "agent_trace",
        desc=f"Concise real-runner execution trace for role {role!r}.",
    )
    return sig.with_updated_fields(
        "answer",
        desc=f"JSON-serialized canonical output observed while optimizing role {role!r}.",
    )


def _compact_enabled(dataset: str) -> bool:
    raw = os.environ.get("GEPA_REFLECTION_COMPACT_DATASETS", "lcb")
    enabled = {item.strip().lower() for item in raw.split(",") if item.strip()}
    return dataset.lower() in enabled or "*" in enabled


def _head_tail(text: Any, max_chars: int, head_ratio: float = 0.55) -> str:
    value = "" if text is None else str(text)
    if len(value) <= max_chars:
        return value
    head = max(0, int(max_chars * head_ratio))
    tail = max(0, max_chars - head)
    return (
        value[:head].rstrip()
        + "\n\n...[truncated for GEPA reflection; execution/scoring used full data]...\n\n"
        + value[-tail:].lstrip()
    )


def _compact_sequence(values: Any, limit: int = 3) -> list[Any]:
    if not isinstance(values, list):
        return []
    compact = []
    for value in values[:limit]:
        if isinstance(value, dict):
            compact.append({k: _head_tail(v, 600) for k, v in list(value.items())[:8]})
        else:
            compact.append(_head_tail(value, 600))
    if len(values) > limit:
        compact.append(f"...[{len(values) - limit} more items omitted from GEPA reflection]...")
    return compact


def compact_task_instance_for_reflection(task_instance: Any, dataset: str) -> Any:
    """Return a trace-only view; real execution still receives the full instance."""
    if not _compact_enabled(dataset) or not isinstance(task_instance, dict):
        return task_instance

    if dataset == "lcb":
        tests = task_instance.get("tests") or []
        test_modes = sorted(
            {
                str(test.get("testtype") or ("functional" if test.get("fn_name") or test.get("func_name") else "stdin"))
                for test in tests
                if isinstance(test, dict)
            }
        )
        return {
            "id": task_instance.get("id"),
            "difficulty": task_instance.get("difficulty"),
            "problem": _head_tail(task_instance.get("problem"), 7000),
            "starter_code": _head_tail(task_instance.get("starter_code"), 3000),
            "test_count": len(tests) if isinstance(tests, list) else 0,
            "test_modes": test_modes,
            "tests": "[hidden tests omitted from GEPA reflection; scoring used full tests]",
        }

    return {
        key: _head_tail(value, 4000) if isinstance(value, str) else value
        for key, value in task_instance.items()
    }


def compact_prediction_for_reflection(prediction: dspy.Prediction, dataset: str) -> dspy.Prediction:
    """Compact only the trace copy of a prediction, not the returned prediction."""
    if not _compact_enabled(dataset):
        return prediction

    fields: dict[str, Any] = {}
    for key in ("tool_calls", "answer", "winner", "vote_summary", "agent_trace", "code", "patch", "itinerary"):
        value = getattr(prediction, key, None)
        if value is None:
            continue
        if key == "tool_calls":
            fields[key] = _compact_sequence(value, limit=2)
        elif key in {"answer", "code", "patch"}:
            fields[key] = _head_tail(value, 6000)
        elif key == "agent_trace":
            fields[key] = _head_tail(value, 3000)
        elif key == "vote_summary":
            fields[key] = _head_tail(value, 1000)
        else:
            fields[key] = value
    return dspy.Prediction(**fields)


def append_reflection_trace(
    adapter: RealRunnerAdapter,
    module: dspy.Predict,
    inputs: dict[str, Any],
    outputs: dspy.Prediction,
) -> None:
    dataset = getattr(adapter, "dataset", "")
    trace_inputs = dict(inputs)
    if "task_instance" in trace_inputs:
        trace_inputs["task_instance"] = compact_task_instance_for_reflection(trace_inputs["task_instance"], dataset)
    append_trace(module, trace_inputs, compact_prediction_for_reflection(outputs, dataset))


def fail_on_adapter_error() -> bool:
    raw = os.environ.get("REAL_RUNNER_FAIL_ON_ADAPTER_ERROR", "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def scoreable_adapter_runtime_error(exc: BaseException) -> bool:
    """Errors that should fail one example, not the whole GEPA pair."""

    # Real-runner execution talks to local/remote OpenAI-compatible servers.
    # A transient endpoint hiccup should make this one example score 0, not
    # destroy an hours-long GEPA pair and leave no meta.json artifact.
    scoreable_names = {
        "APIConnectionError",
        "APIError",
        "APIResponseValidationError",
        "APIStatusError",
        "APITimeoutError",
        "BadRequestError",
        "ConnectError",
        "ConnectionError",
        "ConnectionResetError",
        "InternalServerError",
        "ReadError",
        "ReadTimeout",
        "RemoteProtocolError",
        "TimeoutException",
    }
    return isinstance(exc, TimeoutError) or type(exc).__name__ in scoreable_names


def adapter_runtime_error_output(exc: BaseException) -> dict[str, Any]:
    message = f"{type(exc).__name__}: {exc}"
    return {
        "model_output": [],
        "winner": None,
        "buckets": [],
        "answer_text": f"ERROR: real runner failed before producing a valid final output: {message}",
        "raw": message,
        "error": message,
        "runner_output": {
            "error_type": type(exc).__name__,
            "error": str(exc),
        },
    }


class RealRunnerRolePredict(dspy.Predict):
    """GEPA-mutable prompt holder for one real-runner role.

    The owning AdapterBackedProgram normally runs the adapter once and emits
    traces for all roles. Calling this predictor directly is supported for
    single-role debugging.
    """

    def __init__(self, adapter: RealRunnerAdapter, role: str):
        validate_adapter(adapter)
        super().__init__(_role_signature(role))
        self.adapter = adapter
        self.role = role
        self.signature = self.signature.with_instructions(adapter.get_prompt(role))

    def sync_to_adapter(self) -> None:
        current_prompt = self.signature.instructions or ""
        if current_prompt != self.adapter.get_prompt(self.role):
            self.adapter.set_prompt(self.role, current_prompt)

    def forward(self, **kwargs):
        self.sync_to_adapter()
        task_instance = kwargs["task_instance"]
        try:
            out = self.adapter.run_example(task_instance)
        except Exception as exc:
            if scoreable_adapter_runtime_error(exc):
                out = adapter_runtime_error_output(exc)
            elif fail_on_adapter_error():
                raise
            else:
                out = adapter_runtime_error_output(exc)
        pred = prediction_from_adapter_output(self.adapter, self.role, out)
        if kwargs.pop("_trace", True) and dspy.settings.trace is not None:
            append_reflection_trace(self.adapter, self, kwargs, pred)
        return pred


def prediction_from_adapter_output(
    adapter: RealRunnerAdapter,
    role: str,
    output: Any,
) -> dspy.Prediction:
    """Convert adapter output into a DSPy Prediction GEPA can reflect on."""
    answer_text = None
    if not isinstance(output, dict):
        payload = output
        tool_calls = []
        winner = None
        vote_summary = ""
    else:
        payload = output.get("model_output") or []
        tool_calls = payload if isinstance(payload, list) else []
        winner = output.get("winner")
        vote_summary = json.dumps(output.get("buckets") or [], default=str)
        answer_text = output.get("answer_text")

    if hasattr(adapter, "format_role_trace"):
        agent_trace = adapter.format_role_trace(role, output)
    else:
        agent_trace = str(output)
    if isinstance(output, dict) and output.get("error"):
        agent_trace = f"{agent_trace}\nerror={output['error']}"

    prediction_fields = {
        "tool_calls": tool_calls,
        "answer": answer_text if answer_text is not None else json.dumps(payload, default=str),
        "winner": winner,
        "vote_summary": vote_summary,
        "agent_trace": agent_trace,
    }
    if isinstance(output, dict):
        for key in (
            "predicted_answer",
            "runner_correct",
            "runner_answer_correct",
            "scoring_prev_tool_content",
            "previous_tool_content",
        ):
            if key in output:
                prediction_fields[key] = output[key]
    if isinstance(output, dict):
        for key in (
            "communication_format",
            "communication_parse_ok",
            "communication_all_parse_ok",
            "communication_parse_rate",
            "communication_required_report_count",
            "communication_missing_roles",
            "communication_infra_error",
            "communication_parse_errors",
            "communication_parse_warnings",
            "communication_report_ok_count",
            "communication_report_total",
            "communication_reports",
            "communication_rendered_reports",
            "communication_inflight_handoffs",
            "communication_inflight_handoff_count",
            "communication_inflight_all_parse_ok",
        ):
            if key in output:
                prediction_fields[key] = output[key]
    return dspy.Prediction(**prediction_fields)


def append_trace(module: dspy.Predict, inputs: dict[str, Any], outputs: dspy.Prediction) -> None:
    """Append the exact trace tuple DSPy's GEPA reflective dataset expects."""
    trace = dspy.settings.trace
    if trace is None:
        return
    if len(trace) >= dspy.settings.max_trace_size:
        trace.pop(0)
    trace.append((module, dict(inputs), outputs))


class AdapterBackedProgram(dspy.Module):
    """GEPA-compatible DSPy program backed by one real-runner adapter."""

    def __init__(self, adapter: RealRunnerAdapter):
        super().__init__()
        validate_adapter(adapter)
        self._adapter = adapter
        self._role_predictor_names: dict[str, str] = {}
        for role in adapter.roles():
            name = _safe_attr(role)
            if hasattr(self, name):
                name = f"role_{name}"
            setattr(self, name, RealRunnerRolePredict(adapter, role))
            self._role_predictor_names[role] = name

    def sync_prompts_to_adapter(self) -> None:
        for role, name in self._role_predictor_names.items():
            predictor = getattr(self, name)
            predictor.sync_to_adapter()

    def forward(self, task_instance: dict[str, Any]):
        self.sync_prompts_to_adapter()
        try:
            out = self._adapter.run_example(task_instance)
        except Exception as exc:
            if scoreable_adapter_runtime_error(exc):
                out = adapter_runtime_error_output(exc)
            elif fail_on_adapter_error():
                raise
            else:
                out = adapter_runtime_error_output(exc)

        first_pred = None
        for role, name in self._role_predictor_names.items():
            predictor = getattr(self, name)
            pred = prediction_from_adapter_output(self._adapter, role, out)
            if first_pred is None:
                first_pred = pred
            append_reflection_trace(self._adapter, predictor, {"task_instance": task_instance}, pred)

        return first_pred or dspy.Prediction()


class IndependentBFCLRealRunnerProgram(AdapterBackedProgram):
    """GEPA-compatible program that executes the real independent/BFCL runner."""

    def __init__(self, prompt: str | None = None, n_agents: int | None = None):
        adapter_cls = get_adapter_class("bfcl", "independent")
        super().__init__(adapter_cls(prompt=prompt, n_agents=n_agents))


class RealRunnerProgram(AdapterBackedProgram):
    """GEPA-compatible program for any registered real-runner pair."""

    def __init__(self, dataset: str, topology: str, **adapter_kwargs):
        adapter_cls = get_adapter_class(dataset, topology)
        super().__init__(adapter_cls(**adapter_kwargs))


class BFCLRealRunnerProgram(AdapterBackedProgram):
    """GEPA-compatible program for any implemented BFCL real-runner topology."""

    def __init__(self, topology: str, **adapter_kwargs):
        super().__init__(get_adapter_class("bfcl", topology)(**adapter_kwargs))
