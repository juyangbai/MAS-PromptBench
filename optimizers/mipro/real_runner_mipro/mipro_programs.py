"""MIPRO-aware DSPy bridge for real-runner adapters."""
from __future__ import annotations

import json
from typing import Any

import dspy

from real_runner_mipro.programs import (
    _role_signature,
    _safe_attr,
    _head_tail,
    adapter_runtime_error_output,
    append_reflection_trace,
    fail_on_adapter_error,
    prediction_from_adapter_output,
    scoreable_adapter_runtime_error,
)
from real_runner_mipro.protocol import RealRunnerAdapter, validate_adapter
from real_runner_mipro.registry import get_adapter_class


_DEMO_FIELD_ORDER = (
    "task_instance",
    "question",
    "problem",
    "answer",
    "tool_calls",
    "winner",
    "vote_summary",
    "agent_trace",
    "code",
    "patch",
    "itinerary",
)


def _demo_to_dict(demo: Any) -> dict[str, Any]:
    if demo is None:
        return {}
    if hasattr(demo, "toDict"):
        data = demo.toDict()
    elif isinstance(demo, dict):
        data = dict(demo)
    else:
        data = dict(getattr(demo, "__dict__", {}))
    return {
        str(k): v
        for k, v in data.items()
        if not str(k).startswith("_") and str(k) not in {"input_keys", "demos"}
    }


def _compact_value(value: Any, max_chars: int = 1400) -> Any:
    if isinstance(value, str):
        return _head_tail(value, max_chars)
    if isinstance(value, dict):
        compact: dict[str, Any] = {}
        for idx, key in enumerate(sorted(value)):
            if idx >= 12:
                compact["...omitted_keys"] = len(value) - idx
                break
            compact[str(key)] = _compact_value(value[key], max(300, max_chars // 2))
        return compact
    if isinstance(value, list):
        compact_list = [_compact_value(item, max(300, max_chars // 2)) for item in value[:6]]
        if len(value) > 6:
            compact_list.append(f"...{len(value) - 6} more items omitted")
        return compact_list
    return value


def summarize_demo(demo: Any) -> dict[str, Any]:
    """Return a stable, JSON-serializable summary of a selected DSPy demo."""
    raw = _demo_to_dict(demo)
    ordered: dict[str, Any] = {}
    for key in _DEMO_FIELD_ORDER:
        if key in raw:
            ordered[key] = _compact_value(raw[key])
    for key in sorted(raw):
        if key not in ordered:
            ordered[key] = _compact_value(raw[key], 800)
    return ordered


def selected_demo_summaries(demos: Any) -> list[dict[str, Any]]:
    if not demos:
        return []
    return [summarize_demo(demo) for demo in list(demos)]


def render_instruction_with_demos(instruction: str, demos: Any) -> str:
    """Render MIPRO's selected demos into the prompt real runners execute."""
    base = (instruction or "").strip()
    demo_summaries = selected_demo_summaries(demos)
    if not demo_summaries:
        return base + "\n"

    lines = [base, "", "### MIPROv2 Selected Demonstrations"]
    lines.append("Use these examples as behavioral guidance for this role. Do not copy their final answers unless the new task is identical.")
    for index, demo in enumerate(demo_summaries, start=1):
        lines.extend(
            [
                "",
                f"#### Demonstration {index}",
                "```json",
                json.dumps(demo, ensure_ascii=False, indent=2, default=str),
                "```",
            ]
        )
    return "\n".join(lines).strip() + "\n"


class MIPRORolePredict(dspy.Predict):
    """Prompt holder whose selected MIPRO demos are rendered into adapter prompts."""

    def __init__(self, adapter: RealRunnerAdapter, role: str):
        validate_adapter(adapter)
        super().__init__(_role_signature(role))
        self.adapter = adapter
        self.role = role
        self.signature = self.signature.with_instructions(adapter.get_prompt(role))
        self._last_rendered_prompt: str | None = None

    def rendered_prompt(self) -> str:
        return render_instruction_with_demos(self.signature.instructions or "", getattr(self, "demos", []))

    def demo_summaries(self) -> list[dict[str, Any]]:
        return selected_demo_summaries(getattr(self, "demos", []))

    def sync_to_adapter(self) -> None:
        rendered = self.rendered_prompt()
        if rendered != self._last_rendered_prompt or rendered != self.adapter.get_prompt(self.role):
            self.adapter.set_prompt(self.role, rendered)
            self._last_rendered_prompt = rendered

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


class MIPROAdapterBackedProgram(dspy.Module):
    """DSPy program backed by one real-runner adapter, with demo-aware prompts."""

    def __init__(self, adapter: RealRunnerAdapter):
        super().__init__()
        validate_adapter(adapter)
        self._adapter = adapter
        self._role_predictor_names: dict[str, str] = {}
        for role in adapter.roles():
            name = _safe_attr(role)
            if hasattr(self, name):
                name = f"role_{name}"
            setattr(self, name, MIPRORolePredict(adapter, role))
            self._role_predictor_names[role] = name

    def sync_prompts_to_adapter(self) -> None:
        for name in self._role_predictor_names.values():
            getattr(self, name).sync_to_adapter()

    def role_artifacts(self) -> list[dict[str, Any]]:
        artifacts = []
        for role, name in self._role_predictor_names.items():
            predictor = getattr(self, name)
            artifacts.append(
                {
                    "role": role,
                    "predictor_name": name,
                    "prompt": predictor.rendered_prompt(),
                    "demos": predictor.demo_summaries(),
                }
            )
        return artifacts

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


class MIPRORealRunnerProgram(MIPROAdapterBackedProgram):
    """MIPRO-compatible program for any registered real-runner pair."""

    def __init__(self, dataset: str, topology: str, **adapter_kwargs):
        adapter_cls = get_adapter_class(dataset, topology)
        super().__init__(adapter_cls(**adapter_kwargs))
