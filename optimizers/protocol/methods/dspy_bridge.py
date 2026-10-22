"""DSPy plumbing shared by the GEPA and MIPRO ports.

* :class:`PhysicalCountingLM`, :func:`build_reflection_lm`, :func:`build_task_lm`
  and :class:`LogicalSeedLM`: the DSPy LMs under the common reflection policy
  (logical seeds, thinking, 48,000 tokens, temperature/top-p 1.0);
* :func:`pin_dspy_parallel_executor`: no straggler resubmission (one rollout is charged once);
* :class:`ProtocolRunnerAdapter`: the prompt-mutable adapter API over the protocol
  runner that the bridge's DSPy programs drive;
* :func:`dspy_examples` and :func:`require_grid_cell`.

Importing this module contacts no endpoint.
"""

from __future__ import annotations

import copy
import json
import threading
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from dspy import LM
from dspy.clients.base_lm import BaseLM

from ..cells import in_grid
from ..config import TASK_DECODING, normalize_task
from ..errors import OptimizerContractError, OptimizerInfrastructureFailure
from ..reflection import (
    REFLECTION_DEFAULT_TEMPERATURE,
    REFLECTION_DEFAULT_TOP_P,
    REFLECTION_MAX_OUTPUT_TOKENS,
    REFLECTION_SAMPLING_POLICY,
    capture_reflection_responses,
    model_request_timeout_seconds,
    observe_reflection_response,
    reflection_response_summary,
)
from ..rollouts import example_mapping, ordered_prompt_roles, role_trace_messages
from ..schema import CellSpec, PromptBundle, RunRecord, content_hash, example_id, to_jsonable
from ..seeding import logical_request_seed
from ..settings import ProtocolSettings


# DSPy execution
def pin_dspy_parallel_executor() -> None:
    """Make DSPy wait for every evaluation row instead of resubmitting stragglers.

    ``ParallelExecutor`` resubmits rows still running after ``timeout`` and
    keeps whichever copy finishes first, which would execute (and charge) one
    logical rollout twice. Timeout 0 disables the straggler path.
    """
    try:
        from dspy.utils.parallelizer import ParallelExecutor
    except ImportError:
        return
    if getattr(ParallelExecutor, "_protocol_no_stragglers", False):
        return
    original = ParallelExecutor.__init__

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs["timeout"] = 0
        original(self, *args, **kwargs)

    ParallelExecutor.__init__ = __init__
    ParallelExecutor._protocol_no_stragglers = True


# DSPy LMs
class PhysicalCountingLM(LM):
    """Count completed provider responses separately from cached logical calls.

    Copies share only the accounting sink, never sampling or endpoint state.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "n_calls": 0,
            "model_calls": 0,
            "logical_calls": 0,
            "cache_hits": 0,
        }
        self._usage_lock = threading.Lock()

    def __deepcopy__(self, memo: dict[int, Any]) -> PhysicalCountingLM:
        import copy

        clone = type(self).__new__(type(self))
        memo[id(self)] = clone
        for name, value in self.__dict__.items():
            if name in {"usage", "_usage_lock"}:
                setattr(clone, name, value)
            elif name == "_lock":
                setattr(clone, name, threading.Lock())
            else:
                setattr(clone, name, copy.deepcopy(value, memo))
        return clone

    def _process_lm_response(self, response: Any, prompt: Any, messages: Any, **kwargs: Any) -> Any:
        observe_reflection_response(response)
        cached = bool(getattr(response, "cache_hit", False))
        usage = dict(getattr(response, "usage", {}) or {})
        with self._usage_lock:
            self.usage["logical_calls"] += 1
            self.usage["cache_hits"] += int(cached)
            if not cached:
                self.usage["n_calls"] += 1
                self.usage["model_calls"] += 1
                self.usage["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
                self.usage["completion_tokens"] += int(usage.get("completion_tokens") or 0)
        return super()._process_lm_response(response, prompt, messages, **kwargs)


def build_reflection_lm(**overrides: Any) -> PhysicalCountingLM:
    """Reflection LM under the common policy (wrap with :class:`LogicalSeedLM` per cell)."""
    settings = ProtocolSettings.from_env()
    model = settings.reflection_model
    kwargs = {
        "api_base": settings.reflection_endpoint(),
        "api_key": settings.api_key,
        "temperature": REFLECTION_DEFAULT_TEMPERATURE,
        "top_p": REFLECTION_DEFAULT_TOP_P,
        "max_tokens": REFLECTION_MAX_OUTPUT_TOKENS,
        "timeout": model_request_timeout_seconds(model),
        "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
    }
    kwargs.update(overrides)
    return PhysicalCountingLM(f"openai/{model}", **kwargs)


def build_task_lm(task_model: str | None = None, **overrides: Any) -> PhysicalCountingLM:
    """Auxiliary DSPy task LM (e.g. MIPRO bootstrapping) on the first task endpoint."""
    settings = ProtocolSettings.from_env()
    model = task_model or settings.model_id or settings.task_model or ""
    endpoints = settings.task_endpoints_env
    decoding = TASK_DECODING["optimization"]
    kwargs = {
        "api_base": endpoints[0] if endpoints else settings.vllm_base_url,
        "api_key": settings.api_key,
        "temperature": decoding["temperature"],
        "top_p": decoding["top_p"],
        "max_tokens": decoding["max_output_tokens"],
        "timeout": model_request_timeout_seconds(model),
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }
    kwargs.update(overrides)
    return PhysicalCountingLM(f"openai/{model}", **kwargs)


class LogicalSeedLM(BaseLM):
    """Transparent DSPy-LM proxy that freezes sampling and logical call seeds."""

    def __init__(self, *, cell: CellSpec, lm: Any, phase: str) -> None:
        self.cell = cell
        self.lm = lm
        self.phase = phase
        self.requests: list[dict[str, Any]] = []
        self._counts: dict[str, int] = {}
        self._lock = threading.Lock()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.lm, name)

    def __deepcopy__(self, memo: dict[int, Any]) -> LogicalSeedLM:
        del memo
        return self

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        call_hash = content_hash({"args": to_jsonable(args), "kwargs": to_jsonable(kwargs)})
        with self._lock:
            occurrence = self._counts.get(call_hash, 0)
            self._counts[call_hash] = occurrence + 1
        seed = logical_request_seed(
            self.cell.optimizer_seed, self.cell.cell_id, self.phase, occurrence, call_hash, "reflection", occurrence
        )
        effective = dict(kwargs)
        effective.update(
            {
                "temperature": REFLECTION_DEFAULT_TEMPERATURE,
                "top_p": REFLECTION_DEFAULT_TOP_P,
                "max_tokens": REFLECTION_MAX_OUTPUT_TOKENS,
                "seed": seed,
            }
        )
        # DSPy shallowly merges call kwargs over LM defaults; keep both levels
        # of extra_body while enforcing thinking at the request boundary.
        defaults = getattr(self.lm, "kwargs", {}) or {}
        default_extra = dict(defaults.get("extra_body") or {})
        call_extra = dict(effective.get("extra_body") or {})
        template = dict(default_extra.get("chat_template_kwargs") or {})
        template.update(call_extra.get("chat_template_kwargs") or {})
        template.update(effective.pop("chat_template_kwargs", {}) or {})
        template["enable_thinking"] = True
        effective["extra_body"] = {**default_extra, **call_extra, "chat_template_kwargs": template}
        base = {
            "phase": self.phase,
            "request_seed": seed,
            "request_sha256": call_hash,
            "temperature": REFLECTION_DEFAULT_TEMPERATURE,
            "top_p": REFLECTION_DEFAULT_TOP_P,
            "max_output_tokens": REFLECTION_MAX_OUTPUT_TOKENS,
            "thinking": True,
            "sampling_policy": REFLECTION_SAMPLING_POLICY,
            "occurrence": occurrence,
        }
        observations: list[dict[str, Any]] = []
        try:
            with capture_reflection_responses() as captured:
                try:
                    output = self.lm(*args, **effective)
                finally:
                    observations.extend(captured)
        except Exception as exc:
            with self._lock:
                if self._counts.get(call_hash) == occurrence + 1:
                    self._counts[call_hash] = occurrence
                self.requests.append(
                    {
                        **base,
                        "status": "infrastructure_failure",
                        **reflection_response_summary(observations),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            raise
        with self._lock:
            self.requests.append({**base, "status": "success", **reflection_response_summary(observations)})
        return output


# The protocol runner behind the bridge's DSPy programs
def record_to_adapter_output(record: RunRecord) -> dict[str, Any]:
    """Adapter-output view of a run record, carrying ``runner_score``/``runner_status`` for the metric."""
    final = record.final_output
    output = dict(final) if isinstance(final, Mapping) else {"answer_text": "" if final is None else str(final)}
    output.setdefault("answer_text", "" if final is None else str(final))
    output.setdefault("model_output", [])
    output.setdefault("answer", final)
    output.setdefault("messages", list(record.messages))
    output.setdefault("tool_events", list(record.tool_events))
    output["runner_score"] = record.score
    output["runner_status"] = record.status
    output["runner_output"] = {"common_run_record": record.to_dict(), **dict(output.get("runner_output") or {})}
    if record.error:
        output["error"] = record.error
    return output


class ProtocolRunnerAdapter:
    """Prompt-mutable real-runner adapter API over the protocol runner.

    The bridge's ``AdapterBackedProgram`` / ``MIPROAdapterBackedProgram`` drive
    it like a real adapter; every ``run_example`` is one protocol-runner
    rollout of the current bundle. Deep copies (one per DSPy candidate) isolate
    the prompts but share the runner, the rollout counts and the failures.
    Subclasses define the request seed (:meth:`request_seed`).
    """

    def __init__(self, *, cell: CellSpec, runner: Any, bundle: PromptBundle, phase: str) -> None:
        self.cell = cell
        self.dataset = normalize_task(cell.task)
        self.topology = cell.topology
        self.framework = cell.framework
        self._runner = runner
        self._roles = {role: bundle.roles[role] for role in ordered_prompt_roles(cell, bundle)}
        self._demos = tuple(bundle.demos)
        self._metadata = dict(bundle.metadata)
        self._phase = phase
        self._call_counts: dict[tuple[str, str], int] = {}
        self._call_lock = threading.Lock()
        self._runtime_failures: list[Exception] = []

    def __deepcopy__(self, memo: dict[int, Any]) -> ProtocolRunnerAdapter:
        """Isolate candidate prompts while sharing the one runner and seed ledger."""
        clone = type(self).__new__(type(self))
        memo[id(self)] = clone
        clone.__dict__.update(self.__dict__)
        clone._roles = dict(self._roles)
        clone._demos = copy.deepcopy(self._demos, memo)
        clone._metadata = copy.deepcopy(self._metadata, memo)
        return clone

    @property
    def bundle(self) -> PromptBundle:
        return PromptBundle(roles=dict(self._roles), demos=self._demos, metadata=self._metadata)

    def roles(self) -> list[str]:
        return list(self._roles)

    def get_prompt(self, role: str) -> str:
        self._check_role(role)
        return self._roles[role]

    def set_prompt(self, role: str, text: str) -> None:
        self._check_role(role)
        if not isinstance(text, str) or not text.strip():
            raise ValueError("role prompts must be non-empty strings")
        self._roles[role] = text

    def reset(self) -> None:
        """The protocol runner owns runtime state; prompt state is retained."""

    def request_seed(self, item_id: str, bundle: PromptBundle, occurrence: int) -> int:
        """Seed of the ``occurrence``-th rollout of ``bundle`` on ``item_id``."""
        raise NotImplementedError

    def run_example(self, example: Any) -> dict[str, Any]:
        """One charged rollout of the current bundle; failures are kept for :meth:`raise_for_runtime_failures`."""
        task = example_mapping(example)
        item_id = example_id(task)
        bundle = self.bundle
        with self._call_lock:
            occurrence = self._call_counts.get((bundle.digest, item_id), 0)
            self._call_counts[(bundle.digest, item_id)] = occurrence + 1
        try:
            record = self._runner.run(task, bundle, self.request_seed(item_id, bundle, occurrence))
            if not isinstance(record, RunRecord):
                raise OptimizerContractError(f"runner.run must return RunRecord; got {type(record).__name__}")
            return record_to_adapter_output(record)
        except Exception as exc:
            # DSPy catches evaluation errors and finishes compile with its
            # failure score. Share failures across candidate copies so no
            # optimizer artifact is emitted after a failed rollout.
            with self._call_lock:
                self._runtime_failures.append(exc)
            raise

    def raise_for_runtime_failures(self, method: str) -> None:
        """Fail the optimization if any rollout of any candidate copy failed."""
        with self._call_lock:
            failure = self._runtime_failures[0] if self._runtime_failures else None
        if failure is None:
            return
        message = f"native {method} compile encountered a runtime failure; no optimizer artifact can be accepted"
        if isinstance(failure, OptimizerInfrastructureFailure):
            raise OptimizerInfrastructureFailure(f"{message}: {failure}") from failure
        raise OptimizerContractError(message) from failure

    def format_role_trace(self, role: str, output: Any) -> str:
        """Reflection trace of one role: the run record's status, score, final output and the role's messages."""
        self._check_role(role)
        if not isinstance(output, Mapping):
            return str(output)
        record = (output.get("runner_output") or {}).get("common_run_record") or {}
        messages = role_trace_messages(self.cell, self.roles(), role, record.get("messages", []))
        return "\n".join(
            [
                f"role={role}",
                f"status={record.get('status', output.get('runner_status'))}",
                f"score={record.get('score', output.get('runner_score'))}",
                f"final_output={str(record.get('final_output', output.get('answer_text')))[:2000]}",
                f"messages={json.dumps(messages, ensure_ascii=False, default=str)[:3000]}",
            ]
        )

    def _check_role(self, role: str) -> None:
        if role not in self._roles:
            raise KeyError(f"unknown role {role!r}; choices={list(self._roles)}")


def runner_score(prediction: Any, optimizer: str) -> float:
    """The protocol runner's score carried on a prediction (the GEPA and MIPRO metric)."""
    score = getattr(prediction, "runner_score", None)
    status = getattr(prediction, "runner_status", None)
    if status not in {"success", "semantic_failure"} or score is None:
        raise OptimizerContractError(f"DSPy {optimizer} metric did not receive a usable runner score")
    return float(score)


def dspy_examples(rows: Sequence[Any]) -> list[Any]:
    """DSPy examples whose only input is the complete row (``task_instance``), passed back to the runner unchanged."""
    import dspy

    examples = []
    for index, row in enumerate(rows):
        task = example_mapping(row)
        task.setdefault("id", example_id(row, index))
        examples.append(dspy.Example(id=str(task["id"]), task_instance=task).with_inputs("task_instance"))
    return examples


def require_grid_cell(cell: CellSpec, error: Callable[[str], Exception]) -> None:
    """A cell that claims grid membership (``source_tables``) must be an exact grid cell.

    ``run.py`` admits off-grid cells only with ``--allow-any-cell``; those
    carry no source tables and are reported as non-conformant.
    """
    if cell.source_tables and not in_grid(cell):
        raise error(
            f"{cell.method} cell is outside the experiment grid: {cell.task}/{cell.topology}/"
            f"{cell.framework}/{cell.communication}/r{cell.team_size}/{cell.task_model}"
        )


__all__ = [
    "LogicalSeedLM",
    "PhysicalCountingLM",
    "ProtocolRunnerAdapter",
    "build_reflection_lm",
    "build_task_lm",
    "dspy_examples",
    "pin_dspy_parallel_executor",
    "record_to_adapter_output",
    "require_grid_cell",
    "runner_score",
]
