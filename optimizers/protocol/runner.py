"""Execute, score and charge full-MAS rollouts through the real-runner adapters.

One rollout = take the cell's ``optimizers.bridge`` adapter, install the
bundle's role prompts, set the request seed and the phase decoding
(optimization 0.2 / evaluation 0.0 via ``lm.eval_mode``), point ``MODEL_ID``
at the cell's task model, call ``adapter.run_example(task_instance)`` and score
the prediction with the dataset metric.

Classification follows the protocol:

* usable (``success`` / ``semantic_failure``): a scored observation, charged;
* ``infrastructure_failure``: any exception before a scored observation
  (transport errors, the Agents SDK ``PreObservationFailure``, BadRequest,
  timeouts), returned transport error text, zero observed model calls, or a
  scorer exception. Retried at most twice with the same request seed and
  never charged; after three attempts the record is returned unusable.

Rollouts run one at a time per process: the runners read the request seed,
temperature and model from process-global state. Run jobs as processes.
"""

from __future__ import annotations

import importlib
import inspect
import os
import sys
import threading
import time
import traceback
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Protocol

from topologies.decentralized.openai_agents.agents_sdk_base import FRAMEWORK as SDK_FRAMEWORK
from topologies.decentralized.openai_agents.agents_sdk_base import load_agents_sdk

from .adapter_output import (
    TELEMETRY_KEYS,
    capture_model_responses,
    find_telemetry,
    json_safe,
    mapping_messages,
    synthetic_final_message,
    tool_events,
    transport_error_text,
    transport_failure,
    usage_from_telemetry,
)
from .artifacts import ArtifactStore, scrub_text
from .budget import BudgetLedger
from .cells import registry_key as cell_registry_key
from .config import MAX_INFRASTRUCTURE_RETRIES, TASK_DECODING, TASKS
from .errors import (
    JobError,
    PreObservationInfrastructureFailure,
    RunnerContractError,
    TelemetryError,
    UnsupportedRuntimeBinding,
)
from .schema import CellSpec, PromptBundle, RunRecord, Usage, canonical_json, content_hash, example_id, schema_name

UsableStatus = Literal["success", "semantic_failure"]
SEED_LIMIT = 2**31 - 1
OPTIMIZATION_SPLITS = ("train", "validation")
EVALUATION_PHASES = ("baseline", "final_validation", "test")
# Fields only the scorer may read; removed from the instance the MAS sees.
SCORER_PRIVATE_FIELDS = {
    "bfcl": ("ground_truth",),
    "gpqa": ("correct_letter", "correct_answer", "incorrect_answers", "raw"),
}
# Process-global runner state (seed, temperature, model) forces serial rollouts.
_EXECUTION_LOCK = threading.RLock()


# Records
@dataclass(frozen=True)
class RuntimeInvocation:
    """Complete input delivered to one runtime execution."""

    cell: CellSpec
    example_id: str
    example: Any
    task_input: Mapping[str, Any]
    bundle: PromptBundle
    request_seed: int
    attempt_index: int
    phase: str
    evaluation_seed_identity: str | None = None


@dataclass(frozen=True)
class RuntimeInvocationResult:
    """Unscored result of a complete MAS execution."""

    final_output: Any
    usage: Usage
    messages: tuple[Mapping[str, Any], ...] = ()
    tool_events: tuple[Mapping[str, Any], ...] = ()
    model_id: str = ""
    trace_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    prediction: Any = None


@dataclass(frozen=True)
class ScoreResult:
    """The dataset metric's verdict on one execution."""

    score: float
    status: UsableStatus = "success"
    usage: Usage = field(default_factory=Usage)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BatchExecution:
    """Records of one charged batch and how it was trimmed at the cap."""

    records: tuple[RunRecord, ...]
    requested: int
    scheduled: int
    cap_truncated: int
    budget_snapshot: Mapping[str, int]


def phase_bucket(phase: str) -> str:
    """``optimization`` for every optimization phase, else the evaluation phase itself."""
    if phase == "optimization" or phase.startswith("optimization/"):
        return "optimization"
    if phase in EVALUATION_PHASES:
        return phase
    raise RunnerContractError(f"unsupported runtime phase {phase!r}")


def usage_dict(usage: Usage) -> dict[str, int]:
    """Usage counters as a plain dict of ints."""
    return {key: int(value) for key, value in asdict(usage).items()}


def validate_usage(usage: Usage, *, name: str) -> None:
    """Reject negative, non-integer or inconsistent usage counters."""
    values = asdict(usage)
    if any(not isinstance(value, int) or value < 0 for value in values.values()):
        raise RunnerContractError(f"{name} contains negative or non-integer counters")
    if usage.total_tokens != usage.input_tokens + usage.output_tokens:
        raise RunnerContractError(f"{name}.total_tokens must equal input_tokens + output_tokens")
    if usage.total_tokens and usage.model_calls == 0:
        raise RunnerContractError(f"{name} has tokens but zero model_calls")


# Task data
def example_row(native: Any) -> dict[str, Any]:
    """JSON-safe mapping handed to optimizers for one native example."""
    if hasattr(native, "toDict"):
        value = native.toDict()
    elif isinstance(native, Mapping):
        value = dict(native)
    else:
        value = {key: item for key, item in vars(native).items() if not key.startswith("_")}
    row = json_safe(value)
    row["id"] = example_id(native, fallback=None)
    return row


def task_input_of(native: Any, dataset: str) -> dict[str, Any]:
    """The instance the MAS receives, without scorer-private fields."""
    instance = native.get("task_instance") if isinstance(native, Mapping) else getattr(native, "task_instance", None)
    if instance is None:
        instance = native if isinstance(native, Mapping) else example_row(native)
    if hasattr(instance, "toDict"):
        instance = instance.toDict()
    value = dict(instance)
    for key in SCORER_PRIVATE_FIELDS.get(dataset, ()):
        value.pop(key, None)
    return value


class TaskData:
    """Fixed ordered splits over one dataset's native examples.

    Test rows are released only through :meth:`unlock_test`, which requires
    a locked selection that recorded no test exposure.
    """

    def __init__(self, dataset: str, split_ids: Mapping[str, Sequence[str]], examples: Iterable[Any]) -> None:
        self.dataset = dataset
        self.split_ids = {
            split: tuple(str(item) for item in split_ids.get(split, ())) for split in ("train", "validation", "test")
        }
        ordered = [item for split in ("train", "validation", "test") for item in self.split_ids[split]]
        if len(set(ordered)) != len(ordered):
            raise ValueError(f"{dataset}: train/validation/test IDs must be disjoint and unique")
        index = {example_id(example, fallback=None): example for example in examples}
        missing = [item for item in ordered if item not in index]
        if missing:
            raise ValueError(
                f"{dataset}: {len(missing)} fixed-split IDs are absent from the loaded pool, e.g. {missing[:3]}"
            )
        self._natives = {item: index[item] for item in ordered}
        self._rows: dict[str, dict[str, Any]] = {}
        self._test_unlocked = False
        self.split_hash = content_hash({"dataset": dataset, **{k: list(v) for k, v in self.split_ids.items()}})

    def native(self, example_id: str) -> Any:
        """The dataset's native example."""
        return self._natives[example_id]

    def row(self, example_id: str) -> dict[str, Any]:
        """The JSON-safe row handed to optimizers."""
        if example_id not in self._rows:
            self._rows[example_id] = example_row(self._natives[example_id])
        return self._rows[example_id]

    def split_of(self, example_id: str) -> str | None:
        """The split holding an example, or None."""
        for split, ids in self.split_ids.items():
            if example_id in ids:
                return split
        return None

    def unlock_test(self, selection: Mapping[str, Any]) -> None:
        """Release the test rows after a locked selection without test exposure."""
        if selection.get("test_exposed") is not False or not selection.get("selection_id"):
            raise RunnerContractError("test rows require a locked selection without test exposure")
        self._test_unlocked = True

    def rows(self, split: str) -> tuple[dict[str, Any], ...]:
        """All rows of a split, in the fixed order."""
        if split == "test" and not self._test_unlocked:
            raise RunnerContractError("test rows are released only after the deployment selection is locked")
        if split not in self.split_ids:
            raise ValueError(f"unknown split {split!r}")
        return tuple(self.row(item) for item in self.split_ids[split])


def load_task_data(dataset: str) -> TaskData:
    """Load ``optimizers.bridge.datasets.<dataset>`` restricted to its fixed splits."""
    module = importlib.import_module(f"optimizers.bridge.datasets.{dataset}")
    from optimizers.bridge.datasets.split_utils import fixed_split_ids

    ids = fixed_split_ids(dataset)
    if ids is None:
        raise ValueError(f"{dataset}: benchmarks/{dataset}/{dataset}_splits.json is missing")
    return TaskData(dataset, ids, module.load_all())


# Prompts, roles and execution hooks
def render_prompts(bundle: PromptBundle) -> dict[str, str]:
    """Role prompts as executed; MIPRO demos are rendered into each role prompt."""
    prompts = dict(bundle.roles)
    if not bundle.demos:
        return prompts
    from optimizers.bridge.mipro_programs import render_instruction_with_demos

    return {
        role: render_instruction_with_demos(
            prompt, [demo for demo in bundle.demos if not demo.get("role") or str(demo.get("role")) == role]
        )
        for role, prompt in prompts.items()
    }


_ROLE_ORDERS: dict[tuple[str, str], tuple[str, ...]] = {}


def remember_role_order(task: str, key: str, roles: Sequence[str]) -> None:
    """Cache the adapter's native role order of one runtime condition."""
    _ROLE_ORDERS[(task, key)] = tuple(roles)


def runtime_role_order(cell: CellSpec) -> tuple[str, ...] | None:
    """Native role order of the cell's adapter (``adapter.roles()``), or None."""
    try:
        key = cell_registry_key(cell.topology, cell.framework, cell.communication, cell.team_size)
    except ValueError:
        return None
    if (cell.task, key) not in _ROLE_ORDERS:
        try:
            from optimizers.bridge.registry import get_adapter_class

            adapter_class = get_adapter_class(cell.task, key)
            adapter = adapter_class(**default_adapter_kwargs(adapter_class, key, cell.team_size))
            remember_role_order(cell.task, key, adapter.roles())
        except Exception:
            return None
    return _ROLE_ORDERS[(cell.task, key)]


class ExecutionHook(Protocol):
    """Masked or otherwise modified execution requested through bundle metadata.

    ``build_adapter`` returns the adapter that runs this request (for example
    a coalition with some workers masked); ``verify`` inspects the native
    output and returns acknowledgement metadata. An ack with
    ``allow_zero_model_calls=True`` accepts an execution with no model call
    (an empty coalition).
    """

    def build_adapter(self, runtime: AdapterRuntime, request: RuntimeInvocation, control: Any) -> Any: ...

    def verify(
        self, runtime: AdapterRuntime, request: RuntimeInvocation, control: Any, output: Any
    ) -> Mapping[str, Any]: ...


# Metadata keys that change execution; a bundle carrying one needs its hook.
CONTROL_METADATA_KEYS = ("optimizer_control",)
_EXECUTION_HOOKS: dict[str, ExecutionHook] = {}


def register_execution_hook(metadata_key: str, hook: ExecutionHook | None) -> None:
    """Register (or with ``None`` remove) the hook for one bundle metadata key."""
    if metadata_key not in CONTROL_METADATA_KEYS:
        raise ValueError(f"unknown control metadata key {metadata_key!r}; known: {CONTROL_METADATA_KEYS}")
    if hook is None:
        _EXECUTION_HOOKS.pop(metadata_key, None)
    else:
        _EXECUTION_HOOKS[metadata_key] = hook


def execution_hook_for(bundle: PromptBundle) -> tuple[ExecutionHook | None, Any]:
    """The registered hook and control value a bundle requests (``(None, None)`` for plain execution)."""
    for key in CONTROL_METADATA_KEYS:
        if key in bundle.metadata:
            hook = _EXECUTION_HOOKS.get(key)
            if hook is None:
                raise RunnerContractError(f"bundle metadata requests {key!r} but no execution hook is registered")
            return hook, bundle.metadata[key]
    return None, None


# Runtime
def default_adapter_kwargs(adapter_class: Any, key: str, team_size: int) -> dict[str, Any]:
    """Constructor kwargs: replica count for independent/decentralized, 2 debate rounds."""
    from .cells import parse_registry_key

    parsed = parse_registry_key(key)
    topology = str(parsed["topology"])
    kwargs: dict[str, Any] = {}
    if "team_size" in parsed:
        kwargs["n_agents"] = int(parsed["team_size"])
    elif topology in {"independent", "decentralized"}:
        kwargs["n_agents"] = int(team_size)
    if topology == "decentralized":
        kwargs["n_rounds"] = 2
    parameters = inspect.signature(adapter_class).parameters
    variadic = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values())
    return {name: value for name, value in kwargs.items() if variadic or name in parameters}


def _require_agents_sdk() -> None:
    """Import the OpenAI Agents SDK before the first rollout of an openai_agents
    cell; raises JobError with the reason when this process cannot use it."""
    try:
        load_agents_sdk()
    except RuntimeError as exc:
        raise JobError(str(exc)) from exc


class AdapterRuntime:
    """Bind one cell to its real-runner adapter and execute invocations."""

    supports_demos = True

    def __init__(
        self,
        cell: CellSpec,
        *,
        registry_key: str | None = None,
        adapter_class: Any = None,
        adapter_kwargs: Mapping[str, Any] | None = None,
        capture_usage: bool = True,
        prediction_builder: Callable[[Any, str, Any], Any] | None = None,
        task_environment: Mapping[str, str] | None = None,
    ) -> None:
        self.cell = cell
        self.registry_key = registry_key or cell_registry_key(
            cell.topology, cell.framework, cell.communication, cell.team_size
        )
        if adapter_class is None:
            if cell.framework == SDK_FRAMEWORK:
                _require_agents_sdk()
            importlib.import_module("optimizers.bridge.lm")
            from optimizers.bridge.registry import get_adapter_class

            try:
                adapter_class = get_adapter_class(cell.task, self.registry_key)
            except KeyError as exc:
                raise UnsupportedRuntimeBinding(f"no real-runner adapter for {cell.task}/{self.registry_key}") from exc
            if prediction_builder is None:
                from optimizers.bridge.programs import prediction_from_adapter_output

                prediction_builder = prediction_from_adapter_output
        self.adapter_class = adapter_class
        self.adapter_kwargs = dict(
            adapter_kwargs
            if adapter_kwargs is not None
            else default_adapter_kwargs(adapter_class, self.registry_key, cell.team_size)
        )
        self.capture_usage = capture_usage
        self.prediction_builder = prediction_builder
        self.task_environment = dict(task_environment or default_task_environment(cell.task))
        # Stable runtime identity: condition hashes and every artifact record it.
        self.implementation_id = f"real_runner_gepa/{cell.task}/{self.registry_key}"
        self._adapter = self.new_adapter()
        self.native_roles = tuple(self._adapter.roles())
        if not self.native_roles or len(set(self.native_roles)) != len(self.native_roles):
            raise UnsupportedRuntimeBinding(f"{self.implementation_id} exposes no unique prompt roles")
        self.required_roles = tuple(sorted(self.native_roles))
        self.seed_prompts = {role: self._adapter.get_prompt(role) for role in self.required_roles}
        remember_role_order(cell.task, self.registry_key, self.native_roles)

    def new_adapter(self) -> Any:
        """A fresh adapter instance of the cell's class."""
        adapter = self.adapter_class(**self.adapter_kwargs)
        for name in ("roles", "get_prompt", "set_prompt", "reset", "run_example"):
            if not callable(getattr(adapter, name, None)):
                raise UnsupportedRuntimeBinding(f"adapter {type(adapter).__name__} lacks {name}()")
        return adapter

    def seed_bundle(self) -> PromptBundle:
        """The frozen starting bundle: each role's shipped seed prompt."""
        return PromptBundle(
            roles=dict(self.seed_prompts),
            metadata={"source": "seed_prompts", "runtime_implementation_id": self.implementation_id},
        )

    def environment(self, request_seed: int, evaluation: bool) -> dict[str, str]:
        """Environment of one rollout: task model, request seed and phase decoding."""
        decoding = TASK_DECODING["evaluation" if evaluation else "optimization"]
        return {
            **self.task_environment,
            "MODEL_ID": self.cell.task_model,
            "REQUEST_SEED": str(int(request_seed)),
            "TASK_MODEL_TEMPERATURE": str(decoding["temperature"]),
            "TASK_MODEL_TOP_P": str(decoding["top_p"]),
            "TASK_MODEL_MAX_TOKENS": str(decoding["max_output_tokens"]),
            "TASK_TEMP_OPTIMIZE": str(TASK_DECODING["optimization"]["temperature"]),
            "TASK_TEMP_EVAL": str(TASK_DECODING["evaluation"]["temperature"]),
        }

    @staticmethod
    def _decoding(evaluation: bool):
        lm = sys.modules.get("optimizers.bridge.lm")
        return lm.eval_mode(evaluation) if lm is not None else nullcontext()

    def __call__(self, request: RuntimeInvocation) -> RuntimeInvocationResult:
        """Execute one invocation and return its unscored result.

        Installs the bundle's rendered prompts, runs the adapter under the rollout
        environment and decoding, and reads the usage from the observed model
        responses (else the runner's telemetry). Raises
        ``PreObservationInfrastructureFailure`` for a transport failure, raised or
        returned as output text, and ``RunnerContractError`` for a bundle or a
        telemetry that breaks the runtime contract.
        """
        prompts = render_prompts(request.bundle)
        if tuple(sorted(prompts)) != self.required_roles:
            raise RunnerContractError("bundle roles differ from the runtime roles")
        hook, control = execution_hook_for(request.bundle)
        evaluation = phase_bucket(request.phase) != "optimization"
        environment = self.environment(request.request_seed, evaluation)
        with (
            _EXECUTION_LOCK,
            patched_environment(environment),
            self._decoding(evaluation),
            capture_model_responses(self.cell.task_model, self.capture_usage) as capture,
        ):
            adapter, value = self._execute(request, prompts, hook, control, capture)
            prediction = None
            if self.prediction_builder is not None:
                prediction = self.prediction_builder(adapter, self.native_roles[0], value)
        ack = dict(hook.verify(self, request, control, value)) if hook is not None else None
        error = transport_error_text(value)
        if error:
            raise PreObservationInfrastructureFailure(
                scrub_text(error)[:2000],
                stage="task_runtime",
                usage=_captured_usage(capture),
                metadata={"implementation_id": self.implementation_id},
            )
        safe = json_safe(value)
        if not isinstance(safe, Mapping):
            safe = {"output": safe}
        messages = mapping_messages(value) or (synthetic_final_message(safe, self.cell.method),)
        tools = tool_events(value, messages)
        captured = capture.telemetry()
        usage = self._usage(value, captured, tools, ack)
        metadata = self._result_metadata(request, evaluation, captured is not None, messages, ack)
        return RuntimeInvocationResult(
            final_output=dict(safe),
            usage=usage,
            messages=tuple(dict(m) for m in messages),
            tool_events=tuple(dict(e) for e in tools),
            model_id=self.cell.task_model,
            trace_id=str(safe.get("trace_id")) if safe.get("trace_id") else None,
            metadata=metadata,
            prediction=prediction,
        )

    def _execute(self, request: RuntimeInvocation, prompts: Mapping[str, str], hook, control, capture):
        """Install ``prompts`` in the request's adapter and run the task; returns ``(adapter, output)``.

        A transport failure, or the Agents SDK's ``PreObservationFailure``, becomes a
        ``PreObservationInfrastructureFailure`` that carries the usage observed so far.
        """
        try:
            adapter = hook.build_adapter(self, request, control) if hook is not None else self._adapter
            for role in self.native_roles:
                adapter.set_prompt(role, prompts[role])
            adapter.reset()
            return adapter, adapter.run_example(dict(request.task_input))
        except PreObservationInfrastructureFailure as exc:
            if capture.events and not exc.usage.model_calls:
                exc.usage = _captured_usage(capture)
            raise
        except Exception as exc:
            sdk_pre_observation = type(exc).__name__ == "PreObservationFailure"
            if sdk_pre_observation or transport_failure(exc):
                raise PreObservationInfrastructureFailure(
                    f"{type(exc).__name__}: {exc}",
                    stage="openai_agents_sdk" if sdk_pre_observation else "task_runtime",
                    usage=_captured_usage(capture),
                    metadata={"exception_type": type(exc).__name__, "implementation_id": self.implementation_id},
                ) from exc
            raise

    @staticmethod
    def _usage(
        value: Any, captured: Mapping[str, int] | None, tools: Sequence[Mapping[str, Any]], ack: Mapping | None
    ) -> Usage:
        """Usage of one execution: the observed model responses when captured, else the runner's telemetry.

        The tool-call count is the largest of both telemetries and the tool calls
        found in the messages. Zero model calls are accepted only when the execution
        hook acknowledged them (an empty coalition).
        """
        telemetry = find_telemetry(value)
        if captured is not None:
            telemetry = {
                **captured,
                "n_tool_calls": max(captured["n_tool_calls"], int((telemetry or {}).get("n_tool_calls", 0) or 0)),
            }
        allow_zero = bool(ack and ack.get("allow_zero_model_calls"))
        if telemetry is None and allow_zero:
            telemetry = dict.fromkeys(TELEMETRY_KEYS, 0)
        try:
            return usage_from_telemetry(
                telemetry,
                allow_zero_calls=allow_zero,
                derived_tool_calls=sum(1 for e in tools if e.get("type") == "tool_call"),
            )
        except TelemetryError as exc:
            raise RunnerContractError(str(exc)) from exc

    def _result_metadata(
        self,
        request: RuntimeInvocation,
        evaluation: bool,
        observed_usage: bool,
        messages: Sequence[Mapping[str, Any]],
        ack: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        """The runtime-result metadata recorded with every execution (validated as canonical JSON)."""
        decoding = TASK_DECODING["evaluation" if evaluation else "optimization"]
        metadata = {
            "schema": schema_name("runtime-result"),
            "implementation_id": self.implementation_id,
            "registry_key": self.registry_key,
            "request_seed": request.request_seed,
            "attempt_index": request.attempt_index,
            "prompt_roles": list(self.required_roles),
            "demo_count": len(request.bundle.demos),
            "effective_parameters": {
                "model": self.cell.task_model,
                "temperature": decoding["temperature"],
                "top_p": decoding["top_p"],
                "max_tokens": decoding["max_output_tokens"],
                "thinking": decoding["thinking"],
                "request_seed": request.request_seed,
            },
            "usage_source": "observed-model-responses" if observed_usage else "native-telemetry",
            "message_trace_complete": not any(m.get("synthetic_from_final_output") for m in messages),
            "scorer_private_fields_removed": [k for k in SCORER_PRIVATE_FIELDS.get(self.cell.task, ())],
        }
        if ack is not None:
            metadata["execution_control"] = ack
        canonical_json(metadata)
        return metadata


@contextmanager
def patched_environment(values: Mapping[str, str]) -> Iterator[None]:
    """Set environment variables for the duration of one rollout, then restore them."""
    missing = object()
    previous = {key: os.environ.get(key, missing) for key in values}
    os.environ.update({key: str(value) for key, value in values.items()})
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is missing:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _captured_usage(capture) -> Usage:
    telemetry = capture.telemetry()
    if telemetry is None:
        return Usage()
    return usage_from_telemetry(telemetry, allow_zero_calls=True)


def default_task_environment(task: str) -> dict[str, str]:
    """Task-authoritative execution switches."""
    if task == "toolhop":
        return {"TOOLHOP_ALLOW_DATASET_EXEC": "1"}
    return {}


# Scorer
class DatasetMetricScorer:
    """Score with ``optimizers.bridge.datasets.<dataset>.metric(example, prediction)``.

    Status is ``success`` for a positive score, else ``semantic_failure``; the
    metric feedback is kept in the record for reflective optimizers.
    """

    def __init__(self, dataset: str) -> None:
        self.module = importlib.import_module(f"optimizers.bridge.datasets.{dataset}")
        self.dataset = dataset
        self.scorer_id = f"dataset-metric/{dataset}"

    def __call__(self, example: Any, result: RuntimeInvocationResult) -> ScoreResult:
        if result.prediction is None:
            raise RunnerContractError("dataset metric requires the adapter prediction")
        value = self.module.metric(example, result.prediction)
        score = float(getattr(value, "score", value))
        feedback = getattr(value, "feedback", None)
        return ScoreResult(
            score=score,
            status="success" if score > 0 else "semantic_failure",
            metadata={
                "metric": TASKS.get(self.dataset, {}).get("metric"),
                "feedback": None if feedback is None else str(feedback),
            },
        )


# Executor shared by optimization and evaluation
class CellExecutor:
    """One attempt = runtime + scorer -> (record, outcome, attempt evidence)."""

    def __init__(self, *, cell: CellSpec, runtime: Any, scorer: Any, data: TaskData) -> None:
        self.cell = cell
        self.runtime = runtime
        self.scorer = scorer
        self.data = data
        self.scorer_id = str(getattr(scorer, "scorer_id", type(scorer).__name__))
        self.implementation_id = str(getattr(runtime, "implementation_id", type(runtime).__name__))
        self.required_roles = tuple(runtime.required_roles)
        self.supports_demos = bool(getattr(runtime, "supports_demos", False))
        self._scorer_lock = threading.Lock()

    def validate_bundle(self, bundle: PromptBundle) -> None:
        """Reject a bundle the runtime cannot execute."""
        if not isinstance(bundle, PromptBundle):
            raise RunnerContractError(f"expected PromptBundle, got {type(bundle).__name__}")
        actual = tuple(sorted(bundle.roles))
        if actual != self.required_roles:
            raise RunnerContractError(
                f"prompt roles differ for {self.implementation_id}: expected={self.required_roles}, actual={actual}"
            )
        if bundle.demos and not self.supports_demos:
            raise UnsupportedRuntimeBinding(f"{self.implementation_id} cannot inject demonstrations")
        canonical_json({"roles": dict(bundle.roles), "demos": list(bundle.demos), "metadata": dict(bundle.metadata)})
        execution_hook_for(bundle)

    def resolve(self, example: Any, *, splits: Sequence[str]) -> tuple[str, Any]:
        """Map an optimizer example to the canonical native example of an allowed split."""
        if hasattr(example, "toDict"):
            example = example.toDict()
        if not isinstance(example, Mapping):
            raise RunnerContractError("runner examples must be mappings with an 'id'")
        item_id = example_id(example, fallback=None)
        if self.data.split_of(item_id) not in splits:
            raise RunnerContractError(f"example {item_id!r} is outside the allowed splits {tuple(splits)}")
        if content_hash(json_safe(dict(example))) != content_hash(self.data.row(item_id)):
            raise RunnerContractError(f"example {item_id!r} differs from the canonical row")
        return item_id, self.data.native(item_id)

    def attempt(
        self,
        *,
        example_id: str,
        native: Any,
        bundle: PromptBundle,
        request_seed: int,
        attempt_index: int,
        phase: str,
        evaluation_seed_identity: str | None = None,
        strict: bool = False,
    ) -> tuple[RunRecord, str, dict[str, Any]]:
        """Execute and score once; returns ``(record, outcome, attempt evidence)``.

        The outcome is the record's status. A runtime that fails before an
        observation, or a scorer that fails after one, gives an
        ``infrastructure_failure`` record. With ``strict`` (evaluation) an
        unclassified runtime error is re-raised and a scorer failure raises
        ``RunnerContractError``.
        """
        request = RuntimeInvocation(
            cell=self.cell,
            example_id=example_id,
            example=native,
            task_input=task_input_of(native, self.cell.task),
            bundle=bundle,
            request_seed=request_seed,
            attempt_index=attempt_index,
            phase=phase,
            evaluation_seed_identity=evaluation_seed_identity,
        )
        base = {
            "schema": schema_name("runtime-attempt"),
            "cell_id": self.cell.cell_id,
            "example_id": example_id,
            "bundle_sha256": bundle.digest,
            "request_seed": request_seed,
            "attempt_index": attempt_index,
            "phase": phase,
            "runtime_implementation_id": self.implementation_id,
            "scorer_id": self.scorer_id,
        }
        # Failures are recorded inside their ``except`` clause: ``_failure`` keeps the active traceback.
        started = time.monotonic()
        try:
            raw = self.runtime(request)
        except PreObservationInfrastructureFailure as exc:
            validate_usage(exc.usage, name="infrastructure_failure.usage")
            return self._failure(
                base, request, started, exc, stage=exc.stage, usage=exc.usage, pre_observation=True, extra=exc.metadata
            )
        except RunnerContractError as exc:
            return self._failure(
                base, request, started, exc, stage="runtime_contract", usage=Usage(), pre_observation=True
            )
        except Exception as exc:
            if strict:
                raise
            return self._failure(
                base, request, started, exc, stage="runtime_unclassified", usage=Usage(), pre_observation=True
            )
        self._check_observation(raw)
        try:
            with self._scorer_lock:
                scored = self.scorer(native, raw)
        except Exception as exc:
            if strict:
                raise RunnerContractError("scorer failed after a runtime observation") from exc
            return self._failure(
                base, request, started, exc, stage="scorer", usage=raw.usage, pre_observation=False, raw=raw
            )
        score = self._checked_score(scored)
        return self._scored(base, request, started, raw, scored, score)

    def _check_observation(self, raw: Any) -> None:
        """Reject a runtime result of the wrong type, with invalid usage or from another model."""
        if not isinstance(raw, RuntimeInvocationResult):
            raise RunnerContractError(f"runtime returned {type(raw).__name__}, expected RuntimeInvocationResult")
        validate_usage(raw.usage, name="task_usage")
        if raw.model_id and raw.model_id != self.cell.task_model:
            raise RunnerContractError("runtime returned a model ID different from the cell")

    @staticmethod
    def _checked_score(scored: Any) -> float:
        """The score of a valid ``ScoreResult`` (usable status, score in [0, 1], valid judge usage)."""
        if not isinstance(scored, ScoreResult):
            raise RunnerContractError(f"scorer returned {type(scored).__name__}, expected ScoreResult")
        if scored.status not in {"success", "semantic_failure"}:
            raise RunnerContractError(f"invalid usable status: {scored.status!r}")
        score = float(scored.score)
        if not 0.0 <= score <= 1.0:
            raise RunnerContractError(f"score outside [0,1]: {score}")
        validate_usage(scored.usage, name="judge_usage")
        return score

    def _scored(
        self,
        base: Mapping[str, Any],
        request: RuntimeInvocation,
        started: float,
        raw: RuntimeInvocationResult,
        scored: ScoreResult,
        score: float,
    ) -> tuple[RunRecord, str, dict[str, Any]]:
        """The usable record of a scored observation and its attempt evidence."""
        breakdown = {
            "task": usage_dict(raw.usage),
            "judge": usage_dict(scored.usage),
            "reflection": usage_dict(Usage()),
        }
        latency = time.monotonic() - started
        record = RunRecord(
            cell_id=self.cell.cell_id,
            example_id=request.example_id,
            status=scored.status,
            final_output=raw.final_output,
            score=score,
            messages=list(raw.messages),
            tool_events=list(raw.tool_events),
            usage=raw.usage,
            request_seed=request.request_seed,
            latency_seconds=latency,
            framework=self.cell.framework,
            model_id=raw.model_id or self.cell.task_model,
            trace_id=raw.trace_id,
            metadata={
                "runtime_implementation_id": self.implementation_id,
                "bundle_sha256": request.bundle.digest,
                "runtime_metadata": dict(raw.metadata),
                "scorer_id": self.scorer_id,
                "scorer_metadata": json_safe(dict(scored.metadata)),
                "usage_by_component": breakdown,
            },
        )
        canonical_json(record.to_dict())
        attempt = {
            **base,
            "outcome": scored.status,
            "pre_observation": False,
            "score": score,
            "trace_id": raw.trace_id,
            "latency_seconds": latency,
            "usage_by_component": breakdown,
            "record_sha256": content_hash(record.to_dict()),
        }
        attempt["attempt_id"] = content_hash(attempt)
        return record, scored.status, attempt

    def _failure(self, base, request, started, exc, *, stage, usage, pre_observation, raw=None, extra=None):
        """The ``infrastructure_failure`` record of a failed attempt and its attempt evidence.

        Called inside the ``except`` clause that caught ``exc``: the record keeps
        the active traceback (except for ``task_runtime`` failures).
        """
        latency = time.monotonic() - started
        error = scrub_text(f"{type(exc).__name__}: {exc}")[:2000]
        breakdown = {"task": usage_dict(usage), "judge": usage_dict(Usage()), "reflection": usage_dict(Usage())}
        record = RunRecord(
            cell_id=self.cell.cell_id,
            example_id=request.example_id,
            status="infrastructure_failure",
            final_output=None if raw is None else raw.final_output,
            messages=[] if raw is None else list(raw.messages),
            tool_events=[] if raw is None else list(raw.tool_events),
            usage=usage,
            request_seed=request.request_seed,
            latency_seconds=latency,
            framework=self.cell.framework,
            model_id=self.cell.task_model,
            trace_id=None if raw is None else raw.trace_id,
            error=error,
            metadata={
                "pre_observation": pre_observation,
                "failure_stage": stage,
                "traceback": scrub_text(traceback.format_exc())[-4000:] if stage != "task_runtime" else None,
                "adapter_metadata": json_safe(dict(extra or {})),
                "usage_by_component": breakdown,
            },
        )
        attempt = {
            **base,
            "outcome": "infrastructure_failure",
            "pre_observation": pre_observation,
            "error": error,
            "failure_stage": stage,
            "latency_seconds": latency,
            "usage_by_component": breakdown,
        }
        attempt["attempt_id"] = content_hash(attempt)
        return record, "infrastructure_failure", attempt


# Budget-charging runner handed to optimizers
@dataclass
class _Pending:
    original_index: int
    example_id: str
    native: Any
    request_seed: int
    attempt_index: int = 0
    task_usage: Usage = field(default_factory=Usage)
    judge_usage: Usage = field(default_factory=Usage)
    infrastructure_failures: int = 0
    latency_seconds: float = 0.0
    attempt_ids: list[str] = field(default_factory=list)


class ProtocolRunner:
    """Execute and charge complete MAS observations for one optimization job.

    ``run``/``run_batch`` reserve budget before dispatch,
    trim the last batch at the cap, retry only infrastructure failures (same
    seed, at most ``max_infrastructure_retries``) and charge usable records.
    Only train and validation rows are accepted.
    """

    supports_concurrent = False
    max_concurrent_evaluations = 1

    def __init__(
        self,
        *,
        cell: CellSpec,
        executor: CellExecutor,
        budget: BudgetLedger,
        seed_bundle: PromptBundle | None = None,
        phase: str = "optimization",
        max_infrastructure_retries: int = MAX_INFRASTRUCTURE_RETRIES,
        artifact_store: ArtifactStore | None = None,
        artifact_directory=None,
        event_sink: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> None:
        if phase_bucket(phase) != "optimization":
            raise ValueError("ProtocolRunner is search-budgeted; evaluations use evaluation.Evaluator")
        if max_infrastructure_retries < 0:
            raise ValueError("max_infrastructure_retries cannot be negative")
        if budget.maximum != cell.budget:
            raise ValueError("runner ledger and cell budget differ")
        self.cell = cell
        self.executor = executor
        self.budget = budget
        self.phase = phase
        self.max_infrastructure_retries = max_infrastructure_retries
        self.required_roles = executor.required_roles
        self.seed_bundle = seed_bundle
        self.initial_bundle = seed_bundle
        self.artifact_store = artifact_store
        self.artifact_directory = artifact_directory
        self.event_sink = event_sink
        self.job_identity = None
        self.usage_totals = {"task": Usage(), "judge": Usage()}
        self._write_lock = threading.Lock()
        self._started = time.monotonic()
        if seed_bundle is not None:
            self.validate_bundle(seed_bundle)

    def validate_bundle(self, bundle: PromptBundle) -> None:
        """Reject a bundle the cell's runtime cannot execute."""
        self.executor.validate_bundle(bundle)

    def _prepare(self, examples: Sequence[Any], bundle: PromptBundle, request_seeds: Sequence[int]) -> list[_Pending]:
        self.validate_bundle(bundle)
        self._persist_bundle(bundle)
        if len(examples) != len(request_seeds):
            raise ValueError("examples and request_seeds must have equal length")
        pending: list[_Pending] = []
        seen: set[str] = set()
        for index, (example, seed) in enumerate(zip(examples, request_seeds)):
            example_id, native = self.executor.resolve(example, splits=OPTIMIZATION_SPLITS)
            if example_id in seen:
                raise RunnerContractError(f"duplicate example ID in atomic batch: {example_id}")
            seen.add(example_id)
            if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < SEED_LIMIT:
                raise RunnerContractError(f"invalid request seed for {example_id}: {seed!r}")
            pending.append(_Pending(index, example_id, native, seed))
        return pending

    def run(self, example: Any, bundle: PromptBundle, request_seed: int) -> RunRecord:
        """Execute one charged logical rollout."""
        result = self.run_batch([example], bundle, [request_seed])
        if result.scheduled != 1:
            raise RuntimeError("optimization budget exhausted")
        return result.records[0]

    def run_batch(self, examples: Sequence[Any], bundle: PromptBundle, request_seeds: Sequence[int]) -> BatchExecution:
        """Reserve an atomic batch, trim at B, and retry only infrastructure failures."""
        prepared = self._prepare(examples, bundle, request_seeds)
        requested = len(prepared)
        if requested == 0:
            return BatchExecution((), 0, 0, 0, self.budget.snapshot())
        reservation = self.budget.reserve(requested)
        active = prepared[: reservation.size]
        scheduled = len(active)
        final: dict[int, RunRecord] = {}
        while active:
            outcomes: list[str] = []
            retry: list[_Pending] = []
            try:
                for item in active:
                    record, outcome, _ = self._attempt(item, bundle)
                    outcomes.append(outcome)
                    if outcome == "infrastructure_failure" and item.attempt_index < self.max_infrastructure_retries:
                        item.attempt_index += 1
                        retry.append(item)
                    else:
                        final[item.original_index] = self._finalize(record, item, bundle)
            except Exception:
                self.budget.commit_prefix(reservation, outcomes)
                raise
            self.budget.commit(reservation, outcomes)
            if retry:
                reservation = self.budget.reserve(len(retry))
                if reservation.size != len(retry):
                    raise AssertionError("infrastructure retries unexpectedly lost reserved capacity")
            active = retry
        ordered = tuple(final[index] for index in sorted(final))
        for record in ordered:
            self._record(record)
        return BatchExecution(ordered, requested, scheduled, requested - scheduled, self.budget.snapshot())

    def _attempt(self, item: _Pending, bundle: PromptBundle):
        record, outcome, attempt = self.executor.attempt(
            example_id=item.example_id,
            native=item.native,
            bundle=bundle,
            request_seed=item.request_seed,
            attempt_index=item.attempt_index,
            phase=self.phase,
        )
        item.attempt_ids.append(str(attempt["attempt_id"]))
        item.task_usage += record.usage
        item.latency_seconds += record.latency_seconds
        if outcome == "infrastructure_failure":
            item.infrastructure_failures += 1
        else:
            item.judge_usage += Usage(**record.metadata["usage_by_component"]["judge"])
        with self._write_lock:
            self.usage_totals["task"] += record.usage
            if outcome != "infrastructure_failure":
                self.usage_totals["judge"] += Usage(**record.metadata["usage_by_component"]["judge"])
        self._persist("attempts.jsonl", attempt)
        return record, outcome, attempt

    def _finalize(self, record: RunRecord, item: _Pending, bundle: PromptBundle) -> RunRecord:
        record.usage = item.task_usage
        record.latency_seconds = item.latency_seconds
        record.metadata = {
            **record.metadata,
            "phase": self.phase,
            "bundle_sha256": bundle.digest,
            "execution_attempts": item.attempt_index + 1,
            "infrastructure_failures": item.infrastructure_failures,
            "retry_seed_policy": "same_logical_request_seed",
            "attempt_ids": list(item.attempt_ids),
            "usage_by_component": {
                "task": usage_dict(item.task_usage),
                "judge": usage_dict(item.judge_usage),
                "reflection": usage_dict(Usage()),
            },
        }
        return record

    def _record(self, record: RunRecord) -> None:
        self._persist(
            "records.jsonl",
            {
                "schema": schema_name("run-result"),
                "record": record.to_dict(),
                "record_sha256": content_hash(record.to_dict()),
                "budget_snapshot": self.budget.snapshot(),
            },
        )
        if self.event_sink is not None:
            self.event_sink(
                {
                    "kind": "mas_completed",
                    "usable": record.usable,
                    "status": record.status,
                    "example_id": record.example_id,
                    "usage": usage_dict(record.usage),
                    "infrastructure_failures": int(record.metadata.get("infrastructure_failures", 0)),
                    "latency_seconds": record.latency_seconds,
                    "phase": self.phase,
                    "elapsed_seconds": time.monotonic() - self._started,
                    "budget": self.budget.snapshot(),
                }
            )

    def _persist_bundle(self, bundle: PromptBundle) -> None:
        if self.artifact_store is None or self.artifact_directory is None:
            return
        with self._write_lock:
            self.artifact_store.write_json(
                self.artifact_directory,
                f"bundle-{bundle.digest}.json",
                {
                    "schema": schema_name("prompt-bundle"),
                    "bundle_sha256": bundle.digest,
                    "roles": dict(bundle.roles),
                    "demos": list(bundle.demos),
                    "metadata": dict(bundle.metadata),
                },
            )

    def _persist(self, name: str, value: Mapping[str, Any]) -> None:
        if self.artifact_store is None or self.artifact_directory is None:
            return
        with self._write_lock:
            self.artifact_store.append_jsonl(self.artifact_directory, name, value)


# Endpoint configuration
def configure_task_endpoints(endpoints: Sequence[str], *, task_model: str) -> None:
    """Point every adapter at ``endpoints`` (round robin) serving ``task_model``.

    Must run before ``optimizers.bridge.lm`` is imported, or it re-seeds the
    already created endpoint cycle.
    """
    import itertools

    endpoints = tuple(item.strip() for item in endpoints if item and item.strip())
    if not endpoints:
        raise ValueError("at least one task endpoint is required")
    os.environ["TASK_ENDPOINTS"] = ",".join(endpoints)
    os.environ["VLLM_BASE_URL"] = endpoints[0]
    os.environ["TASK_MODEL"] = task_model
    os.environ["MODEL_ID"] = task_model
    lm = sys.modules.get("optimizers.bridge.lm")
    if lm is not None:
        with lm._adapter_endpoint_lock:
            lm._adapter_endpoint_cycle = itertools.cycle(endpoints)


__all__ = [
    "AdapterRuntime",
    "BatchExecution",
    "CONTROL_METADATA_KEYS",
    "CellExecutor",
    "DatasetMetricScorer",
    "ExecutionHook",
    "ProtocolRunner",
    "RuntimeInvocation",
    "RuntimeInvocationResult",
    "SCORER_PRIVATE_FIELDS",
    "ScoreResult",
    "TaskData",
    "configure_task_endpoints",
    "default_adapter_kwargs",
    "example_row",
    "load_task_data",
    "patched_environment",
    "phase_bucket",
    "register_execution_hook",
    "remember_role_order",
    "render_prompts",
    "runtime_role_order",
    "task_input_of",
]
