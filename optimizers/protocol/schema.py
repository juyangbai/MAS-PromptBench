"""Records and identities shared by the runner, the methods and reporting.

Content hashing, the prompt bundle, the job cell, run records, the row
identity, the stop-reason vocabulary and the one optimizer result schema.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any

from .config import (
    BUDGET,
    COMMUNICATIONS,
    DEFAULT_TASK_MODEL,
    OPTIMIZER_SEEDS,
    PROTOCOL_ID,
    REFLECTION_MODEL_ID,
    TEAM_SIZES,
)


# Hashing
def canonical_json(value: Any) -> str:
    """Sorted, compact JSON: the input of every content hash."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def content_hash(value: Any) -> str:
    """SHA-256 of the canonical JSON of ``value``."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def schema_name(thing: str) -> str:
    """Schema string for one artifact kind."""
    return f"mas-promptbench-{thing}"


# Core records
@dataclass(frozen=True)
class Usage:
    """Model/tool calls and token counts."""

    model_calls: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            model_calls=self.model_calls + other.model_calls,
            tool_calls=self.tool_calls + other.tool_calls,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )


@dataclass(frozen=True)
class PromptBundle:
    """Role prompts (plus optional demos/metadata) that define one MAS candidate.

    ``metadata`` is part of the digest. The runner reads ``metadata`` only for
    registered execution hooks (see ``runner.register_execution_hook``).
    """

    roles: Mapping[str, str]
    demos: tuple[Mapping[str, Any], ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.roles:
            raise ValueError("PromptBundle requires at least one role")
        for role, prompt in self.roles.items():
            if not role or not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"invalid role prompt: {role!r}")

    @property
    def digest(self) -> str:
        """Content hash of roles, demos and metadata."""
        return content_hash({"roles": dict(self.roles), "demos": list(self.demos), "metadata": dict(self.metadata)})


@dataclass(frozen=True)
class CellSpec:
    """One optimizer job identity: method x runtime condition x optimizer seed.

    ``task`` is the dataset key (gpqa, hotpotqa, math, lcb, apps, bfcl, swe,
    apibank, toolhop). The protocol budget is 600; smaller budgets are
    accepted for smoke runs and are reported as non-conformant, and so is a
    reflection model other than the protocol's.
    """

    method: str
    task: str
    topology: str
    framework: str
    communication: str = "freeform"
    team_size: int = 4
    task_model: str = DEFAULT_TASK_MODEL
    reflection_model: str = REFLECTION_MODEL_ID
    optimizer_seed: int = 0
    split_hash: str = ""
    protocol_id: str = PROTOCOL_ID
    protocol_hash: str = ""
    budget: int = BUDGET
    source_tables: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.topology == "single" and self.team_size != 1:
            object.__setattr__(self, "team_size", 1)
        if self.topology != "single" and self.team_size not in TEAM_SIZES:
            raise ValueError(f"invalid multi-agent team size: {self.team_size}")
        if self.communication not in COMMUNICATIONS:
            raise ValueError(f"invalid communication mode: {self.communication}")
        if self.optimizer_seed not in OPTIMIZER_SEEDS:
            raise ValueError(f"invalid optimizer seed: {self.optimizer_seed}")
        if type(self.budget) is not int or not 0 < self.budget <= BUDGET:
            raise ValueError(f"budget must be an integer in 1..{BUDGET}")

    @property
    def protocol_conformant(self) -> bool:
        """Full budget and the protocol reflection model under the frozen protocol."""
        return (
            self.budget == BUDGET and self.protocol_id == PROTOCOL_ID and self.reflection_model == REFLECTION_MODEL_ID
        )

    @property
    def identity(self) -> Mapping[str, Any]:
        """Every field that defines the job (hashed into ``cell_id``)."""
        return {
            "protocol_id": self.protocol_id,
            "protocol_hash": self.protocol_hash,
            "method": self.method,
            "task": self.task,
            "topology": self.topology,
            "framework": self.framework,
            "communication": self.communication,
            "team_size": self.team_size,
            "task_model": self.task_model,
            "reflection_model": self.reflection_model,
            "optimizer_seed": self.optimizer_seed,
            "split_hash": self.split_hash,
            "budget": self.budget,
        }

    @property
    def cell_id(self) -> str:
        """Short content hash of the identity."""
        return content_hash(self.identity)[:24]


@dataclass
class RunRecord:
    """One scored (or failed) full-MAS execution."""

    cell_id: str
    example_id: str
    status: str
    final_output: Any = None
    score: float | None = None
    messages: list[Mapping[str, Any]] = field(default_factory=list)
    tool_events: list[Mapping[str, Any]] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    request_seed: int | None = None
    latency_seconds: float = 0.0
    framework: str = ""
    model_id: str = ""
    trace_id: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def usable(self) -> bool:
        """A scored observation (``success`` or ``semantic_failure``)."""
        return self.status in {"success", "semantic_failure"}

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["usable"] = self.usable
        return value


def bundle_to_dict(bundle: PromptBundle) -> dict[str, Any]:
    """Bundle payload with its ``bundle_sha256``."""
    return {
        "roles": dict(bundle.roles),
        "demos": list(bundle.demos),
        "metadata": dict(bundle.metadata),
        "bundle_sha256": bundle.digest,
    }


def bundle_from_dict(value: Mapping[str, Any]) -> PromptBundle:
    """Rebuild a bundle and verify any stored digest."""
    bundle = PromptBundle(dict(value["roles"]), tuple(value.get("demos", ())), dict(value.get("metadata", {})))
    for key in ("bundle_sha256", "digest", "sha256"):
        if key in value and value[key] != bundle.digest:
            raise ValueError("bundle content hash mismatch")
    return bundle


def record_from_dict(value: Mapping[str, Any]) -> RunRecord:
    """Rebuild a run record from :meth:`RunRecord.to_dict`."""
    data = dict(value)
    data.pop("usable", None)
    data["usage"] = Usage(**data.get("usage", {}))
    return RunRecord(**data)


def incumbent_of(artifact: Any) -> PromptBundle | None:
    """The incumbent bundle an optimizer returned, if any."""
    bundle = getattr(artifact, "incumbent_bundle", None) or getattr(artifact, "selected_bundle", None)
    return bundle if isinstance(bundle, PromptBundle) else None


def sealed(value: Mapping[str, Any]) -> dict[str, Any]:
    """Copy of ``value`` with its own content hash under ``sha256``."""
    result = {key: item for key, item in value.items() if key != "sha256"}
    result["sha256"] = content_hash(result)
    return result


def verify_sealed(value: Mapping[str, Any]) -> dict[str, Any]:
    """``value`` without its ``sha256`` seal; raises when the seal does not match."""
    result = dict(value)
    digest = result.pop("sha256", None)
    if digest != content_hash(result):
        raise ValueError("artifact content hash mismatch")
    return result


# Rows
ROW_ID_KEYS = ("id", "example_id", "task_id", "problem_id", "question_id", "instance_id")


def example_id(example: Any, fallback: int | None = 0) -> str:
    """Stable ID of a task row; ``example-<fallback>`` when it has none (``fallback=None`` raises)."""
    for key in ROW_ID_KEYS:
        value = example.get(key) if isinstance(example, Mapping) else getattr(example, key, None)
        if value is not None and str(value):
            return str(value)
    if fallback is None:
        from .errors import RunnerContractError

        raise RunnerContractError("example has no stable ID")
    return f"example-{fallback}"


# Stop reasons
class StopReason:
    """Every ``stop_reason`` an optimizer result can carry, spelled as serialized.

    Methods inherited different spellings for the same event; both are kept.
    Running out of rollouts is ``BUDGET`` (HiveMind, MAPRO, MASPO) or
    ``ROLLOUT_BUDGET_SPENT`` (GEPA, MIPRO, MASPOB); TAVO names the point at
    which it stopped (``BUDGET_BEFORE_OUTER_ROUND`` / ``BUDGET_BEFORE_RETRY``)
    and MAMUT-GEPA the exact metric-call cap (``METRIC_CALL_CAP``).
    """

    # rollout budget
    BUDGET = "budget"
    ROLLOUT_BUDGET_SPENT = "rollout_budget_spent"
    BUDGET_BEFORE_OUTER_ROUND = "budget_before_outer_round"
    BUDGET_BEFORE_RETRY = "budget_before_retry"
    METRIC_CALL_CAP = "metric_call_cap"
    # native schedule completed
    MAX_FULL_EVALS = "max_full_evals"
    NUM_TRIALS_COMPLETE = "num_trials_complete"
    MAX_CYCLES = "max_cycles"
    MAX_ITERS = "max_iters"
    MAX_OUTER_ROUNDS = "max_outer_rounds"
    MAXIMUM_SEARCH_DEPTH = "maximum_search_depth"
    NATIVE_GEPA_STOP = "native_gepa_stop"
    # native early stopping
    FULL_EVAL_PLATEAU = "full_eval_plateau"
    PATIENCE = "patience"
    NATIVE_PATIENCE = "native_patience"
    # no search
    IDENTITY_NO_SEARCH = "identity_no_search"
    # ``run.py`` default for a result that names none
    NATIVE_RETURN = "native_return"


# Optimizer result
RESULT_SCHEMA = schema_name("native-optimizer-result")
SEARCH_LAYOUT = "search"
JOURNAL_LAYOUT = "journal"


@dataclass(frozen=True)
class LearningCurvePoint:
    """One committed state of a search-layout learning curve."""

    event: str
    iteration: int
    charged_rollouts: int
    score: float | None
    bundle_hash: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


def to_jsonable(value: Any) -> Any:
    """JSON view of results and checkpoints (bundles carry their ``digest``; dataclasses become dicts)."""
    if isinstance(value, PromptBundle):
        return {
            "roles": dict(value.roles),
            "demos": [to_jsonable(item) for item in value.demos],
            "metadata": to_jsonable(dict(value.metadata)),
            "digest": value.digest,
        }
    if is_dataclass(value):
        return {key: to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


@dataclass(frozen=True)
class OptimizerResult:
    """What every method's ``optimize`` returns; serialized as ``optimization/optimizer_result.json``.

    One schema, two serialized layouts (field names and values are part of
    the published artifacts):

    * ``SEARCH_LAYOUT`` (GEPA, MIPRO, MASPOB, TAVO): ``selected_bundle``,
      ``implementation_kind``, ``production_eligible``, ``reflection_requests``;
      bundles without a digest and the stop reason inside ``metadata``.
    * ``JOURNAL_LAYOUT`` (identity and the ``RunnerSession`` methods):
      ``incumbent_bundle``, ``protocol_id``, ``native_iterations``, ``records``,
      ``request_ledger``, ``events``; bundles with ``bundle_sha256`` and a
      top-level ``stop_reason``.
    """

    method: str
    cell_id: str
    seed_bundle: PromptBundle
    incumbent_bundle: PromptBundle
    budget_snapshot: Mapping[str, int]
    stop_reason: str
    layout: str = JOURNAL_LAYOUT
    schema: str = RESULT_SCHEMA
    learning_curve: tuple[Any, ...] = ()
    checkpoints: tuple[Mapping[str, Any], ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    # journal layout
    protocol_id: str = PROTOCOL_ID
    native_iterations: int = 0
    records: tuple[RunRecord, ...] = ()
    request_ledger: tuple[Mapping[str, Any], ...] = ()
    events: tuple[Mapping[str, Any], ...] = ()
    # search layout
    implementation_kind: str = ""
    production_eligible: bool = True
    reflection_requests: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if self.layout not in (SEARCH_LAYOUT, JOURNAL_LAYOUT):
            raise ValueError(f"unknown optimizer result layout {self.layout!r}")

    @property
    def selected_bundle(self) -> PromptBundle:
        """The incumbent (search-layout name)."""
        return self.incumbent_bundle

    def to_dict(self) -> dict[str, Any]:
        """The serialized result in its layout."""
        if self.layout == SEARCH_LAYOUT:

            def plain(bundle: PromptBundle) -> dict[str, Any]:
                return {"roles": dict(bundle.roles), "demos": list(bundle.demos), "metadata": dict(bundle.metadata)}

            return to_jsonable(
                {
                    "schema": self.schema,
                    "method": self.method,
                    "implementation_kind": self.implementation_kind,
                    "production_eligible": self.production_eligible,
                    "cell_id": self.cell_id,
                    "seed_bundle": plain(self.seed_bundle),
                    "selected_bundle": plain(self.incumbent_bundle),
                    "budget_snapshot": self.budget_snapshot,
                    "learning_curve": self.learning_curve,
                    "checkpoints": self.checkpoints,
                    "reflection_requests": self.reflection_requests,
                    "metadata": {**dict(self.metadata), "stop_reason": self.stop_reason},
                }
            )
        return {
            "schema": self.schema,
            "method": self.method,
            "protocol_id": self.protocol_id,
            "cell_id": self.cell_id,
            "seed_bundle": bundle_to_dict(self.seed_bundle),
            "incumbent_bundle": bundle_to_dict(self.incumbent_bundle),
            "native_iterations": self.native_iterations,
            "stop_reason": self.stop_reason,
            "records": [record.to_dict() for record in self.records],
            "request_ledger": list(self.request_ledger),
            "events": list(self.events),
            "checkpoints": list(self.checkpoints),
            "learning_curve": list(self.learning_curve),
            "budget_snapshot": dict(self.budget_snapshot),
            "metadata": dict(self.metadata),
        }


__all__ = [
    "CellSpec",
    "JOURNAL_LAYOUT",
    "LearningCurvePoint",
    "OptimizerResult",
    "PromptBundle",
    "RESULT_SCHEMA",
    "ROW_ID_KEYS",
    "RunRecord",
    "SEARCH_LAYOUT",
    "StopReason",
    "Usage",
    "bundle_from_dict",
    "bundle_to_dict",
    "canonical_json",
    "content_hash",
    "example_id",
    "incumbent_of",
    "record_from_dict",
    "schema_name",
    "sealed",
    "to_jsonable",
    "verify_sealed",
]
