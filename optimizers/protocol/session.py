"""Prompt-mutable, budget-safe session over one protocol-runner cell.

Retained native optimizer loops (HiveMind, MAMUT-GEPA, MAPRO, MASPO) expect
an adapter (``roles``/``get_prompt``/``set_prompt``/``run_example``) they can
mutate between rollouts. :class:`RunnerSession` is that adapter: every rollout
goes through the budget-owning protocol runner with a deterministic logical
seed, and :class:`NativeRolloutBudget` mirrors the native loops' pacing on the
authoritative ledger without charging twice.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from typing import Any

from .errors import NativeInfrastructureExhausted, NativeIntegrationError, PreObservationInfrastructureFailure
from .rollouts import Runner
from .schema import CellSpec, PromptBundle, RunRecord, content_hash, example_id
from .seeding import logical_request_seed


def _normalize_messages(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for raw in messages:
        value = dict(raw)
        # LangChain dumps use ``type``; keep user/system identity distinct from agent output.
        message_type = {"human": "user", "ai": "assistant", "system": "system", "tool": "tool", "function": "tool"}.get(
            value.get("type")
        )
        if not value.get("source"):
            value["source"] = value.get("role") or value.get("name") or message_type or "unknown"
        normalized.append(value)
    return normalized


def record_to_native_output(record: RunRecord) -> dict[str, Any]:
    """Losslessly expose a run record to retained prompt-mutable drivers."""
    if isinstance(record.final_output, Mapping):
        output = dict(record.final_output)
    else:
        output = {"answer_text": "" if record.final_output is None else str(record.final_output)}
    output.setdefault("answer_text", "" if record.final_output is None else str(record.final_output))
    output.setdefault("model_output", [])
    output["messages"] = _normalize_messages(record.messages)
    output["runner_score"] = record.score
    output["runner_status"] = record.status
    output["runner_record"] = record
    output["runner_output"] = {**dict(output.get("runner_output") or {}), "common_run_record": record.to_dict()}
    if record.error:
        output["error"] = record.error
    return output


class NativeRolloutBudget:
    """Native pacing facade backed by the authoritative ledger.

    Retained loops reserve batches before dispatch; this mirrors that pacing
    without charging twice. Infrastructure failures refund native pacing
    because only usable full-MAS observations are counted.
    """

    def __init__(self, ledger: Any) -> None:
        maximum = int(getattr(ledger, "maximum", -1))
        if maximum <= 0:
            raise NativeIntegrationError("native integrations require a positive budget ledger")
        self.ledger = ledger
        snap = dict(ledger.snapshot())
        if int(snap.get("reserved", 0)) != 0:
            raise NativeIntegrationError("optimizer cannot start with open budget reservations")
        self.cap = self.total = maximum
        self.used = int(snap.get("charged", 0))
        self.truncations = self.stops = 0
        self._pending = 0
        self._claimed = 0
        self._lock = threading.RLock()

    def remaining(self) -> int:
        """Slots the native loop may still reserve."""
        with self._lock:
            authoritative = int(self.ledger.snapshot()["remaining"])
            return max(0, min(self.cap - self.used, authoritative - self._pending))

    @property
    def exhausted(self) -> bool:
        """No slot left."""
        return self.remaining() <= 0

    def reserve(self, requested: int) -> int:
        """Reserve up to ``requested`` slots; returns how many were granted."""
        if requested < 0:
            raise ValueError("requested native rollout count cannot be negative")
        if requested == 0:
            return 0
        with self._lock:
            authoritative = int(self.ledger.snapshot()["remaining"])
            available = max(0, min(self.cap - self.used, authoritative - self._pending))
            granted = min(int(requested), available)
            self._pending += granted
            self.used += granted
            if granted < requested:
                self.truncations += 1
                self.stops += 1
            return granted

    def try_charge(self, requested: int) -> bool:
        """Reserve exactly ``requested`` slots, or none (atomic native batch)."""
        if requested <= 0:
            raise ValueError("atomic native batch must be positive")
        with self._lock:
            authoritative = int(self.ledger.snapshot()["remaining"])
            available = max(0, min(self.cap - self.used, authoritative - self._pending))
            if requested > available:
                self.truncations += 1
                self.stops += 1
                return False
            self._pending += requested
            self.used += requested
            return True

    def ensure_one(self) -> bool:
        """Claim one slot for the next dispatch (reserving it when none is pending)."""
        with self._lock:
            if self._pending <= self._claimed and self.reserve(1) != 1:
                return False
            self._claimed += 1
            return True

    def settle(self, status: str) -> None:
        """Settle one dispatched slot; an infrastructure failure gives it back."""
        with self._lock:
            if self._pending <= 0:
                raise NativeIntegrationError("native loop dispatched an unreserved rollout")
            self._pending -= 1
            self._claimed -= 1
            if status == "infrastructure_failure":
                self.used -= 1

    @property
    def pending(self) -> int:
        """Reserved slots not yet settled."""
        with self._lock:
            return self._pending


@dataclass(frozen=True)
class RequestEvent:
    """Ledger entry of one rollout requested by a native loop."""

    event_id: str
    phase: str
    native_iteration: int
    example_id: str
    role: str
    turn: int
    bundle_sha256: str
    request_seed: int
    status: str
    score: float | None
    record_sha256: str
    usage: Mapping[str, int]


class RunnerSession:
    """Prompt-mutable, budget-safe facade over one exact protocol-runner cell."""

    def __init__(
        self,
        *,
        cell: CellSpec,
        runner: Runner,
        budget: Any,
        seed_bundle: PromptBundle,
        phase: str = "optimization",
        max_infrastructure_retries: int = 2,
    ) -> None:
        if getattr(budget, "maximum", None) != cell.budget:
            raise NativeIntegrationError("ledger maximum differs from the cell budget")
        if max_infrastructure_retries < 0:
            raise ValueError("max_infrastructure_retries cannot be negative")
        self.cell = cell
        self.runner = runner
        self.budget = budget
        self.seed_bundle = seed_bundle
        self.pacing = NativeRolloutBudget(budget)
        self.phase = phase
        self.native_iteration = 0
        self.native_event = "initialize"
        self._roles = dict(seed_bundle.roles)
        self._demos = tuple(seed_bundle.demos)
        self._metadata = dict(seed_bundle.metadata)
        self._max_infrastructure_retries = max_infrastructure_retries
        self._turns: dict[tuple[str, int, str, str], int] = {}
        self._lock = threading.RLock()
        self._dispatch_lock = threading.RLock()
        self._dispatch_state = {"allocated": 0, "published": 0, "ready": {}}
        self.records: list[RunRecord] = []
        self.request_events: list[RequestEvent] = []
        owner = getattr(runner, "budget", None)
        if owner is not None and owner is not budget:
            raise NativeIntegrationError("runner and optimizer must share the identical Any")
        self._runner_owns_budget = owner is budget

    @property
    def bundle(self) -> PromptBundle:
        with self._lock:
            return PromptBundle(roles=dict(self._roles), demos=self._demos, metadata=dict(self._metadata))

    def roles(self) -> list[str]:
        with self._lock:
            return list(self._roles)

    def get_prompt(self, role: str) -> str:
        with self._lock:
            self._check_role(role)
            return self._roles[role]

    def set_prompt(self, role: str, text: str) -> None:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("role prompts must be non-empty strings")
        with self._lock:
            self._check_role(role)
            self._roles[role] = text

    def set_bundle_metadata(self, **values: Any) -> None:
        """Bundle metadata passthrough, e.g. ``optimizer_control`` for masked execution."""
        with self._lock:
            self._metadata.update(values)

    def set_context(self, *, iteration: int, event: str) -> None:
        """Native iteration and event that key the next rollouts' seeds."""
        if iteration < 0 or not event:
            raise ValueError("native context requires a non-negative iteration and event")
        with self._lock:
            self.native_iteration = iteration
            self.native_event = event

    def reset(self) -> None:
        """Runtime state is owned by the runner; prompt state intentionally remains."""

    def run_example(self, example: Any) -> dict[str, Any]:
        """One rollout of the current bundle, in the adapter-output shape native loops read."""
        record = self.run_record(example)
        output = record_to_native_output(record)
        if len(self._roles) == 1:
            shared = next(iter(self._roles))
            output["messages"] = [
                {**message, "original_source": message["source"], "source": shared}
                if message["source"] not in {"user", "human", "system"}
                else message
                for message in output["messages"]
            ]
        return output

    def run_record(self, example: Any, *, role: str = "full_mas") -> RunRecord:
        """One charged rollout of the current bundle (``role`` keys the seed)."""
        guard = nullcontext() if getattr(self.runner, "supports_concurrent", False) else self._dispatch_lock
        with guard:
            return self._run_record(example, role=role)

    def _prepare_record(self, example, role):
        if not self.pacing.ensure_one():
            raise NativeIntegrationError("optimization budget exhausted")
        bundle = self.bundle
        eid = example_id(example)
        with self._lock:
            key = (self.native_event, self.native_iteration, bundle.digest, eid)
            turn = self._turns.get(key, 0)
            self._turns[key] = turn + 1
            iteration, native_event = self.native_iteration, self.native_event
        seed = logical_request_seed(
            self.cell.optimizer_seed,
            self.cell.cell_id,
            self.phase,
            iteration,
            eid,
            role if role != "full_mas" else native_event,
            turn,
        )
        with self._dispatch_lock:
            order = self._dispatch_state["allocated"]
            self._dispatch_state["allocated"] += 1
        return dict(bundle=bundle, eid=eid, turn=turn, seed=seed, iteration=iteration, order=order)

    def run_records(self, examples: Sequence[Mapping[str, Any]], *, role: str = "full_mas") -> list[RunRecord]:
        """Evaluate one existing native batch with fixed input order and seeds."""
        rows = tuple(examples)
        width = int(getattr(self.runner, "max_concurrent_evaluations", 1))
        if not getattr(self.runner, "supports_concurrent", False) or width <= 1:
            return [self.run_record(row, role=role) for row in rows]
        prepared = [self._prepare_record(row, role) for row in rows]
        if not rows:
            return []
        batch = getattr(self.runner, "run_batch", None)
        if self._runner_owns_budget and callable(batch):
            result = batch(rows, prepared[0]["bundle"], [item["seed"] for item in prepared])
            if len(result.records) != len(rows):
                raise NativeIntegrationError("native fixed batch was unexpectedly trimmed")
            return [self._finish_record(record, role=role, **item) for record, item in zip(result.records, prepared)]
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=width) as pool:
            return list(
                pool.map(lambda item: self._run_record(item[0], role=role, _prepared=item[1]), zip(rows, prepared))
            )

    def _run_record(self, example: Any, *, role: str = "full_mas", _prepared=None) -> RunRecord:
        prepared = _prepared or self._prepare_record(example, role)
        bundle, eid, seed = prepared["bundle"], prepared["eid"], prepared["seed"]
        attempts = 1 if self._runner_owns_budget else self._max_infrastructure_retries + 1
        record: RunRecord | None = None
        for attempt in range(attempts):
            reservation = None
            before = self.budget.operation_snapshot()
            try:
                if not self._runner_owns_budget:
                    reservation = self.budget.reserve(1)
                    if int(getattr(reservation, "size", 1)) != 1:
                        raise NativeIntegrationError("single rollout reservation was unexpectedly trimmed")
                record = self.runner.run(example, bundle, seed)
            except PreObservationInfrastructureFailure as exc:
                if self._runner_owns_budget:
                    raise NativeIntegrationError(
                        "budget-owning runner leaked an unhandled infrastructure failure"
                    ) from exc
                assert reservation is not None
                self.budget.commit(reservation, ("infrastructure_failure",))
                if attempt + 1 < attempts:
                    continue
                record = RunRecord(
                    cell_id=self.cell.cell_id,
                    example_id=eid,
                    status="infrastructure_failure",
                    request_seed=seed,
                    error="pre-observation infrastructure retries exhausted",
                )
            except Exception as exc:
                if reservation is not None:
                    self.budget.cancel(reservation)
                self.pacing.settle("infrastructure_failure")
                raise NativeIntegrationError(
                    f"runner raised an unclassified {type(exc).__name__}; optimizer aborted"
                ) from exc
            if not isinstance(record, RunRecord):
                if reservation is not None:
                    self.budget.cancel(reservation)
                self.pacing.settle("infrastructure_failure")
                raise NativeIntegrationError(f"runner returned {type(record).__name__}, expected RunRecord")
            if record.status not in {"success", "semantic_failure", "infrastructure_failure"}:
                if reservation is not None:
                    self.budget.cancel(reservation)
                self.pacing.settle("infrastructure_failure")
                raise NativeIntegrationError(f"invalid runner status {record.status!r}")
            if reservation is not None:
                self.budget.commit(reservation, (record.status,))
            after = self.budget.operation_snapshot()
            expected_delta = 0 if record.status == "infrastructure_failure" else 1
            if int(after["charged"]) - int(before["charged"]) != expected_delta:
                self.pacing.settle("infrastructure_failure")
                raise NativeIntegrationError("runner budget charge did not match rollout status")
            if record.status == "infrastructure_failure" and attempt + 1 < attempts:
                continue
            break
        assert record is not None
        return self._finish_record(record, role=role, **prepared)

    def _finish_record(self, record, *, role, bundle, eid, seed, turn, iteration, order):
        if not isinstance(record, RunRecord) or record.status not in {
            "success",
            "semantic_failure",
            "infrastructure_failure",
        }:
            raise NativeIntegrationError("batch returned an invalid rollout record")
        self.pacing.settle(record.status)
        if record.request_seed is None:
            record.request_seed = seed
        elif record.request_seed != seed:
            raise NativeIntegrationError("runner changed the logical request seed")
        event_payload = {
            "phase": self.phase,
            "native_iteration": iteration,
            "example_id": eid,
            "role": role,
            "turn": turn,
            "bundle_sha256": bundle.digest,
            "request_seed": seed,
            "status": record.status,
            "score": record.score,
            "record_sha256": content_hash(record.to_dict()),
        }
        event = RequestEvent(
            event_id=content_hash(event_payload),
            usage={key: int(value) for key, value in asdict(record.usage).items()},
            **event_payload,
        )
        with self._dispatch_lock:
            state = self._dispatch_state
            state["ready"][order] = (record, event)
            while state["published"] in state["ready"]:
                ready_record, ready_event = state["ready"].pop(state["published"])
                self.records.append(ready_record)
                self.request_events.append(ready_event)
                state["published"] += 1
        if record.status == "infrastructure_failure":
            raise NativeInfrastructureExhausted(
                f"infrastructure retries exhausted for {eid}; no score was supplied to native optimizer"
            )
        return record

    def format_role_trace(self, role: str, output: Any) -> str:
        """Reflection trace of one role: status, score, final output and the role's messages."""
        self._check_role(role)
        if not isinstance(output, Mapping):
            return str(output)
        messages = [value for value in output.get("messages", []) if value.get("source") == role]
        return "\n".join(
            (
                f"role={role}",
                f"status={output.get('runner_status')}",
                f"score={output.get('runner_score')}",
                f"final_output={output.get('answer_text', '')}",
                f"messages={json.dumps(messages, ensure_ascii=False, default=str)}",
            )
        )

    def _check_role(self, role: str) -> None:
        if role not in self._roles:
            raise KeyError(f"unknown role {role!r}; choices={list(self._roles)}")


__all__ = ["NativeRolloutBudget", "RequestEvent", "RunnerSession", "record_to_native_output"]
