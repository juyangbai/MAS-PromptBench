"""Rollout surfaces the methods use on top of the protocol runner.

* :func:`load_seed_bundle`, :func:`ordered_prompt_roles`, :func:`native_role_order`,
  :func:`role_trace_messages`: the frozen seed bundle and the native role order;
* :func:`example_mapping`: the complete task mapping the runner expects for a row;
* :class:`BudgetedRunner`: one charged rollout per row through the budget-owning runner
  (or, for endpoint-free fakes, a raw runner under the same ledger rules);
* :class:`BudgetStopRunner`: answers rows requested past B without running them.

Importing this module contacts no endpoint.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any, Protocol

from .budget import budget_exhausted_record, is_budget_exhausted_error, remaining_rollouts
from .errors import NativeIntegrationError, OptimizerContractError, OptimizerInfrastructureFailure
from .schema import CellSpec, PromptBundle, RunRecord, example_id


class Runner(Protocol):
    """What a method needs of a runner: one rollout of a bundle on a row with a request seed."""

    def run(self, example: Any, bundle: PromptBundle, request_seed: int) -> RunRecord: ...


def example_mapping(example: Any) -> dict[str, Any]:
    """Return the complete serializable task mapping the runner expects."""
    if isinstance(example, Mapping):
        return dict(example)
    if hasattr(example, "toDict"):
        value = example.toDict()
        if isinstance(value, Mapping):
            return dict(value)
    value = getattr(example, "task_instance", None)
    if isinstance(value, Mapping):
        result = dict(value)
        result.setdefault("id", example_id(example))
        return result
    values = getattr(example, "__dict__", None)
    if isinstance(values, Mapping):
        result = {str(key): item for key, item in values.items() if not str(key).startswith("_")}
        if result:
            result.setdefault("id", example_id(example))
            return result
    raise OptimizerContractError(f"optimizer example {type(example).__name__} cannot be converted to a task mapping")


# Seed bundle and role order
def load_seed_bundle(cell: CellSpec, runner: Any, supplied: PromptBundle | None = None) -> PromptBundle:
    """The exact runtime seed bundle, with no role-name inference."""
    if supplied is not None:
        bundle = supplied
    else:
        candidate = getattr(runner, "seed_bundle", None) or getattr(runner, "initial_bundle", None)
        if not isinstance(candidate, PromptBundle):
            raise OptimizerContractError("seed PromptBundle is required when the runner does not expose seed_bundle")
        bundle = candidate
    required = tuple(getattr(runner, "required_roles", ()) or tuple(bundle.roles))
    if tuple(sorted(bundle.roles)) != tuple(sorted(required)):
        raise OptimizerContractError(
            f"seed prompt roles disagree with the runtime: expected={sorted(required)}, actual={sorted(bundle.roles)}"
        )
    return PromptBundle(roles=dict(bundle.roles), demos=tuple(bundle.demos), metadata=dict(bundle.metadata))


def _runtime_role_order(cell: CellSpec) -> tuple[str, ...] | None:
    from .runner import runtime_role_order

    source_cell = replace(cell, communication="freeform") if cell.communication != "freeform" else cell
    return runtime_role_order(cell) or runtime_role_order(source_cell)


def ordered_prompt_roles(cell: CellSpec, bundle: PromptBundle) -> tuple[str, ...]:
    """Native component order (the adapter's ``roles()``) for round-robin/seeded loops.

    Bundle hashes are order-independent but predictor order is scientific
    state. Fixtures whose role set differs from the runtime keep their own order.
    """
    roles = _runtime_role_order(cell)
    if roles and set(roles) == set(bundle.roles):
        return tuple(roles)
    return tuple(bundle.roles)


def native_role_order(cell: CellSpec, roles: Sequence[str]) -> list[str]:
    """Stage order of a sequential pipeline from the exact runtime, never sorted prompts."""
    if cell.topology != "sequential":
        return list(roles)
    order = _runtime_role_order(cell)
    if not order or set(order) != set(roles) or len(order) != len(roles):
        raise NativeIntegrationError("sequential prompt roles do not match the exact native stage order")
    return list(order)


def role_trace_messages(
    cell: CellSpec, roles: Sequence[str], role: str, messages: Sequence[Mapping[str, Any]]
) -> list[Mapping[str, Any]]:
    """Keep named native speakers and all replica evidence for a shared prompt."""
    if len(roles) == 1 and cell.topology in {"single", "independent", "decentralized"}:
        return list(messages)
    return [
        message
        for message in messages
        if str(message.get("source") or message.get("name") or message.get("agent") or message.get("role") or "")
        == role
    ]


# Charged rollouts
def validate_budget_owner(cell: CellSpec, runner: Any, budget: Any) -> None:
    """The method's ledger must be the cell's and the runner's."""
    if getattr(budget, "maximum", None) != cell.budget:
        raise OptimizerContractError("optimizer ledger maximum differs from the cell budget")
    owner = getattr(runner, "budget", None)
    if owner is not None and owner is not budget:
        raise OptimizerContractError("runner and optimizer were given different budget ledgers")


class BudgetedRunner:
    """One rollout surface for the protocol runner and raw offline fakes.

    A budget-owning runner (``runner.budget is budget``) already reserves,
    retries and charges, so calls are delegated. The raw-runner branch exists
    for endpoint-free tests and applies the identical ledger rules.
    """

    def __init__(self, *, cell: CellSpec, runner: Runner, budget: Any, max_infrastructure_retries: int = 2) -> None:
        validate_budget_owner(cell, runner, budget)
        if max_infrastructure_retries < 0:
            raise ValueError("max_infrastructure_retries cannot be negative")
        self.cell = cell
        self.runner = runner
        self.budget = budget
        self.max_infrastructure_retries = max_infrastructure_retries
        self.runner_owns_budget = getattr(runner, "budget", None) is budget

    def run(self, example: Any, bundle: PromptBundle, request_seed: int) -> RunRecord:
        """One charged rollout; exhausted infrastructure retries raise."""
        records = self.run_batch((example,), bundle, (request_seed,))
        if len(records) != 1:
            raise OptimizerContractError("single rollout did not return exactly one record")
        return records[0]

    @staticmethod
    def _validate_record(record: Any) -> RunRecord:
        if not isinstance(record, RunRecord):
            raise OptimizerContractError(f"runner returned {type(record).__name__}, expected RunRecord")
        if record.status not in {"success", "semantic_failure", "infrastructure_failure"}:
            raise OptimizerContractError(f"unsupported RunRecord status {record.status!r}")
        if record.status in {"success", "semantic_failure"} and record.score is None:
            raise OptimizerContractError("usable RunRecord is missing its score")
        return record

    def run_batch(self, examples: Sequence[Any], bundle: PromptBundle, seeds: Sequence[int]) -> tuple[RunRecord, ...]:
        """One charged rollout per row, in order; exhausted infrastructure retries raise.

        A runner that owns the ledger charges its own rollouts (as one native
        batch when it has ``run_batch``) and is checked to charge exactly the
        usable ones; any other runner is charged here, one reservation per attempt.
        """
        if len(examples) != len(seeds):
            raise ValueError("examples and seeds must have equal length")
        mapped = tuple(example_mapping(example) for example in examples)
        if not mapped:
            return ()
        native_batch = getattr(self.runner, "run_batch", None)
        if self.runner_owns_budget and callable(native_batch):
            return self._run_native_batch(native_batch, mapped, bundle, seeds)
        run_one = self._run_self_charged if self.runner_owns_budget else self._run_charged
        return tuple(run_one(example, bundle, seed) for example, seed in zip(mapped, seeds))

    def _run_native_batch(
        self, native_batch: Any, mapped: tuple[dict, ...], bundle: PromptBundle, seeds: Sequence[int]
    ) -> tuple[RunRecord, ...]:
        """The budget-owning runner's own batch: it must schedule every row and return only usable records."""
        result = native_batch(mapped, bundle, tuple(seeds))
        records = tuple(getattr(result, "records", ()))
        if int(getattr(result, "scheduled", len(records))) != len(mapped):
            raise OptimizerContractError("native optimizer requested a batch larger than remaining B")
        validated = tuple(self._validate_record(record) for record in records)
        failed = [record.example_id for record in validated if not record.usable]
        if failed:
            raise OptimizerInfrastructureFailure("infrastructure retries exhausted for: " + ", ".join(failed))
        return validated

    def _run_self_charged(self, example: Any, bundle: PromptBundle, seed: Any) -> RunRecord:
        """One rollout of a budget-owning runner, which must charge exactly one slot for a usable record."""
        before = self.budget.operation_snapshot()
        record = self._validate_record(self.runner.run(example, bundle, int(seed)))
        after = self.budget.operation_snapshot()
        if int(after["charged"]) - int(before["charged"]) != (1 if record.usable else 0):
            raise OptimizerContractError("budget-owning runner did not account exactly one logical rollout")
        if not record.usable:
            raise OptimizerInfrastructureFailure(f"infrastructure retries exhausted for {record.example_id}")
        return record

    def _run_charged(self, example: Any, bundle: PromptBundle, seed: Any) -> RunRecord:
        """One rollout charged here: each attempt reserves one slot and commits it with its outcome.

        A runner exception is an infrastructure failure and is retried; an
        invalid record is committed as one and raised.
        """
        for _ in range(self.max_infrastructure_retries + 1):
            reservation = self.budget.reserve(1)
            try:
                raw = self.runner.run(example, bundle, int(seed))
            except Exception:
                self.budget.commit(reservation, ("infrastructure_failure",))
                raw = None
            if raw is None:
                continue
            try:
                record = self._validate_record(raw)
            except Exception:
                self.budget.commit(reservation, ("infrastructure_failure",))
                raise
            self.budget.commit(reservation, (record.status,))
            if record.usable:
                return record
        raise OptimizerInfrastructureFailure(f"infrastructure retries exhausted for {example_id(example)}")


class BudgetStopRunner:
    """Answer rows past B without running them; every other row goes to ``runner``.

    GEPA and MIPRO size batches from their own schedules, so the last
    minibatch or full validation pass can ask for more rows than the ledger
    has left. Spending B is the protocol's stop, not a failure: such a row
    gets ``budget_exhausted_record`` (a semantic failure with the failure
    score, never executed or charged) and the optimizer returns its incumbent.
    ``runner`` is the :class:`BudgetedRunner` over the protocol runner.
    """

    def __init__(self, runner: BudgetedRunner) -> None:
        self._runner = runner
        self.cell = runner.cell
        self.budget = runner.budget
        self.answered = 0
        self._lock = threading.Lock()

    def _spent(self, example: Any) -> RunRecord:
        with self._lock:
            self.answered += 1
        return budget_exhausted_record(self.cell, example)

    def run(self, example: Any, bundle: PromptBundle, request_seed: int) -> RunRecord:
        """One charged rollout, or the budget-exhausted record once B is spent."""
        if remaining_rollouts(self.budget) == 0:
            return self._spent(example)
        try:
            return self._runner.run(example, bundle, request_seed)
        except RuntimeError as exc:
            if not is_budget_exhausted_error(exc):
                raise
            return self._spent(example)


__all__ = [
    "BudgetStopRunner",
    "BudgetedRunner",
    "Runner",
    "example_mapping",
    "load_seed_bundle",
    "native_role_order",
    "ordered_prompt_roles",
    "role_trace_messages",
    "validate_budget_owner",
]
