"""The 600-rollout budget: a thread-safe reservation and charge ledger, and its stop.

Usable observations (``success`` / ``semantic_failure``) are charged.
``infrastructure_failure`` attempts are counted but never charged, and the
ledger never lets ``charged`` exceed ``maximum``.

Spending B is the protocol's stop, not a failure. GEPA and MIPRO size batches
from their own schedules, not from B, so the last minibatch or full
evaluation can ask for rows past the ledger: such a row is answered by
:func:`budget_exhausted_record`, a semantic failure with the failure score
that is never executed or charged, and the optimizer returns its incumbent.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from .config import BUDGET
from .schema import RunRecord, example_id

EXHAUSTED_MESSAGE = "optimization budget exhausted"
BUDGET_EXHAUSTED_ERROR = "rollout budget exhausted; rollout not executed"

Outcome = Literal["success", "semantic_failure", "infrastructure_failure"]
OUTCOMES = frozenset({"success", "semantic_failure", "infrastructure_failure"})


@dataclass(frozen=True)
class BudgetReservation:
    """Slots reserved by one dispatch, closed by exactly one commit or cancel."""

    reservation_id: int
    size: int


class BudgetLedger:
    """Reservation-before-dispatch ledger of charged (usable) and attempted rollouts."""

    def __init__(self, maximum: int = BUDGET) -> None:
        if type(maximum) is not int or not 0 < maximum <= BUDGET:
            raise ValueError(f"budget maximum must be an integer in 1..{BUDGET}")
        self.maximum = maximum
        self.charged = 0
        self.reserved = 0
        self.attempted = 0
        self.successful = 0
        self.semantic_failures = 0
        self.infrastructure_failures = 0
        self.retries = 0
        self._next_id = 1
        self._open: dict[int, int] = {}
        self._lock = threading.Lock()
        self._thread_accounting = threading.local()

    def operation_snapshot(self) -> dict[str, int]:
        """Monotonic counters for commits made by the current caller thread."""
        return dict(getattr(self._thread_accounting, "counts", {"charged": 0, "attempted": 0}))

    @property
    def remaining(self) -> int:
        """Slots neither charged nor reserved."""
        with self._lock:
            return self.maximum - self.charged - self.reserved

    def reserve(self, requested: int) -> BudgetReservation:
        """Reserve up to ``requested`` slots; the last batch is trimmed at the cap."""
        if requested <= 0:
            raise ValueError("requested reservation must be positive")
        with self._lock:
            allocated = min(requested, self.maximum - self.charged - self.reserved)
            if allocated <= 0:
                raise RuntimeError(EXHAUSTED_MESSAGE)
            reservation = BudgetReservation(self._next_id, allocated)
            self._next_id += 1
            self._open[reservation.reservation_id] = allocated
            self.reserved += allocated
            return reservation

    def commit(self, reservation: BudgetReservation, outcomes: Iterable[Outcome]) -> None:
        """Close a reservation with one outcome per reserved slot."""
        values = tuple(outcomes)
        with self._lock:
            expected = self._open.get(reservation.reservation_id)
            if expected is None or expected != reservation.size:
                raise ValueError("unknown or already committed reservation")
            if len(values) != expected:
                raise ValueError(f"expected {expected} outcomes, got {len(values)}")
            invalid = set(values) - OUTCOMES
            if invalid:
                raise ValueError(f"invalid outcomes: {sorted(invalid)}")
            self._commit_locked(reservation, values, expected)

    def commit_prefix(self, reservation: BudgetReservation, outcomes: Iterable[Outcome]) -> None:
        """Commit an attempted prefix and release the unattempted suffix.

        Used when a fail-closed contract error stops an atomic batch part-way:
        attempted executions keep exact accounting, untouched slots are freed.
        """
        values = tuple(outcomes)
        with self._lock:
            expected = self._open.get(reservation.reservation_id)
            if expected is None or expected != reservation.size:
                raise ValueError("unknown or already committed reservation")
            if len(values) > expected:
                raise ValueError(f"reservation permits at most {expected} outcomes, got {len(values)}")
            invalid = set(values) - OUTCOMES
            if invalid:
                raise ValueError(f"invalid outcomes: {sorted(invalid)}")
            self._commit_locked(reservation, values, expected)

    def _commit_locked(self, reservation: BudgetReservation, values: tuple[Outcome, ...], expected: int) -> None:
        local = self.operation_snapshot()
        local["charged"] += sum(value != "infrastructure_failure" for value in values)
        local["attempted"] += len(values)
        self._thread_accounting.counts = local
        self._open.pop(reservation.reservation_id)
        self.reserved -= expected
        self.attempted += len(values)
        self.successful += values.count("success")
        self.semantic_failures += values.count("semantic_failure")
        infra = values.count("infrastructure_failure")
        self.infrastructure_failures += infra
        self.retries += infra
        self.charged += len(values) - infra
        if self.charged > self.maximum:
            raise AssertionError("budget overshoot")

    def cancel(self, reservation: BudgetReservation) -> None:
        """Release a reservation without any outcome."""
        with self._lock:
            expected = self._open.pop(reservation.reservation_id, None)
            if expected is None:
                raise ValueError("unknown or already closed reservation")
            self.reserved -= expected

    def snapshot(self) -> dict[str, int]:
        """All counters (the form recorded in every artifact)."""
        with self._lock:
            return {
                "maximum": self.maximum,
                "charged": self.charged,
                "reserved": self.reserved,
                "remaining": self.maximum - self.charged - self.reserved,
                "attempted": self.attempted,
                "successful": self.successful,
                "semantic_failures": self.semantic_failures,
                "infrastructure_failures": self.infrastructure_failures,
                "retries": self.retries,
            }


def closed_ledger(snapshot: Mapping[str, Any], maximum: int | None = None) -> bool:
    """True for a settled, internally consistent ledger snapshot."""
    names = set(BudgetLedger().snapshot())
    if not isinstance(snapshot, dict) or set(snapshot) != names:
        return False
    if any(type(value) is not int or value < 0 for value in snapshot.values()):
        return False
    cap = snapshot["maximum"] if maximum is None else maximum
    return (
        snapshot["maximum"] == cap
        and snapshot["reserved"] == 0
        and snapshot["charged"] == snapshot["successful"] + snapshot["semantic_failures"]
        and snapshot["attempted"] == snapshot["charged"] + snapshot["infrastructure_failures"]
        and snapshot["retries"] == snapshot["infrastructure_failures"]
        and snapshot["remaining"] == cap - snapshot["charged"]
    )


# Rows past the budget
def remaining_rollouts(budget: Any) -> int | None:
    """Remaining rollouts of a ledger-like object, or None when it reports none."""
    snapshot = getattr(budget, "snapshot", None)
    if not callable(snapshot):
        return None
    try:
        return int(snapshot()["remaining"])
    except (KeyError, TypeError, ValueError):
        return None


def is_budget_exhausted_error(exc: BaseException) -> bool:
    """True for the ledger's own refusal to reserve past B."""
    return isinstance(exc, RuntimeError) and str(exc) == EXHAUSTED_MESSAGE


def budget_exhausted_record(cell: Any, example: Any) -> RunRecord:
    """Uncharged, never-executed semantic failure answering a row requested past B."""
    return RunRecord(
        cell_id=str(getattr(cell, "cell_id", "")),
        example_id=example_id(example),
        status="semantic_failure",
        score=0.0,
        error=BUDGET_EXHAUSTED_ERROR,
        framework=str(getattr(cell, "framework", "")),
        metadata={"budget_exhausted": True},
    )


__all__ = [
    "BUDGET_EXHAUSTED_ERROR",
    "BudgetLedger",
    "BudgetReservation",
    "EXHAUSTED_MESSAGE",
    "OUTCOMES",
    "Outcome",
    "budget_exhausted_record",
    "closed_ledger",
    "is_budget_exhausted_error",
    "remaining_rollouts",
]
