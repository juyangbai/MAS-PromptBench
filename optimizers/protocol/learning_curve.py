"""Zero-call observational learning curve on the charged-rollout grid.

Grid rows at 0, 10, ..., B project the latest fully committed state whose
charged count is at or below the grid value; no evaluation is ever added.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from .config import BUDGET, LEARNING_CURVE


@dataclass(frozen=True)
class CommittedState:
    """The latest fully committed optimizer state."""

    charged_rollouts: int
    bundle_hash: str
    native_iteration: int
    native_score: float | None
    metadata: Mapping[str, Any]


class LearningCurveRecorder:
    """Learning-curve rows on the charged-rollout grid 0, 10, ..., B."""

    def __init__(self, maximum: int = BUDGET, step: int = LEARNING_CURVE["step"]) -> None:
        if type(maximum) is not int or not 0 < maximum <= BUDGET or step != LEARNING_CURVE["step"]:
            raise ValueError(f"learning curve requires maximum in 1..{BUDGET} and step {LEARNING_CURVE['step']}")
        self.maximum = maximum
        self.step = step
        self.rows: list[dict[str, Any]] = []
        self._next_grid = 0
        self._last: CommittedState | None = None

    def observe(
        self,
        charged_rollouts: int,
        bundle_hash: str,
        native_iteration: int,
        native_score: float | None = None,
        reason: str | Iterable[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Record a committed state; returns the rows it emitted (grid rows up to its charge, else one native row)."""
        if charged_rollouts < 0 or charged_rollouts > self.maximum:
            raise ValueError("charged rollout count outside protocol range")
        first_observation = self._last is None
        if first_observation and charged_rollouts != 0:
            raise ValueError("first observed state must be the rollout-0 initial state")
        if self._last and charged_rollouts < self._last.charged_rollouts:
            raise ValueError("charged rollout count cannot decrease")
        if reason is None:
            native_reasons = ("initial_state",) if first_observation else ("native_iteration_end",)
        elif isinstance(reason, str):
            native_reasons = (reason,)
        else:
            native_reasons = tuple(reason)
        if not native_reasons or any(not isinstance(value, str) or not value for value in native_reasons):
            raise ValueError("at least one non-empty learning-curve reason is required")
        if first_observation and "initial_state" not in native_reasons:
            native_reasons = ("initial_state", *native_reasons)
        current = CommittedState(charged_rollouts, bundle_hash, native_iteration, native_score, metadata or {})
        emitted: list[dict[str, Any]] = []
        native_emitted = False
        while self._next_grid <= charged_rollouts and self._next_grid <= self.maximum:
            state = current if self._next_grid == charged_rollouts or self._last is None else self._last
            reasons = ["rollout_grid"]
            if self._next_grid == charged_rollouts:
                reasons.extend(native_reasons)
                native_emitted = True
            row = self._row(state, tuple(reasons), rollout_grid=self._next_grid)
            emitted.append(row)
            self.rows.append(row)
            self._next_grid += self.step
        if not native_emitted:
            native_row = self._row(current, native_reasons, rollout_grid=None)
            emitted.append(native_row)
            self.rows.append(native_row)
        self._last = current
        return emitted

    def carry_forward_after_stop(self) -> list[dict[str, Any]]:
        """Extend the final incumbent to B, marked as carried forward (no new evaluations)."""
        if self._last is None:
            raise RuntimeError("cannot carry forward before an observed state")
        emitted: list[dict[str, Any]] = []
        while self._next_grid <= self.maximum:
            row = self._row(
                self._last, ("post_stop_carry_forward",), rollout_grid=self._next_grid, post_stop_carried_forward=True
            )
            emitted.append(row)
            self.rows.append(row)
            self._next_grid += self.step
        return emitted

    @staticmethod
    def _row(
        state: CommittedState,
        reasons: tuple[str, ...],
        rollout_grid: int | None,
        post_stop_carried_forward: bool = False,
    ) -> dict[str, Any]:
        staleness = None if rollout_grid is None else rollout_grid - state.charged_rollouts
        return {
            "reason": reasons[0],
            "reasons": list(dict.fromkeys(reasons)),
            "rollout_grid": rollout_grid,
            "charged_rollouts": state.charged_rollouts,
            "staleness": staleness,
            "post_stop_carried_forward": post_stop_carried_forward,
            "native_iteration": state.native_iteration,
            "native_observed_score": state.native_score,
            "bundle_hash": state.bundle_hash,
            "metadata": dict(state.metadata),
        }


__all__ = ["CommittedState", "LearningCurveRecorder"]
