"""Seed-only reference method: returns the seed bundle and spends no budget.

Useful to exercise the protocol end to end and to obtain the seed-bundle
baseline of a runtime condition.
"""

from __future__ import annotations

from typing import Any

from ..learning_curve import LearningCurveRecorder
from ..schema import CellSpec, OptimizerResult, PromptBundle, StopReason


class IdentityOptimizer:
    """Return the seed bundle without a rollout (``stop_reason`` ``identity_no_search``)."""

    method = "identity"

    def __init__(self, seed_bundle: PromptBundle) -> None:
        self.seed_bundle = seed_bundle

    def optimize(self, cell: CellSpec, runner: Any, budget: Any, training: list, validation: list) -> OptimizerResult:
        """The seed bundle, a flat learning curve and an untouched ledger."""
        del training, validation
        if getattr(runner, "budget", budget) is not budget:
            raise ValueError("runner and optimizer were given different budget ledgers")
        curve = LearningCurveRecorder(maximum=budget.maximum)
        curve.observe(0, self.seed_bundle.digest, 0, None, ("initial_state", "early_stop"))
        curve.carry_forward_after_stop()
        artifact = OptimizerResult(
            method=cell.method,
            protocol_id=cell.protocol_id,
            cell_id=cell.cell_id,
            seed_bundle=self.seed_bundle,
            incumbent_bundle=self.seed_bundle,
            native_iterations=0,
            stop_reason=StopReason.IDENTITY_NO_SEARCH,
            learning_curve=tuple(curve.rows),
            budget_snapshot=dict(budget.snapshot()),
            metadata={"search": "none"},
        )
        store, directory = getattr(runner, "artifact_store", None), getattr(runner, "artifact_directory", None)
        if store is not None and directory is not None:
            for row in curve.rows:
                store.append_jsonl(directory, "learning_curve.jsonl", row)
        return artifact


__all__ = ["IdentityOptimizer"]
