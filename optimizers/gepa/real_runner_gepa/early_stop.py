"""Early-stop helper copied into the isolated real-runner GEPA workspace."""
from __future__ import annotations

import os
import sys

from gepa.utils.stop_condition import StopperProtocol


class FullEvalPlateauStopper(StopperProtocol):
    """Stop after N full validation evaluations without improvement."""

    def __init__(self, patience: int):
        self._patience = patience
        self._best_score = float("-inf")
        self._no_improve_full_evals = 0
        self._last_full_eval_count = 0
        self._last_logged_full_evals = -1

    def __call__(self, gepa_state) -> bool:
        scores = getattr(gepa_state, "program_full_scores_val_set", None) or []
        n_full_evals = len(scores)

        if n_full_evals > self._last_full_eval_count:
            current_best = max(scores) if scores else 0.0
            if current_best > self._best_score:
                self._best_score = current_best
                self._no_improve_full_evals = 0
            else:
                self._no_improve_full_evals += 1
            self._last_full_eval_count = n_full_evals

            if n_full_evals != self._last_logged_full_evals:
                self._last_logged_full_evals = n_full_evals
                print(
                    f"[early-stop] full-evals={n_full_evals} "
                    f"best_val={self._best_score:.3f} "
                    f"no-improve={self._no_improve_full_evals}/{self._patience}",
                    file=sys.stderr,
                    flush=True,
                )

        fired = self._no_improve_full_evals >= self._patience
        if fired and self._last_logged_full_evals != -2:
            self._last_logged_full_evals = -2
            print(
                "[early-stop] halting after "
                f"{self._no_improve_full_evals} full-evals without improvement",
                file=sys.stderr,
                flush=True,
            )
        return fired


def build_early_stopper(patience: int | None = None) -> FullEvalPlateauStopper | None:
    if patience is None:
        raw = os.environ.get("GEPA_EARLY_STOP_PATIENCE", "8")
        try:
            patience = int(raw)
        except ValueError:
            patience = 8
    if patience <= 0:
        return None
    return FullEvalPlateauStopper(patience)
