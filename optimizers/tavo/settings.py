"""TAVO's knobs: :class:`TAVOSettings` and the ``TAVO_CREDIT`` switch."""

from __future__ import annotations

import os
from dataclasses import dataclass


def trajectory_credit_enabled(value: str | None = None) -> bool:
    """Eq. 3 credit is on by default (upstream ``enable_local_evaluation``); ``TAVO_CREDIT=0`` is the ablation."""
    raw = os.environ.get("TAVO_CREDIT", "1") if value is None else value
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


@dataclass(frozen=True)
class TAVOSettings:
    """TAVO's knobs (upstream ``optimization_pipeline.py`` run configuration).

    Environment: ``TAVO_CREDIT`` (:func:`trajectory_credit_enabled`; default on,
    ``0``/``false``/``no``/``off`` is the no-credit Eq. 3 ablation), read when
    the optimizer is built or an attempt runs without an explicit choice. Budget
    split: fixed train batch ``train_batch_size`` and validation batch
    :meth:`validation_batch_size`; a round or retry starts only if its train
    and validation rollouts both fit.
    """

    train_batch_size: int = 6
    max_outer_rounds: int = 5
    min_validation_batch: int = 3
    adoption_threshold: float = 0.01  # improvement over the seed validation reference
    validation_delta_tolerance: float = 0.0  # improvement over the best adopted candidate
    attempts_per_round: int = 2  # one retry
    patience: int = 2
    rng_seed_base: int = 233  # TAVO_SEED default
    reflection_inflight: int = 4
    reflection_max_retries: int = 3
    default_temperature: float = 0.5
    default_top_p: float = 1.0

    def validation_batch_size(self, budget: int, train_size: int, available: int) -> int:
        """Budget layout ``v + K * (t + v) <= B`` gives ``v <= (B - K * t) / (K + 1)``, at least 3."""
        derived = (int(budget) - self.max_outer_rounds * int(train_size)) // (self.max_outer_rounds + 1)
        return min(int(available), max(self.min_validation_batch, derived))


DEFAULTS = TAVOSettings()


__all__ = ["DEFAULTS", "TAVOSettings", "trajectory_credit_enabled"]
