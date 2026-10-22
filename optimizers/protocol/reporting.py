"""Paired, seed-aware summaries, bootstrap CIs, exact McNemar and Holm correction.

The three optimizer seeds are strata: items are paired within a seed and the
3 x N observations are never pooled as independent samples.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from statistics import mean, stdev

from .config import OPTIMIZER_SEEDS

SEEDS = tuple(OPTIMIZER_SEEDS)


@dataclass(frozen=True)
class PairedObservation:
    """Baseline and deployed score of one test item under one evaluation seed."""

    seed: int
    example_id: str
    baseline: float
    deployed: float

    def __post_init__(self) -> None:
        if self.seed not in SEEDS:
            raise ValueError("evaluation seeds are 0, 1, 2")
        if not self.example_id:
            raise ValueError("example_id is required")
        if not (0 <= self.baseline <= 1 and 0 <= self.deployed <= 1):
            raise ValueError("scores must be in [0,1]")


def paired_summary(rows: Iterable[PairedObservation]) -> dict:
    """Per-seed means and deltas, and the mean/std of the delta over seeds 0/1/2."""
    grouped: dict[int, list[PairedObservation]] = defaultdict(list)
    seen: set[tuple[int, str]] = set()
    for row in rows:
        key = (row.seed, row.example_id)
        if key in seen:
            raise ValueError(f"duplicate paired observation: {key}")
        seen.add(key)
        grouped[row.seed].append(row)
    if set(grouped) != set(SEEDS):
        raise ValueError("all three evaluation seeds are required")
    id_sets = [{row.example_id for row in grouped[seed]} for seed in SEEDS]
    if not id_sets[0] or id_sets[1:] != id_sets[:1] * 2:
        raise ValueError("seed strata must contain identical non-empty example IDs")

    per_seed = []
    for seed in SEEDS:
        ordered = sorted(grouped[seed], key=lambda row: row.example_id)
        base = mean(row.baseline for row in ordered)
        deployed = mean(row.deployed for row in ordered)
        per_seed.append(
            {
                "seed": seed,
                "n": len(ordered),
                "baseline": base,
                "deployed": deployed,
                "delta_pp": 100.0 * (deployed - base),
            }
        )
    deltas = [row["delta_pp"] for row in per_seed]
    return {
        "per_seed": per_seed,
        "mean_delta_pp": mean(deltas),
        "std_delta_pp": stdev(deltas),
        "macro_baseline": mean(row["baseline"] for row in per_seed),
        "macro_deployed": mean(row["deployed"] for row in per_seed),
    }


def seed_stratified_paired_bootstrap_ci(
    rows: Iterable[PairedObservation],
    *,
    replicates: int = 10_000,
    confidence: float = 0.95,
    random_seed: int = 0,
) -> tuple[float, float]:
    """Resample seeds, then paired items within each sampled seed (delta in pp)."""
    values = list(rows)
    paired_summary(values)
    by_seed: dict[int, list[PairedObservation]] = defaultdict(list)
    for row in values:
        by_seed[row.seed].append(row)
    if replicates < 100:
        raise ValueError("at least 100 bootstrap replicates are required")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    rng = random.Random(random_seed)
    estimates: list[float] = []
    for _ in range(replicates):
        sampled_seed_means: list[float] = []
        for seed in (rng.choice(SEEDS) for _ in SEEDS):
            stratum = by_seed[seed]
            sampled = [rng.choice(stratum) for _ in stratum]
            sampled_seed_means.append(mean(row.deployed - row.baseline for row in sampled))
        estimates.append(100.0 * mean(sampled_seed_means))
    estimates.sort()
    alpha = (1.0 - confidence) / 2.0
    low = estimates[max(0, math.floor(alpha * replicates))]
    high = estimates[min(replicates - 1, math.ceil((1.0 - alpha) * replicates) - 1)]
    return low, high


def exact_paired_sign_test(rows: Iterable[PairedObservation]) -> float:
    """Two-sided exact sign test; ties are excluded (exact McNemar for binary scores)."""
    signs = [row.deployed - row.baseline for row in rows]
    wins = sum(value > 0 for value in signs)
    losses = sum(value < 0 for value in signs)
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(0, min(wins, losses) + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values, preserving comparison labels."""
    for label, value in p_values.items():
        if not 0 <= value <= 1:
            raise ValueError(f"invalid p-value for {label}: {value}")
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for index, (label, value) in enumerate(ordered):
        running = max(running, min(1.0, (total - index) * value))
        adjusted[label] = running
    return {label: adjusted[label] for label in p_values}


def summarize_paired_observations(
    observations: Iterable[PairedObservation], *, bootstrap_replicates: int = 10_000
) -> dict:
    """Paired summary, seed-stratified CI and, for binary scores, per-seed exact McNemar."""
    observations = list(observations)
    summary = paired_summary(observations)
    summary["confidence_interval_delta_pp"] = list(
        seed_stratified_paired_bootstrap_ci(observations, replicates=bootstrap_replicates)
    )
    if all(row.baseline in {0, 1} and row.deployed in {0, 1} for row in observations):
        summary["per_seed_exact_mcnemar_p"] = {
            str(seed): exact_paired_sign_test(row for row in observations if row.seed == seed) for seed in SEEDS
        }
    return summary


def adjust_predeclared_families(
    p_values: Mapping[str, float],
    families: Mapping[str, Sequence[str]],
) -> dict[str, dict[str, float]]:
    """Holm within each predeclared comparison family (e.g. one table/dataset)."""
    declared = [label for labels in families.values() for label in labels]
    if set(declared) != set(p_values) or any(len(set(labels)) != len(labels) for labels in families.values()):
        raise ValueError("comparison families must declare every reported comparison exactly once per family")
    return {name: holm_adjust({label: p_values[label] for label in labels}) for name, labels in families.items()}


__all__ = [
    "PairedObservation",
    "adjust_predeclared_families",
    "exact_paired_sign_test",
    "holm_adjust",
    "paired_summary",
    "seed_stratified_paired_bootstrap_ci",
    "summarize_paired_observations",
]
