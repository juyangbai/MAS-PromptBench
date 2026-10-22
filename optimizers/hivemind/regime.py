"""Pure helpers describing the reproducible HiveMind adaptation regime.

No runner or LLM imports. :class:`HiveMindSettings` holds the knobs. The
coalition plan (exact power set, or a Monte-Carlo permutation-prefix plan
above the coalition cap) and the row order are functions of the optimizer seed
and ``hm_seed`` only, so a stored plan can be checked against the plan that
would be launched.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from itertools import combinations

ADAPTATION_ID = "hivemind-adapted/worker-shapley+manager-rotation+minibatch-gate"
ALGORITHM = "CG-OPO_coalition"


@dataclass(frozen=True)
class HiveMindSettings:
    """HiveMind's knobs (no environment variables).

    One CG-OPO cycle evaluates every planned coalition on ``coalition_batch``
    training rows, reflects once on the lowest-credit prompt (the manager every
    ``manager_every_k``-th cycle) and gates the update on ``acceptance_batch``
    validation rows for the current and the candidate prompts. ``max_cycles``
    0 means: until the budget is spent.
    """

    max_cycles: int = 0
    coalition_batch: int = 5
    acceptance_batch: int = 5
    max_coalitions: int = 40
    manager_every_k: int = 3
    max_lessons: int = 6
    fail_below: float = 0.5
    hm_seed: int = 0
    reflection_temperature: float = 0.7
    reflection_top_p: float = 1.0

    def cycle_batches(self, coalitions: int, training: int, validation: int, remaining: int) -> tuple[int, int]:
        """Budget split of one cycle: (coalition batch, acceptance batch), shrunk until one cycle fits."""
        batch = min(self.coalition_batch, training)
        accept_n = min(self.acceptance_batch, validation, training)
        while coalitions * batch + 2 * accept_n > remaining and batch > 1:
            batch -= 1
            accept_n = min(accept_n, batch)
        while coalitions * batch + 2 * accept_n > remaining and accept_n > 1:
            accept_n -= 1
        return batch, accept_n


DEFAULTS = HiveMindSettings()
# Registered method settings (scope: the Table-6 cells, see ``protocol.cells``).
METHOD_SETTINGS = {
    "algorithm": ALGORITHM,
    "fail_below": DEFAULTS.fail_below,
    "fail_below_status": "preregistered_adaptation",
    "scope": "table_6",
    "coalition_batch": DEFAULTS.coalition_batch,
    "acceptance_batch": DEFAULTS.acceptance_batch,
    "max_coalitions": DEFAULTS.max_coalitions,
    "manager_every_k": DEFAULTS.manager_every_k,
    "max_lessons": DEFAULTS.max_lessons,
    "max_cycles": DEFAULTS.max_cycles,
    "hm_seed": DEFAULTS.hm_seed,
}


def coalition_key(coalition: frozenset[str]) -> str:
    """Stable name of a coalition (``manager_only`` for the empty one)."""
    return "+".join(sorted(coalition)) if coalition else "manager_only"


def _powerset(items: list[str]):
    for size in range(len(items) + 1):
        for coalition in combinations(items, size):
            yield frozenset(coalition)


def build_coalition_plan(
    workers: list[str],
    rng: random.Random,
    max_coalitions: int,
) -> tuple[list[frozenset[str]], list[list[str]] | None]:
    """Return the driver's exact or permutation-prefix Monte-Carlo plan.

    In Monte-Carlo mode, sample a fixed number of independent uniform
    permutations.  Every permutation contributes all of its prefixes.  The
    worst-case union size is ``2 + p * (worker_count - 1)``, so choosing ``p``
    from that bound keeps the requested cap strict without conditioning the
    samples on how much their prefixes overlap.
    """
    workers = list(workers)
    if 2 ** len(workers) <= max_coalitions:
        exact = sorted(_powerset(workers), key=lambda s: (len(s), coalition_key(s)))
        return exact, None

    worker_count = len(workers)
    if max_coalitions < worker_count + 1:
        raise ValueError(
            "max_coalitions must fit every prefix of at least one permutation "
            f"({worker_count + 1} required for {worker_count} workers)"
        )
    permutation_count = max(1, (max_coalitions - 2) // (worker_count - 1))
    permutations = [rng.sample(workers, worker_count) for _ in range(permutation_count)]
    needed = {frozenset(permutation[:index]) for permutation in permutations for index in range(worker_count + 1)}
    assert len(needed) <= max_coalitions
    coalitions = sorted(needed, key=lambda s: (len(s), coalition_key(s)))
    return coalitions, permutations


def plan_seed(seed_offset: int, hm_seed: int) -> int:
    """RNG seed of the coalition plan for one optimizer seed."""
    return 90001 * seed_offset + hm_seed


def row_order_seed(seed_offset: int, hm_seed: int) -> int:
    """RNG seed of the train-row order for one optimizer seed."""
    return 100003 * seed_offset + hm_seed


def _digest(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def plan_provenance(
    workers: list[str],
    coalitions: list[frozenset[str]],
    permutations: list[list[str]] | None,
    *,
    max_coalitions: int,
    seed_offset: int,
    hm_seed: int,
    empty_key: str = "manager_only",
) -> dict:
    """Artifact metadata that pins a coalition plan (mode, coalitions, permutations, seeds, hashes)."""
    coalition_names = [coalition_key(s) if s else empty_key for s in coalitions]
    permutation_names = [list(p) for p in (permutations or [])]
    return {
        "hivemind_adaptation_id": ADAPTATION_ID,
        "hivemind_fidelity": "adapted",
        "hivemind_shapley_mode": (
            "exact_worker_shapley" if permutations is None else "monte_carlo_permutation_worker_shapley"
        ),
        "hivemind_max_coalitions": max_coalitions,
        "hivemind_worker_count": len(workers),
        "hivemind_exact_coalition_count": 2 ** len(workers),
        "hivemind_mc_sampling_scheme": ("none" if permutations is None else "iid_uniform_fixed_count"),
        "hivemind_coalition_count": len(coalition_names),
        "hivemind_coalitions": coalition_names,
        "hivemind_coalition_plan_sha256": _digest(coalition_names),
        "hivemind_permutation_count": len(permutation_names),
        "hivemind_permutations": permutation_names,
        "hivemind_permutation_plan_sha256": _digest(permutation_names),
        "hivemind_plan_seed": plan_seed(seed_offset, hm_seed),
        "hivemind_row_order_seed": row_order_seed(seed_offset, hm_seed),
        "hivemind_hm_seed": hm_seed,
    }


def expected_plan_provenance(
    workers: list[str],
    *,
    max_coalitions: int,
    seed_offset: int,
    hm_seed: int,
) -> dict:
    """The provenance the plan of these settings must have."""
    coalitions, permutations = build_coalition_plan(
        workers, random.Random(plan_seed(seed_offset, hm_seed)), max_coalitions
    )
    return plan_provenance(
        workers, coalitions, permutations, max_coalitions=max_coalitions, seed_offset=seed_offset, hm_seed=hm_seed
    )


PLAN_META_FIELDS = (
    "hivemind_adaptation_id",
    "hivemind_fidelity",
    "hivemind_shapley_mode",
    "hivemind_max_coalitions",
    "hivemind_worker_count",
    "hivemind_exact_coalition_count",
    "hivemind_mc_sampling_scheme",
    "hivemind_coalition_count",
    "hivemind_coalitions",
    "hivemind_coalition_plan_sha256",
    "hivemind_permutation_count",
    "hivemind_permutations",
    "hivemind_permutation_plan_sha256",
    "hivemind_plan_seed",
    "hivemind_row_order_seed",
    "hivemind_hm_seed",
)


def plan_metadata_matches(
    meta: dict,
    workers: list[str],
    *,
    max_coalitions: int,
    seed_offset: int,
    hm_seed: int,
) -> bool:
    """Whether recorded metadata carries exactly the expected plan provenance."""
    expected = expected_plan_provenance(
        workers, max_coalitions=max_coalitions, seed_offset=seed_offset, hm_seed=hm_seed
    )
    return all(meta.get(field) == expected[field] for field in PLAN_META_FIELDS)


__all__ = [
    "ADAPTATION_ID",
    "ALGORITHM",
    "DEFAULTS",
    "HiveMindSettings",
    "METHOD_SETTINGS",
    "PLAN_META_FIELDS",
    "build_coalition_plan",
    "coalition_key",
    "expected_plan_provenance",
    "plan_metadata_matches",
    "plan_provenance",
    "plan_seed",
    "row_order_seed",
]
