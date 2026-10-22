"""Deterministic request seeds: logical rollout and reflection seeds, paired evaluation seeds.

Optimizer seed ``s`` offsets every seed by ``REQUEST_SEED_OFFSETS[s]``
(0 / 1000 / 2000); infrastructure retries reuse a seed unchanged.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from .config import REQUEST_SEED_OFFSETS

if TYPE_CHECKING:
    from .schema import CellSpec, PromptBundle

OFFSETS = dict(REQUEST_SEED_OFFSETS)
SEED_MODULUS = 2**31 - 1


def logical_request_seed(
    optimizer_seed: int,
    cell_id: str,
    phase: str,
    iteration: int,
    example_id: str,
    role: str,
    turn: int,
    attempt: int = 0,
) -> int:
    """Seed one logical request; infrastructure retries reuse it unchanged."""
    if optimizer_seed not in OFFSETS:
        raise ValueError(f"optimizer seed must be one of {sorted(OFFSETS)}")
    fields: tuple[Any, ...] = (
        optimizer_seed,
        OFFSETS[optimizer_seed],
        cell_id,
        phase,
        iteration,
        example_id,
        role,
        turn,
        attempt,
    )
    payload = "|".join(str(item) for item in fields).encode("utf-8")
    hashed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return (hashed + OFFSETS[optimizer_seed]) % SEED_MODULUS


def paired_evaluation_seed(condition_id: str, evaluation_seed: int, split: str, example_id: str) -> int:
    """Seed one evaluation item identically across optimizer methods.

    Bundle choice and baseline/candidate provenance are deliberately absent,
    so the seed and the deployed bundle see the same per-item request seed.
    The condition digest binds the frozen execution inputs and seed policy.
    """
    if evaluation_seed not in OFFSETS or type(evaluation_seed) is not int:
        raise ValueError("evaluation seed must be 0, 1, or 2")
    if split not in {"train", "validation", "test"}:
        raise ValueError("invalid evaluation split")
    if len(condition_id) != 64 or any(c not in "0123456789abcdef" for c in condition_id):
        raise ValueError("condition_id must be a lowercase SHA-256")
    from .schema import content_hash, schema_name

    return (
        int(
            content_hash(
                {
                    "schema": schema_name("paired-evaluation-seed"),
                    "condition": condition_id,
                    "seed": evaluation_seed,
                    "offset": OFFSETS[evaluation_seed],
                    "split": split,
                    "example_id": example_id,
                }
            )[:16],
            16,
        )
        % SEED_MODULUS
    )


def request_seeds(
    cell: CellSpec, examples: Sequence[Any], *, phase: str, iteration: int, bundle: PromptBundle
) -> tuple[int, ...]:
    """Logical seeds of one batch of rollouts of ``bundle`` (one per row, by row position)."""
    from .schema import example_id

    return tuple(
        logical_request_seed(
            cell.optimizer_seed, cell.cell_id, phase, iteration, example_id(example, index), bundle.digest, index
        )
        for index, example in enumerate(examples)
    )


def reflection_seed(cell: CellSpec, *, phase: str, iteration: int, role: str, prompt: str, turn: int = 0) -> int:
    """Logical seed of one reflection request, keyed by the prompt text."""
    return logical_request_seed(
        cell.optimizer_seed,
        cell.cell_id,
        phase,
        iteration,
        hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        role,
        turn,
    )


__all__ = [
    "OFFSETS",
    "SEED_MODULUS",
    "logical_request_seed",
    "paired_evaluation_seed",
    "reflection_seed",
    "request_seeds",
]
