"""Template for a real-runner GEPA dataset module."""
from __future__ import annotations

import random

import dspy


def load_all() -> list[dspy.Example]:
    """Return examples with stable ids and `task_instance` input."""
    examples: list[dspy.Example] = []
    # TODO: populate examples.
    return examples


def train_val_split(
    examples: list[dspy.Example],
    train_size: int,
    val_size: int,
    seed: int = 0,
    offset: int = 0,
) -> tuple[list[dspy.Example], list[dspy.Example]]:
    pool = list(examples)[offset:]
    rng = random.Random(seed)
    rng.shuffle(pool)
    need = train_size + val_size
    if len(pool) < need:
        raise ValueError(f"need {need} examples after offset={offset}; got {len(pool)}")
    return pool[:train_size], pool[train_size:need]
