"""Shared train/validation split helpers for GEPA datasets."""
from __future__ import annotations

import json
import os
import random
from functools import lru_cache
from pathlib import Path

import dspy


REPO_ROOT = Path(__file__).resolve().parents[4]


SWE_PROTECTED_SAMPLE_DEFAULT = "balanced_30"


def _truthy(value: str | None) -> bool:
    return value is None or value.lower() not in {"0", "false", "no", "off"}


def exclude_real_eval_ids_enabled() -> bool:
    """Whether GEPA optimization splits should exclude known later eval IDs."""
    return _truthy(os.environ.get("GEPA_EXCLUDE_REAL_EVAL_IDS"))


def swe_protected_sample_name() -> str:
    """Return the SWE sample reserved for protected/report evaluation."""
    return (
        os.environ.get("SWE_GEPA_EVAL_SAMPLE")
        or os.environ.get("SWE_SAMPLE")
        or SWE_PROTECTED_SAMPLE_DEFAULT
    )


@lru_cache(maxsize=None)
def real_eval_id_list(dataset: str) -> tuple[str, ...]:
    """Return protected/report eval IDs from benchmarks/<dataset>/<dataset>_eval_ids.json."""
    path = REPO_ROOT / "benchmarks" / dataset / f"{dataset}_eval_ids.json"
    if not path.is_file():
        return ()
    try:
        manifest = json.loads(path.read_text())
    except json.JSONDecodeError:
        return ()
    return tuple(str(item) for item in manifest.get("ids", []))


@lru_cache(maxsize=None)
def real_eval_ids(dataset: str) -> frozenset[str]:
    """Return IDs reserved for protected/report eval artifacts."""
    return frozenset(real_eval_id_list(dataset))


def train_val_split_excluding_real_eval(
    dataset: str,
    examples: list[dspy.Example],
    train_size: int,
    val_size: int,
    seed: int = 0,
    offset: int = 0,
) -> tuple[list[dspy.Example], list[dspy.Example]]:
    """Shuffle/split while keeping optimization rows outside known real eval IDs."""
    excluded = real_eval_ids(dataset) if exclude_real_eval_ids_enabled() else frozenset()
    pool = [example for example in examples if str(example.id) not in excluded]
    pool = pool[offset:]
    random.Random(seed).shuffle(pool)
    need = train_size + val_size
    if len(pool) < need:
        raise ValueError(
            f"need {need} examples after excluding {len(excluded)} real-eval IDs "
            f"and offset={offset}; got {len(pool)}"
        )
    return pool[:train_size], pool[train_size:need]
