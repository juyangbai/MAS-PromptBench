"""Frozen split manifest helpers for GEPA datasets."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import dspy

from real_runner_gepa.datasets.split_utils import real_eval_ids


REPO_ROOT = Path(__file__).resolve().parents[4]
SIGNAL_TRAIN_SIZE = 25
SIGNAL_VAL_SIZE = 25


def _truthy(value: str | None) -> bool:
    return value is None or value.lower() not in {"0", "false", "no", "off"}


def signal_split_enabled() -> bool:
    """Whether exact 25/25 requests should use frozen signal manifests."""
    return _truthy(os.environ.get("GEPA_USE_SIGNAL_25_SPLITS"))


def signal_manifest_path(dataset: str) -> Path:
    return REPO_ROOT / "benchmarks" / dataset / f"{dataset}_gepa25_signal_split.json"


def is_signal_request(train_size: int, val_size: int, seed: int, offset: int) -> bool:
    return (
        signal_split_enabled()
        and train_size == SIGNAL_TRAIN_SIZE
        and val_size == SIGNAL_VAL_SIZE
        and seed == 0
        and offset == 0
    )


def load_signal_manifest(dataset: str) -> dict[str, Any]:
    path = signal_manifest_path(dataset)
    if not path.is_file():
        raise FileNotFoundError(
            f"{dataset} GEPA 25/25 signal manifest not found: {path}. "
            "Generate it before launching train_size=25,val_size=25 with seed=0,offset=0."
        )
    try:
        manifest = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid {dataset} GEPA signal manifest: {path}") from exc
    if manifest.get("dataset") != dataset:
        raise ValueError(
            f"{path} has dataset={manifest.get('dataset')!r}; expected {dataset!r}"
        )
    return manifest


def _ids_from_manifest(manifest: dict[str, Any], key: str) -> list[str]:
    ids = manifest.get(key)
    if not isinstance(ids, list) or not ids:
        raise ValueError(f"signal manifest {manifest.get('dataset')} has no {key}")
    return [str(item) for item in ids]


def _check_unique(label: str, ids: list[str]) -> None:
    duplicates = sorted({item for item in ids if ids.count(item) > 1})
    if duplicates:
        raise ValueError(f"signal manifest {label} has duplicate IDs: {duplicates[:8]}")


def apply_signal_split_if_requested(
    dataset: str,
    examples: list[dspy.Example],
    train_size: int,
    val_size: int,
    seed: int,
    offset: int,
) -> tuple[list[dspy.Example], list[dspy.Example]] | None:
    """Return the frozen 25/25 split when the request exactly matches it.

    Falls back to the normal seeded split when no frozen manifest ships for the
    dataset (e.g. benchmarks slimmed to eval-id manifests only).
    """
    if not is_signal_request(train_size, val_size, seed, offset):
        return None
    if not signal_manifest_path(dataset).is_file():
        return None

    manifest = load_signal_manifest(dataset)
    train_ids = _ids_from_manifest(manifest, "train_ids")
    val_ids = _ids_from_manifest(manifest, "val_ids")
    if len(train_ids) != train_size or len(val_ids) != val_size:
        raise ValueError(
            f"{dataset} signal manifest size mismatch: "
            f"train={len(train_ids)} val={len(val_ids)} expected {train_size}/{val_size}"
        )
    _check_unique(f"{dataset} train_ids", train_ids)
    _check_unique(f"{dataset} val_ids", val_ids)

    overlap = sorted(set(train_ids) & set(val_ids))
    if overlap:
        raise ValueError(f"{dataset} signal manifest train/val overlap: {overlap[:8]}")

    protected = real_eval_ids(dataset)
    protected_overlap = sorted((set(train_ids) | set(val_ids)) & protected)
    if protected_overlap:
        raise ValueError(
            f"{dataset} signal manifest leaks protected eval IDs: {protected_overlap[:8]}"
        )

    by_id = {str(example.id): example for example in examples}
    missing = [item for item in train_ids + val_ids if item not in by_id]
    if missing:
        raise ValueError(f"{dataset} signal manifest IDs missing from dataset: {missing[:8]}")

    return [by_id[item] for item in train_ids], [by_id[item] for item in val_ids]


def signal_manifest_metadata(
    dataset: str,
    train_size: int,
    val_size: int,
    seed: int,
    offset: int,
) -> dict[str, Any] | None:
    """Return compact metadata for the frozen manifest used by this request."""
    if not is_signal_request(train_size, val_size, seed, offset):
        return None
    if not signal_manifest_path(dataset).is_file():
        return None
    manifest = load_signal_manifest(dataset)
    return {
        "path": str(signal_manifest_path(dataset).relative_to(REPO_ROOT)),
        "dataset": manifest.get("dataset"),
        "version": manifest.get("version"),
        "train_size": manifest.get("train_size"),
        "val_size": manifest.get("val_size"),
        "train_ids": _ids_from_manifest(manifest, "train_ids"),
        "val_ids": _ids_from_manifest(manifest, "val_ids"),
        "protected_source": manifest.get("protected_source"),
        "protected_id_count": manifest.get("protected_id_count"),
        "profile_summary": manifest.get("profile_summary", {}),
        "calibration_summary": manifest.get("calibration_summary", {}),
        "generation_metadata": manifest.get("generation_metadata", {}),
    }
