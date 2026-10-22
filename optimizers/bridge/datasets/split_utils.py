"""The fixed train/validation/test splits of the bridge datasets."""

from __future__ import annotations

import json
from functools import cache

from core.paths import BENCHMARKS_DIR


@cache
def fixed_split_ids(dataset: str) -> dict[str, tuple[str, ...]] | None:
    """Return the fixed train/validation/test IDs, or None when none ship."""
    path = BENCHMARKS_DIR / dataset / f"{dataset}_splits.json"
    if not path.is_file():
        return None
    manifest = json.loads(path.read_text())
    return {split: tuple(str(item) for item in manifest.get(split, [])) for split in ("train", "validation", "test")}
