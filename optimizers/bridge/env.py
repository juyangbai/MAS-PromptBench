"""Environment variables read by the bridge::

    variable                     read by
    TASK_MODEL                   lm.TASK_MODEL (the adapters' model when MODEL_ID is unset)
    REFL_MODEL                   lm.REFL_MODEL
    TASK_ENDPOINTS               lm.task_endpoints (comma-separated)
    REFLECTION_COMPACT_DATASETS  programs trace compaction (default lcb)

MIPRO's program view (``mipro_programs``) reads
``MIPRO_REFLECTION_COMPACT_DATASETS`` first; a non-empty value there
overrides ``REFLECTION_COMPACT_DATASETS``. The reflection endpoint is the
protocol's ``REFLECTION_MODEL_BASE_URL`` (``optimizers.protocol.settings``).
"""

from __future__ import annotations

import os

MIPRO_NAMES = {"REFLECTION_COMPACT_DATASETS": "MIPRO_REFLECTION_COMPACT_DATASETS"}


def get(name: str, default: str | None = None, *, prefer_mipro: bool = False) -> str | None:
    """Value of the bridge variable ``name``; with ``prefer_mipro`` a set MIPRO name wins."""
    if prefer_mipro:
        value = os.environ.get(MIPRO_NAMES[name])
        if value:
            return value
    return os.environ.get(name, default)
