"""Shared run protocol for every prompt optimizer (``mas-promptbench``).

One job = (method, cell, optimizer seed). The method receives a 600-rollout
budget ledger and the fixed train/validation rows; the seed and incumbent
bundles are then compared on the full validation split (uncharged, greedy,
paired seeds) and only the locked deployment is evaluated on the test split.
See ``README.md`` for the optimizer interface and the CLIs.

Importing this package never contacts a model endpoint.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

__all__ = ["REPO_ROOT"]
