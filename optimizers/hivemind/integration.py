"""Stable public entrypoint of HiveMind (``METHODS["hivemind"]``)."""

from __future__ import annotations

from .optimizer import HiveMindOptimizer
from .regime import METHOD_SETTINGS

__all__ = ["HiveMindOptimizer", "METHOD_SETTINGS"]
