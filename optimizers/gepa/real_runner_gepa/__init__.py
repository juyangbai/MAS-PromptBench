"""Real-runner GEPA experiment workspace."""
from __future__ import annotations

import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

__all__ = [
    "AdapterBackedProgram",
    "BFCLRealRunnerProgram",
    "DATASET_ADAPTERS",
    "IndependentBFCLRealRunnerProgram",
    "RealRunnerProgram",
    "adapter_choices",
    "datasets",
    "get_adapter_class",
    "topologies",
]


def __getattr__(name: str):
    """Load heavier DSPy/framework dependencies only when requested."""
    if name in {
        "AdapterBackedProgram",
        "BFCLRealRunnerProgram",
        "IndependentBFCLRealRunnerProgram",
        "RealRunnerProgram",
    }:
        from real_runner_gepa import programs

        return getattr(programs, name)
    if name in {
        "DATASET_ADAPTERS",
        "adapter_choices",
        "datasets",
        "get_adapter_class",
        "topologies",
    }:
        from real_runner_gepa import registry

        return getattr(registry, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
