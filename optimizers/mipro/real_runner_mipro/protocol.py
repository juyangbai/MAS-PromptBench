"""Shared protocol for prompt-mutable real-runner adapters."""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class RealRunnerAdapter(Protocol):
    """One prompt-mutable adapter for one concrete topology/dataset pair."""

    topology: str
    dataset: str

    def roles(self) -> list[str]:
        """Return mutable role names exposed by this adapter."""

    def get_prompt(self, role: str) -> str:
        """Return the current prompt for a role."""

    def set_prompt(self, role: str, text: str) -> None:
        """Replace the current prompt for a role."""

    def reset(self) -> None:
        """Clear cached runtime state after prompt mutation."""

    def run_example(self, example: Any) -> Any:
        """Run one example through the real execution engine."""

    def format_role_trace(self, role: str, output: Any) -> str:
        """Return concise role-specific trace text for GEPA reflection."""


def validate_adapter(adapter: object) -> None:
    """Raise TypeError if an object is missing the adapter surface."""
    required = ("roles", "get_prompt", "set_prompt", "reset", "run_example")
    missing = [name for name in required if not hasattr(adapter, name)]
    if missing:
        raise TypeError(
            f"{type(adapter).__name__} is missing adapter methods: {missing}"
        )
