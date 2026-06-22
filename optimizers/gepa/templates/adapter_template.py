"""Template for a new real-runner GEPA adapter.

Copy this file into `real_runner_gepa/adapters/<topology>_<dataset>.py` and
replace TODO sections. Keep adapter state instance-owned; do not monkey-patch
global prompts in real runner modules unless there is no safer option.
"""
from __future__ import annotations

from typing import Any


DATASET = "TODO_DATASET"
TOPOLOGY = "TODO_TOPOLOGY"
ROLES = ["TODO_ROLE"]


class TODOAdapter:
    dataset = DATASET
    topology = TOPOLOGY

    def __init__(self, prompts: dict[str, str] | None = None):
        self._prompts = prompts or {role: "" for role in ROLES}

    def roles(self) -> list[str]:
        return list(ROLES)

    def get_prompt(self, role: str) -> str:
        self._check_role(role)
        return self._prompts[role]

    def set_prompt(self, role: str, text: str) -> None:
        self._check_role(role)
        self._prompts[role] = text
        self.reset()

    def reset(self) -> None:
        """Clear cached compiled graphs/crews/clients after prompt mutation."""
        return None

    def run_example(self, example: Any) -> dict:
        """Run one example through the real execution engine."""
        raise NotImplementedError

    def format_role_trace(self, role: str, output: Any) -> str:
        """Return concise role-specific text for GEPA reflection."""
        self._check_role(role)
        return str(output)

    @staticmethod
    def _check_role(role: str) -> None:
        if role not in ROLES:
            raise KeyError(f"Unknown role {role!r}; expected one of {ROLES}")
