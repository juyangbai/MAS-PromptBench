"""Fake reflection model and grid-cell runners for offline MAMUT-GEPA tests.

Rollouts use the protocol ``FakeAdapter`` (roles ``planner``/``writer``): seed
prompts answer even-numbered items, a writer prompt containing ``GOOD``
answers every item. The fake reflection model always proposes a new, marked
instruction, so every GEPA proposal changes the candidate.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from optimizers.protocol.cells import QWEN_MODEL
from optimizers.protocol.schema import CellSpec
from optimizers.protocol.tests.fakes import FakeAdapter, FakeReflectionBackend, fake_cell, fake_runtime, protocol_runner

from ..integration import _BEGIN, _END

# Config
MARK = "GOOD"
TASK = "hotpotqa"


class FakeReflection(FakeReflectionBackend):
    """Deterministic reflection: a new, marked instruction per call (revision counted per role)."""

    def __init__(self, *, mark: str = MARK) -> None:
        super().__init__()
        self.mark = mark
        self._revisions: dict[str, int] = {}

    def respond(self, prompt: str, request: Mapping[str, Any]) -> str:
        role = request["role"]
        revision = self._revisions.get(role, 0) + 1
        self._revisions[role] = revision
        return f"Some reasoning first.\n{_BEGIN}\n{self.mark} instruction for {role}, revision {revision}.\n{_END}"


class SharedPromptAdapter(FakeAdapter):
    """One shared prompt (``writer``) whose transcript still names two native speakers."""

    def roles(self) -> list[str]:
        return ["writer"]


def grid_cell(
    *,
    seed: int = 0,
    budget: int = 600,
    task: str = TASK,
    topology: str = "sequential",
    framework: str = "langgraph",
    team_size: int = 4,
    task_model: str = QWEN_MODEL,
    method: str = "mamut_gepa",
) -> CellSpec:
    """A MAMUT-GEPA grid cell over the fake dataset rows."""
    return fake_cell(
        method=method,
        task=task,
        topology=topology,
        framework=framework,
        team_size=team_size,
        task_model=task_model,
        seed=seed,
        budget=budget,
    )


def grid_runner(cell: CellSpec, *, adapter_class: type = FakeAdapter, store=None, directory=None):
    """Budget-owning protocol runner over a fake adapter, data and scorer for ``cell``."""
    return protocol_runner(cell, fake_runtime(cell, adapter_class), store=store, directory=directory)


__all__ = ["FakeReflection", "MARK", "SharedPromptAdapter", "TASK", "grid_cell", "grid_runner"]
