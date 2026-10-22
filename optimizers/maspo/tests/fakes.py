"""Role-echo adapters and a fake proposal/judge model for offline MASPO tests.

Each role's message echoes its prompt, so a changed prompt changes that role's
observable output. Seed prompts answer even-numbered items; a terminal prompt
containing ``MARK`` answers every item. The fake model appends ``MARK`` (plus
the first sample's question, so the two minibatch halves give two distinct
offspring) to the reference prompt, and its pairwise judge prefers the output
with more ``MARK``s, answering ``B`` on ties. All MAS rollouts go through the
protocol ``ProtocolRunner``.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from typing import Any

from optimizers.protocol.cells import QWEN_MODEL
from optimizers.protocol.schema import CellSpec
from optimizers.protocol.tests.fakes import FakeReflectionBackend, fake_cell, fake_runtime, protocol_runner

# Config
MARK = "GOOD"
TASK = "hotpotqa"


class RoleEchoAdapter:
    """Prompt-mutable adapter with configurable native roles (``ROLES``) and final role (``TERMINAL``)."""

    ROLES: tuple[str, ...] = ()
    TERMINAL = ""
    CALLS: list[dict[str, Any]] = []

    def __init__(self) -> None:
        self._prompts = {role: f"Seed prompt for {role}." for role in self.ROLES}

    def roles(self) -> list[str]:
        return list(self.ROLES)

    def get_prompt(self, role: str) -> str:
        return self._prompts[role]

    def set_prompt(self, role: str, text: str) -> None:
        self._prompts[role] = text

    def reset(self) -> None:
        return None

    def run_example(self, example: dict) -> dict:
        item = str(example["id"])
        RoleEchoAdapter.CALLS.append(
            {"id": item, "seed": int(os.environ["REQUEST_SEED"]), "prompts": dict(self._prompts)}
        )
        correct = MARK in self._prompts[self.TERMINAL] or int(item[2:]) % 2 == 0
        messages = [{"source": "user", "content": example["question"]}]
        messages += [{"source": role, "content": f"{role} output under: {self._prompts[role]}"} for role in self.ROLES]
        return {
            "answer": example["question"].upper() if correct else "wrong",
            "runner_output": {"messages": messages},
            "telemetry": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "n_llm_calls": len(self.ROLES),
                "n_tool_calls": 0,
            },
        }


def adapter_class(roles: tuple[str, ...], terminal: str) -> type:
    return type("RoleEchoAdapter", (RoleEchoAdapter,), {"ROLES": tuple(roles), "TERMINAL": terminal})


CENTRALIZED = adapter_class(("manager", "retriever_worker", "reasoner_worker"), "manager")
# Native stage order deliberately differs from the sorted bundle order.
SEQUENTIAL = adapter_class(("zeta_reader", "middle_planner", "alpha_writer"), "alpha_writer")
SHARED = adapter_class(("solver",), "solver")


def _between(text: str, start: str, end: str) -> str:
    head = text.split(start, 1)
    return head[1].split(end, 1)[0] if len(head) == 2 else ""


class FakeModel(FakeReflectionBackend):
    """Deterministic stand-in for the MASPO proposal and judge model."""

    def respond(self, prompt: str, request: Mapping[str, Any]) -> str:
        if request["phase"] == "proposal":
            reference = _between(prompt, "<reference_prompt>\n", "\n</reference_prompt>") or _between(
                prompt, "Reference prompt:\n```\n", "\n```"
            )
            first = re.search(r"Problem 1:\n(.*)", prompt)
            tag = first.group(1).strip()[-3:] if first else "?"
            return f"<analyse>needs detail</analyse><prompt>{reference} {MARK}[{tag}]</prompt>"
        output_a = _between(prompt, "Output A:\n", "Output B:")
        output_b = prompt.split("Output B:\n", 1)[-1]
        return "A" if output_a.count(MARK) > output_b.count(MARK) else "B"


def grid_cell(
    *,
    topology: str,
    seed: int = 0,
    budget: int = 600,
    task: str = TASK,
    framework: str = "langgraph",
    team_size: int = 4,
    task_model: str = QWEN_MODEL,
    communication: str = "freeform",
) -> CellSpec:
    """A MASPO grid cell over the fake dataset rows."""
    return fake_cell(
        method="maspo",
        task=task,
        topology=topology,
        framework=framework,
        communication=communication,
        team_size=team_size,
        task_model=task_model,
        seed=seed,
        budget=budget,
    )


def grid_runner(cell: CellSpec, adapter: type):
    """Budget-owning protocol runner over ``adapter`` and the fake data and scorer."""
    return protocol_runner(cell, fake_runtime(cell, adapter))


__all__ = [
    "CENTRALIZED",
    "FakeModel",
    "MARK",
    "RoleEchoAdapter",
    "SEQUENTIAL",
    "SHARED",
    "TASK",
    "adapter_class",
    "grid_cell",
    "grid_runner",
]
