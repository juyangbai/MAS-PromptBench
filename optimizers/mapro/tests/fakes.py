"""Fake model backends and per-topology adapters for offline MAPRO tests.

Adapters answer every item correctly once all role prompts carry ``MARK``;
seed prompts answer only even-numbered items. The fake reflection model
returns marked pool variants and mutations, the fake judge scores the
candidate response (node) or the upstream output plus downstream prompt
(edge) by the marker, and the fake probe echoes whether the candidate prompt
is marked. With ``prefer_latest`` the judge ranks marked candidates by their
mutation round, so every round selects (and evaluates) a new assignment. All
MAS rollouts go through the protocol ``ProtocolRunner``.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from typing import Any

from optimizers.protocol.cells import LLAMA_MODEL, QWEN_MODEL
from optimizers.protocol.runner import AdapterRuntime
from optimizers.protocol.schema import CellSpec
from optimizers.protocol.tests.fakes import FakeReflectionBackend, fake_cell, fake_runtime, protocol_runner

# Config
MARK = "IMPROVED"
TASK = "hotpotqa"
MODELS = {"qwen": QWEN_MODEL, "llama": LLAMA_MODEL}


def _marked(text: str | None) -> bool:
    return MARK in (text or "")


def _between(text: str, start: str, end: str) -> str:
    head = text.split(start, 1)
    return head[1].split(end, 1)[0] if len(head) == 2 else ""


class FakeBackend(FakeReflectionBackend):
    """Deterministic stand-in for the reflection and task-model tiers."""

    def __init__(self, *, prefer_latest: bool = False) -> None:
        super().__init__()
        self.prefer_latest = prefer_latest

    def respond(self, prompt: str, request: Mapping[str, Any]) -> str:
        return self._respond(prompt, request["phase"], request["system"], prefer_latest=self.prefer_latest)

    @staticmethod
    def _respond(prompt: str, phase: str, system: str | None, *, prefer_latest: bool = False) -> str:
        if phase == "candidate_probe":
            revision = re.search(r"revision (\d+)\.", system or "")
            return (
                f"{MARK} candidate output round {revision.group(1) if revision else 0}"
                if _marked(system)
                else "plain candidate output"
            )
        if phase == "mapro_node_edge_judge":
            if "Agent response (produced under the candidate role prompt):" in prompt:
                response = _between(
                    prompt, "Agent response (produced under the candidate role prompt):", "Preference demonstrations"
                )
                if not _marked(response):
                    return "0.10"
                round_ = re.search(r"candidate output round (\d+)", response)
                return f"{0.5 + 0.05 * int(round_.group(1)):.2f}" if prefer_latest and round_ else "0.95"
            if prefer_latest:
                return "0.50"
            upstream = _between(
                prompt, "Upstream agent's output (the message passed downstream):", "The downstream agent will operate"
            )
            downstream = _between(
                prompt,
                "The downstream agent will operate under this role prompt:",
                "Downstream preference demonstrations",
            )
            return {2: "0.95", 1: "0.50", 0: "0.10"}[int(_marked(upstream)) + int(_marked(downstream))]
        role = _between(prompt, "Agent role: ", "\n").strip()
        if "Produce " in prompt and "VARIANT" in prompt:
            count = int(re.search(r"Produce (\d+) alternative", prompt).group(1))
            return "\n".join(f"VARIANT {i}: {MARK} {role} prompt variant {i}" for i in range(1, count + 1))
        if "BLAME <parent_id>" in prompt:
            parents = re.findall(r"--- parent (\S+)", prompt)
            return "\n".join(f"BLAME {parent}: its output omitted a needed detail" for parent in parents)
        nonce = re.search(r"Revision attempt (\S+) ", prompt)
        return f"{MARK} {role} prompt revision {nonce.group(1) if nonce else 0}"


class TopologyAdapter:
    """Prompt-mutable fake MAS emitting source-tagged messages in its topology's shape."""

    TOPOLOGY = ""
    ROLES: tuple[str, ...] = ()
    CALLS: list[dict[str, Any]] = []
    FAIL_IDS: set[str] = set()  # items whose every attempt fails before observation
    HARD_IDS: set[str] = set()  # items no prompt solves

    def __init__(self, n_agents: int = 4, n_rounds: int = 2) -> None:
        self.n_agents, self.n_rounds = int(n_agents), int(n_rounds)
        self._prompts = {role: f"Seed prompt for the {role}." for role in self.ROLES}

    def roles(self) -> list[str]:
        return list(self.ROLES)

    def get_prompt(self, role: str) -> str:
        return self._prompts[role]

    def set_prompt(self, role: str, text: str) -> None:
        self._prompts[role] = text

    def reset(self) -> None:
        return None

    def _messages(self, question: str) -> list[dict[str, Any]]:
        raise NotImplementedError

    def run_example(self, example: dict) -> dict:
        item = str(example["id"])
        TopologyAdapter.CALLS.append(
            {
                "id": item,
                "seed": int(os.environ["REQUEST_SEED"]),
                "model": os.environ["MODEL_ID"],
                "temperature": float(os.environ["TASK_MODEL_TEMPERATURE"]),
                "prompts": dict(self._prompts),
            }
        )
        if item in TopologyAdapter.FAIL_IDS:
            raise ConnectionError("endpoint refused the connection")
        correct = (
            all(_marked(prompt) for prompt in self._prompts.values()) or int(item[2:]) % 2 == 0
        ) and item not in TopologyAdapter.HARD_IDS
        messages = [{"source": "user", "content": example["question"]}, *self._messages(example["question"])]
        return {
            "answer": example["question"].upper() if correct else "wrong",
            "messages": messages,
            "telemetry": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "n_llm_calls": len(messages) - 1,
                "n_tool_calls": 0,
            },
        }


class CentralizedAdapter(TopologyAdapter):
    TOPOLOGY = "centralized"
    ROLES = ("manager", "coder_worker", "tester_worker", "analyzer_worker")

    def _messages(self, question: str) -> list[dict[str, Any]]:
        workers = self.ROLES[1:]
        calls = [
            {"name": f"delegate_to_{worker}", "args": {"instructions": f"{worker}: handle {question}"}}
            for worker in workers
        ]
        return [
            {"source": "manager", "content": "delegating", "tool_calls": calls},
            *({"source": worker, "content": f"{worker} report"} for worker in workers),
            {"source": "manager", "content": "final answer"},
        ]


class SequentialAdapter(TopologyAdapter):
    TOPOLOGY = "sequential"
    ROLES = ("planner", "writer", "checker")  # native stage order differs from sorted order

    def _messages(self, question: str) -> list[dict[str, Any]]:
        return [{"source": role, "content": f"{role} stage output"} for role in self.ROLES]


class IndependentAdapter(TopologyAdapter):
    TOPOLOGY = "independent"
    ROLES = ("solver",)

    def _messages(self, question: str) -> list[dict[str, Any]]:
        return [
            {"source": "solver", "name": f"replica_{index}", "content": f"replica {index} answer"}
            for index in range(self.n_agents)
        ]


class DecentralizedAdapter(TopologyAdapter):
    TOPOLOGY = "decentralized"
    ROLES = ("debater",)

    def _messages(self, question: str) -> list[dict[str, Any]]:
        return [
            {"source": "debater", "name": f"peer_{peer}", "content": f"round {turn} peer {peer} view"}
            for turn in range(self.n_rounds)
            for peer in range(self.n_agents)
        ]


ADAPTERS = {
    adapter.TOPOLOGY: adapter
    for adapter in (CentralizedAdapter, SequentialAdapter, IndependentAdapter, DecentralizedAdapter)
}


def mapro_cell(
    topology: str,
    *,
    framework: str = "langgraph",
    team_size: int = 4,
    budget: int = 600,
    model: str = "qwen",
    seed: int = 0,
    communication: str = "freeform",
) -> CellSpec:
    """A MAPRO grid cell over the fake dataset rows."""
    return fake_cell(
        method="mapro",
        task=TASK,
        topology=topology,
        framework=framework,
        communication=communication,
        team_size=team_size,
        task_model=MODELS[model],
        seed=seed,
        budget=budget,
    )


def mapro_runtime(cell: CellSpec) -> AdapterRuntime:
    """The cell's runtime over the fake adapter of its topology."""
    return fake_runtime(cell, ADAPTERS[cell.topology])


def mapro_runner(cell: CellSpec):
    """Budget-owning protocol runner over the topology's fake adapter."""
    return protocol_runner(cell, mapro_runtime(cell))


def reset_fake() -> None:
    TopologyAdapter.CALLS = []
    TopologyAdapter.FAIL_IDS = set()
    TopologyAdapter.HARD_IDS = set()


__all__ = [
    "ADAPTERS",
    "FakeBackend",
    "MARK",
    "TASK",
    "TopologyAdapter",
    "mapro_cell",
    "mapro_runner",
    "mapro_runtime",
    "reset_fake",
]
