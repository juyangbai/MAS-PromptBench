"""Fake reflection LM and table-6 protocol runners over the protocol's fake adapter."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

from optimizers.protocol.schema import CellSpec
from optimizers.protocol.tests.fakes import FakeReflectionBackend, fake_cell, protocol_runner

# Config
# The fake adapter answers every item when the writer prompt contains GOOD and none with BAD.
GOOD_OVERLAY = "1. Re-read the question and check the final answer GOOD before replying.\n2. Keep handoffs short."
BAD_OVERLAY = "1. Reply BAD without checking.\n2. Skip the handoff."
PROMPT_KINDS = (
    ("eq3", "Please evaluate the quality and progress of the current iteration"),
    ("credit", "Please analyze the following Agent's performance"),
    ("salience", "You are given a list of short rules/suggestions"),
    ("overlay", "You are to synthesize a reusable TRUCE prompt-rule overlay"),
    ("aggregate", "Aggregate the following trajectory-aware prompt edits"),
    ("refine_meta", "Please optimize the Agent's System Prompt using meta-learned knowledge"),
    ("refine", "Please optimize the Agent's System Prompt based on the following evaluation suggestions"),
)


def prompt_kind(prompt: str) -> str:
    head = prompt.lstrip()
    for kind, marker in PROMPT_KINDS:
        if head.startswith(marker):
            return kind
    raise AssertionError(f"unexpected reflection prompt: {head[:80]!r}")


def credit_payload(role: str) -> dict[str, Any]:
    return {
        "overall_score": 7,
        "strengths": [f"{role} kept the handoff short"],
        "weaknesses": [f"{role} did not verify the final answer"],
        "prompt_suggestions": {
            "result_oriented_improvements": "Tie every step to the final answer",
            "effectiveness_enhancements": "N/A",
            "quality_focus_additions": "Check the answer format before replying",
            "collaboration_optimizations": "N/A",
        },
        "specific_prompt_modifications": {
            "add_instructions": ["Verify the final answer against the question", f"State the {role} output explicitly"],
            "remove_content": ["Remove redundant restatements of the task"],
            "restructure_suggestions": [f"Move the {role} checklist to the end"],
        },
    }


class FakeReflection(FakeReflectionBackend):
    """Routes each TAVO reflection prompt to a canned response; every request records its ``kind``.

    Credit answers arrive in a code fence followed by prose (exercising the
    ``raw_decode`` hardening). ``salience="broken"`` returns non-JSON so the
    frequency fallback runs; ``fail_kinds`` raise a transport error.
    """

    def __init__(self, overlay: str = GOOD_OVERLAY, *, salience: str = "ok", fail_kinds: tuple[str, ...] = ()) -> None:
        super().__init__()
        self.overlay = overlay
        self.salience = salience
        self.fail_kinds = tuple(fail_kinds)

    def complete(
        self,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        *,
        request_seed: int = 0,
        top_p: float = 1.0,
        max_output_tokens: int | None = None,
        thinking: bool = True,
        system: str | None = None,
        phase: str | None = None,
        role: str | None = None,
    ) -> str:
        """The driver's native call-site shape (``temperature``, ``max_tokens``) or the backend shape."""
        return super().complete(
            prompt,
            request_seed=request_seed,
            temperature=0.5 if temperature is None else temperature,
            top_p=top_p,
            max_output_tokens=max_tokens if max_output_tokens is None else max_output_tokens,
            thinking=thinking,
            system=system,
            phase=phase,
            role=role,
        )

    def respond(self, prompt: str, request: Mapping[str, Any]) -> str:
        kind = prompt_kind(prompt)
        request["kind"] = kind
        if kind in self.fail_kinds:
            raise ConnectionError("reflection endpoint unavailable")
        return getattr(self, f"_{kind}")(prompt)

    def kinds(self) -> list[str]:
        return [request["kind"] for request in self.requests]

    def of_kind(self, kind: str) -> list[dict[str, Any]]:
        return [request for request in self.requests if request["kind"] == kind]

    @staticmethod
    def _eq3(prompt: str) -> str:
        return json.dumps(
            {
                "iteration_progress": {"score": 6, "analysis": "advanced", "key_achievements": []},
                "iteration_summary": {"overall_score": 6, "areas_for_improvement": ["verify earlier"]},
            }
        )

    @staticmethod
    def _credit(prompt: str) -> str:
        role = re.search(r"- Agent ID: (\S+)", prompt).group(1)
        return "```json\n" + json.dumps(credit_payload(role)) + "\n```\nThese recommendations target the result."

    def _salience(self, prompt: str) -> str:
        if self.salience == "broken":
            return "I would rank verification first."
        items = list(dict.fromkeys(re.findall(r"^\d+\. (.+)$", prompt, re.M)))
        return json.dumps({"top": [{"text": item, "support_count": 1, "importance": 0} for item in items[:2]]})

    def _overlay(self, prompt: str) -> str:
        return self.overlay

    @staticmethod
    def _aggregate(prompt: str) -> str:
        role = re.search(r"edits for agent (\S+) across", prompt).group(1)
        return json.dumps(
            {
                "specific_prompt_modifications": {
                    "add_instructions": [f"Aggregated rule for {role}"],
                    "remove_content": [],
                    "restructure_suggestions": [],
                },
                "prompt_suggestions": {"result_oriented_improvements": "Check the final answer"},
                "strengths": [],
                "weaknesses": [],
            }
        )

    @staticmethod
    def _refine_meta(prompt: str) -> str:
        role = re.search(r"\*\*Agent ID:\*\* (\S+)", prompt).group(1)
        return f"```text\nMeta-refined prompt for {role}.\n```"

    @staticmethod
    def _refine(prompt: str) -> str:
        role = re.search(r"\*\*Agent ID:\*\* (\S+)", prompt).group(1)
        return f"Refined prompt for {role}."


def tavo_cell(
    *, topology: str = "sequential", task: str = "hotpotqa", seed: int = 0, budget: int = 600, team_size: int = 4
) -> CellSpec:
    """A table-6 TAVO cell whose rows come from the protocol's fake dataset."""
    return fake_cell(method="tavo", task=task, topology=topology, team_size=team_size, seed=seed, budget=budget)


def tavo_runner(cell: CellSpec, *, store: Any = None, directory: Any = None):
    """Budget-owning protocol runner over the fake adapter."""
    return protocol_runner(cell, store=store, directory=directory)


def sequential_trajectory(item: str, score: float) -> dict[str, Any]:
    """A two-stage trajectory shaped like ``_TavoRunnerAdapter.execute_batch`` output."""
    return {
        "id": item,
        "question": f"question {item}",
        "topology": "sequential",
        "team_size": 4,
        "task_model": "Qwen/Qwen3.5-9B",
        "prompt_roles": ["planner", "writer"],
        "messages": [
            {"source": "user", "content": f"question {item}"},
            {"source": "planner", "content": "plan the answer"},
            {"source": "writer", "content": "final answer"},
        ],
        "answer": "QUESTION" if score else "wrong",
        "score": score,
        "error": None,
        "telemetry": {"total_tokens": 15},
        "executed": True,
        "request_seed": 1,
    }


__all__ = [
    "BAD_OVERLAY",
    "FakeReflection",
    "GOOD_OVERLAY",
    "credit_payload",
    "prompt_kind",
    "sequential_trajectory",
    "tavo_cell",
    "tavo_runner",
]
