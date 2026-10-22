"""Decentralized/OpenAI-Agents-SDK BFCL real-runner adapter."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any

from optimizers.bridge.adapters.bfcl_common import (
    coerce_instance,
    compact_role_trace,
    execution_prompt,
    load_prompts,
)
from optimizers.bridge.adapters.module_common import import_real_module, module_lock
from optimizers.bridge.lm import TASK_MODEL, next_task_endpoint

ROLE = "debater"
TOPOLOGY = "decentralized_openai_agents"
PROMPT_TOPOLOGY = "decentralized"
DATASET = "bfcl"
MODULE_NAME = "topologies.decentralized.openai_agents.bfcl.openai_agents_bfcl"


class DecentralizedOpenAIAgentsBFCLAdapter:
    """Runs the real Agents SDK debate runner with the candidate debater prompt.

    The submission is the runner's own aggregate (most common final-round
    output); ground truth is only used by the metric, never for selection.
    """

    topology = TOPOLOGY
    dataset = DATASET

    def __init__(
        self,
        prompts: dict[str, str] | None = None,
        n_agents: int | None = None,
        n_rounds: int | None = None,
    ):
        self._prompts = load_prompts(PROMPT_TOPOLOGY, [ROLE], prompts)
        self.n_agents = n_agents or int(os.environ.get("DECENTRALIZED_N_AGENTS", "4"))
        self.n_rounds = n_rounds or int(os.environ.get("DECENTRALIZED_N_ROUNDS", "2"))

    def roles(self) -> list[str]:
        return [ROLE]

    def get_prompt(self, role: str) -> str:
        self._check_role(role)
        return self._prompts[role]

    def set_prompt(self, role: str, text: str) -> None:
        self._check_role(role)
        self._prompts[role] = text

    def reset(self) -> None:
        return None

    def __getstate__(self):
        return self.__dict__.copy()

    @contextmanager
    def _patched_module(self, module):
        restore: dict[str, Any] = {}

        def patch(name: str, value: Any) -> None:
            restore[name] = getattr(module, name, None)
            setattr(module, name, value)

        patch("SYSTEM_PROMPT", execution_prompt(self._prompts[ROLE], PROMPT_TOPOLOGY, ROLE))
        patch("VLLM_BASE_URL", next_task_endpoint())
        patch("MODEL_ID", os.environ.get("MODEL_ID", TASK_MODEL))
        patch("N_AGENTS", self.n_agents)
        patch("N_ROUNDS", self.n_rounds)
        try:
            yield
        finally:
            for name, value in restore.items():
                setattr(module, name, value)

    def run_example(self, example: Any) -> dict:
        instance = coerce_instance(example)
        module = import_real_module(MODULE_NAME)
        row = {key: instance[key] for key in ("id", "question", "function") if key in instance}
        with module_lock(MODULE_NAME), self._patched_module(module):
            out = module.solve(row)
        return {
            "model_output": out.get("model_output") or [],
            "winner": out.get("winner"),
            "buckets": [],
            "per_peer": [
                {"peer": peer.get("peer"), "call": peer.get("call"), "raw": str(peer.get("raw") or "")[:1200]}
                for peer in out.get("per_peer") or []
            ],
            "runner_output": out,
        }

    def format_role_trace(self, role: str, output: Any) -> str:
        self._check_role(role)
        if not isinstance(output, dict):
            return str(output)
        details = [
            f"peer={peer.get('peer')} call={peer.get('call') or []} raw={str(peer.get('raw') or '')[:500]}"
            for peer in output.get("per_peer") or []
        ]
        return compact_role_trace(
            role=role,
            model_output=output.get("model_output") or [],
            winner=output.get("winner"),
            buckets=output.get("buckets") or [],
            details=details,
        )

    def describe_runtime(self, example: Any | None = None) -> dict:
        instance = coerce_instance(example) if example is not None else {}
        return {
            "topology": self.topology,
            "dataset": self.dataset,
            "framework": "openai_agents",
            "roles": self.roles(),
            "n_agents": self.n_agents,
            "n_rounds": self.n_rounds,
            "tool_count": len(instance.get("function") or []),
            "prompt_prefix": self.get_prompt(ROLE)[:80],
            "module": MODULE_NAME,
        }

    @staticmethod
    def _check_role(role: str) -> None:
        if role != ROLE:
            raise KeyError(f"unknown role {role!r}; expected {ROLE!r}")
