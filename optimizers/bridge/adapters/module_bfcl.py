"""BFCL adapters backed by the real LangGraph topology and team-size modules.

The per-topology BFCL adapters (``<topology>_bfcl.py``) re-build each graph
inside the adapter. Team-size and communication variants instead execute the
real runner module (``teamsizes/<topology>/bfcl/bfcl_r<N>.py`` or the
``topologies/`` base runner), with the adapter's mutable prompts and the
shared decoding helpers patched into the module for the duration of a call.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any

from optimizers.bridge.adapters.bfcl_common import (
    coerce_instance,
    compact_role_trace,
    default_chat_model,
    execution_prompt,
    load_prompts,
)
from optimizers.bridge.adapters.module_common import import_isolated_real_module
from optimizers.bridge.lm import TASK_MODEL, next_task_endpoint

DATASET = "bfcl"
# Scorer-only fields; never shown to the agents.
PRIVATE_FIELDS = ("ground_truth",)


class ModuleBFCLAdapter:
    """Prompt-mutable adapter that delegates one BFCL call to a real module."""

    topology: str
    dataset = DATASET
    framework = "langgraph"
    prompt_topology: str
    roles_: list[str]
    module_name: str

    def __init__(
        self,
        prompts: dict[str, str] | None = None,
        n_agents: int | None = None,
        n_rounds: int | None = None,
    ):
        self._prompts = load_prompts(self.prompt_topology, list(self.roles_), prompts)
        self.n_agents = n_agents
        self.n_rounds = n_rounds

    def roles(self) -> list[str]:
        return list(self.roles_)

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

    def run_example(self, example: Any) -> dict:
        instance = coerce_instance(example)
        task = {key: value for key, value in instance.items() if key not in PRIVATE_FIELDS}
        module = self.load_module()
        try:
            with self._patched_module(module):
                out = module.solve(task)
        except TimeoutError as exc:
            return self._runtime_failure_output(instance, exc)
        except Exception as exc:
            if type(exc).__name__ == "BadRequestError":
                return self._runtime_failure_output(instance, exc)
            raise
        out = dict(out or {})
        if not out.get("telemetry") and out.get("per_agent") and hasattr(module, "langchain_ensemble_telemetry"):
            out["telemetry"] = module.normalize(module.langchain_ensemble_telemetry(out["per_agent"]))
        return {
            "model_output": out.get("model_output") or [],
            "winner": self._winner(out),
            "buckets": out.get("buckets") or [],
            "raw": out.get("raw") or "",
            "runner_output": out,
        }

    def load_module(self):
        """A fresh copy of the runner module for one call (see ``import_isolated_real_module``)."""
        return import_isolated_real_module(self.module_name)

    def _runtime_failure_output(self, instance: dict, exc: BaseException) -> dict:
        raw = (
            "ERROR: real BFCL runner failed before producing a canonical call "
            f"for task {instance.get('id') or '<unknown>'}: {type(exc).__name__}: {exc}"
        )
        return {
            "model_output": [],
            "winner": None,
            "buckets": [],
            "raw": raw,
            "runner_output": {
                "error_type": type(exc).__name__,
                "error": str(exc),
                "id": instance.get("id"),
                "raw": raw,
            },
        }

    def format_role_trace(self, role: str, output: Any) -> str:
        self._check_role(role)
        if not isinstance(output, dict):
            return str(output)
        return compact_role_trace(
            role=role,
            model_output=output.get("model_output") or [],
            winner=output.get("winner"),
            buckets=output.get("buckets") or [],
            details=[self._role_detail(role, output.get("runner_output") or {})],
        )

    def describe_runtime(self, example: Any | None = None) -> dict:
        instance = coerce_instance(example) if example is not None else {}
        return {
            "topology": self.topology,
            "dataset": self.dataset,
            "framework": self.framework,
            "roles": self.roles(),
            "n_agents": self.n_agents,
            "n_rounds": self.n_rounds,
            "tool_count": len(instance.get("function") or []),
            "prompt_prefix": self.get_prompt(self.roles_[0])[:80],
            "example_id": instance.get("id"),
            "module": self.module_name,
        }

    @contextmanager
    def _patched_module(self, module):
        restore = {}

        def patch(name: str, value: Any) -> None:
            restore[name] = getattr(module, name, None)
            setattr(module, name, value)

        if hasattr(module, "_load_prompt"):
            original_load_prompt = module._load_prompt

            def _load_prompt(role: str, *args, **kwargs):
                if role in self._prompts:
                    return self._prompt_for_module(module, role)
                return original_load_prompt(role, *args, **kwargs)

            patch("_load_prompt", _load_prompt)
        if hasattr(module, "SYSTEM_PROMPT"):
            patch("SYSTEM_PROMPT", self._prompt_for_module(module, self.roles_[0]))
        if hasattr(module, "_build_llm"):
            patch("_build_llm", lambda: default_chat_model(0))
        if hasattr(module, "_build_one_agent"):
            # Independent replicas keep their per-replica seed, as in the runner.
            def _build_one_agent(tools, seed: int):
                return module.create_react_agent(
                    model=default_chat_model(seed), tools=tools, prompt=module.SYSTEM_PROMPT
                )

            patch("_build_one_agent", _build_one_agent)
        if hasattr(module, "VLLM_BASE_URL"):
            patch("VLLM_BASE_URL", next_task_endpoint())
        if hasattr(module, "MODEL_ID"):
            patch("MODEL_ID", os.environ.get("MODEL_ID", TASK_MODEL))
            if hasattr(module, "_register_model_with_bfcl"):
                module._register_model_with_bfcl(module.MODEL_ID)
        if hasattr(module, "N_AGENTS") and self.n_agents is not None:
            patch("N_AGENTS", self.n_agents)
        if hasattr(module, "N_ROUNDS") and self.n_rounds is not None:
            patch("N_ROUNDS", self.n_rounds)
        try:
            yield
        finally:
            for name, value in restore.items():
                setattr(module, name, value)

    def _prompt_for_module(self, module, role: str) -> str:
        return execution_prompt(self._prompts[role], self.prompt_topology, role)

    def _winner(self, out: dict) -> Any:
        if out.get("winner") is not None:
            return out.get("winner")
        if "by_stage" in out:
            return self.roles_[-1]
        if self.prompt_topology == "centralized":
            return "manager"
        return None

    def role_detail(self, role: str, out: dict) -> str:
        return self._role_detail(role, out)

    def _role_detail(self, role: str, out: dict) -> str:
        if "by_stage" in out:
            return f"{role}_text={str((out.get('by_stage') or {}).get(role, ''))[:1200]}"
        if isinstance(out.get("per_agent"), list):
            parts = []
            for agent in out["per_agent"]:
                parts.append(
                    f"agent id={agent.get('agent_id')} seed={agent.get('seed')} "
                    f"solve_s={agent.get('solve_s')} output={agent.get('model_output') or []} "
                    f"error={agent.get('error') or 'None'}"
                )
            return "\n".join(parts)
        if isinstance(out.get("per_peer"), list):
            parts = []
            for peer in out["per_peer"]:
                parts.append(
                    f"peer={peer.get('peer')} call={peer.get('call') or []} raw={str(peer.get('raw') or '')[:500]}"
                )
            return "\n".join(parts)
        role_msgs = [
            msg.get("content", "")
            for msg in out.get("messages") or []
            if isinstance(msg, dict) and msg.get("source") == role
        ]
        last = role_msgs[-1] if role_msgs else out.get("raw", "")
        return f"{role}_last_message={str(last)[:1200]}"

    def _check_role(self, role: str) -> None:
        if role not in self.roles_:
            raise KeyError(f"Unknown role {role!r}; expected one of {self.roles_}")
