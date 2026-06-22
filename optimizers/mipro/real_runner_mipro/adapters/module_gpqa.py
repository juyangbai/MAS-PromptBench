"""GPQA adapters backed by the real topology modules."""
from __future__ import annotations

import asyncio
import os
import threading
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

from real_runner_mipro.adapters.gpqa_common import (
    answer_text,
    coerce_instance,
    compact_role_trace,
    default_chat_model,
    execution_prompt,
    load_prompts,
)
from real_runner_mipro.adapters.module_common import _AgentRecursionHeadroom
from real_runner_mipro.adapters.module_common import import_real_module
from real_runner_mipro.lm import TASK_MODEL, next_task_endpoint


DATASET = "gpqa"

_MODULE_LOCKS: dict[str, threading.Lock] = {}
_MODULE_LOCKS_GUARD = threading.Lock()


def _module_lock(module_name: str) -> threading.Lock:
    with _MODULE_LOCKS_GUARD:
        if module_name not in _MODULE_LOCKS:
            _MODULE_LOCKS[module_name] = threading.Lock()
        return _MODULE_LOCKS[module_name]


class ModuleGPQAAdapter:
    """Prompt-mutable adapter that delegates execution to a real GPQA module."""

    topology: str
    dataset = DATASET
    framework: str
    prompt_topology: str
    roles_: list[str]
    module_name: str

    def __init__(
        self,
        prompts: dict[str, str] | None = None,
        n_agents: int | None = None,
        n_rounds: int | None = None,
        model_factory: Callable[[int], Any] | None = None,
        client_factory: Callable[[], Any] | None = None,
    ):
        self._prompts = load_prompts(self.prompt_topology, self.roles_, prompts)
        self.n_agents = n_agents
        self.n_rounds = n_rounds
        self.model_factory = model_factory or default_chat_model
        self.client_factory = client_factory

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
        module = import_real_module(self.module_name)
        lock = _module_lock(self.module_name)
        with lock, self._patched_module(module):
            out = module.solve(instance["question"], instance["choices"])
        final_answer = out.get("answer")
        raw = out.get("raw") or self._fallback_raw(out)
        return {
            "model_output": [],
            "answer": final_answer,
            "answer_text": answer_text(final_answer, raw),
            "winner": self._winner(out),
            "buckets": self._buckets(out),
            "raw": raw,
            "runner_output": out,
        }

    def format_role_trace(self, role: str, output: Any) -> str:
        self._check_role(role)
        if not isinstance(output, dict):
            return str(output)
        runner_output = output.get("runner_output") or {}
        details = [self._role_detail(role, runner_output)]
        return compact_role_trace(
            role=role,
            answer=output.get("answer"),
            winner=output.get("winner"),
            votes=output.get("buckets") or {},
            details=details,
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
            patch("_load_prompt", lambda role: execution_prompt(self._prompts[role], self.prompt_topology, role))
        if hasattr(module, "SYSTEM_PROMPT") and "debater" in self._prompts:
            patch("SYSTEM_PROMPT", execution_prompt(self._prompts["debater"], self.prompt_topology, "debater"))
        if hasattr(module, "_build_llm"):
            if self.framework == "crewai":
                patch("_build_llm", self._default_crewai_llm_factory)
            else:
                patch("_build_llm", lambda: self.model_factory(0))
        if hasattr(module, "_build_client"):
            patch("_build_client", self.client_factory or self._default_openai_like_client(module))
        if hasattr(module, "build_agent"):
            original_build_agent = module.build_agent

            def _build_agent_with_recursion_headroom(*args, **kwargs):
                return _AgentRecursionHeadroom(original_build_agent(*args, **kwargs))

            patch("build_agent", _build_agent_with_recursion_headroom)
        if hasattr(module, "create_react_agent"):
            original_create_react_agent = module.create_react_agent

            def _create_react_agent_with_recursion_headroom(*args, **kwargs):
                return _AgentRecursionHeadroom(original_create_react_agent(*args, **kwargs))

            patch("create_react_agent", _create_react_agent_with_recursion_headroom)
        if hasattr(module, "N_AGENTS") and self.n_agents is not None:
            patch("N_AGENTS", self.n_agents)
        if hasattr(module, "N_ROUNDS") and self.n_rounds is not None:
            patch("N_ROUNDS", self.n_rounds)
        if hasattr(module, "_RECURSION_LIMIT"):
            patch("_RECURSION_LIMIT", max(int(getattr(module, "_RECURSION_LIMIT", 0) or 0), 90))
        try:
            yield
        finally:
            for name, value in restore.items():
                setattr(module, name, value)

    def _default_openai_like_client(self, module):
        if "autogen" in self.framework:
            def build_autogen_client():
                from autogen_ext.models.openai import OpenAIChatCompletionClient

                return OpenAIChatCompletionClient(
                    model=os.environ.get("MODEL_ID", TASK_MODEL),
                    base_url=next_task_endpoint(),
                    api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
                    model_info={
                        "vision": False,
                        "function_calling": True,
                        "json_output": True,
                        "family": "qwen",
                        "structured_output": False,
                    },
                    temperature=0.2,
                    top_p=0.9,
                    seed=0,
                    max_tokens=2048,
                    extra_body={
                        "repetition_penalty": 1.05,
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                )

            return build_autogen_client

        def build_openai_client():
            from openai import OpenAI

            return OpenAI(base_url=next_task_endpoint(), api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"), timeout=300.0, max_retries=5)

        return build_openai_client

    @staticmethod
    def _default_crewai_llm_factory():
        from crewai import LLM

        return LLM(
            model=f"openai/{os.environ.get('MODEL_ID', TASK_MODEL)}",
            base_url=next_task_endpoint(),
            api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
            temperature=0.2,
            top_p=0.9,
            seed=0,
            max_tokens=2048,
            additional_drop_params=[],
            extra_body={
                "repetition_penalty": 1.05,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )

    def _winner(self, out: dict) -> Any:
        if "by_stage" in out:
            return self.roles_[-1]
        if "per_peer" in out:
            for peer in out.get("per_peer") or []:
                if peer.get("letter") == out.get("answer"):
                    return peer.get("peer")
        return "manager" if self.topology.startswith("centralized") else None

    def _buckets(self, out: dict) -> dict:
        if "per_peer" not in out:
            return {}
        buckets: dict[str, int] = {}
        for peer in out.get("per_peer") or []:
            letter = peer.get("letter")
            if letter:
                buckets[letter] = buckets.get(letter, 0) + 1
        return buckets

    def _fallback_raw(self, out: dict) -> str:
        if "by_stage" in out:
            return str((out.get("by_stage") or {}).get(self.roles_[-1], ""))
        if "per_peer" in out:
            for peer in out.get("per_peer") or []:
                if peer.get("letter") == out.get("answer"):
                    return str(peer.get("raw") or peer.get("raw_tail") or "")
        messages = out.get("messages") or []
        if messages:
            return str(messages[-1].get("content") if isinstance(messages[-1], dict) else messages[-1])
        return ""

    def _role_detail(self, role: str, out: dict) -> str:
        if "by_stage" in out:
            text = (out.get("by_stage") or {}).get(role, "")
            return f"{role}_text={str(text)[:1200]}"
        if role == "debater" and "per_peer" in out:
            parts = []
            for peer in out.get("per_peer") or []:
                raw = peer.get("raw") or peer.get("raw_tail") or ""
                parts.append(f"peer={peer.get('peer')} letter={peer.get('letter')} raw_tail={str(raw)[-350:]}")
            return "\n".join(parts)
        messages = out.get("messages") or []
        role_msgs = [
            msg.get("content", "")
            for msg in messages
            if isinstance(msg, dict) and msg.get("source") == role
        ]
        return f"{role}_last_message={str(role_msgs[-1] if role_msgs else '')[:1200]}"

    def _check_role(self, role: str) -> None:
        if role not in self.roles_:
            raise KeyError(f"Unknown role {role!r}; expected one of {self.roles_}")


class SequentialGPQAAdapter(ModuleGPQAAdapter):
    topology = "sequential"
    framework = "langgraph"
    prompt_topology = "sequential"
    roles_ = ["analyzer", "solver", "critic", "verifier"]
    module_name = "topologies.sequential.langgraph.gpqa.langgraph_gpqa"


class SequentialCrewAIGPQAAdapter(ModuleGPQAAdapter):
    topology = "sequential_crewai"
    framework = "crewai"
    prompt_topology = "sequential"
    roles_ = ["analyzer", "solver", "critic", "verifier"]
    module_name = "topologies.sequential.crewai.gpqa.crewai_gpqa"


class DecentralizedGPQAAdapter(ModuleGPQAAdapter):
    topology = "decentralized"
    framework = "langgraph"
    prompt_topology = "decentralized"
    roles_ = ["debater"]
    module_name = "topologies.decentralized.langgraph.gpqa.langgraph_gpqa"

    def __init__(self, *args, n_agents: int | None = None, n_rounds: int | None = None, **kwargs):
        super().__init__(
            *args,
            n_agents=n_agents or int(os.environ.get("DECENTRALIZED_N_AGENTS", "4")),
            n_rounds=n_rounds or int(os.environ.get("DECENTRALIZED_N_ROUNDS", "2")),
            **kwargs,
        )


class DecentralizedOpenAIGPQAAdapter(DecentralizedGPQAAdapter):
    topology = "decentralized_openai"
    framework = "openai"
    module_name = "topologies.decentralized.openai.gpqa.openai_gpqa"


class CentralizedGPQAAdapter(ModuleGPQAAdapter):
    topology = "centralized"
    framework = "langgraph"
    prompt_topology = "centralized"
    roles_ = ["manager", "analyzer_worker", "solver_worker", "verifier_worker"]
    module_name = "topologies.centralized.langgraph.gpqa.langgraph_gpqa"


class CentralizedAutoGenGPQAAdapter(CentralizedGPQAAdapter):
    topology = "centralized_autogen"
    framework = "autogen"
    module_name = "topologies.centralized.autogen.gpqa.autogen_gpqa"

    def run_example(self, example: Any) -> dict:
        return asyncio.run(self._run_example_async(example))

    async def _run_example_async(self, example: Any) -> dict:
        from autogen_agentchat.agents import AssistantAgent
        from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
        from autogen_agentchat.teams import SelectorGroupChat

        module = import_real_module(self.module_name)
        instance = coerce_instance(example)
        client = (self.client_factory or self._default_openai_like_client(module))()

        manager = AssistantAgent(
            "manager",
            description="Coordinator that delegates to 3 workers and synthesizes the final letter.",
            model_client=client,
            system_message=execution_prompt(self._prompts["manager"], self.prompt_topology, "manager")
            + "\n\nWhen you emit the final 'Answer: X' line, immediately follow it "
            "with the literal string TERMINATE on its own line so the group-chat knows to stop.",
        )
        analyzer_worker = AssistantAgent(
            "analyzer_worker",
            description="Analyzes scientific principles and derives each option.",
            model_client=client,
            system_message=execution_prompt(self._prompts["analyzer_worker"], self.prompt_topology, "analyzer_worker"),
        )
        solver_worker = AssistantAgent(
            "solver_worker",
            description="Picks one letter + rationale given the manager's instruction and analysis.",
            model_client=client,
            system_message=execution_prompt(self._prompts["solver_worker"], self.prompt_topology, "solver_worker"),
        )
        verifier_worker = AssistantAgent(
            "verifier_worker",
            description="Sanity-checks the solver's letter against the analyzer's output.",
            model_client=client,
            system_message=execution_prompt(self._prompts["verifier_worker"], self.prompt_topology, "verifier_worker"),
        )

        def selector_func(messages) -> str | None:
            if not messages:
                return manager.name
            if messages[-1].source != manager.name:
                return manager.name
            return None

        selector_prompt = (
            "You are coordinating a 4-agent team on a multiple-choice science question.\n"
            "Select the next agent to act.\n\n{roles}\n\n"
            "Conversation so far:\n{history}\n\n"
            "Pick exactly one agent from {participants}."
        )
        termination = TextMentionTermination("TERMINATE") | MaxMessageTermination(16)
        team = SelectorGroupChat(
            [manager, analyzer_worker, solver_worker, verifier_worker],
            model_client=client,
            termination_condition=termination,
            selector_prompt=selector_prompt,
            selector_func=selector_func,
            allow_repeated_speaker=True,
        )

        try:
            mcq = module.format_mcq(instance["question"], instance["choices"])
            result = await asyncio.wait_for(team.run(task=mcq), timeout=120)
        finally:
            await client.close()

        messages = [
            {
                "source": getattr(msg, "source", None),
                "content": getattr(msg, "content", None)
                if isinstance(getattr(msg, "content", None), str)
                else str(getattr(msg, "content", "")),
            }
            for msg in result.messages
        ]
        manager_msgs = [msg for msg in messages if msg["source"] == "manager"]
        final = manager_msgs[-1]["content"] if manager_msgs else ""
        final_answer = module.extract_answer(final)
        if final_answer is None:
            for msg in reversed(messages):
                final_answer = module.extract_answer(msg.get("content") or "")
                if final_answer is not None:
                    break
        return {
            "model_output": [],
            "answer": final_answer,
            "answer_text": answer_text(final_answer, final),
            "winner": "manager",
            "buckets": {},
            "raw": final,
            "runner_output": {
                "answer": final_answer,
                "raw": final,
                "messages": messages,
            },
        }
