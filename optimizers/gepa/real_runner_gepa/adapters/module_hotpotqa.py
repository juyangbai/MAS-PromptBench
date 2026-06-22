"""HotpotQA adapters backed by the real topology modules."""
from __future__ import annotations

import asyncio
import os
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from real_runner_gepa.adapters.module_common import _AgentRecursionHeadroom
from real_runner_gepa.adapters.module_common import import_isolated_real_module
from real_runner_gepa.adapters.gpqa_common import coerce_instance, default_chat_model
from real_runner_gepa.datasets.hotpotqa import normalize_answer
from real_runner_gepa.lm import TASK_MODEL, next_task_endpoint
from real_runner_gepa.output_contracts import append_output_contract


DATASET = "hotpotqa"

_MODULE_LOCKS: dict[str, threading.Lock] = {}
_MODULE_LOCKS_GUARD = threading.Lock()


def _module_lock(module_name: str) -> threading.Lock:
    with _MODULE_LOCKS_GUARD:
        if module_name not in _MODULE_LOCKS:
            _MODULE_LOCKS[module_name] = threading.Lock()
        return _MODULE_LOCKS[module_name]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def prompt_path(topology: str, role: str) -> Path:
    return repo_root() / "configs" / "prompts" / topology / "hotpotqa" / f"{role}.txt"


def load_prompts(topology: str, roles: list[str], overrides: dict[str, str] | None = None) -> dict[str, str]:
    overrides = overrides or {}
    return {role: overrides.get(role, prompt_path(topology, role).read_text().strip()) for role in roles}


class ModuleHotpotQAAdapter:
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
    ):
        self._prompts = load_prompts(self.prompt_topology, self.roles_, prompts)
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
        module = import_isolated_real_module(self.module_name)
        try:
            with self._patched_module(module):
                out = module.solve(instance["question"])
        except TimeoutError as exc:
            return self._runtime_failure_output(instance, exc)
        except Exception as exc:
            if type(exc).__name__ == "BadRequestError":
                return self._runtime_failure_output(instance, exc)
            raise
        answer = out.get("answer")
        raw = out.get("raw") or self._fallback_raw(out)
        return {
            "model_output": [],
            "answer": answer,
            "answer_text": raw or (f"Answer: {answer}" if answer else ""),
            "winner": self._winner(out),
            "buckets": self._buckets(out),
            "raw": raw,
            "runner_output": out,
        }

    def _runtime_failure_output(self, instance: dict, exc: BaseException) -> dict:
        raw = (
            "ERROR: real HotpotQA runner failed before producing a final "
            f"short-form answer for task {instance.get('id') or '<unknown>'}: "
            f"{type(exc).__name__}: {exc}"
        )
        return {
            "model_output": [],
            "answer": None,
            "answer_text": raw,
            "winner": None,
            "buckets": {},
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
        runner_output = output.get("runner_output") or {}
        return "\n".join(
            [
                f"role={role}",
                f"winner={output.get('winner')}",
                f"votes={output.get('buckets') or {}}",
                f"selected_answer={output.get('answer')}",
                self._role_detail(role, runner_output),
            ]
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
        tool_func_restore = []

        def patch(name: str, value: Any) -> None:
            restore[name] = getattr(module, name, None)
            setattr(module, name, value)

        def patch_tool_func(name: str) -> None:
            tool_obj = getattr(module, name, None)
            func = getattr(tool_obj, "func", None)
            if not callable(func):
                return

            def safe_func(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    return f"ERROR: {type(exc).__name__}: {exc}"

            tool_func_restore.append((tool_obj, func))
            try:
                tool_obj.func = safe_func
            except Exception:
                object.__setattr__(tool_obj, "func", safe_func)

        patch_tool_func("wikipedia_search")
        patch_tool_func("wikipedia_page")
        if hasattr(module, "_load_prompt"):
            patch("_load_prompt", lambda role: self._prompt_for_module(module, role))
        if hasattr(module, "SYSTEM_PROMPT"):
            role = "debater" if "debater" in self._prompts else self.roles_[0]
            patch("SYSTEM_PROMPT", self._prompt_for_module(module, role))
        if hasattr(module, "_build_llm"):
            if self.framework == "crewai":
                patch("_build_llm", self._crewai_llm)
            else:
                patch("_build_llm", lambda: default_chat_model(0, max_tokens=2048))
        if hasattr(module, "VLLM_BASE_URL"):
            patch("VLLM_BASE_URL", next_task_endpoint())
        if hasattr(module, "MODEL_ID"):
            patch("MODEL_ID", os.environ.get("MODEL_ID", TASK_MODEL))
        if hasattr(module, "_build_client"):
            patch("_build_client", self._openai_client if self.framework == "openai" else self._autogen_client)
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
            for tool_obj, func in reversed(tool_func_restore):
                try:
                    tool_obj.func = func
                except Exception:
                    object.__setattr__(tool_obj, "func", func)
            for name, value in restore.items():
                setattr(module, name, value)

    def _prompt_for_module(self, module, role: str) -> str:
        text = self._prompts[role]
        nudge = getattr(module, "_OUTPUT_FORMAT_NUDGE", "")
        if nudge and "Answer: <short-form>" not in text:
            text = text + nudge
        return append_output_contract(text, self.dataset, self.prompt_topology, role)

    @staticmethod
    def _crewai_llm():
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

    @staticmethod
    def _openai_client():
        from openai import OpenAI

        return OpenAI(base_url=next_task_endpoint(), api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"), timeout=300.0, max_retries=5)

    @staticmethod
    def _autogen_client():
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

    def _winner(self, out: dict) -> Any:
        if "by_stage" in out:
            return self.roles_[-1]
        if "per_agent" in out:
            target = normalize_answer(out.get("answer") or "")
            for agent in out.get("per_agent") or []:
                if normalize_answer(agent.get("answer") or "") == target:
                    return agent.get("agent_id")
        if "per_peer" in out:
            target = normalize_answer(out.get("answer") or "")
            for peer in out.get("per_peer") or []:
                if normalize_answer(peer.get("answer") or "") == target:
                    return peer.get("peer")
        return "manager" if self.topology.startswith("centralized") else None

    def _buckets(self, out: dict) -> dict:
        if isinstance(out.get("votes"), dict):
            return out["votes"]
        buckets: dict[str, int] = {}
        for item in (out.get("per_agent") or []) + (out.get("per_peer") or []):
            answer = item.get("answer")
            if answer:
                key = normalize_answer(answer)
                buckets[key] = buckets.get(key, 0) + 1
        return buckets

    def _fallback_raw(self, out: dict) -> str:
        if "by_stage" in out:
            return str((out.get("by_stage") or {}).get(self.roles_[-1], ""))
        if "per_agent" in out:
            return str((out.get("per_agent") or [{}])[0].get("raw", ""))
        if "per_peer" in out:
            return str((out.get("per_peer") or [{}])[0].get("raw", ""))
        messages = out.get("messages") or []
        if messages:
            return str(messages[-1].get("content") if isinstance(messages[-1], dict) else messages[-1])
        return ""

    def _role_detail(self, role: str, out: dict) -> str:
        if "by_stage" in out:
            text = (out.get("by_stage") or {}).get(role, "")
            return f"{role}_text={str(text)[:1200]}"
        if role in {"solver", "debater"}:
            items = out.get("per_agent") or out.get("per_peer") or []
            parts = []
            for item in items:
                ident = item.get("agent_id", item.get("peer"))
                raw = item.get("raw") or ""
                parts.append(f"member={ident} answer={item.get('answer')} raw_tail={str(raw)[-350:]}")
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


class SingleHotpotQAAdapter(ModuleHotpotQAAdapter):
    topology = "single"
    framework = "langgraph"
    prompt_topology = "single"
    roles_ = ["solver"]
    module_name = "topologies.single.hotpotqa.langgraph_hotpotqa"


class IndependentHotpotQAAdapter(ModuleHotpotQAAdapter):
    topology = "independent"
    framework = "langgraph"
    prompt_topology = "independent"
    roles_ = ["solver"]
    module_name = "topologies.independent.hotpotqa.langgraph_hotpotqa"

    def __init__(self, *args, n_agents: int | None = None, **kwargs):
        super().__init__(*args, n_agents=n_agents or int(os.environ.get("INDEPENDENT_N_AGENTS", "4")), **kwargs)


class SequentialHotpotQAAdapter(ModuleHotpotQAAdapter):
    topology = "sequential"
    framework = "langgraph"
    prompt_topology = "sequential"
    roles_ = ["planner", "retriever", "reasoner", "writer"]
    module_name = "topologies.sequential.langgraph.hotpotqa.langgraph_hotpotqa"


class SequentialCrewAIHotpotQAAdapter(SequentialHotpotQAAdapter):
    topology = "sequential_crewai"
    framework = "crewai"
    module_name = "topologies.sequential.crewai.hotpotqa.crewai_hotpotqa"


class DecentralizedHotpotQAAdapter(ModuleHotpotQAAdapter):
    topology = "decentralized"
    framework = "langgraph"
    prompt_topology = "decentralized"
    roles_ = ["debater"]
    module_name = "topologies.decentralized.langgraph.hotpotqa.langgraph_hotpotqa"

    def __init__(self, *args, n_agents: int | None = None, n_rounds: int | None = None, **kwargs):
        super().__init__(
            *args,
            n_agents=n_agents or int(os.environ.get("DECENTRALIZED_N_AGENTS", "4")),
            n_rounds=n_rounds or int(os.environ.get("DECENTRALIZED_N_ROUNDS", "2")),
            **kwargs,
        )


class DecentralizedOpenAIHotpotQAAdapter(DecentralizedHotpotQAAdapter):
    topology = "decentralized_openai"
    framework = "openai"
    module_name = "topologies.decentralized.openai.hotpotqa.openai_hotpotqa"


class CentralizedHotpotQAAdapter(ModuleHotpotQAAdapter):
    topology = "centralized"
    framework = "langgraph"
    prompt_topology = "centralized"
    roles_ = ["manager", "retriever_worker", "reasoner_worker", "writer_worker"]
    module_name = "topologies.centralized.langgraph.hotpotqa.langgraph_hotpotqa"


class CentralizedAutoGenHotpotQAAdapter(CentralizedHotpotQAAdapter):
    topology = "centralized_autogen"
    framework = "autogen"
    module_name = "topologies.centralized.autogen.hotpotqa.autogen_hotpotqa"

    def run_example(self, example: Any) -> dict:
        return asyncio.run(self._run_example_async(example))

    async def _run_example_async(self, example: Any) -> dict:
        module = import_isolated_real_module(self.module_name)
        instance = coerce_instance(example)
        with self._patched_module(module):
            team = module.build_team()
            # module._build_client is patched, but build_team keeps the client
            # internal. AutoGen exposes it as model_client on the team.
            client = getattr(team, "_model_client", None) or getattr(team, "model_client", None)
            try:
                result = await asyncio.wait_for(team.run(task=instance["question"]), timeout=120)
            finally:
                if client is not None and hasattr(client, "close"):
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
        answer = module.extract_answer(final)
        return {
            "model_output": [],
            "answer": answer,
            "answer_text": final or (f"Answer: {answer}" if answer else ""),
            "winner": "manager",
            "buckets": {},
            "raw": final,
            "runner_output": {
                "answer": answer,
                "raw": final,
                "messages": messages,
            },
        }
