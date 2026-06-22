"""Centralized/AutoGen BFCL real-runner adapter."""
from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Callable
from typing import Any, Sequence

from real_runner_mipro.adapters.bfcl_common import (
    coerce_instance,
    compact_role_trace,
    extract_canonical,
    execution_prompt,
    flatten_user_request,
    load_prompts,
    schemas_text,
)
from real_runner_mipro.lm import TASK_MODEL, next_task_endpoint


TOPOLOGY = "centralized_autogen"
PROMPT_TOPOLOGY = "centralized"
DATASET = "bfcl"
ROLES = ["manager", "inspector_worker", "caller_worker", "validator_worker"]

MANAGER_TERMINATE_NUDGE = (
    "\n\nWhen you emit the final fenced ```json``` block containing "
    "the canonical call list, immediately follow it with the literal "
    "string TERMINATE on its own line so the group-chat knows to stop."
)


def format_task(user_request: str, schemas: str) -> str:
    return (
        "USER REQUEST:\n"
        f"{user_request}\n\n"
        "SCHEMAS:\n"
        f"{schemas}\n\n"
        "Emit the final canonical call list as a SINGLE fenced ```json``` "
        "block. Canonical form is a list of dicts; each dict has exactly "
        "ONE key equal to the ACTUAL function name from one of the schemas "
        "above, and the value is the arguments dict."
    )


class CentralizedAutoGenBFCLAdapter:
    topology = TOPOLOGY
    dataset = DATASET

    def __init__(
        self,
        prompts: dict[str, str] | None = None,
        client_factory: Callable[[], Any] | None = None,
    ):
        self._prompts = load_prompts(PROMPT_TOPOLOGY, ROLES, prompts)
        self.client_factory = client_factory or self._default_client_factory

    def roles(self) -> list[str]:
        return list(ROLES)

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
        return asyncio.run(self._run_example_async(example))

    async def _run_example_async(self, example: Any) -> dict:
        instance = coerce_instance(example)
        team, client = self._build_team()
        task = format_task(flatten_user_request(instance["question"]), schemas_text(instance))
        try:
            result = await team.run(task=task)
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
        model_output = extract_canonical(final)
        if model_output is None:
            caller_msgs = [msg for msg in messages if msg["source"] == "caller_worker"]
            if caller_msgs:
                model_output = extract_canonical(caller_msgs[-1]["content"])
        if model_output is None:
            for msg in reversed(messages):
                model_output = extract_canonical(msg.get("content") or "")
                if model_output is not None:
                    break
        return {
            "model_output": model_output or [],
            "winner": "manager",
            "buckets": [],
            "messages": messages,
        }

    def format_role_trace(self, role: str, output: Any) -> str:
        self._check_role(role)
        if not isinstance(output, dict):
            return str(output)
        role_msgs = [msg["content"] for msg in output.get("messages") or [] if msg.get("source") == role]
        last = role_msgs[-1] if role_msgs else ""
        return compact_role_trace(
            role=role,
            model_output=output.get("model_output") or [],
            winner=output.get("winner"),
            buckets=output.get("buckets") or [],
            details=[f"{role}_last_message={last[:1200]}"],
        )

    def describe_runtime(self, example: Any | None = None) -> dict:
        instance = coerce_instance(example) if example is not None else {}
        return {
            "topology": self.topology,
            "dataset": self.dataset,
            "framework": "autogen",
            "roles": self.roles(),
            "tool_count": len(instance.get("function") or []),
            "prompt_prefix": self.get_prompt("manager")[:80],
        }

    def _build_team(self):
        from autogen_agentchat.agents import AssistantAgent
        from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
        from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage
        from autogen_agentchat.teams import SelectorGroupChat

        client = self.client_factory()

        manager = AssistantAgent(
            "manager",
            description="Coordinator that plans the call, dispatches composition + validation, and emits the final canonical JSON.",
            model_client=client,
            system_message=execution_prompt(self._prompts["manager"], PROMPT_TOPOLOGY, "manager") + MANAGER_TERMINATE_NUDGE,
        )
        inspector_worker = AssistantAgent(
            "inspector_worker",
            description="Reads the schema and returns an argument plan.",
            model_client=client,
            system_message=execution_prompt(self._prompts["inspector_worker"], PROMPT_TOPOLOGY, "inspector_worker"),
        )
        caller_worker = AssistantAgent(
            "caller_worker",
            description="Composes the canonical JSON call per manager instruction.",
            model_client=client,
            system_message=execution_prompt(self._prompts["caller_worker"], PROMPT_TOPOLOGY, "caller_worker"),
        )
        validator_worker = AssistantAgent(
            "validator_worker",
            description="Checks the call against the schema.",
            model_client=client,
            system_message=execution_prompt(self._prompts["validator_worker"], PROMPT_TOPOLOGY, "validator_worker"),
        )

        def selector_func(messages: Sequence[BaseAgentEvent | BaseChatMessage]) -> str | None:
            if not messages:
                return manager.name
            if messages[-1].source != manager.name:
                return manager.name
            return None

        selector_prompt = (
            "You are coordinating a 4-agent team on a BFCL function-calling task.\n"
            "Select the next agent to act.\n\n{roles}\n\n"
            "Conversation so far:\n{history}\n\n"
            "Pick exactly one agent from {participants}."
        )
        termination = TextMentionTermination("TERMINATE") | MaxMessageTermination(24)
        team = SelectorGroupChat(
            [manager, inspector_worker, caller_worker, validator_worker],
            model_client=client,
            termination_condition=termination,
            selector_prompt=selector_prompt,
            selector_func=selector_func,
            allow_repeated_speaker=True,
        )
        return team, client

    @staticmethod
    def _default_client_factory():
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
            max_tokens=1024,
            extra_body={
                "repetition_penalty": 1.05,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )

    @staticmethod
    def _check_role(role: str) -> None:
        if role not in ROLES:
            raise KeyError(f"Unknown role {role!r}; expected one of {ROLES}")
