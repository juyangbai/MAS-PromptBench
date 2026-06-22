"""Independent/BFCL real-runner adapter pilot.

This is a contained copy/adaptation of the real runner shape from:
`topologies/independent/bfcl/langgraph_bfcl.py`.

The important difference is that the role prompt is instance-owned and
mutable through the adapter protocol, so an optimizer can replace it without
editing files under `configs/` or `topologies/`.
"""
from __future__ import annotations

import asyncio
import json
import operator
import os
import time
from collections.abc import Callable
from typing import Annotated, Any

from typing_extensions import TypedDict

from real_runner_mipro.adapters.bfcl_common import (
    canonical_key,
    coerce_instance,
    compact_role_trace,
    default_chat_model,
    execution_prompt,
    extract_first_tool_calls,
    load_prompt,
    recursion_limit,
    schema_to_tool,
    to_canonical,
)


ROLE = "caller"
TOPOLOGY = "independent"
DATASET = "bfcl"


class State(TypedDict):
    instance: dict
    prompt: list
    answers: Annotated[list[dict], operator.add]


class AgentInput(TypedDict):
    agent_id: int
    seed: int
    function_schemas: list[dict]
    prompt: list


def majority_vote(answers: list[dict]) -> dict | None:
    valid = [a for a in answers if a.get("model_output")]
    if not valid:
        return None

    buckets: dict[str, list[dict]] = {}
    for answer in valid:
        buckets.setdefault(canonical_key(answer["model_output"]), []).append(answer)

    best_key = None
    best_count = -1
    best_first_id = 10**9
    for key, bucket in buckets.items():
        first_id = min(a["agent_id"] for a in bucket)
        if (len(bucket), -first_id) > (best_count, -best_first_id):
            best_count = len(bucket)
            best_first_id = first_id
            best_key = key
    return min(buckets[best_key], key=lambda a: a["agent_id"])


def load_default_prompt() -> str:
    return load_prompt(TOPOLOGY, ROLE)


class IndependentBFCLAdapter:
    """Prompt-mutable adapter for the real independent/BFCL runner shape."""

    topology = TOPOLOGY
    dataset = DATASET

    def __init__(
        self,
        prompt: str | None = None,
        n_agents: int | None = None,
        model_factory: Callable[[int], Any] | None = None,
        keep_messages: bool | None = None,
    ):
        self._prompts = {ROLE: prompt if prompt is not None else load_default_prompt()}
        self.n_agents = n_agents or int(os.environ.get("INDEPENDENT_N_AGENTS", "4"))
        self.model_factory = model_factory or self._default_model_factory
        self.keep_messages = (
            keep_messages
            if keep_messages is not None
            else os.environ.get("REAL_RUNNER_KEEP_MESSAGES", "0") == "1"
        )
        self._compiled = None

    def roles(self) -> list[str]:
        return [ROLE]

    def get_prompt(self, role: str) -> str:
        self._check_role(role)
        return self._prompts[role]

    def set_prompt(self, role: str, text: str) -> None:
        self._check_role(role)
        self._prompts[role] = text
        self.reset()

    def reset(self) -> None:
        self._compiled = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_compiled"] = None
        return state

    def run_example(self, example: Any) -> dict:
        instance = self._coerce_instance(example)
        compiled = self._compiled_graph()
        prompt = instance["question"][0]
        result = asyncio.run(
            compiled.ainvoke(
                {
                    "instance": instance,
                    "prompt": prompt,
                    "answers": [],
                }
            )
        )
        per_agent = sorted(result["answers"], key=lambda a: a["agent_id"])
        winner = majority_vote(per_agent)

        buckets: dict[str, int] = {}
        for answer in [a for a in per_agent if a.get("model_output")]:
            key = canonical_key(answer["model_output"])
            buckets[key] = buckets.get(key, 0) + 1

        return {
            "model_output": (winner or {}).get("model_output") or [],
            "winner": (winner or {}).get("agent_id"),
            "buckets": sorted(buckets.items(), key=lambda kv: -kv[1]),
            "per_agent": per_agent,
        }

    def format_role_trace(self, role: str, output: Any) -> str:
        """Summarize real per-agent behavior in plain text for GEPA."""
        self._check_role(role)
        if not isinstance(output, dict):
            return str(output)

        details = []
        for agent in output.get("per_agent") or []:
            error = agent.get("error")
            model_output = agent.get("model_output") or []
            solve_s = agent.get("solve_s")
            details.append(
                "agent "
                f"id={agent.get('agent_id')} seed={agent.get('seed')} "
                f"solve_s={solve_s} output={model_output} "
                f"error={error or 'None'}"
            )
        return compact_role_trace(
            role=role,
            model_output=output.get("model_output") or [],
            winner=output.get("winner"),
            buckets=output.get("buckets") or [],
            details=details,
        )

    def describe_runtime(self, example: Any | None = None) -> dict:
        """Fast structural view for tests without making LLM calls."""
        tool_count = 0
        if example is not None:
            instance = self._coerce_instance(example)
            tool_count = len(instance.get("function") or [])
        return {
            "topology": self.topology,
            "dataset": self.dataset,
            "roles": self.roles(),
            "n_agents": self.n_agents,
            "prompt_prefix": self.get_prompt(ROLE)[:80],
            "tool_count": tool_count,
        }

    def _compiled_graph(self):
        if self._compiled is None:
            self._compiled = self._build_graph().compile()
        return self._compiled

    def _build_graph(self):
        from langgraph.constants import END, START
        from langgraph.graph.state import StateGraph

        graph = StateGraph(State)
        for i in range(self.n_agents):
            graph.add_node(f"agent_{i}", self._run_replica)
        graph.add_conditional_edges(START, self._fan_out)
        graph.add_edge([f"agent_{i}" for i in range(self.n_agents)], END)
        return graph

    def _fan_out(self, state: State) -> list:
        from langgraph.types import Send

        return [
            Send(
                f"agent_{i}",
                {
                    "agent_id": i,
                    "seed": i,
                    "function_schemas": state["instance"]["function"],
                    "prompt": state["prompt"],
                },
            )
            for i in range(self.n_agents)
        ]

    async def _run_replica(self, inp: AgentInput) -> dict:
        from langgraph.prebuilt import create_react_agent

        tools = [schema_to_tool(schema) for schema in inp["function_schemas"]]
        agent = create_react_agent(
            model=self.model_factory(inp["seed"]),
            tools=tools,
            prompt=execution_prompt(self._prompts[ROLE], TOPOLOGY, ROLE),
        )

        start = time.time()
        try:
            result = await agent.ainvoke(
                {"messages": inp["prompt"]},
                config={"recursion_limit": recursion_limit("BFCL_INDEPENDENT_RECURSION_LIMIT")},
            )
        except Exception as exc:
            return {
                "answers": [
                    {
                        "agent_id": inp["agent_id"],
                        "seed": inp["seed"],
                        "tool_calls": [],
                        "model_output": [],
                        "error": f"{type(exc).__name__}: {exc}",
                        "solve_s": round(time.time() - start, 2),
                    }
                ]
            }

        tool_calls = extract_first_tool_calls(result["messages"])
        answer = {
            "agent_id": inp["agent_id"],
            "seed": inp["seed"],
            "tool_calls": tool_calls,
            "model_output": to_canonical(tool_calls),
            "solve_s": round(time.time() - start, 2),
        }
        if self.keep_messages:
            answer["messages"] = result["messages"]
        return {
            "answers": [answer]
        }

    @staticmethod
    def _default_model_factory(seed: int):
        return default_chat_model(seed)

    @staticmethod
    def _coerce_instance(example: Any) -> dict:
        return coerce_instance(example)

    @staticmethod
    def _check_role(role: str) -> None:
        if role != ROLE:
            raise KeyError(f"Unknown role {role!r}; expected {ROLE!r}")
