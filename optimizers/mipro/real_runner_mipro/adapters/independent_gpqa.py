"""Independent/GPQA real-runner adapter."""
from __future__ import annotations

import asyncio
import operator
import os
import time
from collections import Counter
from collections.abc import Callable
from typing import Annotated, Any

from typing_extensions import TypedDict

from real_runner_mipro.adapters.gpqa_common import (
    answer_text,
    calculator_tool,
    clean_messages,
    coerce_instance,
    compact_role_trace,
    default_chat_model,
    execution_prompt,
    final_answer_from_messages,
    load_prompt,
    majority_vote,
    prompt_from_instance,
)


ROLE = "solver"
TOPOLOGY = "independent"
DATASET = "gpqa"


class State(TypedDict):
    prompt: str
    answers: Annotated[list[dict], operator.add]


class AgentInput(TypedDict):
    agent_id: int
    seed: int
    prompt: str


class IndependentGPQAAdapter:
    topology = TOPOLOGY
    dataset = DATASET

    def __init__(
        self,
        prompt: str | None = None,
        n_agents: int | None = None,
        model_factory: Callable[[int], Any] | None = None,
        keep_messages: bool | None = None,
    ):
        self._prompts = {ROLE: prompt if prompt is not None else load_prompt(TOPOLOGY, ROLE)}
        self.n_agents = n_agents or int(os.environ.get("INDEPENDENT_N_AGENTS", "4"))
        self.model_factory = model_factory or default_chat_model
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
        instance = coerce_instance(example)
        compiled = self._compiled_graph()
        result = asyncio.run(compiled.ainvoke({"prompt": prompt_from_instance(instance), "answers": []}))
        per_agent = sorted(result["answers"], key=lambda a: a["agent_id"])
        winner_letter = majority_vote(per_agent)
        winner_id = None
        for agent in per_agent:
            if agent.get("answer") == winner_letter:
                winner_id = agent.get("agent_id")
                break
        votes = dict(Counter(a["answer"] for a in per_agent if a.get("answer") is not None))
        winner_raw = ""
        if winner_id is not None:
            winner_raw = next((a.get("raw", "") for a in per_agent if a.get("agent_id") == winner_id), "")
        return {
            "model_output": [],
            "answer": winner_letter,
            "answer_text": answer_text(winner_letter, winner_raw),
            "winner": winner_id,
            "buckets": votes,
            "per_agent": per_agent,
        }

    def format_role_trace(self, role: str, output: Any) -> str:
        self._check_role(role)
        if not isinstance(output, dict):
            return str(output)
        details = []
        for agent in output.get("per_agent") or []:
            details.append(
                "agent "
                f"id={agent.get('agent_id')} seed={agent.get('seed')} "
                f"answer={agent.get('answer')} solve_s={agent.get('solve_s')} "
                f"error={agent.get('error') or 'None'} raw_tail={(agent.get('raw') or '')[-300:]}"
            )
        return compact_role_trace(
            role=role,
            answer=output.get("answer"),
            winner=output.get("winner"),
            votes=output.get("buckets") or {},
            details=details,
        )

    def describe_runtime(self, example: Any | None = None) -> dict:
        return {
            "topology": self.topology,
            "dataset": self.dataset,
            "roles": self.roles(),
            "n_agents": self.n_agents,
            "prompt_prefix": self.get_prompt(ROLE)[:80],
            "example_id": coerce_instance(example).get("id") if example is not None else None,
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
            Send(f"agent_{i}", {"agent_id": i, "seed": i, "prompt": state["prompt"]})
            for i in range(self.n_agents)
        ]

    async def _run_replica(self, inp: AgentInput) -> dict:
        from langgraph.prebuilt import create_react_agent

        agent = create_react_agent(
            model=self.model_factory(inp["seed"]),
            tools=[calculator_tool()],
            prompt=execution_prompt(self._prompts[ROLE], TOPOLOGY, ROLE),
        )
        start = time.time()
        try:
            result = await agent.ainvoke(
                {"messages": [("user", inp["prompt"])]},
                config={"recursion_limit": int(os.environ.get("GPQA_INDEPENDENT_RECURSION_LIMIT", "15"))},
            )
            messages = clean_messages(result["messages"])
            letter, raw = final_answer_from_messages(messages)
            error = None
        except Exception as exc:
            messages = []
            letter = None
            raw = ""
            error = f"{type(exc).__name__}: {exc}"

        answer = {
            "agent_id": inp["agent_id"],
            "seed": inp["seed"],
            "answer": letter,
            "raw": raw,
            "solve_s": round(time.time() - start, 2),
            "error": error,
        }
        if self.keep_messages:
            answer["messages"] = messages
        return {"answers": [answer]}

    @staticmethod
    def _check_role(role: str) -> None:
        if role != ROLE:
            raise KeyError(f"Unknown role {role!r}; expected {ROLE!r}")
