"""Independent GPQA-Diamond runner (LangGraph): N seeded replicas, letter majority vote.

Replica i is a calculator ReAct agent sampling with seed i; the answer is the
most common extracted letter (ties: lowest replica). N is the team size
(``configs/teams/gpqa.yaml``, r=4; ``INDEPENDENT_N_AGENTS`` overrides).
``teamsizes/independent/gpqa`` runs this module with r = 2, 4, 8 and 10.
"""

from __future__ import annotations

import asyncio
import operator
from pathlib import Path
from typing import Annotated

from langchain_core.tools import tool
from langgraph.constants import END, START
from langgraph.graph.state import StateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.types import Send
from typing_extensions import TypedDict

from core import cli, prompts, settings, teams
from core.batch import attempt
from core.calculator import CALCULATOR_DOC, make_calculator
from core.llm import chat_openai
from core.tasks import gpqa as task
from core.tasks.gpqa import extract_answer, format_prompt, load_instances, majority_vote
from core.telemetry import langchain_ensemble_telemetry, normalize
from core.thinking import strip_ai_thinking, strip_thinking  # noqa: F401  (runner API)

TOPOLOGY = "independent"
TEAM_SIZE = globals().get("TEAM_SIZE")  # preset by the teamsizes/ variants (core.variant)
TEAM = teams.spec(TOPOLOGY, task.DATASET, TEAM_SIZE)

VLLM_BASE_URL = settings.base_url()
MODEL_ID = settings.model_id()
N_AGENTS = settings.independent_n_agents(TEAM.n_agents)  # replica i samples with seed i

# Stall guards: each row gets a wall-clock cap and each replica a ReAct recursion limit.
PER_ROW_TIMEOUT_S = 120
_RECURSION_LIMIT = TEAM.recursion_limit

SYSTEM_PROMPT = prompts.role_prompt(TOPOLOGY, task.DATASET, TEAM.role)

calculator = tool(make_calculator(CALCULATOR_DOC))


def _build_one_agent(seed: int):
    """One replica's ReAct agent, sampling with ``seed``."""
    model = chat_openai(model=MODEL_ID, base_url=VLLM_BASE_URL, seed=seed)
    return create_react_agent(model=model, tools=[calculator], prompt=SYSTEM_PROMPT)


class State(TypedDict):
    question: str
    choices: list[str]
    prompt: str
    answers: Annotated[list[dict], operator.add]


class AgentInput(TypedDict):
    agent_id: int
    seed: int
    prompt: str


async def _run_replica(inp: AgentInput) -> dict:
    """Run one replica and return its letter and transcript."""
    agent = _build_one_agent(seed=inp["seed"])
    result = await agent.ainvoke(
        {"messages": [("user", inp["prompt"])]},
        config={"recursion_limit": _RECURSION_LIMIT},
    )
    strip_ai_thinking(result["messages"])
    final = result["messages"][-1].content
    answer = {
        "agent_id": inp["agent_id"],
        "seed": inp["seed"],
        "answer": extract_answer(final),
        "raw": final,
        "messages": result["messages"],
    }
    return {"answers": [answer]}


def _fan_out(state: State) -> list[Send]:
    return [Send(f"agent_{i}", {"agent_id": i, "seed": i, "prompt": state["prompt"]}) for i in range(N_AGENTS)]


def build_graph() -> StateGraph:
    graph = StateGraph(State)
    for i in range(N_AGENTS):
        graph.add_node(f"agent_{i}", _run_replica)
    graph.add_conditional_edges(START, _fan_out)
    graph.add_edge([f"agent_{i}" for i in range(N_AGENTS)], END)
    return graph


def solve(question: str, choices: list[str]) -> dict:
    """Run the ensemble on one question.

    Returns ``{"answer", "per_agent", "votes"}``: the majority letter, each
    replica's ``{agent_id, seed, answer, raw, messages}`` and the letter counts.
    """
    compiled = build_graph().compile()
    prompt = format_prompt(question, choices)

    async def _run():
        return await asyncio.wait_for(
            compiled.ainvoke({"question": question, "choices": choices, "prompt": prompt, "answers": []}),
            timeout=PER_ROW_TIMEOUT_S,
        )

    result = asyncio.run(_run())
    per_agent = sorted(result["answers"], key=lambda a: a["agent_id"])
    return {
        "answer": majority_vote(per_agent),
        "per_agent": per_agent,
        "votes": task.votes(a.get("answer") for a in per_agent),
    }


def run_batch(
    instances: list[dict],
    out_path: Path | None = None,
    verbose: bool = True,
    _propagate_errors: bool = False,
) -> dict:
    """Solve and score every instance (``_propagate_errors`` re-raises a failed row)."""
    fallback = {"answer": None, "per_agent": [], "votes": {}}

    def row(_, inst: dict) -> dict:
        out, latency_s, error = attempt(
            lambda: solve(inst["question"], inst["choices"]), fallback=fallback, propagate=_propagate_errors
        )
        return task.record(
            inst,
            out["answer"],
            votes=out["votes"],
            per_agent=[
                {"agent_id": a["agent_id"], "seed": a["seed"], "answer": a["answer"], "raw": a["raw"]}
                for a in out["per_agent"]
            ],
            latency_s=round(latency_s, 2),
            **normalize(langchain_ensemble_telemetry(out.get("per_agent") or [])),
            error=error,
        )

    label = f"independent/GPQA-Diamond (N={N_AGENTS})"
    return task.run_batch(instances, row, out_path=out_path, verbose=verbose, label=label)


def run_one(instance: dict, out_dir: Path | None = None) -> dict:
    """Score one instance, letting any failure raise so the caller can retry it."""
    return run_batch([instance], out_path=None, verbose=False, _propagate_errors=True)["per_instance"][0]


def _canned_demo() -> None:
    out = solve(task.DEMO_QUESTION, task.DEMO_CHOICES)
    print(f"\n=== Ensemble ({N_AGENTS} replicas) vote: {out['votes']}")
    task.print_demo_answer(out["answer"])
    for a in out["per_agent"]:
        print(f"--- agent_{a['agent_id']} (seed {a['seed']}) -> {a['answer']!r} ---")


def main(argv: list[str] | None = None) -> int:
    return cli.main(
        argv,
        description="Independent-topology GPQA-Diamond runner (LangGraph ensemble).",
        load_instances=load_instances,
        run_batch=run_batch,
        demo=_canned_demo,
        source=task.SOURCE,
        add_arguments=task.add_arguments,
    )


if __name__ == "__main__":
    raise SystemExit(main())
