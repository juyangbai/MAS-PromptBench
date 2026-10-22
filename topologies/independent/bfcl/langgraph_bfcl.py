"""Independent BFCL runner (LangGraph): N seeded ReAct replicas, majority vote over canonical calls.

Replica i is the single-agent tool caller sampling with seed i; the prediction
is the largest bucket of equal canonical outputs (ties: lowest replica). N is
the team size (``configs/teams/bfcl.yaml``, r=4; ``INDEPENDENT_N_AGENTS``
overrides). ``teamsizes/independent/bfcl`` runs this module with r = 2, 4, 8 and 10.
"""

from __future__ import annotations

import asyncio
import json
import operator
import time
from pathlib import Path
from typing import Annotated

from langgraph.constants import END, START
from langgraph.graph.state import StateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.types import Send
from typing_extensions import TypedDict

from core import prompts, settings, teams
from core.communication import CommPolicy
from core.llm import chat_openai
from core.paths import RESULTS_DIR
from core.tasks import bfcl as task
from core.tasks.bfcl import (  # noqa: F401  (runner API)
    AST_CATEGORIES,
    HF_DATASET,
    extract_first_tool_calls,
    load_instances,
    majority_vote,
    schema_to_tool,
    to_canonical,
)
from core.telemetry import langchain_ensemble_telemetry, normalize

TOPOLOGY = "independent"
TEAM_SIZE = globals().get("TEAM_SIZE")  # preset by the teamsizes/ variants (core.variant)
TEAM = teams.spec(TOPOLOGY, task.DATASET, TEAM_SIZE)
DEFAULT_OUT_DIR = RESULTS_DIR / ("bfcl_independent" if TEAM_SIZE is None else f"bfcl_independent_r{TEAM_SIZE}")

VLLM_BASE_URL = settings.base_url()
MODEL_ID = settings.model_id()
COMMUNICATION_FORMAT = globals().get("COMMUNICATION_FORMAT", "freeform")  # preset by the communications/ entries
COMMUNICATION = CommPolicy(COMMUNICATION_FORMAT, task.DATASET, TOPOLOGY)
N_AGENTS = settings.independent_n_agents(TEAM.n_agents)  # replica i samples with seed i

SYSTEM_PROMPT = COMMUNICATION.system_prompt(prompts.role_prompt(TOPOLOGY, task.DATASET, TEAM.role))

_register_model_with_bfcl = task.register_model  # called again by callers that repoint MODEL_ID
_register_model_with_bfcl(MODEL_ID)


def _build_one_agent(tools: list, seed: int):
    """One replica's ReAct agent over ``tools``, sampling with ``seed``."""
    llm = chat_openai(model=MODEL_ID, base_url=VLLM_BASE_URL, seed=seed)
    return create_react_agent(model=llm, tools=tools, prompt=SYSTEM_PROMPT)


def score_one(function_schemas: list[dict], model_output: list[dict], ground_truth: list[dict], category: str) -> dict:
    return task.score_one(function_schemas, model_output, ground_truth, category, MODEL_ID)


class State(TypedDict):
    instance: dict
    prompt: list  # the instance's request turns
    answers: Annotated[list[dict], operator.add]


class AgentInput(TypedDict):
    agent_id: int
    seed: int
    function_schemas: list[dict]
    prompt: list


async def _run_replica(inp: AgentInput) -> dict:
    """Run one replica and return its first-turn calls; a failure becomes an answer with an ``error``."""
    agent = _build_one_agent([schema_to_tool(s) for s in inp["function_schemas"]], seed=inp["seed"])
    answer = {"agent_id": inp["agent_id"], "seed": inp["seed"]}
    start = time.time()
    try:
        result = await agent.ainvoke({"messages": inp["prompt"]}, config={"recursion_limit": TEAM.recursion_limit})
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        answer.update(tool_calls=[], model_output=[], error=error, solve_s=round(time.time() - start, 2), messages=[])
        return {"answers": [answer]}
    elapsed = time.time() - start
    tool_calls = extract_first_tool_calls(result["messages"])
    answer.update(
        tool_calls=tool_calls,
        model_output=to_canonical(tool_calls, inp["function_schemas"]),
        messages=result["messages"],
        solve_s=round(elapsed, 2),
    )
    return {"answers": [answer]}


def _fan_out(state: State) -> list[Send]:
    functions = state["instance"]["function"]
    return [
        Send(f"agent_{i}", {"agent_id": i, "seed": i, "function_schemas": functions, "prompt": state["prompt"]})
        for i in range(N_AGENTS)
    ]


def build_graph() -> StateGraph:
    graph = StateGraph(State)
    for i in range(N_AGENTS):
        graph.add_node(f"agent_{i}", _run_replica)
    graph.add_conditional_edges(START, _fan_out)
    graph.add_edge([f"agent_{i}" for i in range(N_AGENTS)], END)
    return graph


def solve(instance: dict) -> dict:
    """Run the ensemble on one instance.

    Returns ``{"model_output", "winner", "buckets", "per_agent"}``: the winning
    canonical calls (or []), the winning replica, ``(canonical key, count)`` per
    bucket and each replica's ``{agent_id, seed, tool_calls, model_output,
    messages, solve_s, error?}``.
    """
    compiled = build_graph().compile()
    state = {"instance": instance, "prompt": instance["question"][0], "answers": []}
    result = asyncio.run(compiled.ainvoke(state))
    per_agent = sorted(result["answers"], key=lambda a: a["agent_id"])
    winner = majority_vote(per_agent)
    return {
        "model_output": winner.get("model_output") or [],
        "winner": winner["agent_id"],
        "buckets": task.vote_counts(per_agent),
        "per_agent": per_agent,
    }


def _format_per_agent(out: dict) -> str:
    """Trace text: the winner, the buckets and one line per replica."""
    lines = [f"winner: agent_{out.get('winner')}", f"buckets: {out.get('buckets')}", ""]
    for a in out.get("per_agent") or []:
        calls = a.get("model_output") or []
        call_str = json.dumps(calls, sort_keys=True) if calls else "(no call)"
        err = f"  err={a['error']!r}" if a.get("error") else ""
        lines.append(f"agent_{a['agent_id']} seed={a.get('seed')} solve_s={a.get('solve_s')}s  {call_str}{err}")
    return "\n".join(lines)


def run_one(instance: dict, ground_truth: dict, category: str, out_dir: Path) -> dict:
    """Solve one instance, score the voted calls and write the replicas' trace to ``out_dir/traces/<id>.txt``."""
    summary: dict = {"id": instance["id"], "category": category, "n_agents": N_AGENTS}
    try:
        out = solve(instance)
    except Exception as e:
        return task.solve_failed(summary, e)
    summary["winner"] = out.get("winner")
    summary["model_output"] = out.get("model_output") or []
    summary["buckets"] = out.get("buckets") or []
    summary["tool_calls"] = len(out.get("model_output") or [])
    summary.update(normalize(langchain_ensemble_telemetry(out.get("per_agent") or [])))
    task.write_trace(out_dir, instance["id"], _format_per_agent(out))
    return task.add_verdict(summary, score_one, instance, ground_truth, category)


def run_batch(
    category: str,
    limit: int | None = None,
    offset: int = 0,
    only: list[str] | None = None,
    out_dir: Path | None = None,
    verbose: bool = True,
) -> dict:
    """Score a slice of one subset; writes predictions.jsonl and results.jsonl afresh in ``out_dir``."""
    return task.evaluate(
        run_one,
        category,
        limit,
        offset,
        only,
        out_dir or DEFAULT_OUT_DIR,
        model_id=MODEL_ID,
        verbose=verbose,
        note=f"  (N={N_AGENTS})",
        hidden=("buckets",),
    )


def main(argv: list[str] | None = None) -> int:
    return task.cli_main(
        argv,
        description="Independent-topology BFCL ensemble runner.",
        run_one=run_one,
        model_id=MODEL_ID,
        default_out_dir=DEFAULT_OUT_DIR,
        hidden=("buckets",),
    )


if __name__ == "__main__":
    raise SystemExit(main())
