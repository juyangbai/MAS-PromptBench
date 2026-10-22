"""Independent APPS runner (LangGraph): N seeded replicas, majority vote over their programs.

Replica i is a python_exec ReAct agent sampling with seed i. The submission is
the most common whitespace-normalized program (replicas without one abstain;
ties: lowest replica, see :mod:`core.voting`); with tests, only that program is
run. N is the team size (``configs/teams/<dataset>.yaml``, r=4;
``INDEPENDENT_N_AGENTS`` overrides). ``teamsizes/independent/apps`` runs this
module with r = 2, 4, 8 and 10.
"""

from __future__ import annotations

import asyncio
import operator
import os
from pathlib import Path
from typing import Annotated

from langchain_core.tools import tool
from langgraph.constants import END, START
from langgraph.graph.state import StateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.types import Send
from typing_extensions import TypedDict

from core import cli, code_tasks, prompts, settings, teams
from core.batch import attempt
from core.code_tasks import exact_match_score, extract_code  # noqa: F401  (runner API)
from core.llm import chat_openai
from core.tasks import apps as task
from core.tasks.apps import format_prompt, load_instances
from core.telemetry import langchain_ensemble_telemetry, normalize
from core.thinking import strip_ai_thinking, strip_thinking  # noqa: F401  (runner API)

TOPOLOGY = "independent"
TEAM_SIZE = globals().get("TEAM_SIZE")  # preset by the teamsizes/ variants (core.variant)
TEAM = teams.spec(TOPOLOGY, task.DATASET, TEAM_SIZE)

VLLM_BASE_URL = settings.base_url()
MODEL_ID = settings.model_id()
N_AGENTS = settings.independent_n_agents(TEAM.n_agents)  # replica i samples with seed i

_APPS_TEST_TIMEOUT_S = int(os.environ.get("APPS_TEST_TIMEOUT_S", str(task.TEST_TIMEOUT_S)))
_RECURSION_LIMIT = int(os.environ.get("APPS_INDEPENDENT_RECURSION_LIMIT", str(TEAM.recursion_limit)))
PER_ROW_TIMEOUT_S = int(os.environ.get("APPS_INDEPENDENT_ROW_TIMEOUT_S", "180"))

SYSTEM_PROMPT = prompts.role_prompt(TOPOLOGY, task.DATASET, TEAM.role)

python_exec = tool(
    code_tasks.make_python_exec(
        code_tasks.PYTHON_EXEC_DOC,
        default_timeout_s=int(os.environ.get("APPS_TOOL_TIMEOUT_S", str(code_tasks.EXEC_TIMEOUT_S))),
        char_budget=int(os.environ.get("APPS_TOOL_OUTPUT_CHAR_BUDGET", "4000")),
    )
)


def run_tests(code: str, input_output: dict, timeout_s: int = _APPS_TEST_TIMEOUT_S) -> dict:
    """:func:`core.tasks.apps.run_tests` with this runner's default timeout (``APPS_TEST_TIMEOUT_S``)."""
    return task.run_tests(code, input_output, timeout_s=timeout_s)


def _build_one_agent(seed: int):
    """One replica's ReAct agent, sampling with ``seed``."""
    model = chat_openai(model=MODEL_ID, base_url=VLLM_BASE_URL, seed=seed)
    return create_react_agent(model=model, tools=[python_exec], prompt=SYSTEM_PROMPT)


class State(TypedDict):
    problem: str
    starter_code: str | None
    prompt: str
    answers: Annotated[list[dict], operator.add]


class AgentInput(TypedDict):
    agent_id: int
    seed: int
    prompt: str


async def _run_replica(inp: AgentInput) -> dict:
    """Run one replica and return its program and transcript."""
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
        "code": extract_code(final),
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


def solve(
    problem: str,
    starter_code: str | None = None,
    input_output: dict | None = None,
    timeout_s: int = _APPS_TEST_TIMEOUT_S,
) -> dict:
    """Run the ensemble on one problem.

    Returns ``{"code", "pass_rate", "winner", "per_agent"}``: the voted program,
    its pass rate on ``input_output`` (None without tests), its agent id and each
    replica's ``{agent_id, seed, code, raw, messages}``. The vote never sees the tests.
    """
    compiled = build_graph().compile()
    prompt = format_prompt(problem, starter_code)

    async def _run():
        return await asyncio.wait_for(
            compiled.ainvoke({"problem": problem, "starter_code": starter_code, "prompt": prompt, "answers": []}),
            timeout=PER_ROW_TIMEOUT_S,
        )

    result = asyncio.run(_run())
    per_agent = sorted(result["answers"], key=lambda a: a["agent_id"])
    winner = per_agent[code_tasks.select_program([a["code"] for a in per_agent])]
    code = winner["code"]
    pass_rate = None
    if input_output is not None:
        pass_rate = run_tests(code, input_output, timeout_s=timeout_s)["pass_rate"] if code else 0.0
    return {"code": code, "pass_rate": pass_rate, "winner": winner["agent_id"], "per_agent": per_agent}


def run_batch(
    instances: list[dict],
    out_path: Path | None = None,
    verbose: bool = True,
    per_test_timeout_s: int = _APPS_TEST_TIMEOUT_S,
    _propagate_errors: bool = False,
) -> dict:
    """Solve and score every instance (``_propagate_errors`` re-raises a failed row)."""

    def row(_, inst: dict) -> dict:
        out, latency_s, error = attempt(
            lambda: solve(
                inst["problem"],
                starter_code=inst.get("starter_code") or None,
                input_output=inst["input_output"],
                timeout_s=per_test_timeout_s,
            ),
            fallback={"code": None},
            propagate=_propagate_errors,
        )
        code = out.get("code")
        per_agent = out.get("per_agent") or []
        return task.record(
            inst,
            code,
            code_tasks.selection_scores(code, out.get("winner"), out.get("pass_rate") or 0.0),
            per_agent=code_tasks.compact_replicas(per_agent),
            latency_s=round(latency_s, 2),
            **normalize(langchain_ensemble_telemetry(per_agent)),
            error=error,
        )

    label = f"independent/APPS (N={N_AGENTS})"
    return task.run_batch(instances, row, out_path=out_path, verbose=verbose, label=label)


def run_one(instance: dict, out_dir: Path | None = None) -> dict:
    """Score one instance, letting any failure raise so the caller can retry it."""
    return run_batch([instance], out_path=None, verbose=False, _propagate_errors=True)["per_instance"][0]


def _canned_demo() -> None:
    _, problem, _, tests = task.DEMOS[0]
    out = solve(problem, input_output=tests)
    print(f"=== Ensemble ({N_AGENTS} replicas), voted winner: agent_{out['winner']} ===")
    task.print_demo_code(out["code"], tests)


def main(argv: list[str] | None = None) -> int:
    return cli.main(
        argv,
        description="Independent-topology APPS runner (LangGraph majority vote).",
        load_instances=load_instances,
        run_batch=run_batch,
        demo=_canned_demo,
        source=task.SOURCE,
        add_arguments=task.add_arguments,
    )


if __name__ == "__main__":
    raise SystemExit(main())
