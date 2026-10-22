"""Sequential LCB runner (LangGraph): a pipeline of stages, each seeing all earlier outputs.

The stages are the team spec (``configs/teams/<dataset>.yaml``): at r=4 analyzer ->
coder -> tester -> debugger. A stage with tools is a python_exec ReAct agent, one
without is a single model call; the submission is the debugger's last fenced
Python block (else the coder's). ``teamsizes/sequential/lcb`` runs this module
with r = 2, 4, 8 and 10.
"""

from __future__ import annotations

import operator
from pathlib import Path
from typing import Annotated

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent
from typing_extensions import TypedDict

from core import cli, code_tasks, prompts, settings, teams
from core.batch import attempt
from core.code_tasks import exact_match_score, extract_code  # noqa: F401  (runner API)
from core.communication import CommPolicy
from core.llm import chat_openai
from core.paths import RESULTS_DIR
from core.tasks import lcb as task
from core.tasks.lcb import format_prompt, load_instances, run_tests  # noqa: F401
from core.telemetry import langchain_telemetry, normalize

TOPOLOGY = "sequential"
TEAM_SIZE = globals().get("TEAM_SIZE")  # preset by the teamsizes/ variants (core.variant)
TEAM = teams.spec(TOPOLOGY, task.DATASET, TEAM_SIZE)
DEFAULT_PREDICTIONS = (
    RESULTS_DIR
    / ("lcb_sequential_langgraph" if TEAM_SIZE is None else f"lcb_sequential_r{TEAM_SIZE}")
    / "predictions.jsonl"
)

VLLM_BASE_URL = settings.base_url()
MODEL_ID = settings.model_id()
COMMUNICATION_FORMAT = globals().get("COMMUNICATION_FORMAT", "freeform")  # preset by the communications/ entries
COMMUNICATION = CommPolicy(COMMUNICATION_FORMAT, task.DATASET, TOPOLOGY)

python_exec = tool(code_tasks.make_python_exec(code_tasks.PYTHON_EXEC_DOC))
TOOLS = {"python_exec": python_exec}


def _load_prompt(role: str) -> str:
    return COMMUNICATION.system_prompt(prompts.role_prompt(TOPOLOGY, task.DATASET, role))


def _build_llm() -> ChatOpenAI:
    return chat_openai(model=MODEL_ID, base_url=VLLM_BASE_URL)


def _merge_dict(a: dict | None, b: dict | None) -> dict:
    out = dict(a or {})
    out.update(b or {})
    return out


class SequentialState(TypedDict, total=False):
    inputs: dict
    by_stage: Annotated[dict, _merge_dict]
    messages: Annotated[list, operator.add]


def _stage_handoff(role: str, text: str) -> str:
    """A prior stage's output as the next stage receives it (the communication format's report)."""
    return COMMUNICATION.handoff(
        role,
        text,
        next_action="Use this prior-stage report as the only handoff context for your stage.",
        payload={"handoff": "prior_stage"},
    )


def _format_user(template: str, inputs: dict, by_stage: dict, prior_roles: list[str]) -> str:
    body = template.format(**inputs)
    for role in prior_roles:
        body += f"\n\n--- PRIOR STAGE: {role} ---\n{_stage_handoff(role, by_stage.get(role, ''))}"
    return body


def _make_tool_node(role, sys_prompt, tools, llm, template, prior_roles):
    agent = create_react_agent(model=llm, tools=tools, prompt=sys_prompt)

    def node(state: SequentialState) -> dict:
        user = _format_user(template, state["inputs"], state.get("by_stage") or {}, prior_roles)
        res = agent.invoke({"messages": [("user", user)]}, config={"recursion_limit": TEAM.recursion_limit})
        raw = next(
            (
                m.content
                for m in reversed(res["messages"])
                if getattr(m, "type", None) == "ai" and getattr(m, "content", "")
            ),
            "",
        )
        ai_msgs = [m for m in res["messages"] if getattr(m, "type", None) == "ai"]
        return {"by_stage": {role: raw}, "messages": ai_msgs}

    return node


def _make_plain_node(role, sys_prompt, llm, template, prior_roles):
    def node(state: SequentialState) -> dict:
        user = _format_user(template, state["inputs"], state.get("by_stage") or {}, prior_roles)
        ai = llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=user)])
        return {"by_stage": {role: ai.content or ""}, "messages": [ai]}

    return node


def _build_graph(llm: ChatOpenAI):
    """Compile the team's stage pipeline; returns ``(graph, roles)``."""
    graph = StateGraph(SequentialState)
    roles: list[str] = []
    for stage in TEAM.stages:
        sys_prompt = _load_prompt(stage.role)
        tools = [TOOLS[name] for name in stage.tools]
        if tools:
            node = _make_tool_node(stage.role, sys_prompt, tools, llm, stage.task, list(roles))
        else:
            node = _make_plain_node(stage.role, sys_prompt, llm, stage.task, list(roles))
        graph.add_node(stage.role, node)
        roles.append(stage.role)
    graph.add_edge(START, roles[0])
    for a, b in zip(roles, roles[1:]):
        graph.add_edge(a, b)
    graph.add_edge(roles[-1], END)
    return graph.compile(), roles


def solve(problem: str, starter_code: str | None = None) -> dict:
    """Run the pipeline on one problem.

    Returns ``{"code", "raw", "by_stage", "telemetry"}``: the debugger's program
    (the last stage's when the debugger wrote nothing, else the coder's), the last
    stage's text, every stage's text and token/call counts.
    """
    compiled, roles = _build_graph(_build_llm())
    result = compiled.invoke(
        {"inputs": {"problem_prompt": format_prompt(problem, starter_code)}, "by_stage": {}, "messages": []}
    )
    stages = result.get("by_stage") or {}
    final = stages.get(roles[-1], "")
    code = extract_code(stages.get("debugger", "") or final)
    if code is None:
        code = extract_code(stages.get("coder", ""))
    return {
        "code": code,
        "raw": final,
        "by_stage": stages,
        "telemetry": normalize(langchain_telemetry(result.get("messages") or [])),
    }


def run_batch(
    instances: list[dict],
    out_path: Path | None = None,
    verbose: bool = True,
    per_test_timeout_s: int = task.BATCH_TEST_TIMEOUT_S,
) -> dict:
    """Solve every instance and score it on its tests."""

    def row(_, inst: dict) -> dict:
        out, latency_s, error = attempt(
            lambda: solve(inst["problem"], starter_code=inst.get("starter_code") or None), fallback={"code": None}
        )
        code = out["code"]
        return task.record(
            inst,
            code,
            task.test_scores(code, inst["tests"], per_test_timeout_s),
            by_stage=out.get("by_stage") or {},
            latency_s=round(latency_s, 2),
            **(out.get("telemetry") or {}),
            error=error,
        )

    return task.run_batch(instances, row, out_path=out_path, verbose=verbose, label="sequential/LCB")


def _canned_demo() -> None:
    _, problem, _, tests = task.DEMOS[0]
    out = solve(problem)
    for role, text in out["by_stage"].items():
        print(f"\n=== {role.capitalize()} (excerpt) ===\n{text[:400]}...")
    task.print_demo_code(out["code"], tests)


def main(argv: list[str] | None = None) -> int:
    return cli.main(
        argv,
        description="Sequential-topology LCB runner (LangGraph 4-stage).",
        load_instances=load_instances,
        run_batch=run_batch,
        demo=_canned_demo,
        source=task.SOURCE,
        predictions=DEFAULT_PREDICTIONS,
        add_arguments=task.add_platform_arguments,
    )


if __name__ == "__main__":
    raise SystemExit(main())
