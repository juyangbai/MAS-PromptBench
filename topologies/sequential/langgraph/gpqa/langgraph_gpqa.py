"""Sequential GPQA-Diamond runner (LangGraph): a pipeline of stages, each seeing all earlier outputs.

The stages are the team spec (``configs/teams/gpqa.yaml``): at r=4 analyzer ->
solver -> critic -> verifier. Every stage is a ReAct agent with its spec tools;
the answer is the letter extracted from the last stage's output.
``teamsizes/sequential/gpqa`` runs this module with r = 2, 4, 8 and 10.
"""

from __future__ import annotations

import operator
from pathlib import Path
from typing import Annotated

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent
from typing_extensions import TypedDict

from core import cli, prompts, settings, teams
from core.batch import attempt
from core.calculator import CALCULATOR_DOC_DECIMAL_PI, make_calculator
from core.llm import chat_openai
from core.paths import RESULTS_DIR
from core.tasks import gpqa as task
from core.tasks.gpqa import extract_answer, load_instances
from core.telemetry import langchain_telemetry, normalize

TOPOLOGY = "sequential"
TEAM_SIZE = globals().get("TEAM_SIZE")  # preset by the teamsizes/ variants (core.variant)
TEAM = teams.spec(TOPOLOGY, task.DATASET, TEAM_SIZE)
DEFAULT_PREDICTIONS = (
    RESULTS_DIR
    / ("gpqa_sequential_langgraph" if TEAM_SIZE is None else f"gpqa_sequential_r{TEAM_SIZE}")
    / "predictions.jsonl"
)

VLLM_BASE_URL = settings.base_url()
MODEL_ID = settings.model_id()

calculator = tool(make_calculator(CALCULATOR_DOC_DECIMAL_PI))
TOOLS = {"calculator": calculator}

format_mcq = task.format_prompt


def _load_prompt(role: str) -> str:
    return prompts.role_prompt(TOPOLOGY, task.DATASET, role)


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


def _format_user(template: str, inputs: dict, by_stage: dict, prior_roles: list[str]) -> str:
    body = template.format(**inputs)
    for role in prior_roles:
        body += f"\n\n--- PRIOR STAGE: {role} ---\n{by_stage.get(role, '')}"
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


def _build_graph(llm: ChatOpenAI):
    """Compile the team's stage pipeline; returns ``(graph, roles)``."""
    graph = StateGraph(SequentialState)
    roles: list[str] = []
    for stage in TEAM.stages:
        tools = [TOOLS[name] for name in stage.tools]
        node = _make_tool_node(stage.role, _load_prompt(stage.role), tools, llm, stage.task, list(roles))
        graph.add_node(stage.role, node)
        roles.append(stage.role)
    graph.add_edge(START, roles[0])
    for a, b in zip(roles, roles[1:]):
        graph.add_edge(a, b)
    graph.add_edge(roles[-1], END)
    return graph.compile(), roles


def solve(question: str, choices: list[str]) -> dict:
    """Run the pipeline on one question.

    Returns ``{"answer", "raw", "by_stage", "telemetry"}``: the last stage's
    letter and text, every stage's text and token/call counts.
    """
    compiled, roles = _build_graph(_build_llm())
    result = compiled.invoke({"inputs": {"question": format_mcq(question, choices)}, "by_stage": {}, "messages": []})
    stages_out = result.get("by_stage") or {}
    final = stages_out.get(roles[-1], "")
    return {
        "answer": extract_answer(final),
        "raw": final,
        "by_stage": stages_out,
        "telemetry": normalize(langchain_telemetry(result.get("messages") or [])),
    }


def run_batch(instances: list[dict], out_path: Path | None = None, verbose: bool = True) -> dict:
    """Solve and score every instance."""

    def row(_, inst: dict) -> dict:
        out, latency_s, error = attempt(lambda: solve(inst["question"], inst["choices"]))
        return task.record(
            inst,
            out["answer"],
            raw=out.get("raw") or "",
            by_stage=task.stage_excerpts(out.get("by_stage") or {}),
            latency_s=round(latency_s, 2),
            **(out.get("telemetry") or {}),
            error=error,
        )

    return task.run_batch(instances, row, out_path=out_path, verbose=verbose, label="sequential/GPQA-Diamond")


def _canned_demo() -> None:
    out = solve(task.DEMO_QUESTION, task.DEMO_CHOICES)
    for role, text in out["by_stage"].items():
        print(f"\n=== {role.capitalize()} (excerpt) ===\n{text[:400]}...")
    task.print_demo_answer(out["answer"])


def main(argv: list[str] | None = None) -> int:
    return cli.main(
        argv,
        description="Sequential-topology GPQA-Diamond runner (LangGraph).",
        load_instances=load_instances,
        run_batch=run_batch,
        demo=_canned_demo,
        source=task.SOURCE,
        predictions=DEFAULT_PREDICTIONS,
        add_arguments=task.add_arguments,
    )


if __name__ == "__main__":
    raise SystemExit(main())
