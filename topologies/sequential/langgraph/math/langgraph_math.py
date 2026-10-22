"""Sequential MATH runner (LangGraph): a pipeline of stages, each seeing all earlier outputs.

The stages are the team spec (``configs/teams/<dataset>.yaml``): at r=4 decomposer ->
computer -> checker -> verifier. A stage with tools is a calculator ReAct agent,
one without is a single model call; the answer is the last stage's boxed answer.
``teamsizes/sequential/math`` runs this module with r = 2, 4, 8 and 10.
"""

from __future__ import annotations

import hashlib
import operator
from pathlib import Path
from typing import Annotated

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent
from typing_extensions import TypedDict

from core import cli, prompts, settings, teams
from core.batch import attempt, write_trace
from core.calculator import CALCULATOR_DOC_DECIMAL_PI, make_calculator
from core.llm import chat_openai
from core.paths import RESULTS_DIR
from core.tasks import math as task
from core.tasks.math import exact_match_score, extract_answer, extract_boxed, is_equiv, load_instances  # noqa: F401
from core.telemetry import langchain_telemetry, normalize

TOPOLOGY = "sequential"
TEAM_SIZE = globals().get("TEAM_SIZE")  # preset by the teamsizes/ variants (core.variant)
TEAM = teams.spec(TOPOLOGY, task.DATASET, TEAM_SIZE)
DEFAULT_OUT_DIR = RESULTS_DIR / ("math_sequential_langgraph" if TEAM_SIZE is None else f"math_sequential_r{TEAM_SIZE}")

VLLM_BASE_URL = settings.base_url()
MODEL_ID = settings.model_id()

# Telemetry fields a batch record carries (the LLM call count is not among them).
_RECORD_TELEMETRY = ("prompt_tokens", "completion_tokens", "total_tokens", "n_tool_calls")

calculator = tool(make_calculator(CALCULATOR_DOC_DECIMAL_PI))
TOOLS = {"calculator": calculator}


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


def solve(problem: str) -> dict:
    """Run the pipeline on one problem.

    Returns ``{"answer", "raw", "by_stage", "telemetry"}``: the last stage's
    boxed answer and text, every stage's text and token/call counts.
    """
    compiled, roles = _build_graph(_build_llm())
    result = compiled.invoke({"inputs": {"problem": problem}, "by_stage": {}, "messages": []})
    stages_out = result.get("by_stage") or {}
    final = stages_out.get(roles[-1], "")
    return {
        "answer": extract_answer(final),
        "raw": final,
        "by_stage": stages_out,
        "telemetry": normalize(langchain_telemetry(result.get("messages") or [])),
    }


def run_one(instance: dict, out_dir: Path) -> dict:
    """Solve and score one instance and write its stage trace to ``out_dir/traces/<idx>.txt``."""
    rid = instance["id"]
    summary: dict = {"id": rid, **task.meta(instance)}
    idx = instance.get("idx")
    if idx is None:
        idx = int(hashlib.md5(rid.encode("utf-8")).hexdigest(), 16) % 10000

    out, latency_s, error = attempt(lambda: solve(instance["problem"]))
    pred = out.get("answer")
    summary["gold_answer"] = instance["answer"]
    summary["predicted_answer"] = pred
    summary["em"] = task.score(pred, instance["answer"])
    summary["latency_s"] = round(latency_s, 2)
    summary["error"] = error
    summary.update(out.get("telemetry") or {})
    by_stage = out.get("by_stage") or {}
    write_trace(out_dir / "traces" / f"{idx:04d}.txt", ((stage.upper(), text) for stage, text in by_stage.items()))
    summary["by_stage"] = by_stage
    summary["raw"] = out.get("raw") or ""
    return summary


def run_batch(
    instances: list[dict],
    out_path: Path | None = None,
    out_dir: Path | None = None,
    verbose: bool = True,
) -> dict:
    """Score every instance; traces go to ``out_dir`` (default :data:`DEFAULT_OUT_DIR`)."""
    out_dir = Path(out_dir or DEFAULT_OUT_DIR).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    def row(i: int, inst: dict) -> dict:
        summary = run_one({**inst, "idx": inst.get("idx", i)}, out_dir)
        rec = task.record(
            inst,
            summary["predicted_answer"],
            **task.meta(inst),
            by_stage=summary["by_stage"],
            raw=summary["raw"],
            latency_s=summary["latency_s"],
            error=summary["error"],
        )
        rec.update((k, summary[k]) for k in _RECORD_TELEMETRY if k in summary)
        return rec

    return task.run_batch(instances, row, out_path=out_path, verbose=verbose, label=f"sequential/{task.LABEL}")


def _canned_demo() -> None:
    out = solve(task.DEMO_PROBLEM)
    for role, text in out["by_stage"].items():
        print(f"\n=== {role.capitalize()} (excerpt) ===\n{text[:400]}...")
    task.print_demo_answer(out["answer"])


def main(argv: list[str] | None = None) -> int:
    return cli.main(
        argv,
        description="Sequential-topology MATH runner (LangGraph 4-stage).",
        load_instances=load_instances,
        run_batch=run_batch,
        demo=_canned_demo,
        source=task.SOURCE,
    )


if __name__ == "__main__":
    raise SystemExit(main())
