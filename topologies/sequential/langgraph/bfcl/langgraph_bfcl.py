"""Sequential BFCL runner (LangGraph): a pipeline of stages, each seeing all earlier outputs.

The stages are the team spec (``configs/teams/bfcl.yaml``): at r=4 analyzer ->
inspector -> caller -> verifier, each one model call. The prediction is the
verifier's fenced canonical JSON (else the caller's). ``teamsizes/sequential/bfcl``
runs this module with r = 2, 4, 8 and 10.
"""

from __future__ import annotations

import operator
from pathlib import Path
from typing import Annotated

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from core import prompts, settings, teams
from core.communication import CommPolicy
from core.llm import chat_openai
from core.paths import RESULTS_DIR
from core.tasks import bfcl as task
from core.tasks.bfcl import AST_CATEGORIES, HF_DATASET, extract_canonical, load_instances  # noqa: F401
from core.telemetry import langchain_telemetry, normalize

TOPOLOGY = "sequential"
TEAM_SIZE = globals().get("TEAM_SIZE")  # preset by the teamsizes/ variants (core.variant)
TEAM = teams.spec(TOPOLOGY, task.DATASET, TEAM_SIZE)
DEFAULT_OUT_DIR = RESULTS_DIR / ("bfcl_sequential_langgraph" if TEAM_SIZE is None else f"bfcl_sequential_r{TEAM_SIZE}")

VLLM_BASE_URL = settings.base_url()
MODEL_ID = settings.model_id()
COMMUNICATION_FORMAT = globals().get("COMMUNICATION_FORMAT", "freeform")  # preset by the communications/ entries
COMMUNICATION = CommPolicy(COMMUNICATION_FORMAT, task.DATASET, TOPOLOGY)

_register_model_with_bfcl = task.register_model  # called again by callers that repoint MODEL_ID
_register_model_with_bfcl(MODEL_ID)


def _load_prompt(role: str) -> str:
    return COMMUNICATION.system_prompt(prompts.role_prompt(TOPOLOGY, task.DATASET, role))


def _build_llm() -> ChatOpenAI:
    return chat_openai(model=MODEL_ID, base_url=VLLM_BASE_URL)


def score_one(function_schemas: list[dict], model_output: list[dict], ground_truth: list[dict], category: str) -> dict:
    return task.score_one(function_schemas, model_output, ground_truth, category, MODEL_ID)


def _merge_dict(a: dict | None, b: dict | None) -> dict:
    out = dict(a or {})
    out.update(b or {})
    return out


class SequentialState(TypedDict, total=False):
    inputs: dict
    by_stage: Annotated[dict, _merge_dict]
    messages: Annotated[list, operator.add]


def _format_stage_handoff(role: str, text: str) -> str:
    return COMMUNICATION.handoff(
        role,
        text,
        next_action="Use this prior-stage report as the only handoff context for your stage.",
        payload={"handoff": "prior_stage"},
    )


def _format_user(template: str, inputs: dict, by_stage: dict, prior_roles: list[str]) -> str:
    body = template.format(**inputs)
    for role in prior_roles:
        body += f"\n\n--- PRIOR STAGE: {role} ---\n{_format_stage_handoff(role, by_stage.get(role, ''))}"
    return body


def _make_node(role: str, sys_prompt: str, llm: ChatOpenAI, template: str, prior_roles: list[str]):
    """A stage: one model call on the stage task plus the earlier stages' outputs."""

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
        graph.add_node(stage.role, _make_node(stage.role, _load_prompt(stage.role), llm, stage.task, list(roles)))
        roles.append(stage.role)
    graph.add_edge(START, roles[0])
    for a, b in zip(roles, roles[1:]):
        graph.add_edge(a, b)
    graph.add_edge(roles[-1], END)
    return graph.compile(), roles


def solve(instance: dict) -> dict:
    """Run the pipeline on one instance.

    Returns ``{"model_output", "by_stage", "raw", "telemetry"}``: the canonical
    calls (or []), every stage's text, the last stage's text and token/call counts.
    """
    compiled, roles = _build_graph(_build_llm())
    result = compiled.invoke({"inputs": task.stage_inputs(instance), "by_stage": {}, "messages": []})
    stages_out = result.get("by_stage") or {}
    final = stages_out.get(roles[-1], "")
    model_output = extract_canonical(stages_out.get("verifier", "") or final)
    if model_output is None:
        model_output = extract_canonical(stages_out.get("caller", ""))
    return {
        "model_output": model_output or [],
        "by_stage": stages_out,
        "raw": final,
        "telemetry": normalize(langchain_telemetry(result.get("messages") or [])),
    }


def run_one(instance: dict, ground_truth: dict, category: str, out_dir: Path) -> dict:
    """Solve and score one instance and write its stage outputs to ``out_dir/traces/<id>.txt``."""
    summary: dict = {"id": instance["id"], "category": category}
    try:
        out = solve(instance)
    except Exception as e:
        return task.solve_failed(summary, e)
    summary["model_output"] = out.get("model_output") or []
    summary["tool_calls"] = len(summary["model_output"])
    summary.update(out.get("telemetry") or {})
    stages = (out.get("by_stage") or {}).items()
    task.write_trace(out_dir, instance["id"], task.sections_trace((role.upper(), text) for role, text in stages))
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
        run_one, category, limit, offset, only, out_dir or DEFAULT_OUT_DIR, model_id=MODEL_ID, verbose=verbose
    )


def main(argv: list[str] | None = None) -> int:
    return task.cli_main(
        argv,
        description=f"Sequential-topology BFCL runner (LangGraph {len(TEAM.stages)}-stage).",
        run_one=run_one,
        model_id=MODEL_ID,
        default_out_dir=DEFAULT_OUT_DIR,
    )


if __name__ == "__main__":
    raise SystemExit(main())
