"""Sequential SWE-bench Verified runner (LangGraph): a pipeline of stages on one clone.

The stages are the team spec (``configs/teams/swe.yaml``): at r=4 investigator ->
planner -> patcher -> tester. Each stage sees the issue brief and all earlier
stage outputs; a stage with tools is a ReAct agent on the instance clone, one
without is a single model call. The patch is ``git diff HEAD`` of the clone.
``teamsizes/sequential/swe`` runs this module with r = 2, 4, 8 and 10.
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

from core import prompts, settings, teams
from core.communication import CommPolicy
from core.llm import chat_openai
from core.tasks import swe as task
from core.tasks.swe import (  # noqa: F401  (runner API)
    clone_and_checkout,
    is_resolved,
    load_instances,
    run_tests_singularity,
)
from core.telemetry import langchain_telemetry, normalize
from core.thinking import strip_thinking  # noqa: F401  (runner API)

TOPOLOGY = "sequential"
TEAM_SIZE = globals().get("TEAM_SIZE")  # preset by the teamsizes/ variants (core.variant)
TEAM = teams.spec(TOPOLOGY, task.DATASET, TEAM_SIZE)
MODEL_NAME = "mas-promptbench-sequential"
DEFAULT_WORKDIR_ROOT, DEFAULT_OUT_DIR = task.default_dirs(
    "sequential_langgraph" if TEAM_SIZE is None else f"sequential_r{TEAM_SIZE}"
)
# Stage outputs a solve() returns and traces, whatever the team (absent stages are "").
_REPORTED_STAGES = ("investigator", "planner", "patcher", "tester")

VLLM_BASE_URL = settings.base_url()
MODEL_ID = settings.model_id()
COMMUNICATION_FORMAT = globals().get("COMMUNICATION_FORMAT", "freeform")  # preset by the communications/ entries
COMMUNICATION = CommPolicy(COMMUNICATION_FORMAT, task.DATASET, TOPOLOGY)

WORKDIR = task.Workdir(task.env_repo_dir())
TOOLS = {
    "file_read": tool(task.make_file_read(WORKDIR, task.FILE_READ_DOC)),
    "str_replace": tool(
        task.make_str_replace(WORKDIR, task.STR_REPLACE_DOC_TARGETED, not_found=task.NOT_FOUND_READ_FIRST, preview=True)
    ),
    "list_dir": tool(task.make_list_dir(WORKDIR, task.LIST_DIR_DOC)),
    "search_repo": tool(task.make_search_repo(WORKDIR, task.SEARCH_REPO_DOC)),
    "shell_exec": tool(task.make_shell_exec(WORKDIR, task.SHELL_EXEC_DOC_TESTER)),
}

_ensure_sif = task.ensure_sif


def _load_prompt(role: str) -> str:
    return COMMUNICATION.system_prompt(prompts.role_prompt(TOPOLOGY, task.DATASET, role))


def _build_llm() -> ChatOpenAI:
    return chat_openai(model=MODEL_ID, base_url=VLLM_BASE_URL)


def _set_repo_dir(path: Path | str) -> None:
    """Bind the checkout the tools act on (in this context)."""
    WORKDIR.bind(Path(path).resolve())


def format_task_brief(problem_statement: str, instance_id: str | None = None, hints_text: str | None = None) -> str:
    """The issue brief every stage starts from, at the bound checkout."""
    return task.issue_brief(
        problem_statement,
        instance_id,
        hints_text,
        checkout=task.checked_out_at(WORKDIR.get()),
        note=task.NO_TESTS_NOTE_ASCII,
    )


def compute_patch() -> str:
    """``git diff HEAD`` of the bound checkout."""
    return task.compute_patch(WORKDIR.get())


def predictions_entry(instance_id: str, patch: str, model_name: str = MODEL_NAME) -> dict:
    return task.predictions_entry(instance_id, patch, model_name)


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


def solve(instance: dict, eval_mode: str = "singularity") -> dict:
    """Run the pipeline on the instance checked out at the bound checkout.

    Returns ``{"patch", "resolved", "report", "by_stage", "telemetry"}``: the
    clone's patch, its evaluation (``resolved`` None and no report with
    ``eval_mode='none'``; False without a patch), the outputs of the
    investigator, planner, patcher and tester stages, and token/call counts.
    """
    brief = format_task_brief(
        instance["problem_statement"], instance_id=instance.get("instance_id"), hints_text=instance.get("hints_text")
    )
    compiled, _ = _build_graph(_build_llm())
    result = compiled.invoke({"inputs": {"task_brief": brief}, "by_stage": {}, "messages": []})
    stages_out = result.get("by_stage") or {}
    out = {
        "patch": compute_patch(),
        "resolved": None if eval_mode == "none" else False,
        "report": None,
        "by_stage": {role: stages_out.get(role, "") for role in _REPORTED_STAGES},
        "telemetry": normalize(langchain_telemetry(result.get("messages") or [])),
    }
    if eval_mode != "none" and out["patch"]:
        f2p, p2p = task.instance_tests(instance)
        out["report"] = run_tests_singularity(instance, out["patch"], f2p, p2p)
        out["resolved"] = is_resolved(out["report"])
    return out


def run_one(instance: dict, workdir_root: Path, out_dir: Path, eval_mode: str = "singularity") -> dict:
    """Clone, solve and score one instance; writes its patch, prediction and stage trace under ``out_dir``."""
    summary, out = task.solve_in_checkout(
        instance, workdir_root, _set_repo_dir, lambda: solve(instance, eval_mode=eval_mode)
    )
    if out is None:
        return summary
    summary.update(out.get("telemetry") or {})
    patch = out["patch"] or ""
    summary["patch_chars"] = len(patch)
    iid = instance["instance_id"]
    trace = task.sections((stage.upper(), text) for stage, text in (out.get("by_stage") or {}).items())
    task.write_artifacts(out_dir, iid, patch, predictions_entry(iid, patch), trace)
    return {**summary, **task.eval_fields(eval_mode, out.get("report"))}


def _run_instances(
    instances: list[dict],
    out_dir: Path | None = None,
    workdir_root: Path | None = None,
    eval_mode: str = "singularity",
    keep_workdirs: bool = False,
    out_path: Path | None = None,
) -> None:
    """Solve and score loaded instances (the command line's batch)."""
    workdir_root = workdir_root or DEFAULT_WORKDIR_ROOT
    out_dir = out_dir or DEFAULT_OUT_DIR
    task.run_batch(
        instances,
        lambda inst: run_one(inst, workdir_root, out_dir, eval_mode=eval_mode),
        out_dir=out_dir,
        eval_mode=eval_mode,
        workdirs=None if keep_workdirs else lambda inst: [workdir_root / inst["instance_id"]],
        predictions=out_path,
    )


def run_batch(
    subset: str = "test",
    limit: int | None = None,
    offset: int = 0,
    only: list[str] | None = None,
    workdir_root: Path | None = None,
    out_dir: Path | None = None,
    eval_mode: str = "singularity",
    keep_workdirs: bool = False,
) -> None:
    """Solve and score a Verified slice (``eval_mode``: ``singularity`` or ``none``)."""
    instances = load_instances(subset, limit, offset, only)
    task.log_loaded(instances)
    _run_instances(instances, out_dir, workdir_root, eval_mode, keep_workdirs)


def main(argv: list[str] | None = None) -> int:
    return task.cli_main(
        argv,
        description="Sequential-topology SWE-bench Verified agent (LangGraph).",
        run_batch=_run_instances,
        default_out_dir=DEFAULT_OUT_DIR,
        eval_modes=("singularity", "none"),
    )


if __name__ == "__main__":
    raise SystemExit(main())
