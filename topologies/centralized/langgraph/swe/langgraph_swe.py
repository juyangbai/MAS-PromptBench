"""Centralized SWE-bench Verified runner (LangGraph): a manager delegating to tool-using workers.

The team is the team spec (``configs/teams/swe.yaml``): at r=4 a manager with
navigator, patcher and tester workers, each with its own tools on one clone of
the instance repository. The manager hands a turn to a worker by calling its
``delegate_to_<worker>`` tool; every worker reports back to the manager, which
ends the run with ``TERMINATE`` or after the spec's ``max_turns``. The patch is
``git diff HEAD`` of the clone. ``teamsizes/centralized/swe`` runs this module
with r = 2, 4, 8 and 10.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, create_react_agent
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

TOPOLOGY = "centralized"
TEAM_SIZE = globals().get("TEAM_SIZE")  # preset by the teamsizes/ variants (core.variant)
TEAM = teams.spec(TOPOLOGY, task.DATASET, TEAM_SIZE)
MODEL_NAME = "mas-promptbench-centralized"
DEFAULT_WORKDIR_ROOT, DEFAULT_OUT_DIR = task.default_dirs(
    "centralized_langgraph" if TEAM_SIZE is None else f"centralized_r{TEAM_SIZE}"
)

VLLM_BASE_URL = settings.base_url()
MODEL_ID = settings.model_id()
COMMUNICATION_FORMAT = globals().get("COMMUNICATION_FORMAT", "freeform")  # preset by the communications/ entries
COMMUNICATION = CommPolicy(COMMUNICATION_FORMAT, task.DATASET, TOPOLOGY)
MAX_TURNS = TEAM.max_turns

WORKDIR = task.Workdir(task.env_repo_dir())
TOOLS = {
    "file_read": tool(task.make_file_read(WORKDIR, task.FILE_READ_DOC_SHORT)),
    "str_replace": tool(task.make_str_replace(WORKDIR, task.STR_REPLACE_DOC_NARROW)),
    "list_dir": tool(task.make_list_dir(WORKDIR, task.LIST_DIR_DOC)),
    "search_repo": tool(task.make_search_repo(WORKDIR, task.SEARCH_REPO_DOC, terse=True)),
    "shell_exec": tool(task.make_shell_exec(WORKDIR, task.SHELL_EXEC_DOC)),
}

_ensure_sif = task.ensure_sif


def _delegate_tool(worker: teams.Worker):
    """The manager's ``delegate_to_<worker>`` tool: it echoes the instructions for the worker to read."""

    def delegate(instructions: str) -> str:
        return instructions

    delegate.__doc__ = worker.delegate
    return tool(f"delegate_to_{worker.role}")(delegate)


DELEGATION_TOOLS = [_delegate_tool(worker) for worker in TEAM.workers]
globals().update((t.name, t) for t in DELEGATION_TOOLS)  # each tool is also an attribute: delegate_to_<worker>
DELEGATION_NAMES = {t.name for t in DELEGATION_TOOLS}
MANAGER_TOOLS = [TOOLS[name] for name in TEAM.manager_tools] + DELEGATION_TOOLS
_manager_tool_node = ToolNode(MANAGER_TOOLS)

_MANAGER_TERMINATE_NUDGE = task.TERMINATE_NUDGE + "\n\n" + TEAM.delegation_note


def _load_prompt(role: str) -> str:
    return COMMUNICATION.system_prompt(prompts.role_prompt(TOPOLOGY, task.DATASET, role))


def _build_llm() -> ChatOpenAI:
    return chat_openai(model=MODEL_ID, base_url=VLLM_BASE_URL)


def _set_repo_dir(path: Path | str) -> None:
    """Bind the checkout the tools act on (in this context)."""
    WORKDIR.bind(Path(path).resolve())


def format_task_brief(problem_statement: str, instance_id: str | None = None, hints_text: str | None = None) -> str:
    """The manager's first message, at the bound checkout."""
    return task.issue_brief(
        problem_statement, instance_id, hints_text, checkout=task.checked_out_at(WORKDIR.get()), note=task.NO_TESTS_NOTE
    )


def compute_patch() -> str:
    """``git diff HEAD`` of the bound checkout."""
    return task.compute_patch(WORKDIR.get())


def predictions_entry(instance_id: str, patch: str, model_name: str = MODEL_NAME) -> dict:
    return task.predictions_entry(instance_id, patch, model_name)


class CentralizedState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    turn_count: int


def _tag_source(msg: BaseMessage, source: str) -> None:
    """Record the speaking agent as ``additional_kwargs["source"]`` (AutoGen-style)."""
    try:
        kw = dict(getattr(msg, "additional_kwargs", None) or {})
        kw["source"] = source
        msg.additional_kwargs = kw
    except Exception:
        pass


def _manager_system() -> str:
    return _load_prompt(TEAM.manager) + _MANAGER_TERMINATE_NUDGE


def _manager_node(state: CentralizedState) -> dict:
    llm = _build_llm().bind_tools(MANAGER_TOOLS)
    ai = llm.invoke([SystemMessage(content=_manager_system())] + state["messages"])
    _tag_source(ai, "manager")
    return {"messages": [ai], "turn_count": int(state.get("turn_count", 0)) + 1}


def _route_from_manager(state: CentralizedState) -> str:
    msgs = state["messages"]
    if not msgs:
        return "manager"
    last = msgs[-1]
    if int(state.get("turn_count", 0)) >= MAX_TURNS:
        return END
    if isinstance(last, AIMessage):
        content = last.content or ""
        if isinstance(content, str) and "TERMINATE" in content:
            return END
        if getattr(last, "tool_calls", None):
            return "manager_tools"
    return "manager"


def _route_from_manager_tools(state: CentralizedState) -> str:
    """The delegated worker of the manager's latest tool calls, else back to the manager."""
    for m in reversed(state["messages"]):
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            for tc in m.tool_calls:
                name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
                if name in DELEGATION_NAMES:
                    return name.removeprefix("delegate_to_")
            return "manager"
    return "manager"


def _make_worker_node(name: str, tools: list, llm: ChatOpenAI, extra_prompt: str = ""):
    agent = create_react_agent(model=llm, tools=tools, prompt=_load_prompt(name) + extra_prompt)

    def node(state: CentralizedState) -> dict:
        prior = list(state["messages"])
        result = agent.invoke({"messages": prior}, config={"recursion_limit": TEAM.recursion_limit})
        new_msgs = result["messages"][len(prior) :]
        for m in new_msgs:
            if isinstance(m, AIMessage):
                _tag_source(m, name)
        n_turns = sum(1 for m in new_msgs if isinstance(m, AIMessage))
        return {"messages": new_msgs, "turn_count": int(state.get("turn_count", 0)) + n_turns}

    return node


def _build_graph(llm: ChatOpenAI | None = None):
    """Compile the manager/worker graph; returns ``(graph, roles)``."""
    if llm is None:
        llm = _build_llm()
    workers = [worker.role for worker in TEAM.workers]
    graph = StateGraph(CentralizedState)
    graph.add_node("manager", _manager_node)
    graph.add_node("manager_tools", _manager_tool_node)
    for worker in TEAM.workers:
        tools = [TOOLS[t] for t in worker.tools]
        graph.add_node(worker.role, _make_worker_node(worker.role, tools, llm, worker.prompt_suffix))
    graph.add_edge(START, "manager")
    graph.add_conditional_edges(
        "manager", _route_from_manager, {"manager_tools": "manager_tools", "manager": "manager", END: END}
    )
    graph.add_conditional_edges(
        "manager_tools", _route_from_manager_tools, {**{name: name for name in workers}, "manager": "manager"}
    )
    for name in workers:
        graph.add_edge(name, "manager")
    return graph.compile(), ["manager", *workers]


def _communications_source(m: BaseMessage) -> str:
    src = (getattr(m, "additional_kwargs", None) or {}).get("source")
    if src:
        return src
    t = getattr(m, "type", None)
    return {"human": "user", "ai": "assistant", "tool": "tool"}.get(t, t or "?")


def _communications_to_record(m: BaseMessage) -> dict:
    content = getattr(m, "content", "") or ""
    return {"source": _communications_source(m), "content": content if isinstance(content, str) else str(content)}


def solve(instance: dict, eval_mode: str = "singularity") -> dict:
    """Run the team on the instance checked out at the bound checkout.

    Returns ``{"patch", "resolved", "report", "messages", "telemetry"}``: the
    clone's patch, its evaluation (``resolved`` None and no report with
    ``eval_mode='none'``; False without a patch), every turn as
    ``{source, content}`` and token/call counts.
    """
    compiled, _ = _build_graph()
    brief = format_task_brief(
        instance["problem_statement"], instance_id=instance.get("instance_id"), hints_text=instance.get("hints_text")
    )
    result = compiled.invoke(
        {"messages": [HumanMessage(content=brief)], "turn_count": 0}, config={"recursion_limit": MAX_TURNS * 4}
    )
    msgs = result.get("messages") or []
    out = {
        "patch": compute_patch(),
        "resolved": None if eval_mode == "none" else False,
        "report": None,
        "messages": [_communications_to_record(m) for m in msgs],
        "telemetry": normalize(langchain_telemetry(msgs)),
    }
    if eval_mode != "none" and out["patch"]:
        f2p, p2p = task.instance_tests(instance)
        out["report"] = run_tests_singularity(instance, out["patch"], f2p, p2p)
        out["resolved"] = is_resolved(out["report"])
    return out


def run_one(instance: dict, workdir_root: Path, out_dir: Path, eval_mode: str = "singularity") -> dict:
    """Clone, solve and score one instance; writes its patch, prediction and chat trace under ``out_dir``."""
    summary, out = task.solve_in_checkout(
        instance, workdir_root, _set_repo_dir, lambda: solve(instance, eval_mode=eval_mode)
    )
    if out is None:
        return summary
    summary.update(out.get("telemetry") or {})
    patch = out["patch"] or ""
    messages = out.get("messages") or []
    summary.update(patch_chars=len(patch), n_messages=len(messages))
    iid = instance["instance_id"]
    trace = task.sections((str(m.get("source", "?")).upper(), m.get("content", "")) for m in messages)
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
        description="Centralized-topology SWE-bench Verified agent (LangGraph).",
        run_batch=_run_instances,
        default_out_dir=DEFAULT_OUT_DIR,
        eval_modes=("singularity", "none"),
        skip_eval=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
