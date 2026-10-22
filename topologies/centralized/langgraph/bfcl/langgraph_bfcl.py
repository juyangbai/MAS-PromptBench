"""Centralized BFCL runner (LangGraph): a manager delegating to tool-less workers.

The team is the team spec (``configs/teams/bfcl.yaml``): at r=4 a manager with
inspector, caller and validator workers. The manager hands a turn to a worker by
calling its ``delegate_to_<worker>`` tool (its only tools); every worker reports
back to the manager, which ends the run with ``TERMINATE`` or after the spec's
``max_turns``. The prediction is the manager's last fenced canonical JSON (else
the caller's, else the latest in the chat). ``teamsizes/centralized/bfcl`` runs
this module with r = 2, 4, 8 and 10.
"""

from __future__ import annotations

import time
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
from core.paths import RESULTS_DIR
from core.tasks import bfcl as task
from core.tasks.bfcl import AST_CATEGORIES, HF_DATASET, extract_canonical, format_task, load_instances  # noqa: F401
from core.telemetry import langchain_telemetry, normalize

TOPOLOGY = "centralized"
TEAM_SIZE = globals().get("TEAM_SIZE")  # preset by the teamsizes/ variants (core.variant)
TEAM = teams.spec(TOPOLOGY, task.DATASET, TEAM_SIZE)
DEFAULT_OUT_DIR = RESULTS_DIR / (
    "bfcl_centralized_langgraph" if TEAM_SIZE is None else f"bfcl_centralized_r{TEAM_SIZE}"
)

VLLM_BASE_URL = settings.base_url()
MODEL_ID = settings.model_id()
COMMUNICATION_FORMAT = globals().get("COMMUNICATION_FORMAT", "freeform")  # preset by the communications/ entries
COMMUNICATION = CommPolicy(COMMUNICATION_FORMAT, task.DATASET, TOPOLOGY)
MAX_TURNS = TEAM.max_turns

_register_model_with_bfcl = task.register_model  # called again by callers that repoint MODEL_ID
_register_model_with_bfcl(MODEL_ID)


def _format_interagent_handoff(role: str, text: str, *, target: str) -> str:
    return COMMUNICATION.handoff(
        role,
        text,
        next_action=f"Use this report as the handoff context for {target}.",
        payload={"handoff": "manager_worker" if role == "manager" else "worker_manager", "target": target},
    )


def _load_prompt(role: str) -> str:
    return COMMUNICATION.system_prompt(prompts.role_prompt(TOPOLOGY, task.DATASET, role))


def _build_llm() -> ChatOpenAI:
    return chat_openai(model=MODEL_ID, base_url=VLLM_BASE_URL)


def score_one(function_schemas: list[dict], model_output: list[dict], ground_truth: list[dict], category: str) -> dict:
    return task.score_one(function_schemas, model_output, ground_truth, category, MODEL_ID)


def _delegate_tool(worker: teams.Worker):
    """The manager's ``delegate_to_<worker>`` tool: it hands the instructions to the worker as a handoff."""

    def delegate(instructions: str) -> str:
        return _format_interagent_handoff("manager", instructions, target=worker.role)

    delegate.__doc__ = worker.delegate
    return tool(f"delegate_to_{worker.role}")(delegate)


DELEGATION_TOOLS = [_delegate_tool(worker) for worker in TEAM.workers]
globals().update((t.name, t) for t in DELEGATION_TOOLS)  # each tool is also an attribute: delegate_to_<worker>
DELEGATION_NAMES = {t.name for t in DELEGATION_TOOLS}
MANAGER_TOOLS = DELEGATION_TOOLS  # the manager has no other tools
_manager_tool_node = ToolNode(MANAGER_TOOLS)

_MANAGER_TERMINATE_NUDGE = task.TERMINATE_NUDGE + "\n\n" + TEAM.delegation_note


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


def _last_content_ai(messages: list) -> AIMessage | None:
    """The last AIMessage with content and no tool calls, else the last with content."""
    for m in reversed(messages):
        if isinstance(m, AIMessage) and (m.content or "") and not getattr(m, "tool_calls", None):
            return m
    for m in reversed(messages):
        if isinstance(m, AIMessage) and (m.content or ""):
            return m
    return None


def _make_worker_node(name: str, tools: list, llm: ChatOpenAI):
    agent = create_react_agent(model=llm, tools=tools, prompt=_load_prompt(name))

    def node(state: CentralizedState) -> dict:
        prior = list(state["messages"])
        result = agent.invoke({"messages": prior}, config={"recursion_limit": TEAM.recursion_limit})
        new_msgs = result["messages"][len(prior) :]
        for m in new_msgs:
            if isinstance(m, AIMessage):
                _tag_source(m, name)
        if COMMUNICATION_FORMAT != "freeform":
            final = _last_content_ai(new_msgs)
            text = getattr(final, "content", "") if final else ""
            handoff = AIMessage(content=_format_interagent_handoff(name, text, target="manager"))
            _tag_source(handoff, name)
            return {"messages": [handoff], "turn_count": int(state.get("turn_count", 0)) + 1}
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
    for name in workers:
        graph.add_node(name, _make_worker_node(name, [], llm))  # BFCL workers have no tools
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


def _fallback_output(rendered: list[dict]) -> list[dict] | None:
    """Canonical calls of the caller's last message, else of the latest message that has any."""
    caller_msgs = [r for r in rendered if r["source"] == "caller_worker"]
    if caller_msgs:
        model_output = extract_canonical(caller_msgs[-1]["content"])
        if model_output is not None:
            return model_output
    for r in reversed(rendered):
        model_output = extract_canonical(r["content"] or "")
        if model_output is not None:
            return model_output
    return None


def solve(instance: dict) -> dict:
    """Run the team on one instance.

    Returns ``{"model_output", "raw", "messages", "telemetry"}``: the canonical
    calls (or []), the manager's last message, every turn as ``{source,
    content}`` and token/call counts.
    """
    compiled, _ = _build_graph()
    prompt = format_task(task.flatten_question(instance["question"]), task.render_schemas(instance["function"]))
    result = compiled.invoke(
        {"messages": [HumanMessage(content=prompt)], "turn_count": 0},
        config={"recursion_limit": MAX_TURNS * 4},
    )
    msgs = result.get("messages") or []
    rendered = [_communications_to_record(m) for m in msgs]
    manager_msgs = [r for r in rendered if r["source"] == "manager"]
    final = manager_msgs[-1]["content"] if manager_msgs else ""
    model_output = extract_canonical(final)
    if model_output is None:
        model_output = _fallback_output(rendered)
    return {
        "model_output": model_output or [],
        "raw": final,
        "messages": rendered,
        "telemetry": normalize(langchain_telemetry(msgs)),
    }


def run_one(instance: dict, ground_truth: dict, category: str, out_dir: Path) -> dict:
    """Solve and score one instance and write the group chat to ``out_dir/traces/<id>.txt``."""
    summary: dict = {"id": instance["id"], "category": category}
    t0 = time.time()
    try:
        out = solve(instance)
    except Exception as e:
        task.solve_failed(summary, e)
        summary["solve_s"] = round(time.time() - t0, 1)
        return summary
    summary["solve_s"] = round(time.time() - t0, 1)
    summary["model_output"] = out.get("model_output") or []
    summary["n_messages"] = len(out.get("messages") or [])
    summary["tool_calls"] = len(summary["model_output"])
    summary.update(out.get("telemetry") or {})
    task.write_trace(out_dir, instance["id"], task.messages_trace(out.get("messages") or []))
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
        description="Centralized-topology BFCL runner (LangGraph manager/worker).",
        run_one=run_one,
        model_id=MODEL_ID,
        default_out_dir=DEFAULT_OUT_DIR,
    )


if __name__ == "__main__":
    raise SystemExit(main())
