"""Centralized HotpotQA runner (LangGraph): a manager delegating to Wikipedia workers.

The team is the team spec (``configs/teams/hotpotqa.yaml``): at r=4 a manager with
retriever, reasoner and writer workers. The manager hands a turn to a worker by
calling its ``delegate_to_<worker>`` tool; every worker reports back to the
manager, which ends the run with ``TERMINATE`` or after the spec's ``max_turns``.
``teamsizes/centralized/hotpotqa`` runs this module with r = 2, 4, 8 and 10.
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

from core import cli, prompts, settings, teams
from core.batch import attempt
from core.communication import CommPolicy
from core.llm import chat_openai
from core.paths import RESULTS_DIR
from core.tasks import hotpotqa as task
from core.tasks.hotpotqa import exact_match_score, f1_score, load_instances, normalize_answer  # noqa: F401
from core.tasks.hotpotqa import extract_manager_answer as extract_answer
from core.telemetry import langchain_telemetry, normalize

TOPOLOGY = "centralized"
TEAM_SIZE = globals().get("TEAM_SIZE")  # preset by the teamsizes/ variants (core.variant)
TEAM = teams.spec(TOPOLOGY, task.DATASET, TEAM_SIZE)
DEFAULT_PREDICTIONS = (
    RESULTS_DIR
    / ("hotpotqa_centralized_langgraph" if TEAM_SIZE is None else f"hotpotqa_centralized_r{TEAM_SIZE}")
    / "predictions.jsonl"
)

VLLM_BASE_URL = settings.base_url()
MODEL_ID = settings.model_id()
COMMUNICATION_FORMAT = globals().get("COMMUNICATION_FORMAT", "freeform")  # preset by the communications/ entries
COMMUNICATION = CommPolicy(COMMUNICATION_FORMAT, task.DATASET, TOPOLOGY)
MAX_TURNS = TEAM.max_turns

wikipedia_search = tool(task.make_wikipedia_search(task.SEARCH_DOC_CENTRALIZED))
wikipedia_page = tool(task.make_wikipedia_page(task.PAGE_DOC_CENTRALIZED))
TOOLS = {"wikipedia_search": wikipedia_search, "wikipedia_page": wikipedia_page}


def _delegate_tool(worker: teams.Worker):
    """The manager's ``delegate_to_<worker>`` tool: it passes the instructions on as the worker's handoff."""

    def delegate(instructions: str) -> str:
        return _format_interagent_handoff("manager", instructions, target=worker.role)

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


def _format_interagent_handoff(role: str, text: str, *, target: str) -> str:
    return COMMUNICATION.handoff(
        role,
        text,
        next_action=f"Use this report as the handoff context for {target}.",
        payload={"handoff": "manager_worker", "target": target},
    )


def _last_content_ai(messages: list[BaseMessage]) -> AIMessage | None:
    """The last AIMessage with content and no tool calls, else the last AIMessage with content."""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and (msg.content or "") and not getattr(msg, "tool_calls", None):
            return msg
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and (msg.content or ""):
            return msg
    return None


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
            report = getattr(final, "content", "") if final else ""
            handoff = AIMessage(content=_format_interagent_handoff(name, report, target="manager"))
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
    for worker in TEAM.workers:
        graph.add_node(worker.role, _make_worker_node(worker.role, [TOOLS[t] for t in worker.tools], llm))
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


def _last_answer(records: list[dict]) -> str | None:
    """The short-form answer of the latest message that has one."""
    for r in reversed(records):
        answer = extract_answer(r["content"] or "")
        if answer is not None:
            return answer
    return None


def solve(question: str) -> dict:
    """Run the team on one question.

    Returns ``{"answer", "raw", "messages", "telemetry"}``: the short-form answer
    of the manager's last message (else of the latest message that has one),
    that message, every turn as ``{source, content}`` and token/call counts.
    """
    compiled, _ = _build_graph()
    result = compiled.invoke(
        {"messages": [HumanMessage(content=question)], "turn_count": 0},
        config={"recursion_limit": MAX_TURNS * 4},
    )
    msgs = result.get("messages") or []
    rendered = [_communications_to_record(m) for m in msgs]
    manager_msgs = [r for r in rendered if r["source"] == "manager"]
    final = manager_msgs[-1]["content"] if manager_msgs else ""
    answer = extract_answer(final)
    if answer is None:
        answer = _last_answer(rendered)
    return {
        "answer": answer,
        "raw": final,
        "messages": rendered,
        "telemetry": normalize(langchain_telemetry(msgs)),
    }


def run_batch(instances: list[dict], out_path: Path | None = None, verbose: bool = True) -> dict:
    """Solve and score every instance."""

    def row(_, inst: dict) -> dict:
        out, latency_s, error = attempt(lambda: solve(inst["question"]))
        return task.record(
            inst,
            out["answer"],
            **task.meta(inst),
            raw=out.get("raw") or "",
            n_messages=len(out.get("messages") or []),
            latency_s=round(latency_s, 2),
            **(out.get("telemetry") or {}),
            error=error,
        )

    return task.run_batch(
        instances,
        row,
        out_path=out_path,
        verbose=verbose,
        label="centralized/HotpotQA",
        marks=task.ASCII_MARKS,
        detail=task.messages_detail,
    )


def _canned_demo() -> None:
    out = solve(task.DEMO_QUESTION)
    task.print_demo_answer(out["answer"])
    print(f"=== {len(out['messages'])} messages across the group chat ===")
    for m in out["messages"]:
        snippet = m["content"][:300].replace("\n", " ")
        print(f"  [{m['source']}] {snippet}")


def main(argv: list[str] | None = None) -> int:
    return cli.main(
        argv,
        description="Centralized-topology HotpotQA runner (LangGraph manager/worker).",
        load_instances=load_instances,
        run_batch=run_batch,
        demo=_canned_demo,
        source=task.SOURCE,
        predictions=DEFAULT_PREDICTIONS,
    )


if __name__ == "__main__":
    raise SystemExit(main())
