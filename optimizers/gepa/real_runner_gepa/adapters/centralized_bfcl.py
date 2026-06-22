"""Centralized/BFCL real-runner adapter."""
from __future__ import annotations

import json
from collections.abc import Callable
from typing import Annotated, Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, create_react_agent
from typing_extensions import TypedDict

from real_runner_gepa.adapters.bfcl_common import (
    coerce_instance,
    compact_role_trace,
    default_chat_model,
    execution_prompt,
    extract_canonical,
    flatten_user_request,
    load_prompts,
    recursion_limit,
    schemas_text,
    to_canonical,
)


TOPOLOGY = "centralized"
DATASET = "bfcl"
ROLES = ["manager", "inspector_worker", "caller_worker", "validator_worker"]
MAX_TURNS = 24


@tool("delegate_to_inspector_worker")
def delegate_to_inspector_worker(instructions: str) -> str:
    """Hand the next turn to the inspector_worker."""
    return instructions


@tool("delegate_to_caller_worker")
def delegate_to_caller_worker(instructions: str) -> str:
    """Hand the next turn to the caller_worker."""
    return instructions


@tool("delegate_to_validator_worker")
def delegate_to_validator_worker(instructions: str) -> str:
    """Hand the next turn to the validator_worker."""
    return instructions


DELEGATION_TOOLS = [
    delegate_to_inspector_worker,
    delegate_to_caller_worker,
    delegate_to_validator_worker,
]
DELEGATION_NAMES = {tool_.name for tool_ in DELEGATION_TOOLS}


MANAGER_TERMINATE_NUDGE = (
    "\n\nWhen you emit the final fenced ```json``` block containing the "
    "canonical call list, immediately follow it with the literal string "
    "TERMINATE on its own line so the group-chat knows to stop."
    "\n\nDelegation: when you want a specific worker to act, call the matching "
    "delegate_to_<worker> tool with clear instructions. The workers are: "
    "inspector_worker, caller_worker, validator_worker."
)


class CentralizedState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    turn_count: int


def format_task(user_request: str, schemas: str) -> str:
    return (
        "USER REQUEST:\n"
        f"{user_request}\n\n"
        "SCHEMAS:\n"
        f"{schemas}\n\n"
        "Emit the final canonical call list as a SINGLE fenced ```json``` block. "
        "Canonical form is a list of dicts; each dict has exactly ONE key equal "
        "to the ACTUAL function name from one of the schemas above, and the "
        "value is the arguments dict."
    )


def _tag_source(msg: BaseMessage, source: str) -> None:
    try:
        kwargs = dict(getattr(msg, "additional_kwargs", None) or {})
        kwargs["source"] = source
        msg.additional_kwargs = kwargs
    except Exception:
        pass


def _communications_source(msg: BaseMessage) -> str:
    kwargs = getattr(msg, "additional_kwargs", None) or {}
    source = kwargs.get("source")
    if source:
        return source
    msg_type = getattr(msg, "type", None)
    return {"human": "user", "ai": "assistant", "tool": "tool"}.get(msg_type, msg_type or "?")


def _communications_to_record(msg: BaseMessage) -> dict:
    content = getattr(msg, "content", "") or ""
    if not isinstance(content, str):
        content = str(content)
    tool_calls = getattr(msg, "tool_calls", None) or []
    return {
        "source": _communications_source(msg),
        "content": content[:1500],
        "tool_calls": tool_calls,
    }


class CentralizedBFCLAdapter:
    topology = TOPOLOGY
    dataset = DATASET

    def __init__(
        self,
        prompts: dict[str, str] | None = None,
        model_factory: Callable[[int], Any] | None = None,
    ):
        self._prompts = load_prompts(TOPOLOGY, ROLES, prompts)
        self.model_factory = model_factory or default_chat_model
        self._compiled = None

    def roles(self) -> list[str]:
        return list(ROLES)

    def get_prompt(self, role: str) -> str:
        self._check_role(role)
        return self._prompts[role]

    def set_prompt(self, role: str, text: str) -> None:
        self._check_role(role)
        self._prompts[role] = text
        self.reset()

    def reset(self) -> None:
        self._compiled = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_compiled"] = None
        return state

    def run_example(self, example: Any) -> dict:
        instance = coerce_instance(example)
        compiled = self._compiled_graph()
        task = format_task(flatten_user_request(instance["question"]), schemas_text(instance))
        result = compiled.invoke(
            {"messages": [HumanMessage(content=task)], "turn_count": 0},
            config={"recursion_limit": recursion_limit("BFCL_CENTRALIZED_GRAPH_RECURSION_LIMIT", MAX_TURNS * 4)},
        )
        messages = [_communications_to_record(msg) for msg in result.get("messages") or []]
        manager_msgs = [m for m in messages if m["source"] == "manager"]
        final = manager_msgs[-1]["content"] if manager_msgs else ""
        model_output = extract_canonical(final)
        if model_output is None:
            caller_msgs = [m for m in messages if m["source"] == "caller_worker"]
            if caller_msgs:
                model_output = extract_canonical(caller_msgs[-1]["content"])
        if model_output is None:
            for msg in reversed(messages):
                model_output = extract_canonical(msg["content"])
                if model_output is not None:
                    break
        if model_output is None:
            # Qwen can occasionally try to call the BFCL schema function from
            # the manager even though the centralized manager only exposes
            # delegation tools. LangGraph records that as an invalid tool call;
            # for BFCL scoring it is still a useful first commitment.
            for msg in messages:
                direct_calls = [
                    call
                    for call in msg.get("tool_calls") or []
                    if call.get("name") not in DELEGATION_NAMES
                ]
                if direct_calls:
                    model_output = to_canonical(direct_calls)
                    break
        return {
            "model_output": model_output or [],
            "winner": "manager",
            "buckets": [],
            "messages": messages,
        }

    def format_role_trace(self, role: str, output: Any) -> str:
        self._check_role(role)
        if not isinstance(output, dict):
            return str(output)
        role_msgs = [m["content"] for m in output.get("messages") or [] if m.get("source") == role]
        last = role_msgs[-1] if role_msgs else ""
        return compact_role_trace(
            role=role,
            model_output=output.get("model_output") or [],
            winner=output.get("winner"),
            buckets=output.get("buckets") or [],
            details=[f"{role}_last_message={last[:1200]}"],
        )

    def describe_runtime(self, example: Any | None = None) -> dict:
        instance = coerce_instance(example) if example is not None else {}
        return {
            "topology": self.topology,
            "dataset": self.dataset,
            "roles": self.roles(),
            "tool_count": len(instance.get("function") or []),
            "prompt_prefix": self.get_prompt("manager")[:80],
        }

    def _compiled_graph(self):
        if self._compiled is None:
            self._compiled = self._build_graph().compile()
        return self._compiled

    def _build_graph(self):
        graph = StateGraph(CentralizedState)
        graph.add_node("manager", self._manager_node)
        graph.add_node("manager_tools", ToolNode(DELEGATION_TOOLS))
        for role in ROLES[1:]:
            graph.add_node(role, self._make_worker_node(role))
        graph.add_edge(START, "manager")
        graph.add_conditional_edges(
            "manager",
            self._route_from_manager,
            {"manager_tools": "manager_tools", "manager": "manager", END: END},
        )
        graph.add_conditional_edges(
            "manager_tools",
            self._route_from_manager_tools,
            {
                "inspector_worker": "inspector_worker",
                "caller_worker": "caller_worker",
                "validator_worker": "validator_worker",
                "manager": "manager",
            },
        )
        for role in ROLES[1:]:
            graph.add_edge(role, "manager")
        return graph

    def _manager_node(self, state: CentralizedState) -> dict:
        llm = self.model_factory(0).bind_tools(DELEGATION_TOOLS)
        sys_msg = SystemMessage(content=execution_prompt(self._prompts["manager"], TOPOLOGY, "manager") + MANAGER_TERMINATE_NUDGE)
        ai = llm.invoke([sys_msg] + state["messages"])
        _tag_source(ai, "manager")
        return {"messages": [ai], "turn_count": int(state.get("turn_count", 0)) + 1}

    def _make_worker_node(self, role: str):
        agent = create_react_agent(model=self.model_factory(ROLES.index(role)), tools=[], prompt=execution_prompt(self._prompts[role], TOPOLOGY, role))

        def node(state: CentralizedState) -> dict:
            prior = list(state["messages"])
            result = agent.invoke(
                {"messages": prior},
                config={"recursion_limit": recursion_limit("BFCL_CENTRALIZED_WORKER_RECURSION_LIMIT")},
            )
            new_msgs = result["messages"][len(prior):]
            for msg in new_msgs:
                if isinstance(msg, AIMessage):
                    _tag_source(msg, role)
            n_turns = sum(1 for msg in new_msgs if isinstance(msg, AIMessage))
            return {"messages": new_msgs, "turn_count": int(state.get("turn_count", 0)) + n_turns}

        return node

    @staticmethod
    def _route_from_manager(state: CentralizedState) -> str:
        messages = state.get("messages") or []
        if not messages:
            return "manager"
        last = messages[-1]
        if int(state.get("turn_count", 0)) >= MAX_TURNS:
            return END
        if isinstance(last, AIMessage):
            content = last.content or ""
            if isinstance(content, str) and "TERMINATE" in content:
                return END
            if getattr(last, "tool_calls", None):
                return "manager_tools"
        return "manager"

    @staticmethod
    def _route_from_manager_tools(state: CentralizedState) -> str:
        for msg in reversed(state.get("messages") or []):
            if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                for call in msg.tool_calls:
                    name = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
                    if name in DELEGATION_NAMES:
                        return name.removeprefix("delegate_to_")
                return "manager"
        return "manager"

    @staticmethod
    def _check_role(role: str) -> None:
        if role not in ROLES:
            raise KeyError(f"Unknown role {role!r}; expected one of {ROLES}")
