"""Coalition masking for the centralized (manager + workers) runtimes.

A coalition S is a subset of the 3 workers available to the manager (the manager is
always present). Masking a worker means its `delegate_to_<worker>` tool does not
exist for that rollout: the manager cannot bind it, the ToolNode cannot execute it,
and the router cannot route to the worker node. All changes are additive subclasses
/ per-rollout module patches in THIS package; no shared runner file is edited.

Implementation per runner family:
- hotpotqa / lcb (module-backed LangGraph runners): the adapters import a FRESH
  isolated module copy per rollout (`import_isolated_real_module`) and patch its
  globals via `_patched_module` / `patched_module`. We extend that pattern: after
  the base patches, `mask_langgraph_module` additionally patches the module's
  DELEGATION_TOOLS / DELEGATION_NAMES / MANAGER_TOOLS / _manager_tool_node (and the
  manager terminate-nudge) so masked delegate tools do not exist. `_build_graph`'s
  worker nodes still exist but are unreachable: `_route_from_manager_tools` reads
  the patched DELEGATION_NAMES, so hallucinated masked-tool calls produce an
  invalid-tool ToolMessage and route back to the manager.
- bfcl: the runner IS the adapter (CentralizedBFCLAdapter builds its own graph), so
  `MaskedCentralizedBFCLAdapter` overrides `_manager_node` / `_build_graph` /
  the manager_tools router: only active delegate tools are bound, only active worker
  nodes are added to the graph, and only active names are routable.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from typing import Any

from optimizers.bridge.adapters import centralized_bfcl as cb
from optimizers.bridge.adapters.bfcl_common import execution_prompt
from optimizers.bridge.adapters.module_hotpotqa import CentralizedHotpotQAAdapter
from optimizers.bridge.adapters.module_lcb import CentralizedLCBAdapter

from .topology import CENTRALIZED_ACK_SCHEMA


def _worker_of(tool_name: str) -> str:
    return tool_name.removeprefix("delegate_to_")


def availability_note(active_workers: list[str], masked_workers: list[str]) -> str:
    """Manager-facing notice so it does not waste turns calling nonexistent tools."""
    masked_tools = ", ".join(f"delegate_to_{w}" for w in sorted(masked_workers))
    if active_workers:
        active_tools = ", ".join(f"delegate_to_{w}" for w in active_workers)
        return (
            "\n\nAvailability notice (this session): the ONLY delegation tools that "
            f"exist are: {active_tools}. The following tools DO NOT EXIST and must "
            f"never be called: {masked_tools}. If a needed worker is unavailable, do "
            "that part of the work yourself."
        )
    return (
        "\n\nAvailability notice (this session): NO delegation tools exist — no "
        "workers are available. Solve the entire task yourself and emit the final "
        "answer directly."
    )


def _control_evidence(active_workers, masked_workers, delegation_tools):
    return {
        "ack_schema": CENTRALIZED_ACK_SCHEMA,
        "applied": True,
        "manager": "manager",
        "active_workers": sorted(active_workers),
        "masked_workers": sorted(masked_workers),
        "routable_workers": sorted(active_workers),
        "bound_delegation_tools": sorted(tool.name for tool in delegation_tools),
        "enforcement": "delegation-tool-and-route-mask",
    }


# ---------------------------------------------------------------------------------
# module-backed runners (hotpotqa, lcb): per-rollout module-global patch
# ---------------------------------------------------------------------------------
@contextmanager
def mask_langgraph_module(module: Any, masked_workers: frozenset[str], evidence: dict | None = None) -> Iterator[None]:
    """Patch a fresh isolated topology-module copy so masked delegate tools vanish."""
    restore = {}

    def patch(name, value):
        restore[name] = getattr(module, name)
        setattr(module, name, value)

    orig_delegation = list(module.DELEGATION_TOOLS)
    orig_delegation_names = {t.name for t in orig_delegation}
    active = [t for t in orig_delegation if _worker_of(t.name) not in masked_workers]
    active_workers = [_worker_of(t.name) for t in active]
    manager_tools = [t for t in module.MANAGER_TOOLS if t.name not in orig_delegation_names] + active
    if evidence is not None:
        evidence.update(_control_evidence(active_workers, sorted(masked_workers), active))
    if not masked_workers:
        yield
        return

    patch("DELEGATION_TOOLS", active)
    patch("DELEGATION_NAMES", {t.name for t in active})
    patch("MANAGER_TOOLS", manager_tools)
    if hasattr(module, "_manager_tool_node"):
        tool_node_cls = getattr(module, "ToolNode", None)
        if tool_node_cls is None:  # pragma: no cover - both topologies import it
            from langgraph.prebuilt import ToolNode as tool_node_cls
        patch("_manager_tool_node", tool_node_cls(manager_tools))
    if hasattr(module, "_MANAGER_TERMINATE_NUDGE"):
        patch(
            "_MANAGER_TERMINATE_NUDGE",
            module._MANAGER_TERMINATE_NUDGE + availability_note(active_workers, sorted(masked_workers)),
        )
    try:
        yield
    finally:
        for name, value in restore.items():
            setattr(module, name, value)


class _ModuleCoalitionMask:
    """Masked workers checked against the adapter roles; mask applied inside the module patch."""

    def __init__(self, *args, masked_workers=(), **kwargs):
        super().__init__(*args, **kwargs)
        self.masked_workers = frozenset(masked_workers)
        self._coalition_control_evidence = None
        unknown = self.masked_workers - set(self.roles_[1:])
        if unknown:
            raise ValueError(f"unknown masked workers {sorted(unknown)}")

    @contextmanager
    def _masked_module(self, base_patch, module):
        evidence = {}
        try:
            with base_patch(module), mask_langgraph_module(module, self.masked_workers, evidence):
                yield
        finally:
            self._coalition_control_evidence = evidence

    def coalition_control_evidence(self) -> dict:
        """What the last applied mask enforced."""
        if self._coalition_control_evidence is None:
            raise RuntimeError("coalition mask was not applied")
        return dict(self._coalition_control_evidence)


class MaskedCentralizedLCBAdapter(_ModuleCoalitionMask, CentralizedLCBAdapter):
    """Centralized LCB runtime with a per-rollout coalition mask."""

    def patched_module(self, module: Any) -> AbstractContextManager:
        """The base module patch plus the coalition mask."""
        return self._masked_module(super().patched_module, module)


class MaskedCentralizedHotpotQAAdapter(_ModuleCoalitionMask, CentralizedHotpotQAAdapter):
    """Centralized HotpotQA runtime with a per-rollout coalition mask."""

    def _patched_module(self, module):
        return self._masked_module(super()._patched_module, module)


# ---------------------------------------------------------------------------------
# bfcl: the adapter builds its own graph -> additive subclass
# ---------------------------------------------------------------------------------
class MaskedCentralizedBFCLAdapter(cb.CentralizedBFCLAdapter):
    """Centralized BFCL runtime that binds and routes only the active workers."""

    def __init__(self, prompts=None, model_factory=None, masked_workers=()):
        super().__init__(prompts=prompts, model_factory=model_factory)
        self.masked_workers = frozenset(masked_workers)
        unknown = self.masked_workers - set(cb.ROLES[1:])
        if unknown:
            raise ValueError(f"unknown masked workers {sorted(unknown)}")
        self._active_delegation = [t for t in cb.DELEGATION_TOOLS if _worker_of(t.name) not in self.masked_workers]
        self._active_names = {t.name for t in self._active_delegation}
        self._active_workers = [r for r in cb.ROLES[1:] if r not in self.masked_workers]

    def coalition_control_evidence(self) -> dict:
        """What the mask enforced: active/masked workers and the bound delegation tools."""
        return _control_evidence(self._active_workers, self.masked_workers, self._active_delegation)

    def _manager_node(self, state: cb.CentralizedState) -> dict:
        from langchain_core.messages import SystemMessage

        llm = self.model_factory(0)
        if self._active_delegation:
            llm = llm.bind_tools(self._active_delegation)
        nudge = cb.MANAGER_TERMINATE_NUDGE
        if self.masked_workers:
            nudge = nudge + availability_note(self._active_workers, sorted(self.masked_workers))
        sys_msg = SystemMessage(content=execution_prompt(self._prompts["manager"], cb.TOPOLOGY, "manager") + nudge)
        ai = llm.invoke([sys_msg] + state["messages"])
        cb._tag_source(ai, "manager")
        return {"messages": [ai], "turn_count": int(state.get("turn_count", 0)) + 1}

    def _masked_route_from_manager_tools(self, state: cb.CentralizedState) -> str:
        from langchain_core.messages import AIMessage

        for msg in reversed(state.get("messages") or []):
            if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                for call in msg.tool_calls:
                    name = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
                    if name in self._active_names:
                        return name.removeprefix("delegate_to_")
                return "manager"
        return "manager"

    def _build_graph(self):
        if not self.masked_workers:
            return super()._build_graph()
        from langgraph.graph import END, START, StateGraph
        from langgraph.prebuilt import ToolNode

        graph = StateGraph(cb.CentralizedState)
        graph.add_node("manager", self._manager_node)
        # ToolNode over the ACTIVE delegate tools only: a hallucinated masked-tool
        # call yields an invalid-tool ToolMessage and routes back to the manager.
        graph.add_node("manager_tools", ToolNode(self._active_delegation))
        for role in self._active_workers:
            graph.add_node(role, self._make_worker_node(role))
        graph.add_edge(START, "manager")
        graph.add_conditional_edges(
            "manager",
            self._route_from_manager,
            {"manager_tools": "manager_tools", "manager": "manager", END: END},
        )
        route_map = {role: role for role in self._active_workers}
        route_map["manager"] = "manager"
        graph.add_conditional_edges("manager_tools", self._masked_route_from_manager_tools, route_map)
        for role in self._active_workers:
            graph.add_edge(role, "manager")
        return graph


# ---------------------------------------------------------------------------------
# factory
# ---------------------------------------------------------------------------------
MASKED_CENTRALIZED: dict[type, type] = {
    cb.CentralizedBFCLAdapter: MaskedCentralizedBFCLAdapter,
    CentralizedHotpotQAAdapter: MaskedCentralizedHotpotQAAdapter,
    CentralizedLCBAdapter: MaskedCentralizedLCBAdapter,
}


def build_masked_runner(base_cls: type, masked_workers: frozenset[str], **adapter_kwargs: Any) -> Any:
    """Masked subclass of the cell's own centralized runtime class (exact match only).

    ``adapter_kwargs`` are the runtime's constructor arguments (none for the
    centralized cells; tests inject a model factory through them).
    """
    masked_cls = MASKED_CENTRALIZED.get(base_cls)
    if masked_cls is None:
        raise ValueError(f"no coalition-masked runner for {base_cls.__module__}.{base_cls.__qualname__}")
    return masked_cls(masked_workers=frozenset(masked_workers), **adapter_kwargs)


__all__ = [
    "MASKED_CENTRALIZED",
    "MaskedCentralizedBFCLAdapter",
    "MaskedCentralizedHotpotQAAdapter",
    "MaskedCentralizedLCBAdapter",
    "availability_note",
    "build_masked_runner",
    "mask_langgraph_module",
]
