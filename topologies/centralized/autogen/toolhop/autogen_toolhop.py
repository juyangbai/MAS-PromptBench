"""Centralized ToolHop runner, AutoGen label: the LangGraph centralized runner under style ``centralized_autogen``.

ToolHop runners call the endpoint directly, so the AutoGen and LangGraph
centralized runners differ only in their style label (records, default output
folder ``results/toolhop/centralized_autogen``). The runner's source executes in
this module's namespace (see core.variant), so this module owns its hooks.
"""

from core import variant

variant.load(globals(), "topologies.centralized.langgraph.toolhop.langgraph_toolhop", STYLE="centralized_autogen")
