"""Centralized API-Bank runner, AutoGen label: the LangGraph centralized runner under style ``centralized_autogen``.

API-Bank runners call the endpoint directly, so the AutoGen and LangGraph
centralized runners differ only in their style label (records, default output
folder ``results/apibank/centralized_autogen``). The runner's source executes in
this module's namespace (see core.variant), so this module owns its hooks.
"""

from core import variant

variant.load(globals(), "topologies.centralized.langgraph.apibank.langgraph_apibank", STYLE="centralized_autogen")
