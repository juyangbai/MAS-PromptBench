"""Sequential ToolHop runner, CrewAI label: the LangGraph sequential runner under style ``sequential_crewai``.

ToolHop runners call the endpoint directly, so the CrewAI and LangGraph
sequential runners differ only in their style label (records, default output
folder ``results/toolhop/sequential_crewai``). The runner's source executes in
this module's namespace (see core.variant), so this module owns its hooks.
"""

from core import variant

variant.load(globals(), "topologies.sequential.langgraph.toolhop.langgraph_toolhop", STYLE="sequential_crewai")
