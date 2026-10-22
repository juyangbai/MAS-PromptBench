"""Independent ToolHop with team size r=8: a majority vote over 8 solver replicas.

Runs teamsizes.toolhop_common for the independent topology with TEAM_SIZE=8 (see core.variant).
"""

from core import variant

variant.load(globals(), "teamsizes.toolhop_common", TOPOLOGY="independent", TEAM_SIZE=8)
