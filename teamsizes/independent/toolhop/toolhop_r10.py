"""Independent ToolHop with team size r=10: a majority vote over 10 solver replicas.

Runs teamsizes.toolhop_common for the independent topology with TEAM_SIZE=10 (see core.variant).
"""

from core import variant

variant.load(globals(), "teamsizes.toolhop_common", TOPOLOGY="independent", TEAM_SIZE=10)
