"""Sequential ToolHop with team size r=4: a majority vote over 4 verifier replicas.

Runs teamsizes.toolhop_common for the sequential topology with TEAM_SIZE=4 (see core.variant).
"""

from core import variant

variant.load(globals(), "teamsizes.toolhop_common", TOPOLOGY="sequential", TEAM_SIZE=4)
