"""Sequential ToolHop with team size r=2: a majority vote over 2 verifier replicas.

Runs teamsizes.toolhop_common for the sequential topology with TEAM_SIZE=2 (see core.variant).
"""

from core import variant

variant.load(globals(), "teamsizes.toolhop_common", TOPOLOGY="sequential", TEAM_SIZE=2)
