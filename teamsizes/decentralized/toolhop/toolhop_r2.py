"""Decentralized ToolHop with team size r=2: a majority vote over 2 debater replicas.

Runs teamsizes.toolhop_common for the decentralized topology with TEAM_SIZE=2 (see core.variant).
"""

from core import variant

variant.load(globals(), "teamsizes.toolhop_common", TOPOLOGY="decentralized", TEAM_SIZE=2)
