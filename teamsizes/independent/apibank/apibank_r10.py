"""Independent API-Bank with team size r=10: a majority vote over 10 solver replicas.

Runs teamsizes.apibank_common for the independent topology with TEAM_SIZE=10 (see core.variant).
"""

from core import variant

variant.load(globals(), "teamsizes.apibank_common", TOPOLOGY="independent", TEAM_SIZE=10)
