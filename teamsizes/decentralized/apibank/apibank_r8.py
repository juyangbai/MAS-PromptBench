"""Decentralized API-Bank with team size r=8: a majority vote over 8 debater replicas.

Runs teamsizes.apibank_common for the decentralized topology with TEAM_SIZE=8 (see core.variant).
"""

from core import variant

variant.load(globals(), "teamsizes.apibank_common", TOPOLOGY="decentralized", TEAM_SIZE=8)
