"""Centralized API-Bank with team size r=10: a majority vote over 10 manager replicas.

Runs teamsizes.apibank_common for the centralized topology with TEAM_SIZE=10 (see core.variant).
"""

from core import variant

variant.load(globals(), "teamsizes.apibank_common", TOPOLOGY="centralized", TEAM_SIZE=10)
