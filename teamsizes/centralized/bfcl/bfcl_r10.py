"""Centralized BFCL with team size r=10: a manager and 9 workers.

Runs topologies.centralized.langgraph.bfcl.langgraph_bfcl with the r=10 team of configs/teams/bfcl.yaml (see core.variant).
"""

from core import variant

variant.load(globals(), "topologies.centralized.langgraph.bfcl.langgraph_bfcl", TEAM_SIZE=10)
