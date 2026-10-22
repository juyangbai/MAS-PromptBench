"""Centralized BFCL with team size r=2: a manager and 1 worker.

Runs topologies.centralized.langgraph.bfcl.langgraph_bfcl with the r=2 team of configs/teams/bfcl.yaml (see core.variant).
"""

from core import variant

variant.load(globals(), "topologies.centralized.langgraph.bfcl.langgraph_bfcl", TEAM_SIZE=2)
