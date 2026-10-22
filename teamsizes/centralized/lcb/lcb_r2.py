"""Centralized LCB with team size r=2: a manager and 1 worker.

Runs topologies.centralized.langgraph.lcb.langgraph_lcb with the r=2 team of configs/teams/lcb.yaml (see core.variant).
"""

from core import variant

variant.load(globals(), "topologies.centralized.langgraph.lcb.langgraph_lcb", TEAM_SIZE=2)
