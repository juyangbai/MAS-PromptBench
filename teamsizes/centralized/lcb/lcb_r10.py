"""Centralized LCB with team size r=10: a manager and 9 workers.

Runs topologies.centralized.langgraph.lcb.langgraph_lcb with the r=10 team of configs/teams/lcb.yaml (see core.variant).
"""

from core import variant

variant.load(globals(), "topologies.centralized.langgraph.lcb.langgraph_lcb", TEAM_SIZE=10)
