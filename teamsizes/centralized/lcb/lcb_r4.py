"""Centralized LCB with team size r=4: a manager and 3 workers.

Runs topologies.centralized.langgraph.lcb.langgraph_lcb with the r=4 team of configs/teams/lcb.yaml (see core.variant).
"""

from core import variant

variant.load(globals(), "topologies.centralized.langgraph.lcb.langgraph_lcb", TEAM_SIZE=4)
