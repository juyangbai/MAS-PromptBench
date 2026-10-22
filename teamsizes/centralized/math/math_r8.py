"""Centralized MATH with team size r=8: a manager and 7 workers.

Runs topologies.centralized.langgraph.math.langgraph_math with the r=8 team of configs/teams/<dataset>.yaml (see core.variant).
"""

from core import variant

variant.load(globals(), "topologies.centralized.langgraph.math.langgraph_math", TEAM_SIZE=8)
