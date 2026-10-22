"""Centralized MATH with team size r=10: a manager and 9 workers.

Runs topologies.centralized.langgraph.math.langgraph_math with the r=10 team of configs/teams/<dataset>.yaml (see core.variant).
"""

from core import variant

variant.load(globals(), "topologies.centralized.langgraph.math.langgraph_math", TEAM_SIZE=10)
