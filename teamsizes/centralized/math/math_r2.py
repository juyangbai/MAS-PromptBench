"""Centralized MATH with team size r=2: a manager and 1 worker.

Runs topologies.centralized.langgraph.math.langgraph_math with the r=2 team of configs/teams/<dataset>.yaml (see core.variant).
"""

from core import variant

variant.load(globals(), "topologies.centralized.langgraph.math.langgraph_math", TEAM_SIZE=2)
