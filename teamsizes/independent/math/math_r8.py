"""Independent MATH with team size r=8: 8 replicas.

Runs topologies.independent.math.langgraph_math with the r=8 team of configs/teams/<dataset>.yaml (see core.variant).
"""

from core import variant

variant.load(globals(), "topologies.independent.math.langgraph_math", TEAM_SIZE=8)
