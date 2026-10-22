"""Sequential MATH with team size r=4: a 4-stage pipeline.

Runs topologies.sequential.langgraph.math.langgraph_math with the r=4 team of configs/teams/<dataset>.yaml (see core.variant).
"""

from core import variant

variant.load(globals(), "topologies.sequential.langgraph.math.langgraph_math", TEAM_SIZE=4)
