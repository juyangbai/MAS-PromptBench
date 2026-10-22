"""Sequential LCB with team size r=2: a 2-stage pipeline.

Runs topologies.sequential.langgraph.lcb.langgraph_lcb with the r=2 team of configs/teams/lcb.yaml (see core.variant).
"""

from core import variant

variant.load(globals(), "topologies.sequential.langgraph.lcb.langgraph_lcb", TEAM_SIZE=2)
