"""Sequential LCB with team size r=4: a 4-stage pipeline.

Runs topologies.sequential.langgraph.lcb.langgraph_lcb with the r=4 team of configs/teams/lcb.yaml (see core.variant).
"""

from core import variant

variant.load(globals(), "topologies.sequential.langgraph.lcb.langgraph_lcb", TEAM_SIZE=4)
