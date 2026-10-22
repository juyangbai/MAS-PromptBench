"""Independent LCB with team size r=10: 10 seeded replicas.

Runs topologies.independent.lcb.langgraph_lcb with the r=10 team of configs/teams/lcb.yaml (see core.variant).
"""

from core import variant

variant.load(globals(), "topologies.independent.lcb.langgraph_lcb", TEAM_SIZE=10)
