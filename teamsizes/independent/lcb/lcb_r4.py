"""Independent LCB with team size r=4: 4 seeded replicas.

Runs topologies.independent.lcb.langgraph_lcb with the r=4 team of configs/teams/lcb.yaml (see core.variant).
"""

from core import variant

variant.load(globals(), "topologies.independent.lcb.langgraph_lcb", TEAM_SIZE=4)
