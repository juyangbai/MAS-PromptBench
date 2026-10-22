"""Independent BFCL with team size r=4: 4 seeded replicas.

Runs topologies.independent.bfcl.langgraph_bfcl with the r=4 team of configs/teams/bfcl.yaml (see core.variant).
"""

from core import variant

variant.load(globals(), "topologies.independent.bfcl.langgraph_bfcl", TEAM_SIZE=4)
