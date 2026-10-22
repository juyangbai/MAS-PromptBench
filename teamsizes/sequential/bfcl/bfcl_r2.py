"""Sequential BFCL with team size r=2: a 2-stage pipeline.

Runs topologies.sequential.langgraph.bfcl.langgraph_bfcl with the r=2 team of configs/teams/bfcl.yaml (see core.variant).
"""

from core import variant

variant.load(globals(), "topologies.sequential.langgraph.bfcl.langgraph_bfcl", TEAM_SIZE=2)
