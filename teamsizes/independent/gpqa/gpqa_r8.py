"""Independent GPQA with team size r=8: 8 replicas.

Runs topologies.independent.gpqa.langgraph_gpqa with the r=8 team of configs/teams/gpqa.yaml (see core.variant).
"""

from core import variant

variant.load(globals(), "topologies.independent.gpqa.langgraph_gpqa", TEAM_SIZE=8)
