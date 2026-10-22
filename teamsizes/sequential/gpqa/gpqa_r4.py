"""Sequential GPQA with team size r=4: a 4-stage pipeline.

Runs topologies.sequential.langgraph.gpqa.langgraph_gpqa with the r=4 team of configs/teams/gpqa.yaml (see core.variant).
"""

from core import variant

variant.load(globals(), "topologies.sequential.langgraph.gpqa.langgraph_gpqa", TEAM_SIZE=4)
