"""Sequential GPQA with team size r=2: a 2-stage pipeline.

Runs topologies.sequential.langgraph.gpqa.langgraph_gpqa with the r=2 team of configs/teams/gpqa.yaml (see core.variant).
"""

from core import variant

variant.load(globals(), "topologies.sequential.langgraph.gpqa.langgraph_gpqa", TEAM_SIZE=2)
