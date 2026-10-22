"""Sequential HotpotQA with team size r=2: a 2-stage pipeline.

Runs topologies.sequential.langgraph.hotpotqa.langgraph_hotpotqa with the r=2 team of configs/teams/hotpotqa.yaml (see core.variant).
"""

from core import variant

variant.load(globals(), "topologies.sequential.langgraph.hotpotqa.langgraph_hotpotqa", TEAM_SIZE=2)
