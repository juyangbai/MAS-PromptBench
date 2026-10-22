"""Independent HotpotQA with team size r=2: 2 replicas.

Runs topologies.independent.hotpotqa.langgraph_hotpotqa with the r=2 team of configs/teams/hotpotqa.yaml (see core.variant).
"""

from core import variant

variant.load(globals(), "topologies.independent.hotpotqa.langgraph_hotpotqa", TEAM_SIZE=2)
