"""Centralized HotpotQA with team size r=10: a manager and 9 workers.

Runs topologies.centralized.langgraph.hotpotqa.langgraph_hotpotqa with the r=10 team of configs/teams/hotpotqa.yaml (see core.variant).
"""

from core import variant

variant.load(globals(), "topologies.centralized.langgraph.hotpotqa.langgraph_hotpotqa", TEAM_SIZE=10)
