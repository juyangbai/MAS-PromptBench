"""Centralized HotpotQA with team size r=8: a manager and 7 workers.

Runs topologies.centralized.langgraph.hotpotqa.langgraph_hotpotqa with the r=8 team of configs/teams/hotpotqa.yaml (see core.variant).
"""

from core import variant

variant.load(globals(), "topologies.centralized.langgraph.hotpotqa.langgraph_hotpotqa", TEAM_SIZE=8)
