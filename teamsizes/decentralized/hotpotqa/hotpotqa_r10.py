"""Decentralized HotpotQA with team size r=10: 10 debating peers.

Runs topologies.decentralized.langgraph.hotpotqa.langgraph_hotpotqa with the r=10 team of configs/teams/hotpotqa.yaml (see core.variant).
"""

from core import variant

variant.load(globals(), "topologies.decentralized.langgraph.hotpotqa.langgraph_hotpotqa", TEAM_SIZE=10)
