"""Decentralized HotpotQA with team size r=8: 8 debating peers.

Runs topologies.decentralized.langgraph.hotpotqa.langgraph_hotpotqa with the r=8 team of configs/teams/hotpotqa.yaml (see core.variant).
"""

from core import variant

variant.load(globals(), "topologies.decentralized.langgraph.hotpotqa.langgraph_hotpotqa", TEAM_SIZE=8)
