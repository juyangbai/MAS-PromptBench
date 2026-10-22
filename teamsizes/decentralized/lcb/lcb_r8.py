"""Decentralized LCB with team size r=8: 8 debating peers.

Runs topologies.decentralized.langgraph.lcb.langgraph_lcb with the r=8 team of configs/teams/lcb.yaml (see core.variant).
"""

from core import variant

variant.load(globals(), "topologies.decentralized.langgraph.lcb.langgraph_lcb", TEAM_SIZE=8)
