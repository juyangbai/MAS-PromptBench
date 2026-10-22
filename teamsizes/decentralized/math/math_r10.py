"""Decentralized MATH with team size r=10: 10 debating peers.

Runs topologies.decentralized.langgraph.math.langgraph_math with the r=10 team of configs/teams/<dataset>.yaml (see core.variant).
"""

from core import variant

variant.load(globals(), "topologies.decentralized.langgraph.math.langgraph_math", TEAM_SIZE=10)
