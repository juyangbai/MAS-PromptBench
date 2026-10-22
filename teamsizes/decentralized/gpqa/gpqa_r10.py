"""Decentralized GPQA with team size r=10: 10 debating peers.

Runs topologies.decentralized.langgraph.gpqa.langgraph_gpqa with the r=10 team of configs/teams/gpqa.yaml (see core.variant).
"""

from core import variant

variant.load(globals(), "topologies.decentralized.langgraph.gpqa.langgraph_gpqa", TEAM_SIZE=10)
