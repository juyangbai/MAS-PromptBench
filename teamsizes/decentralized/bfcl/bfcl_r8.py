"""Decentralized BFCL with team size r=8: 8 debating peers.

Runs topologies.decentralized.langgraph.bfcl.langgraph_bfcl with the r=8 team of configs/teams/bfcl.yaml (see core.variant).
"""

from core import variant

variant.load(globals(), "topologies.decentralized.langgraph.bfcl.langgraph_bfcl", TEAM_SIZE=8)
