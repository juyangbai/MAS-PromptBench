"""Decentralized SWE-bench Verified with team size r=4: 4 debating peers.

Runs topologies.decentralized.langgraph.swe.langgraph_swe with the r=4 team
of configs/teams/swe.yaml (see core.variant).
"""

from core import variant

variant.load(globals(), "topologies.decentralized.langgraph.swe.langgraph_swe", TEAM_SIZE=4)
