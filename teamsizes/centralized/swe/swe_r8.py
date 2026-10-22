"""Centralized SWE-bench Verified with team size r=8: a manager and 7 workers.

Runs topologies.centralized.langgraph.swe.langgraph_swe with the r=8 team
of configs/teams/swe.yaml (see core.variant).
"""

from core import variant

variant.load(globals(), "topologies.centralized.langgraph.swe.langgraph_swe", TEAM_SIZE=8)
