"""Centralized SWE-bench Verified with team size r=2: a manager and 1 worker.

Runs topologies.centralized.langgraph.swe.langgraph_swe with the r=2 team
of configs/teams/swe.yaml (see core.variant).
"""

from core import variant

variant.load(globals(), "topologies.centralized.langgraph.swe.langgraph_swe", TEAM_SIZE=2)
