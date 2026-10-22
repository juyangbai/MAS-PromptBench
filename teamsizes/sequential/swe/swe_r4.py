"""Sequential SWE-bench Verified with team size r=4: a 4-stage pipeline.

Runs topologies.sequential.langgraph.swe.langgraph_swe with the r=4 team
of configs/teams/swe.yaml (see core.variant).
"""

from core import variant

variant.load(globals(), "topologies.sequential.langgraph.swe.langgraph_swe", TEAM_SIZE=4)
