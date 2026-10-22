"""Independent SWE-bench Verified with team size r=10: 10 seeded replicas.

Runs topologies.independent.swe.langgraph_swe with the r=10 team
of configs/teams/swe.yaml (see core.variant).
"""

from core import variant

variant.load(globals(), "topologies.independent.swe.langgraph_swe", TEAM_SIZE=10)
