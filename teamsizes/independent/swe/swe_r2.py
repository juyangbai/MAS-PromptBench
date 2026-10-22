"""Independent SWE-bench Verified with team size r=2: 2 seeded replicas.

Runs topologies.independent.swe.langgraph_swe with the r=2 team
of configs/teams/swe.yaml (see core.variant).
"""

from core import variant

variant.load(globals(), "topologies.independent.swe.langgraph_swe", TEAM_SIZE=2)
