"""Independent APPS with team size r=10: 10 seeded replicas.

Runs topologies.independent.apps.langgraph_apps with the r=10 team of configs/teams/apps.yaml (see core.variant).
"""

from core import variant

variant.load(globals(), "topologies.independent.apps.langgraph_apps", TEAM_SIZE=10)
