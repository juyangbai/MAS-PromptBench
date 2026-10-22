"""Independent APPS with team size r=4: 4 seeded replicas.

Runs topologies.independent.apps.langgraph_apps with the r=4 team of configs/teams/apps.yaml (see core.variant).
"""

from core import variant

variant.load(globals(), "topologies.independent.apps.langgraph_apps", TEAM_SIZE=4)
