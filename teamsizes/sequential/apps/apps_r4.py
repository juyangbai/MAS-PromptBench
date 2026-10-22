"""Sequential APPS with team size r=4: a 4-stage pipeline.

Runs topologies.sequential.langgraph.apps.langgraph_apps with the r=4 team of configs/teams/apps.yaml (see core.variant).
"""

from core import variant

variant.load(globals(), "topologies.sequential.langgraph.apps.langgraph_apps", TEAM_SIZE=4)
