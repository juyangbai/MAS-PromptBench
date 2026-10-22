"""Sequential APPS with team size r=2: a 2-stage pipeline.

Runs topologies.sequential.langgraph.apps.langgraph_apps with the r=2 team of configs/teams/apps.yaml (see core.variant).
"""

from core import variant

variant.load(globals(), "topologies.sequential.langgraph.apps.langgraph_apps", TEAM_SIZE=2)
