"""Centralized APPS with team size r=4: a manager and 3 workers.

Runs topologies.centralized.langgraph.apps.langgraph_apps with the r=4 team of configs/teams/apps.yaml (see core.variant).
"""

from core import variant

variant.load(globals(), "topologies.centralized.langgraph.apps.langgraph_apps", TEAM_SIZE=4)
