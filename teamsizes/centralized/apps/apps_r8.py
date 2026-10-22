"""Centralized APPS with team size r=8: a manager and 7 workers.

Runs topologies.centralized.langgraph.apps.langgraph_apps with the r=8 team of configs/teams/apps.yaml (see core.variant).
"""

from core import variant

variant.load(globals(), "topologies.centralized.langgraph.apps.langgraph_apps", TEAM_SIZE=8)
