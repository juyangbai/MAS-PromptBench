"""Centralized APPS with team size r=2: a manager and 1 worker.

Runs topologies.centralized.langgraph.apps.langgraph_apps with the r=2 team of configs/teams/apps.yaml (see core.variant).
"""

from core import variant

variant.load(globals(), "topologies.centralized.langgraph.apps.langgraph_apps", TEAM_SIZE=2)
