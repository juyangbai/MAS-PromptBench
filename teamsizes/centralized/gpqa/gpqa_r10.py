"""Centralized GPQA with team size r=10: a manager and 9 workers.

Runs topologies.centralized.langgraph.gpqa.langgraph_gpqa with the r=10 team of configs/teams/gpqa.yaml (see core.variant).
"""

from core import variant

variant.load(globals(), "topologies.centralized.langgraph.gpqa.langgraph_gpqa", TEAM_SIZE=10)
