"""Centralized GPQA with team size r=8: a manager and 7 workers.

Runs topologies.centralized.langgraph.gpqa.langgraph_gpqa with the r=8 team of configs/teams/gpqa.yaml (see core.variant).
"""

from core import variant

variant.load(globals(), "topologies.centralized.langgraph.gpqa.langgraph_gpqa", TEAM_SIZE=8)
