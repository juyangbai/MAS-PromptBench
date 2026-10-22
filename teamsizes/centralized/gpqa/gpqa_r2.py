"""Centralized GPQA with team size r=2: a manager and 1 worker.

Runs topologies.centralized.langgraph.gpqa.langgraph_gpqa with the r=2 team of configs/teams/gpqa.yaml (see core.variant).
"""

from core import variant

variant.load(globals(), "topologies.centralized.langgraph.gpqa.langgraph_gpqa", TEAM_SIZE=2)
