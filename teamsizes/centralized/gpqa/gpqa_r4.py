"""Centralized GPQA with team size r=4: a manager and 3 workers.

Runs topologies.centralized.langgraph.gpqa.langgraph_gpqa with the r=4 team of configs/teams/gpqa.yaml (see core.variant).
"""

from core import variant

variant.load(globals(), "topologies.centralized.langgraph.gpqa.langgraph_gpqa", TEAM_SIZE=4)
