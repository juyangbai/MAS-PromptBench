"""ToolHop team-size runner: a majority vote over r seeded replicas of one role.

Executed by ``teamsizes/<topology>/toolhop/toolhop_r<r>.py`` (see core.variant)
with ``TOPOLOGY`` and ``TEAM_SIZE`` preset. Each replica is one tool loop of the
single-topology solver under the prompt of ``<topology>``'s answering role
(replica i samples with seed i); the most common answer wins (ties: the earliest
replica) and is scored again. Records go to ``results/toolhop/<topology>_toolhop_r<r>``
by default.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

from core.tasks import toolhop as task
from core.tasks.toolhop import dataset_summary, load_instances  # noqa: F401  (runner API)
from topologies.single.toolhop import langgraph_toolhop as _base

_ANSWERING_ROLES = {
    "independent": "solver",
    "decentralized": "debater",
    "sequential": "verifier",
    "centralized": "manager",
}

TOPOLOGY = globals()["TOPOLOGY"]
TEAM_SIZE = globals()["TEAM_SIZE"]
ROLE = _ANSWERING_ROLES[TOPOLOGY]
STYLE = f"{TOPOLOGY}_toolhop_r{TEAM_SIZE}"

MODEL_ID = _base.MODEL_ID
VLLM_BASE_URL = _base.VLLM_BASE_URL


def run_one(instance: dict, out_dir: Path) -> dict:
    replicas = partial(
        task.solve_replicas, _base.solve, instance, style=STYLE, topology=TOPOLOGY, role=ROLE, n=TEAM_SIZE
    )
    return task.run_replicas(instance, Path(out_dir), style=STYLE, team_size=TEAM_SIZE, solve=replicas)


def run_batch(
    limit: int | None = None,
    offset: int = 0,
    only: list[str | int] | None = None,
    out_dir: Path | None = None,
    verbose: bool = True,
) -> dict:
    """Solve and record the selected rows; ``out_dir`` defaults to ``results/toolhop/<STYLE>``."""
    instances = load_instances(limit=limit, offset=offset, only=only)
    return task.run_rows(
        instances, run_one, style=STYLE, model_id=MODEL_ID, out_dir=out_dir, team_size=TEAM_SIZE, verbose=verbose
    )


def main(argv: list[str] | None = None) -> int:
    return task.main(
        argv,
        description=f"ToolHop team-size runner ({STYLE}).",
        run_one=run_one,
        style=STYLE,
        model_id=MODEL_ID,
        team_size=TEAM_SIZE,
    )


if __name__ == "__main__":
    raise SystemExit(main())
