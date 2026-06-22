"""Independent ToolHop team-size runner, r=4."""

from __future__ import annotations

import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
for _PARENT in _THIS.parents:
    if (_PARENT / "teamsizes").is_dir() and (_PARENT / "topologies").is_dir():
        if str(_PARENT) not in sys.path:
            sys.path.insert(0, str(_PARENT))
        break

from teamsizes.toolhop_common import (  # noqa: E402
    MODEL_ID,
    VLLM_BASE_URL,
    dataset_summary,
    load_instances,
    main as _main,
    run_batch as _run_batch,
    run_one as _run_one,
)

TEAM_SIZE = 4
STYLE = "independent_toolhop_r4"
TOPOLOGY = "independent"
ROLE = "solver"


def run_one(instance: dict, out_dir: Path) -> dict:
    return _run_one(
        instance, Path(out_dir), style=STYLE, topology=TOPOLOGY, role=ROLE, team_size=TEAM_SIZE
    )


def run_batch(
    limit: int | None = None,
    offset: int = 0,
    only: list[str | int] | None = None,
    out_dir: Path | None = None,
    verbose: bool = True,
) -> dict:
    return _run_batch(
        style=STYLE,
        topology=TOPOLOGY,
        role=ROLE,
        team_size=TEAM_SIZE,
        limit=limit,
        offset=offset,
        only=only,
        out_dir=out_dir,
        verbose=verbose,
    )


if __name__ == "__main__":
    raise SystemExit(_main(style=STYLE, topology=TOPOLOGY, role=ROLE, team_size=TEAM_SIZE))
