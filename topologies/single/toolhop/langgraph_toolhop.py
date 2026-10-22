"""Single-agent ToolHop runner: one tool-calling loop answers the question.

The solver prompt is ``configs/prompts/single/toolhop/solver.txt``; data, the
sandboxed dataset tools, the tool loop, scoring and records live in
:mod:`core.tasks.toolhop` (solving needs ``TOOLHOP_ALLOW_DATASET_EXEC=1``).
"""

from __future__ import annotations

import time
from functools import partial
from pathlib import Path

from core import agent_runs, settings
from core.tasks import toolhop as task
from core.tasks.toolhop import (  # noqa: F401  (runner API)
    DEFAULT_MAX_TURNS,
    HF_DATASET,
    dataset_summary,
    extract_answer,
    load_instances,
    score_answer,
)
from core.tasks.toolhop import last_assistant_content as _last_assistant_content  # noqa: F401  (runner API)
from core.tasks.toolhop import previous_tool_content as _previous_tool_content  # noqa: F401  (runner API)

STYLE = "single_langgraph"
TOPOLOGY = "single"
ROLE = "solver"

VLLM_BASE_URL = settings.base_url()
MODEL_ID = settings.model_id()
# Team-shape knobs; the optimizer adapters set all three on every runner of the dataset.
INDEPENDENT_N_AGENTS = settings.independent_n_agents(dataset=task.DATASET)
DECENTRALIZED_N_AGENTS = settings.decentralized_n_agents(dataset=task.DATASET)
DECENTRALIZED_N_ROUNDS = settings.decentralized_n_rounds(dataset=task.DATASET)


def _system_prompt(topology: str, role: str, style: str, prompt_suffix: str = "") -> str:
    return task.system_prompt(topology, role, style, prompt_suffix)


def solve(
    sample: dict,
    *,
    style: str,
    topology: str,
    role: str,
    max_turns: int = DEFAULT_MAX_TURNS,
    seed: int | None = None,
    prompt_suffix: str = "",
    extra_context: str = "",
) -> dict:
    """Run one tool loop on ``sample``: ``{"messages", "solve_s", "telemetry"}``."""
    functions = task.function_map(sample)
    client = task.client(VLLM_BASE_URL)
    user = task.user_message(sample, extra_context)
    system = _system_prompt(topology, role, style, prompt_suffix)
    return task.tool_loop(
        client, sample, functions, system, user, model=MODEL_ID, role=role, seed=seed, max_turns=max_turns
    )


def solve_topology(
    sample: dict,
    *,
    style: str,
    topology: str,
    role: str,
    prompt_suffix: str = "",
    roles: list[str] | tuple[str, ...] | None = None,
    worker_roles: list[str] | tuple[str, ...] | None = None,
) -> dict:
    """Solve one row with a single scored tool loop (seed ``REQUEST_SEED``)."""
    agent_runs.check_topology(topology, TOPOLOGY)
    start = time.time()
    agent = task.solve_agent(
        solve,
        sample,
        style=style,
        topology=topology,
        role=role,
        seed=settings.request_seed(),
        prompt_suffix=prompt_suffix,
    )
    return {
        **agent,
        "n_agents": 1,
        "messages": agent.get("messages") or [],
        "solve_s": agent.get("solve_s", round(time.time() - start, 1)),
        "telemetry": agent.get("telemetry") or {},
    }


def run_one(instance: dict, out_dir: Path, *, style: str, topology: str, role: str) -> dict:
    solve_row = partial(solve_topology, instance, style=style, topology=topology, role=role)
    return task.run_instance(instance, out_dir, style=style, solve=solve_row)


def run_batch(
    *,
    style: str,
    topology: str,
    role: str,
    limit: int | None = None,
    offset: int = 0,
    only: list[str | int] | None = None,
    out_dir: Path | None = None,
    verbose: bool = True,
) -> dict:
    """Solve and record the selected rows; ``out_dir`` defaults to ``results/toolhop/<style>``."""
    instances = load_instances(limit=limit, offset=offset, only=only)
    run = partial(run_one, style=style, topology=topology, role=role)
    return task.run_rows(instances, run, style=style, model_id=MODEL_ID, out_dir=out_dir, verbose=verbose)


def main(argv: list[str] | None = None) -> int:
    run = partial(run_one, style=STYLE, topology=TOPOLOGY, role=ROLE)
    return task.main(argv, description=f"ToolHop runner ({STYLE}).", run_one=run, style=STYLE, model_id=MODEL_ID)


if __name__ == "__main__":
    raise SystemExit(main())
