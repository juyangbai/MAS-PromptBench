"""Independent ToolHop runner: replicas of one solver vote on the answer.

``INDEPENDENT_N_AGENTS`` replicas (replica i samples with seed i) each run a
tool loop; the most common answer wins (ties: the earliest replica). Data, tools,
the tool loop, scoring and records live in :mod:`core.tasks.toolhop`.
"""

from __future__ import annotations

import time
from functools import partial
from pathlib import Path

from core import agent_runs, settings
from core.communication import CommPolicy
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
from core.telemetry import sum_telemetry

STYLE = "independent_langgraph"
TOPOLOGY = "independent"
ROLE = "solver"

VLLM_BASE_URL = settings.base_url()
MODEL_ID = settings.model_id()
# Preset by the communications/ entries (core.variant). Outside a communications run (None)
# the agents read each other's outputs as plain text, not as freeform reports.
COMMUNICATION_FORMAT = globals().get("COMMUNICATION_FORMAT")
COMMUNICATION = CommPolicy(COMMUNICATION_FORMAT or "freeform", task.DATASET, TOPOLOGY)
# Team-shape knobs; the optimizer adapters set all three on every runner of the dataset.
INDEPENDENT_N_AGENTS = settings.independent_n_agents(dataset=task.DATASET)
DECENTRALIZED_N_AGENTS = settings.decentralized_n_agents(dataset=task.DATASET)
DECENTRALIZED_N_ROUNDS = settings.decentralized_n_rounds(dataset=task.DATASET)


def _system_prompt(topology: str, role: str, style: str, prompt_suffix: str = "") -> str:
    return COMMUNICATION.system_prompt(task.system_prompt(topology, role, style, prompt_suffix))


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
    """Solve one row with ``INDEPENDENT_N_AGENTS`` replicas and a majority vote."""
    agent_runs.check_topology(topology, TOPOLOGY)
    start = time.time()
    per_agent = [
        task.solve_agent(
            solve,
            sample,
            style=style,
            topology=topology,
            role=role,
            seed=seed,
            prompt_suffix=prompt_suffix,
        )
        for seed in range(INDEPENDENT_N_AGENTS)
    ]
    winner = task.choose_winner(per_agent)
    selected = per_agent[winner] if winner is not None else {}
    return {
        "topology": topology,
        "n_agents": INDEPENDENT_N_AGENTS,
        "per_agent": [task.compact_agent(agent) for agent in per_agent],
        "winner": winner,
        "buckets": task.vote_buckets(per_agent),
        **task.answer_fields(selected),
        **task.usage(per_agent),
        "solve_s": round(time.time() - start, 1),
        "telemetry": sum_telemetry(per_agent),
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
