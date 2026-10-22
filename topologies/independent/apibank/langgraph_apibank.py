"""Independent API-Bank runner: replicas of one solver vote on the next API call.

``INDEPENDENT_N_AGENTS`` replicas (replica i samples with seed i) each make one
model call; the most common API call wins (ties: the earliest replica). Data,
prompts, scoring and records live in :mod:`core.tasks.apibank`.
"""

from __future__ import annotations

import time
from functools import partial
from pathlib import Path

from core import agent_runs, settings
from core.communication import CommPolicy
from core.tasks import apibank as task
from core.tasks.apibank import (  # noqa: F401  (runner API)
    APIBANK_LEVEL,
    BENCHMARK_NAME,
    dataset_summary,
    extract_api_call,
    format_prompt,
    load_instances,
    normalize_level,
    parse_api_call,
    score_prediction,
)
from core.tasks.apibank import format_chat_history as _format_chat_history  # noqa: F401  (runner API)
from core.telemetry import sum_telemetry

DATASET_NAME = task.DATASET
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


def _load_prompt(topology: str, role: str, style: str, prompt_suffix: str = "") -> str:
    return COMMUNICATION.system_prompt(task.system_prompt(topology, role, style, prompt_suffix))


def solve(
    instance: dict,
    *,
    style: str,
    topology: str,
    role: str,
    seed: int | None = None,
    prompt_suffix: str = "",
    extra_context: str = "",
) -> dict:
    """One model call on ``instance``: ``{"messages", "raw", "solve_s", "telemetry"}``."""
    client = task.client(VLLM_BASE_URL)
    user = task.user_message(instance, extra_context)
    system = _load_prompt(topology, role, style, prompt_suffix)
    return task.complete(client, system, user, model=MODEL_ID, seed=seed)


def solve_topology(
    instance: dict,
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
            instance,
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
        "tool_calls": 0,
        "turns": len(per_agent),
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
    level: str | int | None = None,
) -> dict:
    """Solve and record the selected rows; ``out_dir`` defaults to ``results/apibank/<style>``."""
    instances = load_instances(limit=limit, offset=offset, only=only, level=level)
    run = partial(run_one, style=style, topology=topology, role=role)
    return task.run_rows(instances, run, style=style, model_id=MODEL_ID, out_dir=out_dir, level=level, verbose=verbose)


def main(argv: list[str] | None = None) -> int:
    run = partial(run_one, style=STYLE, topology=TOPOLOGY, role=ROLE)
    return task.main(argv, description=f"{BENCHMARK_NAME} runner", run_one=run, style=STYLE, model_id=MODEL_ID)


if __name__ == "__main__":
    raise SystemExit(main())
