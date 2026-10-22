"""Sequential API-Bank runner: a pipeline of stages, each reading the earlier stages' reports.

Each stage makes one model call on the dialogue plus the reports so far (stage i
samples with seed i); the last stage's API call (the verifier's) is the prediction.
Data, prompts, scoring and records live in :mod:`core.tasks.apibank`. The CrewAI
runner executes this module with ``STYLE`` preset (API-Bank runners call the
endpoint directly, so the two differ only in their style label).
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
STYLE = globals().get("STYLE", "sequential_langgraph")
TOPOLOGY = "sequential"
ROLE = "verifier"
SEQUENTIAL_ROLES = ("dialogue_reader", "schema_mapper", "argument_planner", "verifier")

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
    """Run the stages ``roles`` (default ``SEQUENTIAL_ROLES``) in order; the last one answers."""
    agent_runs.check_topology(topology, TOPOLOGY)
    start = time.time()
    stage_roles = tuple(roles or SEQUENTIAL_ROLES)
    by_stage: dict[str, str] = {}
    stages: list[dict] = []
    context = ""
    for seed, stage_role in enumerate(stage_roles):
        stage = task.solve_agent(
            solve,
            instance,
            style=style,
            topology=topology,
            role=stage_role,
            seed=seed,
            prompt_suffix=prompt_suffix,
            extra_context=context,
        )
        stages.append(stage)
        by_stage[stage_role] = stage.get("raw") or ""
        context = agent_runs.reports_context(stages, COMMUNICATION_FORMAT, dataset=task.DATASET)
    final = stages[-1] if stages else {}
    return {
        "topology": topology,
        "n_agents": len(stage_roles),
        "by_stage": by_stage,
        "stage_outputs": [task.compact_agent(stage) for stage in stages],
        **task.answer_fields(final),
        "tool_calls": 0,
        "turns": len(stages),
        "solve_s": round(time.time() - start, 1),
        "telemetry": sum_telemetry(stages),
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
