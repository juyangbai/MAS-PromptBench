"""Decentralized API-Bank runner on the OpenAI Agents SDK: peers debate with the row's APIs as tools.

Every peer gets the formatted dialogue and one function tool per API the row
describes; ``run_decentralized_debate`` runs ``DECENTRALIZED_N_AGENTS`` peers for
``DECENTRALIZED_N_ROUNDS`` rounds and selects the final call. Data, prompts,
scoring and records live in :mod:`core.tasks.apibank`.
"""

from __future__ import annotations

import time
from functools import partial
from pathlib import Path

from core import agent_runs, settings
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
from core.telemetry import normalize
from topologies.decentralized.openai_agents.agents_sdk_base import (
    DebateRecord,
    ToolSpec,
    build_task_invoker,
    json_schema,
    raise_if_pre_observation_failure,
    reexec_with_sdk_first,
    require_agents_sdk,
    run_decentralized_debate,
)

DATASET_NAME = task.DATASET
STYLE = "decentralized_openai_agents"
TOPOLOGY = "decentralized"
ROLE = "debater"

VLLM_BASE_URL = settings.base_url()
MODEL_ID = settings.model_id()
# Team-shape knobs; the optimizer adapters set all three on every runner of the dataset.
INDEPENDENT_N_AGENTS = settings.independent_n_agents(dataset=task.DATASET)
DECENTRALIZED_N_AGENTS = settings.decentralized_n_agents(dataset=task.DATASET)
DECENTRALIZED_N_ROUNDS = settings.decentralized_n_rounds(dataset=task.DATASET)


def _load_prompt(topology: str, role: str, style: str, prompt_suffix: str = "") -> str:
    return task.system_prompt(topology, role, style, prompt_suffix)


def _agents_tools(instance: dict) -> tuple[ToolSpec, ...]:
    """The row's API tools as Agents SDK function tools."""
    return tuple(
        ToolSpec(
            name=tool.name, description=tool.description, parameters=json_schema(tool.parameters), handler=tool.call
        )
        for tool in task.api_tools(instance)
    )


def _build_invoker():
    """Agents SDK invoker for the module's current ``VLLM_BASE_URL`` / ``MODEL_ID``."""
    return build_task_invoker(base_url=VLLM_BASE_URL, model_id=MODEL_ID)


def agents_input(instance: dict) -> str:
    """Task text given to every peer: the formatted dialogue prompt."""
    return str(format_prompt(instance))


def run_debate(
    instance: dict,
    *,
    style: str,
    topology: str,
    role: str,
    prompt_suffix: str = "",
) -> DebateRecord:
    """Run N peers x R rounds on one API-Bank row with the row's API tools."""
    return run_decentralized_debate(
        invoker=_build_invoker(),
        example_id=str(instance.get("id")),
        question=agents_input(instance),
        roles={role: _load_prompt(topology, role, style, prompt_suffix)},
        tools=_agents_tools(instance),
        n_agents=DECENTRALIZED_N_AGENTS,
        n_rounds=DECENTRALIZED_N_ROUNDS,
    )


def _turn_agent(instance: dict, turn: dict, role: str) -> dict:
    """Summary of one completed peer turn (one Agents SDK run), with its call scored."""
    usage = turn.get("usage") or {}
    return {
        "role": role,
        "peer": turn.get("peer"),
        "round": turn.get("round"),
        "seed": turn.get("request_seed"),
        "solve_s": round(float(turn.get("latency_s") or 0.0), 1),
        **task.scored_call(instance, str(turn.get("output") or "")),
        "turns": int(usage.get("model_calls") or 0),
        "tool_calls": int(usage.get("tool_calls") or 0),
    }


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
    """Run the debate; a failed debate is reported with stage ``debate``."""
    agent_runs.check_topology(topology, TOPOLOGY, suffixes=("_openai_agents", "_openai"))
    start = time.time()
    record = run_debate(instance, style=style, topology=topology, role=role, prompt_suffix=prompt_suffix)
    raise_if_pre_observation_failure(record)
    rounds = [
        [_turn_agent(instance, turn, role) for turn in sorted(record.turns(r), key=lambda turn: turn["peer"])]
        for r in range(record.rounds)
    ]
    final_round = rounds[-1] if record.ok else []
    winner = record.selected_peer
    answer = task.answer_fields(final_round[winner] if winner is not None else {})
    if not record.ok:
        answer.update(stage="debate", error=record.error)
    return {
        "topology": topology,
        "n_agents": record.team_size,
        "n_rounds": record.rounds,
        "per_peer": [task.compact_agent(agent) for agent in final_round],
        "rounds": [[task.compact_agent(agent) for agent in round_agents] for round_agents in rounds],
        "winner": winner,
        "buckets": dict(record.votes),
        "raw": record.final_output or "",
        **answer,
        "tool_calls": record.usage.tool_calls,
        "turns": record.usage.model_calls,
        "solve_s": round(time.time() - start, 1),
        "telemetry": normalize(record.telemetry()),
        "status": record.status,
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
    return task.main(
        argv,
        description=f"{BENCHMARK_NAME} runner",
        run_one=run,
        style=STYLE,
        model_id=MODEL_ID,
        preflight=require_agents_sdk,
    )


if __name__ == "__main__":
    reexec_with_sdk_first()
    raise SystemExit(main())
