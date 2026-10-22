"""Decentralized ToolHop runner: peers debate the answer over rounds.

``DECENTRALIZED_N_AGENTS`` peers each run a tool loop per round (seed = round *
peers + peer); from the second round on, a peer sees its own previous answer and
the others' reports. The most common answer of the last round wins (ties: the
earliest peer). Data, tools, the tool loop, scoring and records live in
:mod:`core.tasks.toolhop`.
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

STYLE = "decentralized_langgraph"
TOPOLOGY = "decentralized"
ROLE = "debater"

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


def _peer_context(previous_round: list[dict], peer_idx: int, round_idx: int) -> str:
    """What a peer reads in a later round: its own previous answer and the other peers' reports."""
    own = previous_round[peer_idx] if peer_idx < len(previous_round) else {}
    others = [peer for idx, peer in enumerate(previous_round) if idx != peer_idx]
    return (
        f"Debate round {round_idx + 1}. Your previous answer:\n"
        f"{own.get('final_content', '')[-1200:]}\n\n"
        "Other peers' previous-round answers:\n"
        + agent_runs.reports_context(others, COMMUNICATION_FORMAT, dataset=task.DATASET)
        + "\n\nRevise only if peer evidence is stronger."
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
    """Debate ``DECENTRALIZED_N_ROUNDS`` rounds among ``DECENTRALIZED_N_AGENTS`` peers; vote on the last round."""
    agent_runs.check_topology(topology, TOPOLOGY)
    start = time.time()
    rounds: list[list[dict]] = []
    previous_round: list[dict] = []
    for round_idx in range(DECENTRALIZED_N_ROUNDS):
        current_round = []
        for peer_idx in range(DECENTRALIZED_N_AGENTS):
            context = _peer_context(previous_round, peer_idx, round_idx) if previous_round else ""
            peer = task.solve_agent(
                solve,
                sample,
                style=style,
                topology=topology,
                role=role,
                seed=round_idx * DECENTRALIZED_N_AGENTS + peer_idx,
                prompt_suffix=prompt_suffix,
                extra_context=context,
            )
            current_round.append(peer)
        rounds.append(current_round)
        previous_round = current_round
    final_round = rounds[-1] if rounds else []
    winner = task.choose_winner(final_round)
    selected = final_round[winner] if winner is not None else {}
    agents = [agent for round_agents in rounds for agent in round_agents]
    return {
        "topology": topology,
        "n_agents": DECENTRALIZED_N_AGENTS,
        "n_rounds": DECENTRALIZED_N_ROUNDS,
        "per_peer": [task.compact_agent(agent) for agent in final_round],
        "rounds": [[task.compact_agent(agent) for agent in round_agents] for round_agents in rounds],
        "winner": winner,
        "buckets": task.vote_buckets(final_round),
        **task.answer_fields(selected),
        **task.usage(agents),
        "solve_s": round(time.time() - start, 1),
        "telemetry": sum_telemetry(agents),
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
