"""Decentralized ToolHop runner on the OpenAI Agents SDK: peers debate with the row's tools.

Every peer gets the question and one function tool per ToolHop schema, backed by
the row's sandboxed functions (``TOOLHOP_ALLOW_DATASET_EXEC=1``);
``run_decentralized_debate`` runs ``DECENTRALIZED_N_AGENTS`` peers for
``DECENTRALIZED_N_ROUNDS`` rounds and selects the final answer. Data, tools,
scoring and records live in :mod:`core.tasks.toolhop`.
"""

from __future__ import annotations

import time
from functools import partial
from pathlib import Path
from typing import Any

from core import agent_runs, settings
from core.tasks import toolhop as task
from core.tasks.toolhop import (  # noqa: F401  (runner API)
    HF_DATASET,
    dataset_summary,
    extract_answer,
    load_instances,
    score_answer,
)
from core.tasks.toolhop import last_assistant_content as _last_assistant_content  # noqa: F401  (runner API)
from core.tasks.toolhop import previous_tool_content as _previous_tool_content
from core.telemetry import normalize
from topologies.decentralized.openai_agents.agents_sdk_base import (
    DebateRecord,
    ToolSpec,
    build_task_invoker,
    chat_view,
    json_schema,
    raise_if_pre_observation_failure,
    reexec_with_sdk_first,
    require_agents_sdk,
    run_decentralized_debate,
)

STYLE = "decentralized_openai_agents"
TOPOLOGY = "decentralized"
ROLE = "debater"

VLLM_BASE_URL = settings.base_url()
MODEL_ID = settings.model_id()
# Team-shape knobs; the optimizer adapters set all three on every runner of the dataset.
INDEPENDENT_N_AGENTS = settings.independent_n_agents(dataset=task.DATASET)
DECENTRALIZED_N_AGENTS = settings.decentralized_n_agents(dataset=task.DATASET)
DECENTRALIZED_N_ROUNDS = settings.decentralized_n_rounds(dataset=task.DATASET)


def _system_prompt(topology: str, role: str, style: str, prompt_suffix: str = "") -> str:
    return task.system_prompt(topology, role, style, prompt_suffix)


def _agents_tools(sample: dict, functions: dict[str, Any]) -> tuple[ToolSpec, ...]:
    """The row's tools as Agents SDK function tools."""
    return tuple(
        ToolSpec(
            name=tool.name, description=tool.description, parameters=json_schema(tool.parameters), handler=tool.call
        )
        for tool in task.function_tools(sample, functions)
    )


def _build_invoker():
    """Agents SDK invoker for the module's current ``VLLM_BASE_URL`` / ``MODEL_ID``."""
    return build_task_invoker(base_url=VLLM_BASE_URL, model_id=MODEL_ID)


def agents_input(sample: dict) -> str:
    """Task text given to every peer: the question itself."""
    return str(sample["question"])


def run_debate(
    sample: dict,
    *,
    style: str,
    topology: str,
    role: str,
    prompt_suffix: str = "",
) -> DebateRecord:
    """Run N peers x R rounds on one ToolHop row with the row's tools."""
    functions = task.function_map(sample)
    return run_decentralized_debate(
        invoker=_build_invoker(),
        example_id=str(sample.get("id")),
        question=agents_input(sample),
        roles={role: _system_prompt(topology, role, style, prompt_suffix)},
        tools=_agents_tools(sample, functions),
        n_agents=DECENTRALIZED_N_AGENTS,
        n_rounds=DECENTRALIZED_N_ROUNDS,
    )


def _turn_agent(sample: dict, turn: dict, role: str) -> dict:
    """Summary of one completed peer turn (one Agents SDK run), with its answer scored."""
    final_content = str(turn.get("output") or "")
    messages = [{"role": "user", "content": turn.get("input", "")}, *chat_view(turn.get("sdk_items") or [])]
    prev_tool_content = _previous_tool_content(messages)
    usage = turn.get("usage") or {}
    return {
        "role": role,
        "peer": turn.get("peer"),
        "round": turn.get("round"),
        "seed": turn.get("request_seed"),
        "solve_s": round(float(turn.get("latency_s") or 0.0), 1),
        **task.scored_answer(sample, final_content, prev_tool_content),
        "turns": int(usage.get("model_calls") or 0),
        "tool_calls": int(usage.get("tool_calls") or 0),
        "final_content": final_content,
        "previous_tool_content": prev_tool_content,
    }


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
    """Run the debate; a failed debate's error is part of the output."""
    agent_runs.check_topology(topology, TOPOLOGY, suffixes=("_openai_agents", "_openai"))
    start = time.time()
    record = run_debate(sample, style=style, topology=topology, role=role, prompt_suffix=prompt_suffix)
    raise_if_pre_observation_failure(record)
    rounds = [
        [_turn_agent(sample, turn, role) for turn in sorted(record.turns(r), key=lambda turn: turn["peer"])]
        for r in range(record.rounds)
    ]
    final_round = rounds[-1] if record.ok else []
    winner = record.selected_peer
    out = {
        "topology": topology,
        "n_agents": record.team_size,
        "n_rounds": record.rounds,
        "per_peer": final_round,
        "rounds": rounds,
        "winner": winner,
        "buckets": dict(record.votes),
        "final_content": record.final_output or "",
        **task.answer_fields(final_round[winner] if winner is not None else {}),
        "tool_calls": record.usage.tool_calls,
        "turns": record.usage.model_calls,
        "solve_s": round(time.time() - start, 1),
        "telemetry": normalize(record.telemetry()),
        "status": record.status,
    }
    if record.error:
        out["error"] = record.error
    return out


def run_one(instance: dict, out_dir: Path, *, style: str, topology: str, role: str) -> dict:
    solve_row = partial(solve_topology, instance, style=style, topology=topology, role=role)
    return task.run_instance(instance, out_dir, style=style, solve=solve_row, debate_errors=True)


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
    return task.main(
        argv,
        description=f"ToolHop runner ({STYLE}).",
        run_one=run,
        style=STYLE,
        model_id=MODEL_ID,
        preflight=require_agents_sdk,
    )


if __name__ == "__main__":
    reexec_with_sdk_first()
    raise SystemExit(main())
