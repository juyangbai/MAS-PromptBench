"""Decentralized MATH runner (OpenAI Agents SDK): N peers debate for R rounds.

The debate engine is :mod:`topologies.decentralized.openai_agents.agents_sdk_base`;
this module supplies the debater prompt, the calculator tool and MATH scoring.
The submitted output is the most common final-round output (ties: lowest peer).
"""

from __future__ import annotations

from pathlib import Path

from core import cli, prompts, settings, teams
from core.batch import attempt
from core.calculator import CALCULATOR_DESCRIPTION, CALCULATOR_PARAMETERS, evaluate
from core.tasks import math as task
from core.tasks.math import exact_match_score, extract_answer, extract_boxed, is_equiv, load_instances  # noqa: F401
from core.telemetry import normalize
from topologies.decentralized.openai_agents.agents_sdk_base import (
    DebateRecord,
    ToolSpec,
    build_task_invoker,
    coerce_tool_arguments,
    content_example_id,
    json_schema,
    peer_contexts,
    raise_if_pre_observation_failure,
    reexec_with_sdk_first,
    require_agents_sdk,
    required,
    run_decentralized_debate,
)

TOPOLOGY = "decentralized"
TEAM = teams.spec(TOPOLOGY, task.DATASET)

VLLM_BASE_URL = settings.base_url()
MODEL_ID = settings.model_id()
N_AGENTS = settings.decentralized_n_agents(TEAM.n_agents)
N_ROUNDS = settings.decentralized_n_rounds(TEAM.n_rounds)


def _load_prompt(role: str) -> str:
    return prompts.role_prompt(TOPOLOGY, task.DATASET, role)


SYSTEM_PROMPT = _load_prompt(TEAM.role)

calculator = evaluate


def _agents_tools() -> tuple[ToolSpec, ...]:
    return (
        ToolSpec(
            name="calculator",
            description=CALCULATOR_DESCRIPTION,
            parameters=json_schema(CALCULATOR_PARAMETERS),
            handler=lambda arguments: calculator(**coerce_tool_arguments(arguments, {"expression": required(str)})),
        ),
    )


def _build_invoker():
    """Endpoint and model are read from the module at call time, so callers can repoint them between rows."""
    return build_task_invoker(base_url=VLLM_BASE_URL, model_id=MODEL_ID)


def agents_input(problem: str) -> str:
    """Task text given to every peer: the problem statement itself."""
    return str(problem)


def run_debate(problem: str) -> DebateRecord:
    return run_decentralized_debate(
        invoker=_build_invoker(),
        example_id=content_example_id(task.DATASET, problem),
        question=agents_input(problem),
        roles={TEAM.role: SYSTEM_PROMPT},
        tools=_agents_tools(),
        n_agents=N_AGENTS,
        n_rounds=N_ROUNDS,
    )


def solve(problem: str) -> dict:
    """Run the debate on one problem.

    Returns ``{"answer", "raw", "winner", "per_peer", "all_contexts", "telemetry",
    "status"}`` (plus ``"error"`` when the debate recorded one): the boxed answer
    of the selected final-round output, that output, the selected peer, each
    peer's ``{peer, answer, raw}``, the peers' transcripts, token/call counts and
    the debate status.
    """
    record = run_debate(problem)
    raise_if_pre_observation_failure(record)
    raw = record.final_output or ""
    out = {
        "answer": extract_answer(raw) if raw else None,
        "raw": raw,
        "winner": record.selected_peer,
        "per_peer": [
            {"peer": i, "answer": extract_answer(text), "raw": text} for i, text in enumerate(record.peer_final_outputs)
        ],
        "all_contexts": peer_contexts(record, {TEAM.role: SYSTEM_PROMPT}),
        "telemetry": normalize(record.telemetry()),
        "status": record.status,
    }
    if record.error:
        out["error"] = record.error
    return out


def run_batch(instances: list[dict], out_path: Path | None = None, verbose: bool = True) -> dict:
    """Solve and score every instance; a debate error is recorded as the row's error."""

    def row(_, inst: dict) -> dict:
        out, latency_s, error = attempt(lambda: solve(inst["problem"]))
        return task.record(
            inst,
            out["answer"],
            **task.meta(inst),
            winner=out.get("winner"),
            per_peer=[
                {"peer": p["peer"], "answer": p["answer"], "raw": (p.get("raw") or "")[:2000]}
                for p in out.get("per_peer") or []
            ],
            latency_s=round(latency_s, 2),
            **(out.get("telemetry") or {}),
            error=error or out.get("error"),
        )

    label = f"decentralized/{task.LABEL} (N={N_AGENTS}, R={N_ROUNDS})"
    return task.run_batch(instances, row, out_path=out_path, verbose=verbose, label=label)


def _canned_demo() -> None:
    out = solve(task.DEMO_PROBLEM)
    print(f"\n=== Debate: N={N_AGENTS} peers x R={N_ROUNDS} rounds, selected peer {out['winner']} ===")
    for p in out["per_peer"]:
        print(f"  peer {p['peer']}: boxed={p['answer'] if p['answer'] is not None else '(none)'!r}")
    task.print_demo_answer(out["answer"])


def main(argv: list[str] | None = None) -> int:
    return cli.main(
        argv,
        description="Decentralized-topology MATH runner (OpenAI Agents SDK debate).",
        load_instances=load_instances,
        run_batch=run_batch,
        demo=_canned_demo,
        source=task.SOURCE,
        preflight=require_agents_sdk,
    )


if __name__ == "__main__":
    reexec_with_sdk_first()
    raise SystemExit(main())
