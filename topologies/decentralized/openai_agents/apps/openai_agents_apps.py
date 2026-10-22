"""Decentralized APPS runner (OpenAI Agents SDK): N peers debate for R rounds.

The debate engine is :mod:`topologies.decentralized.openai_agents.agents_sdk_base`;
this module supplies the debater prompt and APPS scoring (the peers get no function
tools). The submission is the fenced program of the most common final-round output
(ties: lowest peer); with tests, it is scored and the report attached to that peer.
"""

from __future__ import annotations

from pathlib import Path

from core import cli, code_tasks, prompts, settings, teams
from core.batch import attempt
from core.code_tasks import agents_input, exact_match_score, extract_code  # noqa: F401  (runner API)
from core.tasks import apps as task
from core.tasks.apps import load_instances, run_tests  # noqa: F401
from core.telemetry import normalize
from topologies.decentralized.openai_agents.agents_sdk_base import (
    DebateRecord,
    ToolSpec,
    build_task_invoker,
    content_example_id,
    peer_contexts,
    raise_if_pre_observation_failure,
    reexec_with_sdk_first,
    require_agents_sdk,
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


def _agents_tools() -> tuple[ToolSpec, ...]:
    return ()


def _build_invoker():
    """Endpoint and model are read from the module at call time, so callers can repoint them between rows."""
    return build_task_invoker(base_url=VLLM_BASE_URL, model_id=MODEL_ID)


def run_debate(problem: str, starter_code: str | None = None) -> DebateRecord:
    return run_decentralized_debate(
        invoker=_build_invoker(),
        example_id=content_example_id(task.DATASET, problem),
        question=agents_input(problem, starter_code),
        roles={TEAM.role: SYSTEM_PROMPT},
        tools=_agents_tools(),
        n_agents=N_AGENTS,
        n_rounds=N_ROUNDS,
    )


def solve(problem: str, starter_code: str | None = None, input_output: dict | None = None) -> dict:
    """Run the debate on one problem.

    Returns ``{"code", "raw", "winner", "per_peer", "all_contexts", "telemetry",
    "status"}`` (plus ``"error"`` when the debate recorded one): the program of the
    selected final-round output, that output, the selected peer, each peer's
    ``{peer, code, raw}`` (with tests, the selected peer's test report), the peers'
    transcripts, token/call counts and the debate status. The selection never sees the tests.
    """
    record = run_debate(problem, starter_code=starter_code)
    raise_if_pre_observation_failure(record)
    raw = record.final_output or ""
    winner = record.selected_peer
    code = extract_code(raw) if raw else None
    per_peer = [
        {"peer": i, "code": extract_code(text), "raw": text} for i, text in enumerate(record.peer_final_outputs)
    ]
    if input_output and winner is not None:
        task.score_selected_peer(per_peer, winner, code, input_output)
    out = {
        "code": code,
        "raw": raw,
        "winner": winner,
        "per_peer": per_peer,
        "all_contexts": peer_contexts(record, {TEAM.role: SYSTEM_PROMPT}),
        "telemetry": normalize(record.telemetry()),
        "status": record.status,
    }
    if record.error:
        out["error"] = record.error
    return out


def run_batch(
    instances: list[dict],
    out_path: Path | None = None,
    verbose: bool = True,
    per_test_timeout_s: int = task.TEST_TIMEOUT_S,
) -> dict:
    """Solve and score every instance; a debate error is recorded as the row's error."""

    def row(_, inst: dict) -> dict:
        out, latency_s, error = attempt(
            lambda: solve(
                inst["problem"], starter_code=inst.get("starter_code") or None, input_output=inst["input_output"]
            ),
            fallback={"code": None},
        )
        code = out.get("code")
        return task.record(
            inst,
            code,
            code_tasks.selection_scores(code, out.get("winner"), code_tasks.winner_pass_rate(out)),
            per_peer=code_tasks.compact_peers(out.get("per_peer") or []),
            latency_s=round(latency_s, 2),
            **(out.get("telemetry") or {}),
            error=error or out.get("error"),
        )

    label = f"decentralized/APPS (N={N_AGENTS}, R={N_ROUNDS})"
    return task.run_batch(instances, row, out_path=out_path, verbose=verbose, label=label)


def _canned_demo() -> None:
    for mode, problem, starter, tests in task.DEMOS:
        print(f"\n========== {mode} MODE ==========")
        out = solve(problem, starter_code=starter, input_output=tests)
        print(f"=== Debate: N={N_AGENTS} peers x R={N_ROUNDS} rounds, selected peer {out['winner']} ===")
        task.print_demo_code(out["code"], tests)


def main(argv: list[str] | None = None) -> int:
    return cli.main(
        argv,
        description="Decentralized-topology APPS runner (OpenAI Agents SDK debate).",
        load_instances=load_instances,
        run_batch=run_batch,
        demo=_canned_demo,
        source=task.SOURCE,
        add_arguments=task.add_arguments,
        preflight=require_agents_sdk,
    )


if __name__ == "__main__":
    reexec_with_sdk_first()
    raise SystemExit(main())
