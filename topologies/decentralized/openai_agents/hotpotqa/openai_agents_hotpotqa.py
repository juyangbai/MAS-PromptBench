"""Decentralized HotpotQA runner (OpenAI Agents SDK): N peers debate for R rounds.

The debate engine is :mod:`topologies.decentralized.openai_agents.agents_sdk_base`;
this module supplies the debater prompt, the Wikipedia tools and HotpotQA scoring.
The submitted output is the most common final-round output (ties: lowest peer).
"""

from __future__ import annotations

from pathlib import Path

from core import cli, prompts, settings, teams
from core.batch import attempt
from core.tasks import hotpotqa as task
from core.tasks.hotpotqa import (  # noqa: F401  (runner API)
    exact_match_score,
    extract_answer,
    f1_score,
    load_instances,
    normalize_answer,
)
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

_OUTPUT_FORMAT_NUDGE = task.OUTPUT_FORMAT_NUDGE


def _load_prompt(role: str) -> str:
    return prompts.role_prompt(TOPOLOGY, task.DATASET, role)


SYSTEM_PROMPT = _load_prompt(TEAM.role) + _OUTPUT_FORMAT_NUDGE

wikipedia_search = task.make_wikipedia_search(task.SEARCH_DOC_SHORT)
wikipedia_page = task.make_wikipedia_page(task.PAGE_DOC_SHORT)


def _agents_tools() -> tuple[ToolSpec, ...]:
    return (
        ToolSpec(
            name="wikipedia_search",
            description=task.SEARCH_DESCRIPTION,
            parameters=json_schema(task.SEARCH_PARAMETERS),
            handler=lambda arguments: wikipedia_search(
                **coerce_tool_arguments(arguments, {"query": required(str), "top_k": (int, 3)})
            ),
        ),
        ToolSpec(
            name="wikipedia_page",
            description=task.PAGE_DESCRIPTION,
            parameters=json_schema(task.PAGE_PARAMETERS),
            handler=lambda arguments: wikipedia_page(**coerce_tool_arguments(arguments, {"title": required(str)})),
        ),
    )


def _build_invoker():
    """Endpoint and model are read from the module at call time, so callers can repoint them between rows."""
    return build_task_invoker(base_url=VLLM_BASE_URL, model_id=MODEL_ID)


def agents_input(question: str) -> str:
    """Task text given to every peer: the question itself."""
    return str(question)


def run_debate(question: str) -> DebateRecord:
    return run_decentralized_debate(
        invoker=_build_invoker(),
        example_id=content_example_id(task.DATASET, question),
        question=agents_input(question),
        roles={TEAM.role: SYSTEM_PROMPT},
        tools=_agents_tools(),
        n_agents=N_AGENTS,
        n_rounds=N_ROUNDS,
    )


def solve(question: str) -> dict:
    """Run the debate on one question.

    Returns ``{"answer", "raw", "winner", "per_peer", "all_contexts", "telemetry",
    "status"}`` (plus ``"error"`` when the debate recorded one): the short-form
    answer of the selected final-round output, that output, the selected peer,
    each peer's ``{peer, answer, raw}``, the peers' transcripts, token/call
    counts and the debate status.
    """
    record = run_debate(question)
    raise_if_pre_observation_failure(record)
    raw = record.final_output or ""
    out = {
        "answer": (extract_answer(raw) or None) if raw else None,
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
        out, latency_s, error = attempt(lambda: solve(inst["question"]))
        return task.record(
            inst,
            out["answer"],
            winner=out.get("winner"),
            per_peer=[
                {"peer": p["peer"], "answer": p["answer"], "raw_tail": (p["raw"] or "")[-300:]}
                for p in out.get("per_peer") or []
            ],
            **task.meta(inst),
            latency_s=round(latency_s, 2),
            **(out.get("telemetry") or {}),
            error=error or out.get("error"),
        )

    return task.run_batch(
        instances,
        row,
        out_path=out_path,
        verbose=verbose,
        label="decentralized/HotpotQA",
        team=f"(N={N_AGENTS} peers × R={N_ROUNDS} rounds)",
        width=30,
        detail=task.peers_detail,
    )


def _canned_demo() -> None:
    out = solve(task.DEMO_QUESTION)
    print(f"\n=== Debate: N={N_AGENTS} peers × R={N_ROUNDS} rounds ===")
    for p in out["per_peer"]:
        print(f"  peer {p['peer']}: answer={p['answer'] or '(none)'!r}")
    task.print_demo_answer(out["answer"], label=f"Selected final answer (peer {out['winner']})")


def main(argv: list[str] | None = None) -> int:
    return cli.main(
        argv,
        description="Decentralized-topology HotpotQA runner (OpenAI Agents SDK debate).",
        load_instances=load_instances,
        run_batch=run_batch,
        demo=_canned_demo,
        source=task.SOURCE,
        preflight=require_agents_sdk,
    )


if __name__ == "__main__":
    reexec_with_sdk_first()
    raise SystemExit(main())
