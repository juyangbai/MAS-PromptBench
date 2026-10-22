"""Decentralized BFCL runner (OpenAI Agents SDK): N peers debate for R rounds.

The debate engine is :mod:`topologies.decentralized.openai_agents.agents_sdk_base`;
this module supplies the debater prompt, the instance's functions as
accept-only tools and BFCL scoring. The submission is the canonical call list of
the most common final-round output (ties: lowest peer).
"""

from __future__ import annotations

import json
from pathlib import Path

from core import prompts, settings, teams
from core.paths import RESULTS_DIR
from core.tasks import bfcl as task
from core.tasks.bfcl import AST_CATEGORIES, HF_DATASET, agents_input, extract_canonical, load_instances  # noqa: F401
from core.telemetry import normalize
from topologies.decentralized.openai_agents.agents_sdk_base import (
    DebateRecord,
    ToolSpec,
    build_task_invoker,
    content_example_id,
    json_schema,
    peer_contexts,
    raise_if_pre_observation_failure,
    reexec_with_sdk_first,
    require_agents_sdk,
    run_decentralized_debate,
)

TOPOLOGY = "decentralized"
TEAM = teams.spec(TOPOLOGY, task.DATASET)
DEFAULT_OUT_DIR = RESULTS_DIR / "bfcl_decentralized_openai_agents"

VLLM_BASE_URL = settings.base_url()
MODEL_ID = settings.model_id()
N_AGENTS = settings.decentralized_n_agents(TEAM.n_agents)
N_ROUNDS = settings.decentralized_n_rounds(TEAM.n_rounds)


def _load_prompt(role: str) -> str:
    return prompts.role_prompt(TOPOLOGY, task.DATASET, role)


SYSTEM_PROMPT = _load_prompt(TEAM.role)

_register_model_with_bfcl = task.register_model  # called again by callers that repoint MODEL_ID
_register_model_with_bfcl(MODEL_ID)


def score_one(function_schemas: list[dict], model_output: list[dict], ground_truth: list[dict], category: str) -> dict:
    return task.score_one(function_schemas, model_output, ground_truth, category, MODEL_ID)


def _accept(name: str):
    """A tool handler that acknowledges a call to ``name`` with its arguments; nothing is executed."""
    return lambda arguments: {"accepted_function": name, "arguments": dict(arguments)}


def _agents_tools(instance: dict) -> tuple[ToolSpec, ...]:
    """Every function schema of the instance as an accept-only tool."""
    return tuple(
        ToolSpec(
            name=str(schema["name"]),
            description=str(schema.get("description") or ""),
            parameters=json_schema(schema.get("parameters")),
            handler=_accept(str(schema["name"])),
        )
        for schema in instance.get("function") or []
    )


def _build_invoker():
    """Endpoint and model are read from the module at call time, so callers can repoint them between rows."""
    return build_task_invoker(base_url=VLLM_BASE_URL, model_id=MODEL_ID)


def _example_id(instance: dict) -> str:
    """The instance id, else a digest of its request turns."""
    return str(instance.get("id") or content_example_id(task.DATASET, json.dumps(instance["question"], sort_keys=True)))


def run_debate(instance: dict) -> DebateRecord:
    return run_decentralized_debate(
        invoker=_build_invoker(),
        example_id=_example_id(instance),
        question=agents_input(instance),
        roles={TEAM.role: SYSTEM_PROMPT},
        tools=_agents_tools(instance),
        n_agents=N_AGENTS,
        n_rounds=N_ROUNDS,
    )


def _verdict(instance: dict, model_output: list[dict], ground_truth: dict, category: str) -> dict:
    """The AST checker's verdict on the submission, as attached to the winner's per-peer entry."""
    if not model_output:
        return {"valid": False, "report": None}
    try:
        report = score_one(instance["function"], model_output, ground_truth["ground_truth"], category)
        return {"valid": bool(report.get("valid")), "error_type": report.get("error_type"), "report": report}
    except Exception as e:
        return {"valid": False, "report": {"error": f"{type(e).__name__}: {e}"}}


def solve(instance: dict, ground_truth: dict | None = None, category: str = "simple") -> dict:
    """Run the debate on one instance.

    Returns ``{"model_output", "raw", "winner", "per_peer", "all_contexts",
    "telemetry", "status"}`` (plus ``"error"`` when the debate recorded one).
    With ``ground_truth``, the AST checker's verdict on the submission is
    attached to the winner's per-peer entry; selection never sees it.
    """
    record = run_debate(instance)
    raise_if_pre_observation_failure(record)
    raw = record.final_output or ""
    winner = record.selected_peer
    model_output = (extract_canonical(raw) if raw else None) or []
    per_peer = [
        {"peer": i, "call": extract_canonical(text), "raw": text} for i, text in enumerate(record.peer_final_outputs)
    ]
    if ground_truth is not None and winner is not None:
        per_peer[winner].update(_verdict(instance, model_output, ground_truth, category))
    out = {
        "model_output": model_output,
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


def run_one(instance: dict, ground_truth: dict, category: str, out_dir: Path) -> dict:
    """Solve and score one instance and write the peers' calls to ``out_dir/traces/<id>.txt``."""
    summary: dict = {"id": instance["id"], "category": category, "n_peers": N_AGENTS, "n_rounds": N_ROUNDS}
    try:
        out = solve(instance, ground_truth=ground_truth, category=category)
    except Exception as e:
        return task.solve_failed(summary, e)
    summary["winner"] = out.get("winner")
    summary["model_output"] = out.get("model_output") or []
    if out.get("error"):
        summary["error"] = out["error"]
        summary["stage"] = "debate"
    summary["tool_calls"] = len(summary["model_output"])
    summary.update(out.get("telemetry") or {})
    task.write_trace(out_dir, instance["id"], task.peer_trace(out))
    return task.add_verdict(summary, score_one, instance, ground_truth, category)


def run_batch(
    category: str,
    limit: int | None = None,
    offset: int = 0,
    only: list[str] | None = None,
    out_dir: Path | None = None,
    verbose: bool = True,
) -> dict:
    """Score a slice of one subset; writes predictions.jsonl and results.jsonl afresh in ``out_dir``."""
    return task.evaluate(
        run_one,
        category,
        limit,
        offset,
        only,
        out_dir or DEFAULT_OUT_DIR,
        model_id=MODEL_ID,
        verbose=verbose,
        note=f"  (N={N_AGENTS}, R={N_ROUNDS})",
    )


def main(argv: list[str] | None = None) -> int:
    return task.cli_main(
        argv,
        description="Decentralized-topology BFCL runner (OpenAI Agents SDK debate).",
        run_one=run_one,
        model_id=MODEL_ID,
        default_out_dir=DEFAULT_OUT_DIR,
        preflight=require_agents_sdk,
    )


if __name__ == "__main__":
    reexec_with_sdk_first()
    raise SystemExit(main())
