"""Decentralized SWE-bench Verified runner (OpenAI Agents SDK): N peers debate for R rounds.

The debate engine is :mod:`topologies.decentralized.openai_agents.agents_sdk_base`;
this module supplies the debater prompt and the repository tools. Peer k edits
its own local clone of the instance repository; the submitted patch is
``git diff HEAD`` of the peer whose final-round output was selected (most common
normalized output, ties: lowest peer), and only that patch is evaluated.
"""

from __future__ import annotations

import shutil
import tempfile
import time
from contextvars import ContextVar
from pathlib import Path

from core import prompts, settings, swe_sandbox, teams
from core.tasks import swe as task
from core.tasks.swe import (  # noqa: F401  (runner API)
    PeerWorktrees,
    agents_input,
    clone_and_checkout,
    git_diff,
    is_resolved,
    load_instances,
    local_clone,
    run_tests_singularity,
)
from core.telemetry import normalize
from topologies.decentralized.openai_agents.agents_sdk_base import (
    DebateRecord,
    ToolSpec,
    active_agent_name,
    build_task_invoker,
    json_schema,
    peer_contexts,
    raise_if_pre_observation_failure,
    reexec_with_sdk_first,
    require_agents_sdk,
    run_decentralized_debate,
)

TOPOLOGY = "decentralized"
TEAM = teams.spec(TOPOLOGY, task.DATASET)
MODEL_NAME = "mas-promptbench-decentralized"
DEFAULT_WORKDIR_ROOT, DEFAULT_OUT_DIR = task.default_dirs("decentralized_openai_agents")

VLLM_BASE_URL = settings.base_url()
MODEL_ID = settings.model_id()
N_AGENTS = settings.decentralized_n_agents(TEAM.n_agents)
N_ROUNDS = settings.decentralized_n_rounds(TEAM.n_rounds)


def _load_prompt(role: str) -> str:
    return prompts.role_prompt(TOPOLOGY, task.DATASET, role)


SYSTEM_PROMPT = _load_prompt(TEAM.role)

_ensure_sif = task.ensure_sif

# A checkout bound here seeds the per-peer clones of a solve() without peer_workdirs.
_SEED_CHECKOUT: ContextVar[Path | None] = ContextVar("swe_seed_checkout", default=None)


def _set_repo_dir(path: Path | str) -> None:
    """Bind an existing base checkout as the seed for per-peer clones."""
    _SEED_CHECKOUT.set(Path(path).resolve())


def prepare_peer_worktrees(instance: dict, root: Path, seed: Path | None = None) -> tuple[str, str, list[Path]]:
    """N per-peer clones under ``root`` (see :func:`core.tasks.swe.prepare_peer_worktrees`)."""
    return task.prepare_peer_worktrees(instance, root, N_AGENTS, seed)


def predictions_entry(instance_id: str, patch: str, model_name: str = MODEL_NAME) -> dict:
    return task.predictions_entry(instance_id, patch, model_name)


def _agents_tools(worktrees: PeerWorktrees) -> tuple[ToolSpec, ...]:
    """The repository tools; each call acts on the worktree of the calling peer."""

    def peer() -> str:
        value = active_agent_name()
        if value is None:
            raise RuntimeError("SWE tool called outside an active debate peer")
        return value

    handlers = {
        "file_read": lambda a: worktrees.file_read(
            peer(), str(a.get("path", "")), int(a.get("offset", 0)), a.get("limit")
        ),
        "str_replace": lambda a: worktrees.str_replace(
            peer(), str(a.get("path", "")), str(a.get("old", "")), str(a.get("new", ""))
        ),
        "list_dir": lambda a: worktrees.list_dir(peer(), str(a.get("path", "."))),
        "search_repo": lambda a: worktrees.search_repo(
            peer(), str(a.get("pattern", "")), str(a.get("path", ".")), int(a.get("max_matches", 50))
        ),
        "shell_exec": lambda a: worktrees.shell_exec(
            peer(), str(a.get("command", "")), int(a.get("timeout_s", task.SHELL_TIMEOUT_S))
        ),
    }
    return tuple(
        ToolSpec(name=name, description=description, parameters=json_schema(parameters), handler=handlers[name])
        for name, (description, parameters) in task.AGENTS_TOOLS.items()
    )


def _build_invoker():
    """Endpoint and model are read from the module at call time, so callers can repoint them between rows."""
    return build_task_invoker(base_url=VLLM_BASE_URL, model_id=MODEL_ID)


def run_debate(instance: dict, worktrees: PeerWorktrees) -> DebateRecord:
    """Run N peers x R rounds; peer k's tools act on worktree k only."""
    return run_decentralized_debate(
        invoker=_build_invoker(),
        example_id=str(instance["instance_id"]),
        question=agents_input(instance),
        roles={TEAM.role: SYSTEM_PROMPT},
        tools=_agents_tools(worktrees),
        n_agents=N_AGENTS,
        n_rounds=N_ROUNDS,
    )


def _peer_patches(worktrees: list[Path], outputs: tuple[str, ...]) -> list[dict]:
    per_peer = []
    for i, workdir in enumerate(worktrees):
        entry = {
            "peer": i,
            "workdir": str(workdir),
            "raw": outputs[i] if i < len(outputs) else "",
            "patch": "",
            "report": None,
            "resolved": None,
            "score": None,
        }
        try:
            entry["patch"] = git_diff(workdir)
        except Exception as e:
            entry["patch_error"] = f"{type(e).__name__}: {e}"
        per_peer.append(entry)
    return per_peer


def solve(instance: dict, peer_workdirs: list[Path] | None = None, eval_mode: str = "singularity") -> dict:
    """Run the debate on one instance; peer k edits ``peer_workdirs[k]``.

    Without ``peer_workdirs`` the clones are prepared from the checkout bound
    by ``_set_repo_dir`` (else a fresh clone) and removed afterwards. Returns
    ``{"patch", "resolved", "raw", "winner", "per_peer", "all_contexts",
    "telemetry", "status"}`` (plus ``"error"`` when the debate recorded one).
    """
    iid = instance["instance_id"]
    owned_root: Path | None = None
    if peer_workdirs is None:
        owned_root = Path(tempfile.mkdtemp(prefix=f"{iid}-"))
        stage, error, peer_workdirs = prepare_peer_worktrees(instance, owned_root, seed=_SEED_CHECKOUT.get())
        if error:
            shutil.rmtree(owned_root, ignore_errors=True)
            raise RuntimeError(f"{stage}: {error}")
    worktrees = [Path(p).resolve() for p in peer_workdirs]
    if len(worktrees) != N_AGENTS:
        raise ValueError(f"expected {N_AGENTS} workdirs, got {len(worktrees)}")
    for workdir in worktrees:
        swe_sandbox.register_worktree(workdir, lambda iid=iid: _ensure_sif(iid))
    try:
        record = run_debate(instance, PeerWorktrees(worktrees))
        raise_if_pre_observation_failure(record)
        winner = record.selected_peer
        per_peer = _peer_patches(worktrees, record.peer_final_outputs)
        evaluated = winner is not None and eval_mode != "none"
        if evaluated:
            task.score_selected(per_peer[winner], instance)
        out = {
            "patch": per_peer[winner]["patch"] if winner is not None else "",
            "resolved": per_peer[winner]["resolved"] if evaluated else None,
            "raw": record.final_output or "",
            "winner": winner,
            "per_peer": per_peer,
            "all_contexts": peer_contexts(record, {TEAM.role: SYSTEM_PROMPT}),
            "telemetry": normalize(record.telemetry()),
            "status": record.status,
        }
        if record.error:
            out["error"] = record.error
        return out
    finally:
        for workdir in worktrees:
            swe_sandbox.unregister_worktree(workdir)
        if owned_root is not None:
            shutil.rmtree(owned_root, ignore_errors=True)


def run_one(instance: dict, workdir_root: Path, out_dir: Path, eval_mode: str = "singularity") -> dict:
    """Clone the repository once and once per peer under ``workdir_root/<id>``, run the debate, write its artifacts."""
    iid = instance["instance_id"]
    summary = task.record_head(instance, n_peers=N_AGENTS, n_rounds=N_ROUNDS)
    start = time.time()
    stage, error, peer_workdirs = prepare_peer_worktrees(instance, workdir_root / iid)
    if error:
        return {**summary, "error": error, "stage": stage}
    summary["clone_s"] = round(time.time() - start, 1)
    start = time.time()
    try:
        out = solve(instance, peer_workdirs, eval_mode=eval_mode)
    except Exception as e:
        return {**summary, "error": f"{type(e).__name__}: {e}", "stage": "solve"}
    summary["solve_s"] = round(time.time() - start, 1)
    summary.update(out.get("telemetry") or {})
    if out.get("error"):
        summary.update(error=out["error"], stage="debate")
    patch = out.get("patch") or ""
    summary.update(patch_chars=len(patch), winner=out.get("winner"))
    task.write_artifacts(out_dir, iid, patch, predictions_entry(iid, patch), task.peer_trace(out))
    per_peer = out.get("per_peer") or []
    summary["per_peer"] = task.peer_rates(per_peer)
    return {**summary, **task.winner_eval_fields(eval_mode, out, per_peer, "peer")}


def _run_instances(
    instances: list[dict],
    out_dir: Path | None = None,
    workdir_root: Path | None = None,
    eval_mode: str = "singularity",
    keep_workdirs: bool = False,
    out_path: Path | None = None,
) -> None:
    """Solve and score loaded instances (the command line's batch)."""
    workdir_root = workdir_root or DEFAULT_WORKDIR_ROOT
    out_dir = out_dir or DEFAULT_OUT_DIR
    task.run_batch(
        instances,
        lambda inst: run_one(inst, workdir_root, out_dir, eval_mode=eval_mode),
        out_dir=out_dir,
        eval_mode=eval_mode,
        workdirs=None if keep_workdirs else lambda inst: [workdir_root / inst["instance_id"]],
        omit="per_peer",
        predictions=out_path,
    )


def run_batch(
    subset: str = "test",
    limit: int | None = None,
    offset: int = 0,
    only: list[str] | None = None,
    workdir_root: Path | None = None,
    out_dir: Path | None = None,
    eval_mode: str = "singularity",
    keep_workdirs: bool = False,
) -> None:
    """Solve and score a Verified slice (``eval_mode``: ``singularity`` or ``none``)."""
    instances = load_instances(subset, limit, offset, only)
    task.log_loaded(instances, f" (N={N_AGENTS}, R={N_ROUNDS})")
    _run_instances(instances, out_dir, workdir_root, eval_mode, keep_workdirs)


def main(argv: list[str] | None = None) -> int:
    return task.cli_main(
        argv,
        description="Decentralized-topology SWE-bench Verified agent (OpenAI Agents SDK debate).",
        run_batch=_run_instances,
        default_out_dir=DEFAULT_OUT_DIR,
        eval_modes=("singularity", "none"),
        preflight=require_agents_sdk,
    )


if __name__ == "__main__":
    reexec_with_sdk_first()
    raise SystemExit(main())
