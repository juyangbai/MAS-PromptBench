"""Independent SWE-bench Verified runner (LangGraph): N seeded replicas, each in its own clone.

Replica k clones the instance repository into ``WORKDIR_ROOT/<id>_a<k>`` and runs
a ReAct patcher sampling with seed k; the submitted patch is the most common
whitespace-normalized non-empty patch (ties: lowest replica, see
:mod:`core.voting`), and only that patch is evaluated. N is the team size
(``configs/teams/swe.yaml``, r=4; ``INDEPENDENT_N_AGENTS`` overrides).
``teamsizes/independent/swe`` runs this module with r = 2, 4, 8 and 10.
"""

from __future__ import annotations

import asyncio
import operator
import os
import time
from pathlib import Path
from typing import Annotated

from langchain_core.tools import tool
from langgraph.constants import END, START
from langgraph.graph.state import StateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.types import Send
from typing_extensions import TypedDict

from core import prompts, settings, swe_sandbox, teams
from core.communication import CommPolicy
from core.llm import chat_openai
from core.tasks import swe as task
from core.tasks.swe import (  # noqa: F401  (runner API)
    clone_and_checkout,
    compute_patch,
    exact_match_score,
    is_resolved,
    load_instances,
    run_tests_singularity,
)
from core.telemetry import langchain_ensemble_telemetry, normalize
from core.thinking import strip_ai_thinking, strip_thinking  # noqa: F401  (runner API)

TOPOLOGY = "independent"
TEAM_SIZE = globals().get("TEAM_SIZE")  # preset by the teamsizes/ variants (core.variant)
TEAM = teams.spec(TOPOLOGY, task.DATASET, TEAM_SIZE)
MODEL_NAME = "mas-promptbench-independent"
DEFAULT_WORKDIR_ROOT, DEFAULT_OUT_DIR = task.default_dirs(TOPOLOGY if TEAM_SIZE is None else f"{TOPOLOGY}_r{TEAM_SIZE}")

VLLM_BASE_URL = settings.base_url()
MODEL_ID = settings.model_id()
COMMUNICATION_FORMAT = globals().get("COMMUNICATION_FORMAT", "freeform")  # preset by the communications/ entries
COMMUNICATION = CommPolicy(COMMUNICATION_FORMAT, task.DATASET, TOPOLOGY)
N_AGENTS = settings.independent_n_agents(TEAM.n_agents)  # replica k samples with seed k
# Where the replicas clone: WORKDIR_ROOT/<instance_id>_a<k> (run_one rebinds it).
WORKDIR_ROOT = Path(os.environ.get("SWE_WORKDIR_ROOT", str(DEFAULT_WORKDIR_ROOT))).resolve()

SYSTEM_PROMPT = COMMUNICATION.system_prompt(prompts.role_prompt(TOPOLOGY, task.DATASET, TEAM.role))

WORKDIR = task.Workdir()  # bound by each replica to its own clone
file_read = tool(task.make_file_read(WORKDIR, task.FILE_READ_DOC))
file_write = tool(task.make_file_write(WORKDIR, task.FILE_WRITE_DOC))
list_dir = tool(task.make_list_dir(WORKDIR, task.LIST_DIR_DOC))
search_repo = tool(task.make_search_repo(WORKDIR, task.SEARCH_REPO_DOC))
shell_exec = tool(task.make_shell_exec(WORKDIR, task.SHELL_EXEC_DOC))
TOOLS = [file_read, file_write, list_dir, search_repo, shell_exec]

_ensure_sif = task.ensure_sif


def format_prompt(
    problem_statement: str,
    repo_dir: Path,
    instance_id: str | None = None,
    hints_text: str | None = None,
) -> str:
    """User message for one instance checked out at ``repo_dir``: issue, hints and the tool workflow."""
    return task.issue_brief(
        problem_statement, instance_id, hints_text, checkout=task.checked_out_at(repo_dir), note=task.FIX_NOTE
    )


def _build_one_agent(seed: int):
    """One replica's ReAct agent, sampling with ``seed``."""
    model = chat_openai(model=MODEL_ID, base_url=VLLM_BASE_URL, seed=seed)
    return create_react_agent(model=model, tools=TOOLS, prompt=SYSTEM_PROMPT)


def predictions_entry(instance_id: str, patch: str, model_name: str = MODEL_NAME) -> dict:
    return task.predictions_entry(instance_id, patch, model_name)


class State(TypedDict):
    instance: dict
    answers: Annotated[list[dict], operator.add]


class AgentInput(TypedDict):
    agent_id: int
    seed: int
    instance: dict


def _answer(inp: AgentInput, workdir: Path, **fields) -> dict:
    return {"answers": [{"agent_id": inp["agent_id"], "seed": inp["seed"], **fields, "workdir": str(workdir)}]}


def _failed(inp: AgentInput, workdir: Path, error: str, clone_s: float, solve_s: float) -> dict:
    return _answer(inp, workdir, patch="", raw=None, messages=[], error=error, clone_s=clone_s, solve_s=solve_s)


async def _run_replica(inp: AgentInput) -> dict:
    """Clone the repository for this replica, run its agent there and return its patch and transcript."""
    instance = inp["instance"]
    iid = instance["instance_id"]
    workdir = WORKDIR_ROOT / f"{iid}_a{inp['agent_id']}"
    start = time.time()
    err = clone_and_checkout(instance["repo"], instance["base_commit"], workdir)
    clone_s = time.time() - start
    if err:
        return _failed(inp, workdir, err, clone_s, 0.0)
    swe_sandbox.register_worktree(workdir, lambda: _ensure_sif(iid))
    token = WORKDIR.bind(workdir)
    try:
        agent = _build_one_agent(seed=inp["seed"])
        prompt = format_prompt(
            instance["problem_statement"], repo_dir=workdir, instance_id=iid, hints_text=instance.get("hints_text")
        )
        start = time.time()
        try:
            result = await agent.ainvoke(
                {"messages": [("user", prompt)]}, config={"recursion_limit": TEAM.recursion_limit}
            )
        except Exception as e:
            return _failed(inp, workdir, f"{type(e).__name__}: {e}", clone_s, time.time() - start)
        solve_s = time.time() - start
        strip_ai_thinking(result["messages"])
        return _answer(
            inp,
            workdir,
            patch=compute_patch(workdir),
            raw=result["messages"][-1].content if result["messages"] else "",
            messages=result["messages"],
            clone_s=round(clone_s, 1),
            solve_s=round(solve_s, 1),
        )
    finally:
        WORKDIR.reset(token)


def _fan_out(state: State) -> list[Send]:
    return [Send(f"agent_{i}", {"agent_id": i, "seed": i, "instance": state["instance"]}) for i in range(N_AGENTS)]


def build_graph() -> StateGraph:
    graph = StateGraph(State)
    for i in range(N_AGENTS):
        graph.add_node(f"agent_{i}", _run_replica)
    graph.add_conditional_edges(START, _fan_out)
    graph.add_edge([f"agent_{i}" for i in range(N_AGENTS)], END)
    return graph


def solve(instance: dict, eval_mode: str = "singularity") -> dict:
    """Run the ensemble on one instance.

    Returns ``{"patch", "resolved", "winner", "per_agent"}``: the voted
    replica's patch, whether it resolves the instance (None without
    evaluation), its agent id and every replica's answer (the voted one with
    ``report``, ``resolved`` and ``score`` when evaluated).
    """
    result = asyncio.run(build_graph().compile().ainvoke({"instance": instance, "answers": []}))
    per_agent = sorted(result["answers"], key=lambda a: a["agent_id"])
    winner = per_agent[task.select_patch([a["patch"] for a in per_agent])]
    evaluated = eval_mode != "none"
    if evaluated:
        task.score_selected(winner, instance)
    return {
        "patch": winner["patch"],
        "resolved": winner["resolved"] if evaluated else None,
        "winner": winner["agent_id"],
        "per_agent": per_agent,
    }


def _replica_trace(out: dict) -> str:
    """The winner, then every replica's timings, rates and error."""
    blocks = [f"winner: agent_{out.get('winner')}  resolved={out.get('resolved')}\n\n"]
    for a in out.get("per_agent") or []:
        report = a.get("report") or {}
        blocks.append(
            f"=== agent_{a['agent_id']} seed={a.get('seed')} ===\n"
            f"  patch_chars={len(a.get('patch') or '')}\n"
            f"  clone_s={a.get('clone_s')}  solve_s={a.get('solve_s')}\n"
            f"  f2p_rate={report.get('f2p_rate')}  p2p_rate={report.get('p2p_rate')}\n"
            f"  resolved={a.get('resolved')}  error={a.get('error')!r}\n\n"
        )
    return "".join(blocks)


def _replica_rates(per_agent: list[dict]) -> list[dict]:
    return [
        {
            "agent_id": a["agent_id"],
            "seed": a.get("seed"),
            "patch_chars": len(a.get("patch") or ""),
            "f2p_rate": (a.get("report") or {}).get("f2p_rate"),
            "p2p_rate": (a.get("report") or {}).get("p2p_rate"),
            "resolved": a.get("resolved"),
            "error": a.get("error"),
        }
        for a in per_agent
    ]


def run_one(instance: dict, workdir_root: Path, out_dir: Path, eval_mode: str = "singularity") -> dict:
    """Run the ensemble on one instance (replica clones under ``workdir_root``) and write its artifacts."""
    global WORKDIR_ROOT
    WORKDIR_ROOT = Path(workdir_root).resolve()
    iid = instance["instance_id"]
    summary = task.record_head(instance)
    start = time.time()
    try:
        out = solve(instance, eval_mode=eval_mode)
    except Exception as e:
        return {**summary, "error": f"{type(e).__name__}: {e}", "stage": "solve"}
    summary["solve_s"] = round(time.time() - start, 1)
    patch = out.get("patch") or ""
    per_agent = out.get("per_agent") or []
    summary.update(patch_chars=len(patch), winner=out.get("winner"), n_agents=N_AGENTS)
    summary.update(normalize(langchain_ensemble_telemetry(per_agent)))
    task.write_artifacts(out_dir, iid, patch, predictions_entry(iid, patch), _replica_trace(out))
    summary["per_agent"] = _replica_rates(per_agent)
    return {**summary, **task.winner_eval_fields(eval_mode, out, per_agent, "agent_id")}


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

    def clones(inst: dict) -> list[Path]:
        return [workdir_root / f"{inst['instance_id']}_a{k}" for k in range(N_AGENTS)]

    task.run_batch(
        instances,
        lambda inst: run_one(inst, workdir_root, out_dir, eval_mode=eval_mode),
        out_dir=out_dir,
        eval_mode=eval_mode,
        workdirs=None if keep_workdirs else clones,
        omit="per_agent",
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
    task.log_loaded(instances, f" (N={N_AGENTS})")
    _run_instances(instances, out_dir, workdir_root, eval_mode, keep_workdirs)


def main(argv: list[str] | None = None) -> int:
    return task.cli_main(
        argv,
        description="Independent-topology SWE-bench Verified agent (ensemble).",
        run_batch=_run_instances,
        default_out_dir=DEFAULT_OUT_DIR,
        eval_modes=("singularity", "none"),
    )


if __name__ == "__main__":
    raise SystemExit(main())
