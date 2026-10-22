"""Decentralized SWE-bench Verified runner (LangGraph): N peers debate for R rounds, each in its own clone.

Every peer is the same ReAct debater with its own history and its own clone of
the instance repository. From the second round on, each peer reads the other
peers' previous final summaries and may revise its edits. The submitted patch
is the most common whitespace-normalized non-empty patch of the peers' clones
(ties: lowest peer, see :mod:`core.voting`), and only that patch is evaluated.
N is the team size (``configs/teams/swe.yaml``, r=4; ``DECENTRALIZED_N_AGENTS``
overrides) and R is ``DECENTRALIZED_N_ROUNDS`` (default 2).
``teamsizes/decentralized/swe`` runs this module with r = 2, 4, 8 and 10.
"""

from __future__ import annotations

import time
from pathlib import Path

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent
from typing_extensions import TypedDict

from core import prompts, settings, swe_sandbox, teams
from core.communication import CommPolicy
from core.llm import chat_openai
from core.tasks import swe as task
from core.tasks.swe import (  # noqa: F401  (runner API)
    clone_and_checkout,
    compute_patch,
    is_resolved,
    load_instances,
    run_tests_singularity,
)
from core.telemetry import langchain_telemetry, normalize
from core.thinking import strip_thinking  # noqa: F401  (runner API)

TOPOLOGY = "decentralized"
TEAM_SIZE = globals().get("TEAM_SIZE")  # preset by the teamsizes/ variants (core.variant)
TEAM = teams.spec(TOPOLOGY, task.DATASET, TEAM_SIZE)
MODEL_NAME = "mas-promptbench-decentralized-langgraph"
DEFAULT_WORKDIR_ROOT, DEFAULT_OUT_DIR = task.default_dirs(
    "decentralized_langgraph" if TEAM_SIZE is None else f"decentralized_r{TEAM_SIZE}"
)

VLLM_BASE_URL = settings.base_url()
MODEL_ID = settings.model_id()
COMMUNICATION_FORMAT = globals().get("COMMUNICATION_FORMAT", "freeform")  # preset by the communications/ entries
COMMUNICATION = CommPolicy(COMMUNICATION_FORMAT, task.DATASET, TOPOLOGY)
N_AGENTS = settings.decentralized_n_agents(TEAM.n_agents)
N_ROUNDS = settings.decentralized_n_rounds(TEAM.n_rounds)
_RECURSION_LIMIT = TEAM.recursion_limit  # each peer turn's ReAct loop


def _load_prompt(role: str) -> str:
    return prompts.role_prompt(TOPOLOGY, task.DATASET, role)


SYSTEM_PROMPT = COMMUNICATION.system_prompt(_load_prompt(TEAM.role))

WORKDIR = task.Workdir(Path("."), explain_escapes=False)  # bound to each peer's clone for its turn
file_read = tool(task.make_file_read(WORKDIR, task.FILE_READ_DOC))
str_replace = tool(
    task.make_str_replace(
        WORKDIR, task.STR_REPLACE_DOC_UNIQUE, not_found=task.NOT_FOUND_TERSE, ambiguous=task.AMBIGUOUS_TERSE
    )
)
list_dir = tool(task.make_list_dir(WORKDIR, task.LIST_DIR_DOC_REPO))
search_repo = tool(task.make_search_repo(WORKDIR, task.SEARCH_REPO_DOC_REPO, terse=True))
shell_exec = tool(task.make_shell_exec(WORKDIR, task.SHELL_EXEC_DOC_REPO))
TOOLS = [file_read, str_replace, list_dir, search_repo, shell_exec]

_ensure_sif = task.ensure_sif


def _build_llm() -> ChatOpenAI:
    return chat_openai(model=MODEL_ID, base_url=VLLM_BASE_URL)


def _build_agent():
    """The debater agent shared by all peers of a round (each peer passes its own history and clone)."""
    return create_react_agent(model=_build_llm(), tools=TOOLS, prompt=SYSTEM_PROMPT)


def format_task_brief(problem_statement: str, instance_id: str | None = None, hints_text: str | None = None) -> str:
    """Every peer's first message."""
    return task.issue_brief(
        problem_statement, instance_id, hints_text, checkout=task.PEER_CHECKOUT, note=task.NO_TESTS_NOTE
    )


def predictions_entry(instance_id: str, patch: str, model_name: str = MODEL_NAME) -> dict:
    return task.predictions_entry(instance_id, patch, model_name)


def _peer_injection(others_final: list[BaseMessage], brief: str) -> HumanMessage:
    """The message showing a peer the other peers' previous final summaries."""
    others = []
    for m in others_final:
        content = getattr(m, "content", "") or ""
        others.append(content if isinstance(content, str) else str(content))
    return HumanMessage(content=task.peer_message(others, brief))


class DebateState(TypedDict, total=False):
    contexts: list[list[BaseMessage]]  # per-peer message histories (without the system prompt)
    round_finals: list[list[BaseMessage]]  # per round, each peer's final AIMessage
    round: int
    brief: str
    peer_workdirs: list[Path]


def _last_ai(msgs: list[BaseMessage]) -> BaseMessage | None:
    """The last AIMessage with content and no tool calls, else the last AIMessage."""
    for m in reversed(msgs):
        if isinstance(m, AIMessage) and (m.content or "") and not getattr(m, "tool_calls", None):
            return m
    for m in reversed(msgs):
        if isinstance(m, AIMessage):
            return m
    return None


def _is_bad_request(exc: Exception) -> bool:
    """An endpoint 400 (e.g. a truncated tool-call argument in the history), recognized without importing openai."""
    return "BadRequest" in type(exc).__name__ or "BadRequest" in str(exc)[:120] or "400" in str(exc)[:20]


def _invoke_peer_with_recovery(agent, ctx: list[BaseMessage]) -> list[BaseMessage]:
    """Run one peer turn; after a bad request, drop the failed tool exchange and retry once with a nudge.

    If the retry fails too, the history ends with an ``ERROR`` AIMessage so the
    debate can go on (the peer's clone may still carry partial edits).
    """
    try:
        return agent.invoke({"messages": ctx}, config={"recursion_limit": _RECURSION_LIMIT})["messages"]
    except Exception as e:
        if not _is_bad_request(e):
            raise
    repaired = list(ctx)
    while repaired and getattr(repaired[-1], "type", None) == "tool":
        repaired.pop()
    if repaired and getattr(repaired[-1], "type", None) == "ai":
        repaired.pop()
    repaired.append(HumanMessage(content=task.RETRY_NUDGE))
    try:
        return agent.invoke({"messages": repaired}, config={"recursion_limit": _RECURSION_LIMIT})["messages"]
    except Exception as e2:
        repaired.append(AIMessage(content=f"ERROR: peer crashed on retry: {type(e2).__name__}: {e2}"))
        return repaired


def _round_node(state: DebateState) -> dict:
    """One debate round: every peer in turn, its tools bound to its own clone."""
    agent = _build_agent()
    r = int(state.get("round", 0))
    contexts = [list(c) for c in state["contexts"]]
    prev_finals = state.get("round_finals") or []
    this_round_finals: list[BaseMessage] = []
    for i, workdir in enumerate(state["peer_workdirs"]):
        ctx = contexts[i]
        if r > 0 and prev_finals:
            others = [prev_finals[-1][j] for j in range(len(contexts)) if j != i]
            ctx = ctx + [_peer_injection(others, state["brief"])]
        token = WORKDIR.bind(Path(workdir))
        try:
            try:
                new_msgs = _invoke_peer_with_recovery(agent, ctx)
            except Exception as e:
                new_msgs = ctx + [AIMessage(content=f"ERROR: peer {i} round {r} crashed: {type(e).__name__}: {e}")]
        finally:
            WORKDIR.reset(token)
        contexts[i] = new_msgs
        this_round_finals.append(_last_ai(new_msgs) or AIMessage(content=""))
    return {"contexts": contexts, "round_finals": prev_finals + [this_round_finals], "round": r + 1}


def _route(state: DebateState) -> str:
    return END if int(state.get("round", 0)) >= N_ROUNDS else "round"


def _build_graph():
    g = StateGraph(DebateState)
    g.add_node("round", _round_node)
    g.add_edge(START, "round")
    g.add_conditional_edges("round", _route, {"round": "round", END: END})
    return g.compile()


def _init_contexts(n: int, brief: str) -> list[list[BaseMessage]]:
    """Each peer starts from the brief alone; the agent supplies the system prompt."""
    return [[HumanMessage(content=brief)] for _ in range(n)]


def solve(instance: dict, peer_workdirs: list[Path], eval_mode: str = "singularity") -> dict:
    """Run the debate on one instance; peer k edits ``peer_workdirs[k]`` (N clones the caller prepared).

    Returns ``{"patch", "resolved", "winner", "per_peer", "all_contexts",
    "telemetry"}``: the voted peer's patch, whether it resolves the instance
    (None without evaluation), its index, every peer's ``{peer, workdir,
    patch, report, resolved, score}`` (the last three set for the voted peer
    when evaluated), the peers' histories and token/call counts over all of them.
    """
    brief = format_task_brief(
        instance["problem_statement"], instance_id=instance.get("instance_id"), hints_text=instance.get("hints_text")
    )
    init_state: DebateState = {
        "contexts": _init_contexts(N_AGENTS, brief),
        "round_finals": [],
        "round": 0,
        "brief": brief,
        "peer_workdirs": list(peer_workdirs),
    }
    result = _build_graph().invoke(init_state, config={"recursion_limit": 200})
    contexts = result.get("contexts") or []
    per_peer = [
        {"peer": i, "workdir": str(w), "patch": compute_patch(Path(w)), "report": None, "resolved": None, "score": None}
        for i, w in enumerate(peer_workdirs)
    ]
    winner = task.select_patch([p["patch"] for p in per_peer])
    evaluated = eval_mode != "none"
    if evaluated:
        task.score_selected(per_peer[winner], instance)
    return {
        "patch": per_peer[winner]["patch"],
        "resolved": per_peer[winner]["resolved"] if evaluated else None,
        "winner": winner,
        "per_peer": per_peer,
        "all_contexts": contexts,
        "telemetry": normalize(langchain_telemetry([m for ctx in contexts for m in ctx])),
    }


def run_one(instance: dict, workdir_root: Path, out_dir: Path, eval_mode: str = "singularity") -> dict:
    """Clone one repository per peer under ``workdir_root/<id>``, run the debate and write its artifacts."""
    iid = instance["instance_id"]
    summary = task.record_head(instance, n_peers=N_AGENTS, n_rounds=N_ROUNDS)
    peer_workdirs = [workdir_root / iid / f"peer_{i}" for i in range(N_AGENTS)]
    start = time.time()
    for i, workdir in enumerate(peer_workdirs):
        err = clone_and_checkout(instance["repo"], instance["base_commit"], workdir)
        if err:
            return {**summary, "error": err, "stage": f"clone/peer_{i}", "clone_s": round(time.time() - start, 1)}
        swe_sandbox.register_worktree(workdir, lambda: _ensure_sif(iid))
    summary["clone_s"] = round(time.time() - start, 1)
    start = time.time()
    try:
        out = solve(instance, peer_workdirs, eval_mode=eval_mode)
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        return {**summary, "error": error, "stage": "solve", "solve_s": round(time.time() - start, 1)}
    summary["solve_s"] = round(time.time() - start, 1)
    summary.update(out.get("telemetry") or {})
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
        description="Decentralized-topology SWE-bench Verified agent (LangGraph debate).",
        run_batch=_run_instances,
        default_out_dir=DEFAULT_OUT_DIR,
        eval_modes=("singularity", "none"),
        skip_eval=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
