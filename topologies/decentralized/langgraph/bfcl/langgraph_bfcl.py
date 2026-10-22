"""Decentralized BFCL runner (LangGraph): N peers debate for R rounds.

Every peer answers with one model call per round on its own history; from the
second round on, each peer sees the other peers' previous final calls and may
revise. The prediction is the most common canonical final call list (peers
without a parseable call abstain; ties: lowest peer, see :mod:`core.voting`),
and only that call list is scored. N is the team size (``configs/teams/bfcl.yaml``,
r=4; ``DECENTRALIZED_N_AGENTS`` overrides) and R is ``DECENTRALIZED_N_ROUNDS``
(default 2). ``teamsizes/decentralized/bfcl`` runs this module with r = 2, 4, 8 and 10.
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from core import prompts, settings, teams
from core.communication import CommPolicy
from core.llm import chat_openai
from core.paths import RESULTS_DIR
from core.tasks import bfcl as task
from core.tasks.bfcl import AST_CATEGORIES, HF_DATASET, extract_canonical, load_instances  # noqa: F401
from core.telemetry import langchain_telemetry, normalize

TOPOLOGY = "decentralized"
TEAM_SIZE = globals().get("TEAM_SIZE")  # preset by the teamsizes/ variants (core.variant)
TEAM = teams.spec(TOPOLOGY, task.DATASET, TEAM_SIZE)
DEFAULT_OUT_DIR = RESULTS_DIR / (
    "bfcl_decentralized_langgraph" if TEAM_SIZE is None else f"bfcl_decentralized_r{TEAM_SIZE}"
)

VLLM_BASE_URL = settings.base_url()
MODEL_ID = settings.model_id()
COMMUNICATION_FORMAT = globals().get("COMMUNICATION_FORMAT", "freeform")  # preset by the communications/ entries
COMMUNICATION = CommPolicy(COMMUNICATION_FORMAT, task.DATASET, TOPOLOGY)
N_AGENTS = settings.decentralized_n_agents(TEAM.n_agents)
N_ROUNDS = settings.decentralized_n_rounds(TEAM.n_rounds)


def _load_prompt(role: str) -> str:
    return prompts.role_prompt(TOPOLOGY, task.DATASET, role)


SYSTEM_PROMPT = COMMUNICATION.system_prompt(_load_prompt(TEAM.role))

_register_model_with_bfcl = task.register_model  # called again by callers that repoint MODEL_ID
_register_model_with_bfcl(MODEL_ID)


def _build_llm() -> ChatOpenAI:
    return chat_openai(model=MODEL_ID, base_url=VLLM_BASE_URL, timeout=600.0, max_retries=5)


def score_one(function_schemas: list[dict], model_output: list[dict], ground_truth: list[dict], category: str) -> dict:
    return task.score_one(function_schemas, model_output, ground_truth, category, MODEL_ID)


def _peer_injection(others_final: list[BaseMessage], user_content: str) -> HumanMessage:
    """The message showing a peer the other peers' previous final calls."""
    body = ["These are the final calls from other peer agents in the previous round:"]
    for i, m in enumerate(others_final):
        content = getattr(m, "content", "") or ""
        if not isinstance(content, str):
            content = str(content)
        body.append(f"\nPeer {i + 1}:\n```\n{content}\n```")
    body.append(
        "\nCompare their function calls against yours. Revise ONLY if a peer "
        "picked a better function or caught an error in yours. Re-emit your "
        "final canonical call as a SINGLE fenced ```json``` block at the "
        "end.\n\nOriginal request:\n" + user_content
    )
    return HumanMessage(content="\n".join(body))


class DebateState(TypedDict, total=False):
    contexts: list[list[BaseMessage]]  # per-peer message histories (without the system prompt)
    round_finals: list[list[BaseMessage]]  # per round, each peer's final AIMessage
    round: int
    user_content: str


def _last_ai(msgs: list[BaseMessage]) -> BaseMessage | None:
    """The last AIMessage with content, else the last AIMessage."""
    for m in reversed(msgs):
        if isinstance(m, AIMessage) and (m.content or ""):
            return m
    for m in reversed(msgs):
        if isinstance(m, AIMessage):
            return m
    return None


def _round_node(state: DebateState) -> dict:
    llm = _build_llm()
    r = int(state.get("round", 0))
    contexts = [list(c) for c in state["contexts"]]
    prev_finals = state.get("round_finals") or []
    this_round_finals: list[BaseMessage] = []
    for i in range(len(contexts)):
        ctx = contexts[i]
        if r > 0 and prev_finals:
            others = [prev_finals[-1][j] for j in range(len(contexts)) if j != i]
            ctx = ctx + [_peer_injection(others, state["user_content"])]
        resp = llm.invoke([SystemMessage(content=SYSTEM_PROMPT)] + ctx)
        contexts[i] = ctx + [resp]
        this_round_finals.append(resp)
    return {"contexts": contexts, "round_finals": prev_finals + [this_round_finals], "round": r + 1}


def _route(state: DebateState) -> str:
    return END if int(state.get("round", 0)) >= N_ROUNDS else "round"


def _build_graph():
    g = StateGraph(DebateState)
    g.add_node("round", _round_node)
    g.add_edge(START, "round")
    g.add_conditional_edges("round", _route, {"round": "round", END: END})
    return g.compile()


def _init_contexts(n: int, user_content: str) -> list[list[BaseMessage]]:
    """Each peer starts from the task alone; the round node prepends the system prompt."""
    return [[HumanMessage(content=user_content)] for _ in range(n)]


def solve(instance: dict) -> dict:
    """Run the debate on one instance.

    Returns ``{"model_output", "winner", "per_peer", "all_contexts", "telemetry"}``:
    the voted call list (or []), its peer, each peer's ``{peer, call, raw}``, the
    peers' histories and token/call counts over all of them.
    """
    user_content = task.format_debate_task(
        task.flatten_question(instance["question"]), task.render_schemas(instance["function"])
    )
    state: DebateState = {
        "contexts": _init_contexts(N_AGENTS, user_content),
        "round_finals": [],
        "round": 0,
        "user_content": user_content,
    }
    contexts = _build_graph().invoke(state).get("contexts") or []
    per_peer = []
    for i, ctx in enumerate(contexts):
        final_msg = _last_ai(ctx)
        final = getattr(final_msg, "content", "") or "" if final_msg else ""
        if not isinstance(final, str):
            final = str(final)
        per_peer.append({"peer": i, "call": extract_canonical(final), "raw": final})
    telemetry = normalize(langchain_telemetry([m for ctx in contexts for m in ctx]))
    winner = task.select_call([p["call"] for p in per_peer])
    return {
        "model_output": per_peer[winner]["call"] or [],
        "winner": winner,
        "per_peer": per_peer,
        "all_contexts": contexts,
        "telemetry": telemetry,
    }


def run_one(instance: dict, ground_truth: dict, category: str, out_dir: Path) -> dict:
    """Solve and score one instance and write the peers' calls to ``out_dir/traces/<id>.txt``."""
    summary: dict = {"id": instance["id"], "category": category, "n_peers": N_AGENTS, "n_rounds": N_ROUNDS}
    try:
        out = solve(instance)
    except Exception as e:
        return task.solve_failed(summary, e)
    summary["winner"] = out.get("winner")
    summary["model_output"] = out.get("model_output") or []
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
        description="Decentralized-topology BFCL runner (LangGraph debate).",
        run_one=run_one,
        model_id=MODEL_ID,
        default_out_dir=DEFAULT_OUT_DIR,
    )


if __name__ == "__main__":
    raise SystemExit(main())
