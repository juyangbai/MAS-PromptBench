"""Decentralized APPS runner (LangGraph): N peers debate for R rounds.

Every peer is the same python_exec ReAct debater with its own history. From the
second round on, each peer sees the other peers' previous final outputs and may
revise. The submission is the most common whitespace-normalized final-round
program (peers without one abstain; ties: lowest peer, see :mod:`core.voting`);
with tests, only that program is run.
N is the team size (``configs/teams/<dataset>.yaml``, r=4; ``DECENTRALIZED_N_AGENTS``
overrides) and R is ``DECENTRALIZED_N_ROUNDS`` (default 2).
``teamsizes/decentralized/apps`` runs this module with r = 2, 4, 8 and 10.
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent
from typing_extensions import TypedDict

from core import cli, code_tasks, prompts, settings, teams
from core.batch import attempt
from core.code_tasks import exact_match_score, extract_code  # noqa: F401  (runner API)
from core.llm import chat_openai
from core.paths import RESULTS_DIR
from core.tasks import apps as task
from core.tasks.apps import format_prompt, load_instances, run_tests  # noqa: F401
from core.telemetry import langchain_telemetry, normalize

TOPOLOGY = "decentralized"
TEAM_SIZE = globals().get("TEAM_SIZE")  # preset by the teamsizes/ variants (core.variant)
TEAM = teams.spec(TOPOLOGY, task.DATASET, TEAM_SIZE)
DEFAULT_PREDICTIONS = (
    RESULTS_DIR
    / ("apps_decentralized_langgraph" if TEAM_SIZE is None else f"apps_decentralized_r{TEAM_SIZE}")
    / "predictions.jsonl"
)

VLLM_BASE_URL = settings.base_url()
MODEL_ID = settings.model_id()
N_AGENTS = settings.decentralized_n_agents(TEAM.n_agents)
N_ROUNDS = settings.decentralized_n_rounds(TEAM.n_rounds)
_RECURSION_LIMIT = TEAM.recursion_limit  # each peer turn's ReAct loop


def _load_prompt(role: str) -> str:
    return prompts.role_prompt(TOPOLOGY, task.DATASET, role)


SYSTEM_PROMPT = _load_prompt(TEAM.role)

python_exec = tool(code_tasks.make_python_exec(code_tasks.PYTHON_EXEC_DOC))


def _build_llm() -> ChatOpenAI:
    return chat_openai(model=MODEL_ID, base_url=VLLM_BASE_URL)


def _build_agent():
    """The debater agent shared by all peers of a round (each peer passes its own history)."""
    return create_react_agent(model=_build_llm(), tools=[python_exec], prompt=SYSTEM_PROMPT)


def _peer_injection(others_final: list[BaseMessage], prompt: str) -> HumanMessage:
    """The message showing a peer the other peers' previous final outputs."""
    body = ["These are the final solutions from other peer agents in the previous round:"]
    for i, m in enumerate(others_final):
        content = getattr(m, "content", "") or ""
        if not isinstance(content, str):
            content = str(content)
        body.append(f"\nPeer {i + 1}:\n```\n{content}\n```")
    body.append(task.PEER_REVISION_NOTE + prompt)
    return HumanMessage(content="\n".join(body))


class DebateState(TypedDict, total=False):
    contexts: list[list[BaseMessage]]  # per-peer message histories (without the system prompt)
    round_finals: list[list[BaseMessage]]  # per round, each peer's final AIMessage
    round: int
    prompt: str


def _last_ai(msgs: list[BaseMessage]) -> BaseMessage | None:
    """The last AIMessage with content and no tool calls, else the last AIMessage."""
    for m in reversed(msgs):
        if isinstance(m, AIMessage) and (m.content or "") and not getattr(m, "tool_calls", None):
            return m
    for m in reversed(msgs):
        if isinstance(m, AIMessage):
            return m
    return None


def _round_node(state: DebateState) -> dict:
    agent = _build_agent()
    r = int(state.get("round", 0))
    contexts = [list(c) for c in state["contexts"]]
    prev_finals = state.get("round_finals") or []
    this_round_finals: list[BaseMessage] = []
    for i in range(len(contexts)):
        ctx = contexts[i]
        if r > 0 and prev_finals:
            others = [prev_finals[-1][j] for j in range(len(contexts)) if j != i]
            ctx = ctx + [_peer_injection(others, state["prompt"])]
        result = agent.invoke({"messages": ctx}, config={"recursion_limit": _RECURSION_LIMIT})
        contexts[i] = result["messages"]
        this_round_finals.append(_last_ai(contexts[i]) or AIMessage(content=""))
    return {"contexts": contexts, "round_finals": prev_finals + [this_round_finals], "round": r + 1}


def _route(state: DebateState) -> str:
    return END if int(state.get("round", 0)) >= N_ROUNDS else "round"


def _build_graph():
    g = StateGraph(DebateState)
    g.add_node("round", _round_node)
    g.add_edge(START, "round")
    g.add_conditional_edges("round", _route, {"round": "round", END: END})
    return g.compile()


def _init_contexts(n: int, user_prompt: str) -> list[list[BaseMessage]]:
    """Each peer starts from the problem alone; the agent supplies the system prompt."""
    return [[HumanMessage(content=user_prompt)] for _ in range(n)]


def _peer_code(ctx: list[BaseMessage], final) -> str | None:
    """The program of the peer's final output, else the latest program in its history."""
    code = extract_code(final if isinstance(final, str) else "")
    if code is not None:
        return code
    for m in reversed(ctx):
        content = getattr(m, "content", "") or ""
        code = extract_code(content if isinstance(content, str) else "")
        if code:
            return code
    return None


def solve(problem: str, starter_code: str | None = None, input_output: dict | None = None) -> dict:
    """Run the debate on one problem.

    Returns ``{"code", "winner", "per_peer", "all_contexts", "telemetry"}``: the
    voted program and its peer, each peer's ``{peer, code, raw}`` (with tests, the
    selected peer's test report; the vote never sees the tests), the peers'
    histories and token/call counts over all of them.
    """
    compiled = _build_graph()
    user_prompt = format_prompt(problem, starter_code)
    result = compiled.invoke(
        {"contexts": _init_contexts(N_AGENTS, user_prompt), "round_finals": [], "round": 0, "prompt": user_prompt}
    )
    contexts = result.get("contexts") or []
    per_peer = []
    for i, ctx in enumerate(contexts):
        final_msg = _last_ai(ctx)
        final = getattr(final_msg, "content", "") or "" if final_msg else ""
        per_peer.append({"peer": i, "code": _peer_code(ctx, final), "raw": final})
    telemetry = normalize(langchain_telemetry([m for ctx in contexts for m in ctx]))
    winner = code_tasks.select_program([p["code"] for p in per_peer])
    code = per_peer[winner]["code"]
    if input_output:
        task.score_selected_peer(per_peer, winner, code, input_output)
    return {"code": code, "winner": winner, "per_peer": per_peer, "all_contexts": contexts, "telemetry": telemetry}


def run_batch(
    instances: list[dict],
    out_path: Path | None = None,
    verbose: bool = True,
    per_test_timeout_s: int = task.TEST_TIMEOUT_S,
) -> dict:
    """Solve and score every instance (majority vote over the final-round programs)."""

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
            error=error,
        )

    label = f"decentralized/APPS (N={N_AGENTS}, R={N_ROUNDS})"
    return task.run_batch(instances, row, out_path=out_path, verbose=verbose, label=label)


def _canned_demo() -> None:
    for mode, problem, starter, tests in task.DEMOS:
        print(f"\n========== {mode} MODE ==========")
        out = solve(problem, starter_code=starter, input_output=tests)
        print(f"=== Debate: N={N_AGENTS} peers x R={N_ROUNDS} rounds, winner: peer {out['winner']} ===")
        task.print_demo_code(out["code"], tests)


def main(argv: list[str] | None = None) -> int:
    return cli.main(
        argv,
        description="Decentralized-topology APPS runner (LangGraph debate).",
        load_instances=load_instances,
        run_batch=run_batch,
        demo=_canned_demo,
        source=task.SOURCE,
        predictions=DEFAULT_PREDICTIONS,
        add_arguments=task.add_arguments,
    )


if __name__ == "__main__":
    raise SystemExit(main())
