"""Decentralized MATH runner (LangGraph): N peers debate for R rounds.

Every peer is the same calculator ReAct debater with its own history. From the
second round on, each peer sees the other peers' previous final answers and may
revise; the answer is the equivalence majority of the final round. N is the team
size (``configs/teams/<dataset>.yaml``, r=4; ``DECENTRALIZED_N_AGENTS`` overrides) and R is
``DECENTRALIZED_N_ROUNDS`` (default 2). ``teamsizes/decentralized/math`` runs
this module with r = 2, 4, 8 and 10.
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent
from typing_extensions import TypedDict

from core import cli, prompts, settings, teams
from core.batch import attempt
from core.calculator import CALCULATOR_DOC, make_calculator
from core.llm import chat_openai
from core.runtime import row_timeout
from core.tasks import math as task
from core.tasks.math import (  # noqa: F401  (runner API)
    best_of_n,
    exact_match_score,
    extract_answer,
    extract_boxed,
    is_equiv,
    load_instances,
)
from core.telemetry import langchain_telemetry, normalize

TOPOLOGY = "decentralized"
TEAM_SIZE = globals().get("TEAM_SIZE")  # preset by the teamsizes/ variants (core.variant)
TEAM = teams.spec(TOPOLOGY, task.DATASET, TEAM_SIZE)

VLLM_BASE_URL = settings.base_url()
MODEL_ID = settings.model_id()
N_AGENTS = settings.decentralized_n_agents(TEAM.n_agents)
N_ROUNDS = settings.decentralized_n_rounds(TEAM.n_rounds)

# Stall guards: symbolic problems make the calculator fail and peers retry, so
# each row gets a wall-clock cap and each peer turn a ReAct recursion limit.
PER_ROW_TIMEOUT_S = 120
_RECURSION_LIMIT = TEAM.recursion_limit


def _load_prompt(role: str) -> str:
    return prompts.role_prompt(TOPOLOGY, task.DATASET, role)


SYSTEM_PROMPT = _load_prompt(TEAM.role)

calculator = tool(make_calculator(CALCULATOR_DOC))
TOOLS = [calculator]


def _build_llm() -> ChatOpenAI:
    return chat_openai(model=MODEL_ID, base_url=VLLM_BASE_URL)


def _build_agent():
    """The debater agent shared by all peers of a round (each peer passes its own history)."""
    return create_react_agent(model=_build_llm(), tools=TOOLS, prompt=SYSTEM_PROMPT)


def _peer_injection(others_final: list[BaseMessage], problem: str) -> HumanMessage:
    """The message showing a peer the other peers' previous final answers."""
    body = ["These are the final solutions from other peer agents in the previous round:"]
    for i, m in enumerate(others_final):
        content = getattr(m, "content", "") or ""
        if not isinstance(content, str):
            content = str(content)
        body.append(f"\nPeer {i + 1}:\n```\n{content}\n```")
    body.append(
        "\nCompare their derivations and final boxed answers against your own. "
        "Revise your answer ONLY if a peer catches an error in your work or "
        "presents concretely stronger reasoning. Re-emit your final answer "
        "inside \\boxed{...} at the end.\n\nOriginal problem:\n" + problem
    )
    return HumanMessage(content="\n".join(body))


class DebateState(TypedDict, total=False):
    contexts: list[list[BaseMessage]]  # per-peer message histories (without the system prompt)
    round_finals: list[list[BaseMessage]]  # per round, each peer's final AIMessage
    round: int
    problem: str


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
            ctx = ctx + [_peer_injection(others, state["problem"])]
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


def _init_contexts(n: int, problem: str) -> list[list[BaseMessage]]:
    """Each peer starts from the problem alone; the agent supplies the system prompt."""
    return [[HumanMessage(content=problem)] for _ in range(n)]


def solve(problem: str) -> dict:
    """Run the debate on one problem.

    Returns ``{"answer", "per_peer", "all_contexts", "telemetry"}``: the majority
    final-round answer, each peer's ``{peer, answer, raw}``, the peers' histories
    and token/call counts over all of them.
    """
    compiled = _build_graph()
    init_state: DebateState = {
        "contexts": _init_contexts(N_AGENTS, problem),
        "round_finals": [],
        "round": 0,
        "problem": problem,
    }
    with row_timeout(PER_ROW_TIMEOUT_S):
        result = compiled.invoke(init_state)
    contexts = result.get("contexts") or []
    per_peer = []
    answers: list[str | None] = []
    for i, ctx in enumerate(contexts):
        final_msg = _last_ai(ctx)
        final = getattr(final_msg, "content", "") or "" if final_msg else ""
        if not isinstance(final, str):
            final = str(final)
        answer = extract_answer(final)
        per_peer.append({"peer": i, "answer": answer, "raw": final})
        if answer is not None:
            answers.append(answer)
    return {
        "answer": best_of_n(answers),
        "per_peer": per_peer,
        "all_contexts": contexts,
        "telemetry": normalize(langchain_telemetry([m for ctx in contexts for m in ctx])),
    }


def run_batch(instances: list[dict], out_path: Path | None = None, verbose: bool = True) -> dict:
    """Solve and score every instance."""

    def row(_, inst: dict) -> dict:
        out, latency_s, error = attempt(lambda: solve(inst["problem"]))
        return task.record(
            inst,
            out["answer"],
            **task.meta(inst),
            per_peer=[
                {"peer": p["peer"], "answer": p["answer"], "raw": (p.get("raw") or "")[:2000]}
                for p in out.get("per_peer") or []
            ],
            latency_s=round(latency_s, 2),
            **(out.get("telemetry") or {}),
            error=error,
        )

    label = f"decentralized/{task.LABEL} (N={N_AGENTS}, R={N_ROUNDS})"
    return task.run_batch(instances, row, out_path=out_path, verbose=verbose, label=label)


def _canned_demo() -> None:
    out = solve(task.DEMO_PROBLEM)
    print(f"\n=== Debate: N={N_AGENTS} peers x R={N_ROUNDS} rounds ===")
    for p in out["per_peer"]:
        print(f"  peer {p['peer']}: boxed={p['answer'] if p['answer'] is not None else '(none)'!r}")
    task.print_demo_answer(out["answer"])


def main(argv: list[str] | None = None) -> int:
    return cli.main(
        argv,
        description="Decentralized-topology MATH runner (LangGraph debate).",
        load_instances=load_instances,
        run_batch=run_batch,
        demo=_canned_demo,
        source=task.SOURCE,
    )


if __name__ == "__main__":
    raise SystemExit(main())
