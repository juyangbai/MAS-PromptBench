"""Decentralized HotpotQA runner (LangGraph): N peers debate for R rounds.

Every peer is the same Wikipedia ReAct debater with its own history. From the
second round on, each peer sees the other peers' previous final responses and
may revise; the answer is the normalized-answer majority of the final round. N
is the team size (``configs/teams/hotpotqa.yaml``, r=4; ``DECENTRALIZED_N_AGENTS``
overrides) and R is ``DECENTRALIZED_N_ROUNDS`` (default 2).
``teamsizes/decentralized/hotpotqa`` runs this module with r = 2, 4, 8 and 10.
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
from core.communication import CommPolicy
from core.llm import chat_openai
from core.paths import RESULTS_DIR
from core.tasks import hotpotqa as task
from core.tasks.hotpotqa import (  # noqa: F401  (runner API)
    best_of_n,
    exact_match_score,
    extract_answer,
    f1_score,
    load_instances,
    normalize_answer,
)
from core.telemetry import langchain_telemetry, normalize

TOPOLOGY = "decentralized"
TEAM_SIZE = globals().get("TEAM_SIZE")  # preset by the teamsizes/ variants (core.variant)
TEAM = teams.spec(TOPOLOGY, task.DATASET, TEAM_SIZE)
DEFAULT_PREDICTIONS = (
    RESULTS_DIR
    / ("hotpotqa_decentralized_langgraph" if TEAM_SIZE is None else f"hotpotqa_decentralized_r{TEAM_SIZE}")
    / "predictions.jsonl"
)

VLLM_BASE_URL = settings.base_url()
MODEL_ID = settings.model_id()
COMMUNICATION_FORMAT = globals().get("COMMUNICATION_FORMAT", "freeform")  # preset by the communications/ entries
COMMUNICATION = CommPolicy(COMMUNICATION_FORMAT, task.DATASET, TOPOLOGY)
N_AGENTS = settings.decentralized_n_agents(TEAM.n_agents)
N_ROUNDS = settings.decentralized_n_rounds(TEAM.n_rounds)

# Each peer turn's ReAct recursion limit: about three messages per tool loop,
# for the four tool loops of the Agents SDK debate.
_RECURSION_LIMIT = TEAM.recursion_limit

_OUTPUT_FORMAT_NUDGE = task.OUTPUT_FORMAT_NUDGE


def _load_prompt(role: str) -> str:
    return prompts.role_prompt(TOPOLOGY, task.DATASET, role)


SYSTEM_PROMPT = COMMUNICATION.system_prompt(_load_prompt(TEAM.role) + _OUTPUT_FORMAT_NUDGE)

wikipedia_search = tool(task.make_wikipedia_search(task.SEARCH_DOC_SHORT))
wikipedia_page = tool(task.make_wikipedia_page(task.PAGE_DOC_SHORT, options_label="options:"))
TOOLS = [wikipedia_search, wikipedia_page]


def _build_llm() -> ChatOpenAI:
    return chat_openai(model=MODEL_ID, base_url=VLLM_BASE_URL)


def _build_agent():
    """The debater agent shared by all peers of a round (each peer passes its own history)."""
    return create_react_agent(model=_build_llm(), tools=TOOLS, prompt=SYSTEM_PROMPT)


def _peer_report(i: int, content: str) -> str:
    """Peer ``i``'s previous final response as the other peers see it."""
    if COMMUNICATION_FORMAT == "freeform":
        return f"\nPeer {i + 1}:\n```\n{content}\n```"
    rendered = COMMUNICATION.handoff(
        f"peer_{i + 1}",
        content,
        next_action="Use this previous-round peer report when deciding whether to revise.",
        payload={"handoff": "peer_previous_round", "peer": i + 1},
    )
    return f"\nPeer {i + 1}:\n{rendered}"


def _peer_injection(others_final: list[BaseMessage], question: str) -> HumanMessage:
    """The message showing a peer the other peers' previous final responses."""
    body = ["These are the final responses from other peer agents in the previous round:"]
    for i, m in enumerate(others_final):
        content = getattr(m, "content", "") or ""
        body.append(_peer_report(i, content if isinstance(content, str) else str(content)))
    body.append(
        "\nCompare their reasoning + Wikipedia evidence against your own. "
        "Revise your answer ONLY if a peer cites concretely stronger "
        "evidence. Re-emit a single `Answer: <short-form>` line at the "
        "end.\n\nOriginal question:\n" + question
    )
    return HumanMessage(content="\n".join(body))


class DebateState(TypedDict, total=False):
    contexts: list[list[BaseMessage]]  # per-peer message histories (without the system prompt)
    round_finals: list[list[BaseMessage]]  # per round, each peer's final AIMessage
    round: int
    question: str


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
            ctx = ctx + [_peer_injection(others, state["question"])]
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


def _init_contexts(n: int, question: str) -> list[list[BaseMessage]]:
    """Each peer starts from the question alone; the agent supplies the system prompt."""
    return [[HumanMessage(content=question)] for _ in range(n)]


def _peer_answer(ctx: list[BaseMessage], final: str) -> str | None:
    """The answer of a peer's final response, else of its latest message that has one."""
    answer = extract_answer(final)
    if answer is not None:
        return answer
    for m in reversed(ctx):
        content = getattr(m, "content", "") or ""
        answer = extract_answer(content if isinstance(content, str) else "")
        if answer is not None:
            return answer
    return None


def solve(question: str) -> dict:
    """Run the debate on one question.

    Returns ``{"answer", "per_peer", "all_contexts", "telemetry"}``: the majority
    final-round answer, each peer's ``{peer, answer, raw}``, the peers' histories
    and token/call counts over all of them.
    """
    init_state: DebateState = {
        "contexts": _init_contexts(N_AGENTS, question),
        "round_finals": [],
        "round": 0,
        "question": question,
    }
    result = _build_graph().invoke(init_state)
    contexts = result.get("contexts") or []
    per_peer = []
    for i, ctx in enumerate(contexts):
        final_msg = _last_ai(ctx)
        final = getattr(final_msg, "content", "") or "" if final_msg else ""
        per_peer.append({"peer": i, "answer": _peer_answer(ctx, final), "raw": final})
    return {
        "answer": best_of_n([p["answer"] for p in per_peer]),
        "per_peer": per_peer,
        "all_contexts": contexts,
        "telemetry": normalize(langchain_telemetry([m for ctx in contexts for m in ctx])),
    }


def run_batch(instances: list[dict], out_path: Path | None = None, verbose: bool = True) -> dict:
    """Solve and score every instance; records keep the last 300 characters of each peer's response."""

    def row(_, inst: dict) -> dict:
        out, latency_s, error = attempt(lambda: solve(inst["question"]))
        return task.record(
            inst,
            out["answer"],
            per_peer=[
                {"peer": p["peer"], "answer": p["answer"], "raw_tail": (p["raw"] or "")[-300:]}
                for p in out.get("per_peer") or []
            ],
            **task.meta(inst),
            latency_s=round(latency_s, 2),
            **(out.get("telemetry") or {}),
            error=error,
        )

    return task.run_batch(
        instances,
        row,
        out_path=out_path,
        verbose=verbose,
        label="decentralized/HotpotQA",
        team=f"(N={N_AGENTS} peers x R={N_ROUNDS} rounds)",
        marks=task.ASCII_MARKS,
        width=30,
        detail=task.peers_detail,
    )


def _canned_demo() -> None:
    out = solve(task.DEMO_QUESTION)
    print(f"\n=== Debate: N={N_AGENTS} peers x R={N_ROUNDS} rounds ===")
    for p in out["per_peer"]:
        print(f"  peer {p['peer']}: answer={p['answer'] or '(none)'!r}")
    task.print_demo_answer(out["answer"], label="Majority-vote final answer")


def main(argv: list[str] | None = None) -> int:
    return cli.main(
        argv,
        description="Decentralized-topology HotpotQA runner (LangGraph debate).",
        load_instances=load_instances,
        run_batch=run_batch,
        demo=_canned_demo,
        source=task.SOURCE,
        predictions=DEFAULT_PREDICTIONS,
    )


if __name__ == "__main__":
    raise SystemExit(main())
