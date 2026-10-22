"""Decentralized GPQA-Diamond runner (LangGraph): N peers debate for R rounds.

Every peer is the same calculator ReAct debater with its own history. From the
second round on, each peer sees the other peers' previous final responses and
may revise; the answer is the majority letter of the final round (ties: lowest
peer). N is the team size (``configs/teams/gpqa.yaml``, r=4;
``DECENTRALIZED_N_AGENTS`` overrides) and R is ``DECENTRALIZED_N_ROUNDS``
(default 2). ``teamsizes/decentralized/gpqa`` runs this module with r = 2, 4, 8 and 10.
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
from core.calculator import CALCULATOR_DOC_BRIEF, make_calculator
from core.llm import chat_openai
from core.paths import RESULTS_DIR
from core.runtime import row_timeout
from core.tasks import gpqa as task
from core.tasks.gpqa import best_of_n, extract_answer, load_instances
from core.telemetry import langchain_telemetry, normalize

TOPOLOGY = "decentralized"
TEAM_SIZE = globals().get("TEAM_SIZE")  # preset by the teamsizes/ variants (core.variant)
TEAM = teams.spec(TOPOLOGY, task.DATASET, TEAM_SIZE)
DEFAULT_PREDICTIONS = (
    RESULTS_DIR
    / ("gpqa_decentralized_langgraph" if TEAM_SIZE is None else f"gpqa_decentralized_r{TEAM_SIZE}")
    / "predictions.jsonl"
)

VLLM_BASE_URL = settings.base_url()
MODEL_ID = settings.model_id()
N_AGENTS = settings.decentralized_n_agents(TEAM.n_agents)
N_ROUNDS = settings.decentralized_n_rounds(TEAM.n_rounds)

# Stall guards: each row gets a wall-clock cap and each peer turn a ReAct recursion limit.
PER_ROW_TIMEOUT_S = 120
_RECURSION_LIMIT = TEAM.recursion_limit


def _load_prompt(role: str) -> str:
    return prompts.role_prompt(TOPOLOGY, task.DATASET, role)


SYSTEM_PROMPT = _load_prompt(TEAM.role)

calculator = tool(make_calculator(CALCULATOR_DOC_BRIEF))
TOOLS = [calculator]

format_mcq = task.format_prompt


def _build_llm() -> ChatOpenAI:
    return chat_openai(model=MODEL_ID, base_url=VLLM_BASE_URL)


def _build_agent():
    """The debater agent shared by all peers of a round (each peer passes its own history)."""
    return create_react_agent(model=_build_llm(), tools=TOOLS, prompt=SYSTEM_PROMPT)


def _text(msg: BaseMessage | None) -> str:
    """A message's content as text ("" for no message or empty content)."""
    content = getattr(msg, "content", "") or "" if msg is not None else ""
    return content if isinstance(content, str) else str(content)


def _peer_injection(others_final: list[BaseMessage], question: str) -> HumanMessage:
    """The message showing a peer the other peers' previous final responses."""
    return HumanMessage(content=task.peer_review_prompt([_text(m) for m in others_final], question))


class DebateState(TypedDict, total=False):
    contexts: list[list[BaseMessage]]  # per-peer message histories (without the system prompt)
    round_finals: list[list[BaseMessage]]  # per round, each peer's final AIMessage
    round: int
    mcq: str


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
            ctx = ctx + [_peer_injection(others, state["mcq"])]
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


def _init_contexts(n: int, mcq: str) -> list[list[BaseMessage]]:
    """Each peer starts from the question alone; the agent supplies the system prompt."""
    return [[HumanMessage(content=mcq)] for _ in range(n)]


def solve(question: str, choices: list[str]) -> dict:
    """Run the debate on one question.

    Returns ``{"answer", "per_peer", "all_contexts", "telemetry"}``: the majority
    final-round letter, each peer's ``{peer, letter, raw}``, the peers' histories
    and token/call counts over all of them.
    """
    compiled = _build_graph()
    mcq = format_mcq(question, choices)
    init_state: DebateState = {"contexts": _init_contexts(N_AGENTS, mcq), "round_finals": [], "round": 0, "mcq": mcq}
    with row_timeout(PER_ROW_TIMEOUT_S):
        result = compiled.invoke(init_state)
    contexts = result.get("contexts") or []
    per_peer = []
    for i, ctx in enumerate(contexts):
        final = _text(_last_ai(ctx))
        per_peer.append({"peer": i, "letter": extract_answer(final), "raw": final})
    return {
        "answer": best_of_n(p["letter"] for p in per_peer),
        "per_peer": per_peer,
        "all_contexts": contexts,
        "telemetry": normalize(langchain_telemetry([m for ctx in contexts for m in ctx])),
    }


def run_batch(instances: list[dict], out_path: Path | None = None, verbose: bool = True) -> dict:
    """Solve and score every instance."""

    def row(_, inst: dict) -> dict:
        out, latency_s, error = attempt(lambda: solve(inst["question"], inst["choices"]))
        return task.record(
            inst,
            out["answer"],
            per_peer=task.per_peer_tails(out.get("per_peer") or []),
            latency_s=round(latency_s, 2),
            **(out.get("telemetry") or {}),
            error=error,
        )

    label = f"decentralized/GPQA-Diamond (N={N_AGENTS}, R={N_ROUNDS})"
    return task.run_batch(instances, row, out_path=out_path, verbose=verbose, label=label)


def _canned_demo() -> None:
    out = solve(task.DEMO_QUESTION, task.DEMO_CHOICES)
    print(f"\n=== Debate: N={N_AGENTS} peers x R={N_ROUNDS} rounds ===")
    for p in out["per_peer"]:
        print(f"  peer {p['peer']}: letter={p['letter']}")
    task.print_demo_answer(out["answer"])


def main(argv: list[str] | None = None) -> int:
    return cli.main(
        argv,
        description="Decentralized-topology GPQA-Diamond runner (LangGraph debate).",
        load_instances=load_instances,
        run_batch=run_batch,
        demo=_canned_demo,
        source=task.SOURCE,
        predictions=DEFAULT_PREDICTIONS,
        add_arguments=task.add_arguments,
    )


if __name__ == "__main__":
    raise SystemExit(main())
