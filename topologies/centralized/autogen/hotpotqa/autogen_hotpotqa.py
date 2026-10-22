"""Centralized HotpotQA runner (AutoGen): a manager and Wikipedia workers in a SelectorGroupChat.

The workers are the r=4 centralized team (``configs/teams/hotpotqa.yaml``). Control
returns to the manager after every other speaker; the manager may hand the turn
to a worker, and the chat ends on ``TERMINATE`` or after the team's
``max_turns`` messages. The answer is the ``Answer:`` line of the manager's last message.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage
from autogen_agentchat.teams import SelectorGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient

from core import cli, prompts, settings, teams
from core.batch import attempt
from core.llm import autogen_client
from core.tasks import hotpotqa as task
from core.tasks.hotpotqa import exact_match_score, f1_score, load_instances, normalize_answer  # noqa: F401
from core.tasks.hotpotqa import extract_manager_answer as extract_answer
from core.telemetry import autogen_telemetry, normalize

TOPOLOGY = "centralized"
TEAM = teams.spec(TOPOLOGY, task.DATASET)

VLLM_BASE_URL = settings.base_url()
MODEL_ID = settings.model_id()

# Stall guards: on hard bridge questions the retriever can re-fetch pages for
# many turns, so each row gets a wall-clock cap and the chat a message cap.
PER_ROW_TIMEOUT_S = 120
_MAX_MESSAGES = TEAM.max_turns

_MANAGER_TERMINATE_NUDGE = task.TERMINATE_NUDGE

wikipedia_search = task.make_wikipedia_search(task.SEARCH_DOC_CENTRALIZED)
wikipedia_page = task.make_wikipedia_page(task.PAGE_DOC_CENTRALIZED)

# AutoGen agent descriptions, which the selector shows as the roster.
_DESCRIPTIONS = {
    "manager": "Coordinator that plans the multi-hop retrieval, delegates, and writes the final answer.",
    "retriever_worker": "Performs one Wikipedia lookup per manager instruction and returns structured facts.",
    "reasoner_worker": "Chains retrieved facts into a logical derivation per manager instruction.",
    "writer_worker": "Formats the final short-form answer per manager instruction.",
}
_SELECTOR_PROMPT = (
    "You are coordinating a 4-agent team on a multi-hop Wikipedia question.\n"
    "Select the next agent to act.\n\n{roles}\n\n"
    "Conversation so far:\n{history}\n\n"
    "Pick exactly one agent from {participants}."
)


def _load_prompt(role: str) -> str:
    return prompts.role_prompt(TOPOLOGY, task.DATASET, role)


def _build_client() -> OpenAIChatCompletionClient:
    return autogen_client(model=MODEL_ID, base_url=VLLM_BASE_URL)


def build_team() -> SelectorGroupChat:
    """The manager and its workers, all sharing one client and the Wikipedia tools."""
    client = _build_client()
    agents = [
        AssistantAgent(
            role,
            description=_DESCRIPTIONS[role],
            model_client=client,
            system_message=_load_prompt(role) + (_MANAGER_TERMINATE_NUDGE if role == TEAM.manager else ""),
            tools=[wikipedia_search, wikipedia_page],
        )
        for role in TEAM.roles
    ]
    manager = agents[0]

    def _selector_func(messages: Sequence[BaseAgentEvent | BaseChatMessage]) -> str | None:
        if not messages or messages[-1].source != manager.name:
            return manager.name
        return None

    termination = TextMentionTermination("TERMINATE") | MaxMessageTermination(_MAX_MESSAGES)
    return SelectorGroupChat(
        agents,
        model_client=client,
        termination_condition=termination,
        selector_prompt=_SELECTOR_PROMPT,
        selector_func=_selector_func,
        allow_repeated_speaker=True,
    )


async def solve_async(question: str) -> dict:
    """Run the team on one question.

    Returns ``{"answer", "raw", "messages", "telemetry"}``: the short-form answer
    of the manager's last message, that message, every message as
    ``{source, content}`` and token/call counts.
    """
    result = await asyncio.wait_for(build_team().run(task=question), timeout=PER_ROW_TIMEOUT_S)
    messages = [
        {
            "source": getattr(m, "source", None),
            "content": getattr(m, "content", None)
            if isinstance(getattr(m, "content", None), str)
            else str(getattr(m, "content", "")),
        }
        for m in result.messages
    ]
    manager_msgs = [m for m in messages if m["source"] == "manager"]
    final = manager_msgs[-1]["content"] if manager_msgs else ""
    return {
        "answer": extract_answer(final),
        "raw": final,
        "messages": messages,
        "telemetry": normalize(autogen_telemetry(result)),
    }


def solve(question: str) -> dict:
    return asyncio.run(solve_async(question))


def run_batch(instances: list[dict], out_path: Path | None = None, verbose: bool = True) -> dict:
    """Solve and score every instance."""

    def row(_, inst: dict) -> dict:
        out, latency_s, error = attempt(lambda: solve(inst["question"]))
        return task.record(
            inst,
            out["answer"],
            **task.meta(inst),
            raw=out.get("raw") or "",
            n_messages=len(out.get("messages") or []),
            latency_s=round(latency_s, 2),
            **(out.get("telemetry") or {}),
            error=error,
        )

    return task.run_batch(
        instances, row, out_path=out_path, verbose=verbose, label="centralized/HotpotQA", detail=task.messages_detail
    )


def _canned_demo() -> None:
    out = solve(task.DEMO_QUESTION)
    task.print_demo_answer(out["answer"])
    print(f"=== {len(out['messages'])} messages across the group chat ===")
    for m in out["messages"]:
        snippet = m["content"][:300].replace("\n", " ")
        print(f"  [{m['source']}] {snippet}")


def main(argv: list[str] | None = None) -> int:
    return cli.main(
        argv,
        description="Centralized-topology HotpotQA runner (AutoGen).",
        load_instances=load_instances,
        run_batch=run_batch,
        demo=_canned_demo,
        source=task.SOURCE,
    )


if __name__ == "__main__":
    raise SystemExit(main())
