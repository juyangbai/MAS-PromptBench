"""Centralized topology: 1 manager (PlanningAgent) + 3 workers, hub-and-spoke.

Pattern follows AutoGen's `SelectorGroupChat` tutorial
(python/docs/src/user-guide/agentchat-user-guide/selector-group-chat.ipynb):
the manager is forced back into the loop after every worker turn via
`selector_func`, so workers never speak to each other directly.
"""

import asyncio
import os
from typing import Sequence

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient


def build_team() -> SelectorGroupChat:
    model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")

    planner = AssistantAgent(
        "PlanningAgent",
        description="Coordinator that breaks down the task and delegates to workers.",
        model_client=model_client,
        system_message=(
            "You are the planning agent. Break the user's task into subtasks and "
            "delegate each one to exactly one of the workers below. After every "
            "worker reply, decide the next step. Workers are:\n"
            "  Researcher: gathers facts and primary sources.\n"
            "  Analyst: synthesizes findings into structured insights.\n"
            "  Writer: drafts the final answer.\n"
            "Delegate by addressing the worker by name. When the task is fully "
            "complete, reply with a final summary ending in the word TERMINATE."
        ),
    )

    researcher = AssistantAgent(
        "Researcher",
        description="Gathers facts and sources on a given subtopic.",
        model_client=model_client,
        system_message="You research facts. Reply only with findings; do not delegate.",
    )

    analyst = AssistantAgent(
        "Analyst",
        description="Synthesizes findings into structured insights.",
        model_client=model_client,
        system_message="You synthesize. Reply only with structured insights; do not delegate.",
    )

    writer = AssistantAgent(
        "Writer",
        description="Drafts polished prose from analyst insights.",
        model_client=model_client,
        system_message="You write. Reply only with the drafted prose; do not delegate.",
    )

    def selector_func(messages: Sequence[BaseAgentEvent | BaseChatMessage]) -> str | None:
        if messages[-1].source != planner.name:
            return planner.name
        return None

    selector_prompt = (
        "Select the next agent to act.\n\n{roles}\n\n"
        "Conversation so far:\n{history}\n\n"
        "Pick exactly one agent from {participants}."
    )

    termination = TextMentionTermination("TERMINATE") | MaxMessageTermination(20)

    return SelectorGroupChat(
        [planner, researcher, analyst, writer],
        model_client=model_client,
        termination_condition=termination,
        selector_prompt=selector_prompt,
        selector_func=selector_func,
        allow_repeated_speaker=True,
    )


async def main() -> None:
    assert os.environ.get("OPENAI_API_KEY"), "set OPENAI_API_KEY"
    team = build_team()
    task = "Write a 1-paragraph briefing on the trade-offs of async Python for I/O-bound services."
    await Console(team.run_stream(task=task))


if __name__ == "__main__":
    asyncio.run(main())
