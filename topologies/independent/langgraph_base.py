"""Independent topology: 4 LLM agents fan out from START, no inter-agent communication."""

import asyncio
import operator
import os
from typing import Annotated

from typing_extensions import TypedDict

from openai import AsyncOpenAI

from langgraph.constants import END, START
from langgraph.graph.state import StateGraph
from langgraph.types import Send


AGENTS = [
    {"name": "agent_1", "model": "gpt-4o-mini", "system": "You are a concise analyst."},
    {"name": "agent_2", "model": "gpt-4o-mini", "system": "You are a creative brainstormer."},
    {"name": "agent_3", "model": "gpt-4o-mini", "system": "You are a skeptical critic."},
    {"name": "agent_4", "model": "gpt-4o-mini", "system": "You are a pragmatic engineer."},
]

class State(TypedDict):
    prompt: str
    answers: Annotated[list[dict], operator.add]


class AgentInput(TypedDict):
    prompt: str
    name: str
    model: str
    system: str


_client = AsyncOpenAI()


async def run_agent(state: AgentInput) -> dict:
    resp = await _client.chat.completions.create(
        model=state["model"],
        messages=[
            {"role": "system", "content": state["system"]},
            {"role": "user", "content": state["prompt"]},
        ],
    )
    return {"answers": [{"agent": state["name"], "text": resp.choices[0].message.content}]}


def fan_out(state: State) -> list[Send]:
    return [Send(a["name"], {"prompt": state["prompt"], **a}) for a in AGENTS]


def build_graph() -> StateGraph:
    g = StateGraph(State)
    for a in AGENTS:
        g.add_node(a["name"], run_agent)
    g.add_conditional_edges(START, fan_out)
    g.add_edge([a["name"] for a in AGENTS], END)
    return g


if __name__ == "__main__":
    assert os.environ.get("OPENAI_API_KEY"), "set OPENAI_API_KEY"

    graph = build_graph().compile()
    prompt = "In 2 sentences: what are the trade-offs of async Python?"
    result = asyncio.run(graph.ainvoke({"prompt": prompt, "answers": []}))

    for a in result["answers"]:
        print(f"--- {a['agent']} ---\n{a['text']}\n")
