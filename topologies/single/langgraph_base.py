"""Single-agent ReAct topology using LangGraph's prebuilt agent."""

import os

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent


@tool
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


@tool
def web_search(query: str) -> str:
    """Mock web search. Returns a stubbed snippet for the given query."""
    return f"[mock result for '{query}']"


SYSTEM_PROMPT = (
    "You are a helpful assistant with access to tools. "
    "Use tools when they would help. "
    "When you have the final answer, respond directly without further tool calls."
)


def build_agent():
    return create_react_agent(
        model=ChatOpenAI(model="gpt-4o-mini", temperature=0),
        tools=[add, multiply, web_search],
        prompt=SYSTEM_PROMPT,
    )


if __name__ == "__main__":
    assert os.environ.get("OPENAI_API_KEY"), "set OPENAI_API_KEY"

    agent = build_agent()
    task = "What is 12 + 30, then multiply that by 7? Also search for 'async python'."
    result = agent.invoke(
        {"messages": [("user", task)]},
        config={"recursion_limit": 25},
    )

    for msg in result["messages"]:
        msg.pretty_print()
