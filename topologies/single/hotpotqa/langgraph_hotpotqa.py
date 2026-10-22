"""Single-agent HotpotQA runner (LangGraph): one ReAct agent with Wikipedia search and page tools.

The solver prompt is ``configs/prompts/single/hotpotqa/solver.txt`` plus the
short-form ``Answer:`` nudge; the answer is the last ``Answer:`` line.
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from core import cli, prompts, settings
from core.batch import attempt
from core.llm import chat_openai
from core.tasks import hotpotqa as task
from core.tasks.hotpotqa import (  # noqa: F401  (runner API)
    exact_match_score,
    extract_answer,
    f1_score,
    format_prompt,
    load_instances,
    normalize_answer,
)
from core.telemetry import langchain_telemetry, normalize
from core.thinking import strip_ai_thinking, strip_thinking  # noqa: F401  (runner API)

VLLM_BASE_URL = settings.base_url()
MODEL_ID = settings.model_id()

_OUTPUT_FORMAT_NUDGE = task.OUTPUT_FORMAT_NUDGE
SYSTEM_PROMPT = prompts.role_prompt("single", task.DATASET, "solver", suffix=_OUTPUT_FORMAT_NUDGE)

wikipedia_search = tool(task.make_wikipedia_search(task.SEARCH_DOC))
wikipedia_page = tool(task.make_wikipedia_page(task.PAGE_DOC))


def build_agent():
    model = chat_openai(model=MODEL_ID, base_url=VLLM_BASE_URL)
    return create_react_agent(model=model, tools=[wikipedia_search, wikipedia_page], prompt=SYSTEM_PROMPT)


def solve(question: str, agent=None) -> dict:
    """Run the agent on one question (``agent`` lets a batch reuse one agent).

    Returns ``{"answer", "raw", "messages"}`` with reasoning stripped from every AI message.
    """
    if agent is None:
        agent = build_agent()
    result = agent.invoke({"messages": [("user", format_prompt(question))]}, config={"recursion_limit": 25})
    strip_ai_thinking(result["messages"])
    final = result["messages"][-1].content
    return {"answer": extract_answer(final), "raw": final, "messages": result["messages"]}


def run_batch(instances: list[dict], out_path: Path | None = None, verbose: bool = True) -> dict:
    """Solve and score every instance with one shared agent."""
    agent = build_agent()

    def row(_, inst: dict) -> dict:
        out, latency_s, error = attempt(lambda: solve(inst["question"], agent=agent))
        return task.record(
            inst,
            out["answer"],
            **task.meta(inst),
            raw=out.get("raw") or "",
            latency_s=round(latency_s, 2),
            **normalize(langchain_telemetry(out.get("messages") or [])),
            error=error,
        )

    return task.run_batch(instances, row, out_path=out_path, verbose=verbose)


def _canned_demo() -> None:
    out = solve(task.DEMO_QUESTION)
    task.print_demo_answer(out["answer"])
    print("=== Full message trace ===")
    for msg in out["messages"]:
        msg.pretty_print()


def main(argv: list[str] | None = None) -> int:
    return cli.main(
        argv,
        description="Single-topology HotpotQA runner (LangGraph).",
        load_instances=load_instances,
        run_batch=run_batch,
        demo=_canned_demo,
        source=task.SOURCE,
    )


if __name__ == "__main__":
    raise SystemExit(main())
