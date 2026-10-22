"""Single-agent GPQA-Diamond runner (LangGraph): one ReAct agent with a calculator.

The solver prompt is ``configs/prompts/single/gpqa/solver.txt``; the answer is
the option letter extracted from the agent's final message.
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from core import cli, prompts, settings
from core.batch import attempt
from core.calculator import CALCULATOR_DOC, make_calculator
from core.llm import chat_openai
from core.tasks import gpqa as task
from core.tasks.gpqa import extract_answer, format_prompt, load_instances
from core.telemetry import langchain_telemetry, normalize
from core.thinking import strip_ai_thinking, strip_thinking  # noqa: F401  (runner API)

VLLM_BASE_URL = settings.base_url()
MODEL_ID = settings.model_id()

SYSTEM_PROMPT = prompts.role_prompt("single", task.DATASET, "solver")

calculator = tool(make_calculator(CALCULATOR_DOC))


def build_agent():
    model = chat_openai(model=MODEL_ID, base_url=VLLM_BASE_URL)
    return create_react_agent(model=model, tools=[calculator], prompt=SYSTEM_PROMPT)


def solve(question: str, choices: list[str], agent=None) -> dict:
    """Run the agent on one question (``agent`` lets a batch reuse one agent).

    Returns ``{"answer", "raw", "messages"}`` with reasoning stripped from every AI message.
    """
    if agent is None:
        agent = build_agent()
    prompt = format_prompt(question, choices)
    result = agent.invoke({"messages": [("user", prompt)]}, config={"recursion_limit": 25})
    strip_ai_thinking(result["messages"])
    final = result["messages"][-1].content
    return {"answer": extract_answer(final), "raw": final, "messages": result["messages"]}


def run_batch(instances: list[dict], out_path: Path | None = None, verbose: bool = True) -> dict:
    """Solve and score every instance with one shared agent."""
    agent = build_agent()
    fallback = {"answer": None, "raw": "", "messages": []}

    def row(_, inst: dict) -> dict:
        out, latency_s, error = attempt(
            lambda: solve(inst["question"], inst["choices"], agent=agent), fallback=fallback
        )
        return task.record(
            inst,
            out["answer"],
            raw=out["raw"],
            latency_s=round(latency_s, 2),
            **normalize(langchain_telemetry(out.get("messages") or [])),
            error=error,
        )

    return task.run_batch(instances, row, out_path=out_path, verbose=verbose)


def _canned_demo() -> None:
    out = solve(task.DEMO_QUESTION, task.DEMO_CHOICES)
    task.print_demo_answer(out["answer"])
    print("=== Full message trace ===")
    for msg in out["messages"]:
        msg.pretty_print()


def main(argv: list[str] | None = None) -> int:
    return cli.main(
        argv,
        description="Single-topology GPQA-Diamond runner (LangGraph).",
        load_instances=load_instances,
        run_batch=run_batch,
        demo=_canned_demo,
        source=task.SOURCE,
        add_arguments=task.add_arguments,
    )


if __name__ == "__main__":
    raise SystemExit(main())
