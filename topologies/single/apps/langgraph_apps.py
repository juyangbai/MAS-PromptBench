"""Single-agent APPS runner (LangGraph): one ReAct agent with a python_exec tool.

The solver prompt is ``configs/prompts/single/apps/solver.txt``; the submission is
the last fenced Python block of the agent's final message, scored on the
problem's tests (strict accuracy).
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from core import cli, code_tasks, prompts, settings
from core.batch import attempt
from core.code_tasks import exact_match_score, extract_code  # noqa: F401  (runner API)
from core.llm import chat_openai
from core.tasks import apps as task
from core.tasks.apps import format_prompt, load_instances, run_tests  # noqa: F401  (runner API)
from core.telemetry import langchain_telemetry, normalize
from core.thinking import strip_ai_thinking, strip_thinking  # noqa: F401  (runner API)

VLLM_BASE_URL = settings.base_url()
MODEL_ID = settings.model_id()

SYSTEM_PROMPT = prompts.role_prompt("single", task.DATASET, "solver")

python_exec = tool(code_tasks.make_python_exec(code_tasks.PYTHON_EXEC_DOC))


def build_agent():
    model = chat_openai(model=MODEL_ID, base_url=VLLM_BASE_URL)
    return create_react_agent(model=model, tools=[python_exec], prompt=SYSTEM_PROMPT)


def extract_answer(text: str) -> str | None:
    """Alias of :func:`extract_code` (the other datasets' runner API)."""
    return extract_code(text)


def solve(problem: str, starter_code: str | None = None, agent=None) -> dict:
    """Run the agent on one problem (call-based mode with ``starter_code``, standard input without).

    ``agent`` lets a batch reuse one agent. Returns ``{"code", "raw", "messages"}`` with
    reasoning stripped from every AI message.
    """
    if agent is None:
        agent = build_agent()
    result = agent.invoke(
        {"messages": [("user", format_prompt(problem, starter_code))]}, config={"recursion_limit": 25}
    )
    strip_ai_thinking(result["messages"])
    final = result["messages"][-1].content
    return {"code": extract_code(final), "raw": final, "messages": result["messages"]}


def run_batch(
    instances: list[dict],
    out_path: Path | None = None,
    verbose: bool = True,
    per_test_timeout_s: int = task.TEST_TIMEOUT_S,
) -> dict:
    """Solve every instance with one shared agent and score it on its tests."""
    agent = build_agent()

    def row(_, inst: dict) -> dict:
        out, latency_s, error = attempt(
            lambda: solve(inst["problem"], starter_code=inst.get("starter_code") or None, agent=agent),
            fallback={"code": None},
        )
        code = out["code"]
        return task.record(
            inst,
            code,
            task.test_scores(code, inst["input_output"], per_test_timeout_s),
            raw=(out.get("raw") or "")[:2000],
            latency_s=round(latency_s, 2),
            **normalize(langchain_telemetry(out.get("messages") or [])),
            error=error,
        )

    return task.run_batch(instances, row, out_path=out_path, verbose=verbose, label="single/APPS")


def _canned_demo() -> None:
    for mode, problem, starter, tests in task.DEMOS:
        print(f"\n========== {mode} MODE ==========")
        task.print_demo_code(solve(problem, starter_code=starter)["code"], tests)


def main(argv: list[str] | None = None) -> int:
    return cli.main(
        argv,
        description="Single-topology APPS runner (LangGraph, codeparrot/apps).",
        load_instances=load_instances,
        run_batch=run_batch,
        demo=_canned_demo,
        source=task.SOURCE,
        add_arguments=task.add_arguments,
    )


if __name__ == "__main__":
    raise SystemExit(main())
