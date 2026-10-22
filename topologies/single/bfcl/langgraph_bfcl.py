"""Single-agent BFCL runner (LangGraph): one ReAct agent with the instance's functions as tools.

Every function schema becomes a no-op tool; the prediction is the first turn's
native tool calls in canonical form, scored by bfcl-eval's AST checker.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from langgraph.prebuilt import create_react_agent

from core import prompts, settings
from core.llm import chat_openai
from core.paths import RESULTS_DIR
from core.tasks import bfcl as task
from core.tasks.bfcl import (  # noqa: F401  (runner API)
    AST_CATEGORIES,
    HF_DATASET,
    extract_first_tool_calls,
    load_instances,
    schema_to_tool,
    to_canonical,
)
from core.telemetry import langchain_telemetry, normalize

DEFAULT_OUT_DIR = RESULTS_DIR / "bfcl"

VLLM_BASE_URL = settings.base_url()
MODEL_ID = settings.model_id()

SYSTEM_PROMPT = prompts.role_prompt("single", task.DATASET, "solver")

_register_model_with_bfcl = task.register_model  # called again by callers that repoint MODEL_ID
_register_model_with_bfcl(MODEL_ID)


def build_agent(tools: list):
    """The ReAct agent over ``tools`` (thinking off: Qwen3 drifts into text-form calls when it thinks)."""
    llm = chat_openai(model=MODEL_ID, base_url=VLLM_BASE_URL)
    return create_react_agent(model=llm, tools=tools, prompt=SYSTEM_PROMPT)


def score_one(function_schemas: list[dict], model_output: list[dict], ground_truth: list[dict], category: str) -> dict:
    return task.score_one(function_schemas, model_output, ground_truth, category, MODEL_ID)


def solve(instance: dict) -> dict:
    """Run the agent on one instance.

    Returns ``{"messages", "tool_calls", "model_output", "solve_s"}``: the
    transcript, the first turn's tool calls, their canonical form and the
    agent's wall time.
    """
    agent = build_agent([schema_to_tool(s) for s in instance["function"]])
    start = time.time()
    result = agent.invoke({"messages": instance["question"][0]}, config={"recursion_limit": 25})
    elapsed = time.time() - start
    tool_calls = extract_first_tool_calls(result["messages"])
    return {
        "messages": result["messages"],
        "tool_calls": tool_calls,
        "model_output": to_canonical(tool_calls, instance["function"]),
        "solve_s": elapsed,
    }


def _format_trace(messages: list) -> str:
    """Trace text of the transcript: each message under its type, tool calls as ``name(args)``."""
    lines: list[str] = []
    for msg in messages:
        kind = getattr(msg, "type", "unknown")
        if kind in ("human", "system", "tool"):
            lines += [f"=== {kind.upper()} ===", str(msg.content)]
        elif kind == "ai":
            lines.append("=== AI ===")
            for tc in getattr(msg, "tool_calls", None) or []:
                lines.append(f"[tool_call] {tc['name']}({json.dumps(tc.get('args') or {})})")
            if getattr(msg, "content", ""):
                lines.append(str(msg.content))
        lines.append("")
    return "\n".join(lines)


def run_one(instance: dict, ground_truth: dict, category: str, out_dir: Path) -> dict:
    """Solve and score one instance and write its transcript to ``out_dir/traces/<id>.txt``."""
    summary: dict = {"id": instance["id"], "category": category}
    try:
        solution = solve(instance)
    except Exception as e:
        return task.solve_failed(summary, e)
    summary["solve_s"] = round(solution["solve_s"], 1)
    summary["tool_calls"] = len(solution["tool_calls"])
    summary["model_output"] = solution["model_output"]
    summary.update(normalize(langchain_telemetry(solution.get("messages") or [])))
    task.write_trace(out_dir, instance["id"], _format_trace(solution["messages"]))
    return task.add_verdict(summary, score_one, instance, ground_truth, category)


def run_batch(
    category: str,
    limit: int | None = None,
    offset: int = 0,
    only: list[str] | None = None,
    out_dir: Path | None = None,
    verbose: bool = True,
) -> dict:
    """Score a slice of one subset; writes predictions.jsonl and results.jsonl afresh in ``out_dir``."""
    return task.evaluate(
        run_one, category, limit, offset, only, out_dir or DEFAULT_OUT_DIR, model_id=MODEL_ID, verbose=verbose
    )


def main(argv: list[str] | None = None) -> int:
    return task.cli_main(
        argv,
        description="Run the single-topology BFCL agent on one of the AST subsets.",
        run_one=run_one,
        model_id=MODEL_ID,
        default_out_dir=DEFAULT_OUT_DIR,
    )


if __name__ == "__main__":
    raise SystemExit(main())
