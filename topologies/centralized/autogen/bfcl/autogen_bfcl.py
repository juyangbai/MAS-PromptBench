"""Centralized BFCL runner (AutoGen): a manager and tool-less workers in a SelectorGroupChat.

The workers are the r=4 centralized team (``configs/teams/bfcl.yaml``). Control
returns to the manager after every other speaker; the chat ends on
``TERMINATE`` or after the team's ``max_turns`` messages. The prediction is the
manager's last fenced canonical JSON (else the caller's).
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

from core import prompts, settings, teams
from core.llm import autogen_client
from core.paths import RESULTS_DIR
from core.tasks import bfcl as task
from core.tasks.bfcl import AST_CATEGORIES, HF_DATASET, extract_canonical, format_task, load_instances  # noqa: F401
from core.telemetry import autogen_telemetry, normalize

TOPOLOGY = "centralized"
TEAM = teams.spec(TOPOLOGY, task.DATASET)
DEFAULT_OUT_DIR = RESULTS_DIR / "bfcl_centralized"

VLLM_BASE_URL = settings.base_url()
MODEL_ID = settings.model_id()

_MAX_MESSAGES = TEAM.max_turns
_MANAGER_TERMINATE_NUDGE = task.TERMINATE_NUDGE

_register_model_with_bfcl = task.register_model  # called again by callers that repoint MODEL_ID
_register_model_with_bfcl(MODEL_ID)

# AutoGen agent descriptions, which the selector shows as the roster.
_DESCRIPTIONS = {
    "manager": "Coordinator that plans the call, dispatches composition + validation, and emits the final canonical JSON.",
    "inspector_worker": "Reads the schema and returns an argument plan.",
    "caller_worker": "Composes the canonical JSON call per manager instruction.",
    "validator_worker": "Checks the call against the schema (name exists, required params present, types valid).",
}
_SELECTOR_PROMPT = (
    "You are coordinating a 4-agent team on a BFCL function-calling task.\n"
    "Select the next agent to act.\n\n{roles}\n\n"
    "Conversation so far:\n{history}\n\n"
    "Pick exactly one agent from {participants}."
)


def _load_prompt(role: str) -> str:
    return prompts.role_prompt(TOPOLOGY, task.DATASET, role)


def _build_client() -> OpenAIChatCompletionClient:
    return autogen_client(model=MODEL_ID, base_url=VLLM_BASE_URL)


def score_one(function_schemas: list[dict], model_output: list[dict], ground_truth: list[dict], category: str) -> dict:
    return task.score_one(function_schemas, model_output, ground_truth, category, MODEL_ID)


def build_team() -> SelectorGroupChat:
    """The manager and its workers, all sharing one client."""
    client = _build_client()
    agents = [
        AssistantAgent(
            role,
            description=_DESCRIPTIONS[role],
            model_client=client,
            system_message=_load_prompt(role) + (_MANAGER_TERMINATE_NUDGE if role == TEAM.manager else ""),
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


async def solve_async(instance: dict) -> dict:
    """Run the team on one instance.

    Returns ``{"model_output", "raw", "messages", "telemetry"}``: the canonical
    calls (or []), the manager's last message, every message as ``{source,
    content}`` and token/call counts.
    """
    team = build_team()
    prompt = format_task(task.flatten_question(instance["question"]), task.render_schemas(instance["function"]))
    result = await team.run(task=prompt)
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
    model_output = extract_canonical(final)
    if model_output is None:
        caller_msgs = [m for m in messages if m["source"] == "caller_worker"]
        if caller_msgs:
            model_output = extract_canonical(caller_msgs[-1]["content"])
    return {
        "model_output": model_output or [],
        "raw": final,
        "messages": messages,
        "telemetry": normalize(autogen_telemetry(result)),
    }


def solve(instance: dict) -> dict:
    return asyncio.run(solve_async(instance))


def run_one(instance: dict, ground_truth: dict, category: str, out_dir: Path) -> dict:
    """Solve and score one instance and write the group chat to ``out_dir/traces/<id>.txt``."""
    summary: dict = {"id": instance["id"], "category": category}
    try:
        out = solve(instance)
    except Exception as e:
        return task.solve_failed(summary, e)
    summary["model_output"] = out.get("model_output") or []
    summary["n_messages"] = len(out.get("messages") or [])
    summary["tool_calls"] = len(summary["model_output"])
    summary.update(out.get("telemetry") or {})
    task.write_trace(out_dir, instance["id"], task.messages_trace(out.get("messages") or []))
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
        description="Centralized-topology BFCL runner (AutoGen SelectorGroupChat).",
        run_one=run_one,
        model_id=MODEL_ID,
        default_out_dir=DEFAULT_OUT_DIR,
    )


if __name__ == "__main__":
    raise SystemExit(main())
