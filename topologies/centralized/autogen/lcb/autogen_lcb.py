"""Centralized LCB runner (AutoGen): a manager and python_exec workers in a SelectorGroupChat.

The workers are the r=4 centralized team (``configs/teams/<dataset>.yaml``). Control
returns to the manager after every other speaker; the manager may hand the turn
to a worker, and the chat ends on ``TERMINATE`` or after the team's
``max_turns`` messages. The submission is the manager's last fenced Python block
(else the coder worker's).
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

from core import cli, code_tasks, prompts, settings, teams
from core.batch import attempt
from core.code_tasks import exact_match_score  # noqa: F401  (runner API)
from core.llm import autogen_client
from core.tasks import lcb as task
from core.tasks.lcb import format_prompt, load_instances, run_tests  # noqa: F401
from core.telemetry import autogen_telemetry, normalize

TOPOLOGY = "centralized"
TEAM = teams.spec(TOPOLOGY, task.DATASET)

VLLM_BASE_URL = settings.base_url()
MODEL_ID = settings.model_id()

_MAX_MESSAGES = TEAM.max_turns
_MANAGER_TERMINATE_NUDGE = code_tasks.TERMINATE_NUDGE

extract_code = code_tasks.extract_code_before_terminate
python_exec = code_tasks.make_python_exec(code_tasks.PYTHON_EXEC_DOC)

# AutoGen agent descriptions, which the selector shows as the roster.
_DESCRIPTIONS = {
    "manager": "Coordinator that plans, dispatches, validates via python_exec, and emits the final program.",
    "analyzer_worker": "Returns an algorithmic approach, complexity analysis, and edge cases.",
    "coder_worker": "Writes a Python implementation given the manager's spec.",
    "tester_worker": "Runs supplied tests against candidate code via python_exec and reports results.",
}
_SELECTOR_PROMPT = (
    "You are coordinating a 4-agent team on a programming problem.\n"
    "Select the next agent to act.\n\n{roles}\n\n"
    "Conversation so far:\n{history}\n\n"
    "Pick exactly one agent from {participants}."
)


def _load_prompt(role: str) -> str:
    return prompts.role_prompt(TOPOLOGY, task.DATASET, role)


def _build_client() -> OpenAIChatCompletionClient:
    return autogen_client(model=MODEL_ID, base_url=VLLM_BASE_URL)


def build_team() -> SelectorGroupChat:
    """The manager and its workers, all sharing one client and python_exec."""
    client = _build_client()
    agents = [
        AssistantAgent(
            role,
            description=_DESCRIPTIONS[role],
            model_client=client,
            system_message=_load_prompt(role) + (_MANAGER_TERMINATE_NUDGE if role == TEAM.manager else ""),
            tools=[python_exec],
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


async def solve_async(problem: str, starter_code: str | None = None) -> dict:
    """Run the team on one problem (functional mode with ``starter_code``, stdin mode without).

    Returns ``{"code", "raw", "messages", "telemetry"}``: the program of the manager's
    last message (else of the coder worker's last message), that message, every
    message as ``{source, content}`` and token/call counts.
    """
    result = await build_team().run(task=format_prompt(problem, starter_code))
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
    code = extract_code(final)
    if code is None:
        coder_msgs = [m for m in messages if m["source"] == "coder_worker"]
        if coder_msgs:
            code = extract_code(coder_msgs[-1]["content"])
    return {"code": code, "raw": final, "messages": messages, "telemetry": normalize(autogen_telemetry(result))}


def solve(problem: str, starter_code: str | None = None) -> dict:
    return asyncio.run(solve_async(problem, starter_code))


def run_batch(
    instances: list[dict],
    out_path: Path | None = None,
    verbose: bool = True,
    per_test_timeout_s: int = task.BATCH_TEST_TIMEOUT_S,
) -> dict:
    """Solve every instance and score it on its tests."""

    def row(_, inst: dict) -> dict:
        out, latency_s, error = attempt(
            lambda: solve(inst["problem"], starter_code=inst.get("starter_code") or None), fallback={"code": None}
        )
        code = out["code"]
        return task.record(
            inst,
            code,
            task.test_scores(code, inst["tests"], per_test_timeout_s),
            n_messages=len(out.get("messages") or []),
            raw=(out.get("raw") or "")[:2000],
            latency_s=round(latency_s, 2),
            **(out.get("telemetry") or {}),
            error=error,
        )

    return task.run_batch(instances, row, out_path=out_path, verbose=verbose, label="centralized/LCB")


def _canned_demo() -> None:
    for mode, problem, starter, tests in task.DEMOS:
        print(f"\n========== {mode} MODE ==========")
        out = solve(problem, starter_code=starter)
        task.print_demo_code(out["code"], tests)
        print(f"=== {len(out['messages'])} messages across the group chat ===")


def main(argv: list[str] | None = None) -> int:
    return cli.main(
        argv,
        description="Centralized-topology LCB runner (AutoGen SelectorGroupChat).",
        load_instances=load_instances,
        run_batch=run_batch,
        demo=_canned_demo,
        source=task.SOURCE,
        add_arguments=task.add_arguments,
    )


if __name__ == "__main__":
    raise SystemExit(main())
