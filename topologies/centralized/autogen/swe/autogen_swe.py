"""Centralized SWE-bench Verified runner (AutoGen): a manager and tool-using workers in a SelectorGroupChat.

The agents are the r=4 centralized team (``configs/teams/swe.yaml``) on one clone
of the instance repository. Control returns to the manager after every other
speaker; the manager may hand the turn to a worker, and the chat ends on
``TERMINATE`` or after the team's ``max_turns`` messages. The patch is
``git diff HEAD`` of the clone.
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
from core.tasks import swe as task
from core.tasks.swe import clone_and_checkout, is_resolved, load_instances, run_tests_singularity  # noqa: F401
from core.telemetry import autogen_telemetry, normalize

TOPOLOGY = "centralized"
TEAM = teams.spec(TOPOLOGY, task.DATASET)
MODEL_NAME = "mas-promptbench-centralized"
DEFAULT_WORKDIR_ROOT, DEFAULT_OUT_DIR = task.default_dirs(TOPOLOGY)

VLLM_BASE_URL = settings.base_url()
MODEL_ID = settings.model_id()
_MAX_MESSAGES = TEAM.max_turns

# AutoGen runs tools in executor threads that do not inherit the context, so
# the last bound checkout also serves as the default.
WORKDIR = task.Workdir(task.env_repo_dir(), sticky=True, explain_escapes=False)
TOOLS = {
    "file_read": task.make_file_read(WORKDIR, task.FILE_READ_DOC_SHORT),
    "str_replace": task.make_str_replace(WORKDIR, task.STR_REPLACE_DOC_NARROW),
    "list_dir": task.make_list_dir(WORKDIR, task.LIST_DIR_DOC),
    "search_repo": task.make_search_repo(WORKDIR, task.SEARCH_REPO_DOC, terse=True),
    "shell_exec": task.make_shell_exec(WORKDIR, task.SHELL_EXEC_DOC),
}

_ensure_sif = task.ensure_sif

# AutoGen agent descriptions, which the selector shows as the roster.
_DESCRIPTIONS = {
    "manager": "Coordinator for the SWE pipeline; plans, dispatches, synthesizes.",
    "navigator_worker": "Read-only repo exploration (list_dir, search_repo, file_read).",
    "patcher_worker": "Targeted file edits (file_read + str_replace).",
    "tester_worker": "Shell-based sanity checks (shell_exec + file_read).",
}
_SELECTOR_PROMPT = (
    "You are coordinating a 4-agent team resolving a GitHub issue.\n"
    "Select the next agent to act.\n\n{roles}\n\n"
    "Conversation so far:\n{history}\n\n"
    "Pick exactly one agent from {participants}."
)
_MANAGER_TERMINATE_NUDGE = task.TERMINATE_NUDGE


def _load_prompt(role: str) -> str:
    return prompts.role_prompt(TOPOLOGY, task.DATASET, role)


def _build_client() -> OpenAIChatCompletionClient:
    return autogen_client(model=MODEL_ID, base_url=VLLM_BASE_URL)


def _set_repo_dir(path: Path | str) -> None:
    """Bind the checkout the tools act on."""
    WORKDIR.bind(Path(path).resolve())


def format_task_brief(problem_statement: str, instance_id: str | None = None, hints_text: str | None = None) -> str:
    """The group chat's task, at the bound checkout."""
    return task.issue_brief(
        problem_statement, instance_id, hints_text, checkout=task.checked_out_at(WORKDIR.get()), note=task.NO_TESTS_NOTE
    )


def compute_patch() -> str:
    """``git diff HEAD`` of the bound checkout."""
    return task.compute_patch(WORKDIR.get())


def predictions_entry(instance_id: str, patch: str, model_name: str = MODEL_NAME) -> dict:
    return task.predictions_entry(instance_id, patch, model_name)


def _agent(role: str, tools: tuple[str, ...], suffix: str, client: OpenAIChatCompletionClient) -> AssistantAgent:
    """An agent of the team: its role prompt plus ``suffix``, with the tools named ``tools``."""
    return AssistantAgent(
        role,
        description=_DESCRIPTIONS[role],
        model_client=client,
        system_message=_load_prompt(role) + suffix,
        tools=[TOOLS[name] for name in tools],
    )


def build_team() -> SelectorGroupChat:
    """The manager (inspection tools, TERMINATE nudge) and its workers, all sharing one client."""
    client = _build_client()
    manager = _agent(TEAM.manager, TEAM.manager_tools, _MANAGER_TERMINATE_NUDGE, client)
    workers = [_agent(worker.role, worker.tools, worker.prompt_suffix, client) for worker in TEAM.workers]
    agents = [manager, *workers]

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


async def solve_async(instance: dict, eval_mode: str = "singularity") -> dict:
    """Run the team on the instance checked out at the bound checkout.

    Returns ``{"patch", "resolved", "report", "messages", "telemetry"}``: the
    clone's patch, its evaluation (``resolved`` None and no report with
    ``eval_mode='none'``; False without a patch), every message as
    ``{source, content}`` and token/call counts.
    """
    team = build_team()
    brief = format_task_brief(
        instance["problem_statement"], instance_id=instance.get("instance_id"), hints_text=instance.get("hints_text")
    )
    result = await team.run(task=brief)
    messages = [
        {
            "source": getattr(m, "source", None),
            "content": getattr(m, "content", None)
            if isinstance(getattr(m, "content", None), str)
            else str(getattr(m, "content", "")),
        }
        for m in result.messages
    ]
    out = {
        "patch": compute_patch(),
        "resolved": None if eval_mode == "none" else False,
        "report": None,
        "messages": messages,
        "telemetry": normalize(autogen_telemetry(result)),
    }
    if eval_mode != "none" and out["patch"]:
        f2p, p2p = task.instance_tests(instance)
        out["report"] = run_tests_singularity(instance, out["patch"], f2p, p2p)
        out["resolved"] = is_resolved(out["report"])
    return out


def solve(instance: dict, eval_mode: str = "singularity") -> dict:
    return asyncio.run(solve_async(instance, eval_mode))


def run_one(instance: dict, workdir_root: Path, out_dir: Path, eval_mode: str = "singularity") -> dict:
    """Clone, solve and score one instance; writes its patch, prediction and chat trace under ``out_dir``."""
    summary, out = task.solve_in_checkout(
        instance, workdir_root, _set_repo_dir, lambda: solve(instance, eval_mode=eval_mode)
    )
    if out is None:
        return summary
    summary.update(out.get("telemetry") or {})
    patch = out["patch"] or ""
    messages = out.get("messages") or []
    summary.update(patch_chars=len(patch), n_messages=len(messages))
    iid = instance["instance_id"]
    trace = task.sections((str(m.get("source", "?")).upper(), m.get("content", "")) for m in messages)
    task.write_artifacts(out_dir, iid, patch, predictions_entry(iid, patch), trace)
    return {**summary, **task.eval_fields(eval_mode, out.get("report"))}


def _run_instances(
    instances: list[dict],
    out_dir: Path | None = None,
    workdir_root: Path | None = None,
    eval_mode: str = "singularity",
    keep_workdirs: bool = False,
    out_path: Path | None = None,
) -> None:
    """Solve and score loaded instances (the command line's batch)."""
    workdir_root = workdir_root or DEFAULT_WORKDIR_ROOT
    out_dir = out_dir or DEFAULT_OUT_DIR
    task.run_batch(
        instances,
        lambda inst: run_one(inst, workdir_root, out_dir, eval_mode=eval_mode),
        out_dir=out_dir,
        eval_mode=eval_mode,
        workdirs=None if keep_workdirs else lambda inst: [workdir_root / inst["instance_id"]],
        predictions=out_path,
    )


def run_batch(
    subset: str = "test",
    limit: int | None = None,
    offset: int = 0,
    only: list[str] | None = None,
    workdir_root: Path | None = None,
    out_dir: Path | None = None,
    eval_mode: str = "singularity",
    keep_workdirs: bool = False,
) -> None:
    """Solve and score a Verified slice (``eval_mode``: ``singularity`` or ``none``)."""
    instances = load_instances(subset, limit, offset, only)
    task.log_loaded(instances)
    _run_instances(instances, out_dir, workdir_root, eval_mode, keep_workdirs)


def main(argv: list[str] | None = None) -> int:
    return task.cli_main(
        argv,
        description="Centralized-topology SWE-bench Verified agent (AutoGen).",
        run_batch=_run_instances,
        default_out_dir=DEFAULT_OUT_DIR,
        eval_modes=("singularity", "none"),
    )


if __name__ == "__main__":
    raise SystemExit(main())
