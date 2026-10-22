"""Single-agent SWE-bench Verified runner (LangGraph): one ReAct agent with repository tools.

The solver prompt is ``configs/prompts/single/swe/solver.txt``. The agent edits a
clone of the instance repository; the patch is ``git diff HEAD`` of that clone,
scored with pytest on the host (``local``), in the instance image
(``singularity``) or not at all (``none``).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from core import prompts, settings
from core.llm import chat_openai
from core.tasks import swe as task
from core.tasks.swe import (  # noqa: F401  (runner API)
    clone_and_checkout,
    exact_match_score,
    is_resolved,
    load_instances,
    run_tests_singularity,
)
from core.telemetry import langchain_telemetry, normalize
from core.thinking import strip_ai_thinking, strip_thinking  # noqa: F401  (runner API)

logger = logging.getLogger(__name__)

VLLM_BASE_URL = settings.base_url()
MODEL_ID = settings.model_id()
MODEL_NAME = "mas-promptbench-single"
DEFAULT_WORKDIR_ROOT, DEFAULT_OUT_DIR = task.default_dirs("")

SYSTEM_PROMPT = prompts.role_prompt("single", task.DATASET, "solver")

WORKDIR = task.Workdir(task.env_repo_dir())
file_read = tool(task.make_file_read(WORKDIR, task.FILE_READ_DOC_SEARCH_FIRST))
file_write = tool(task.make_file_write(WORKDIR, task.FILE_WRITE_DOC_PARENTS))
list_dir = tool(task.make_list_dir(WORKDIR, task.LIST_DIR_DOC))
search_repo = tool(task.make_search_repo(WORKDIR, task.SEARCH_REPO_DOC_GREP))
shell_exec = tool(task.make_shell_exec(WORKDIR, task.SHELL_EXEC_DOC_TIMEOUT))
TOOLS = [file_read, file_write, list_dir, search_repo, shell_exec]

_ensure_sif = task.ensure_sif


def _set_repo_dir(path: Path | str) -> None:
    """Bind the checkout the tools act on (in this context)."""
    WORKDIR.bind(Path(path).resolve())


def format_prompt(problem_statement: str, instance_id: str | None = None, hints_text: str | None = None) -> str:
    """User message for one instance: issue, hints and the tool workflow, at the bound checkout."""
    return task.issue_brief(
        problem_statement, instance_id, hints_text, checkout=task.checked_out_at(WORKDIR.get()), note=task.FIX_NOTE
    )


def build_agent():
    model = chat_openai(model=MODEL_ID, base_url=VLLM_BASE_URL, include_max_tokens=False)
    return create_react_agent(model=model, tools=TOOLS, prompt=SYSTEM_PROMPT)


def extract_answer(text: str) -> str | None:
    """The answer of a SWE run is its patch (:func:`compute_patch`), not text."""
    return None


def compute_patch() -> str:
    """``git diff HEAD`` of the bound checkout."""
    return task.compute_patch(WORKDIR.get())


def apply_test_patch(test_patch: str) -> str:
    """Apply the instance's test patch to the bound checkout (local evaluation); "" or the error."""
    return task.apply_test_patch(WORKDIR.get(), test_patch)


def run_tests_local(fail_to_pass: list[str], pass_to_pass: list[str], timeout_s: int = 600) -> dict:
    """Both test groups with pytest in the bound checkout, on the host."""
    return task.run_tests_local(WORKDIR.get(), fail_to_pass, pass_to_pass, timeout_s)


def predictions_entry(instance_id: str, patch: str, model_name: str = MODEL_NAME) -> dict:
    return task.predictions_entry(instance_id, patch, model_name)


def solve(problem_statement: str, instance_id: str | None = None, hints_text: str | None = None) -> dict:
    """Run the agent on the instance checked out at the bound checkout.

    Returns ``{"patch", "raw", "messages"}`` with reasoning stripped from every AI message.
    """
    agent = build_agent()
    prompt = format_prompt(problem_statement, instance_id, hints_text)
    result = agent.invoke({"messages": [("user", prompt)]}, config={"recursion_limit": 100})
    strip_ai_thinking(result["messages"])
    return {"patch": compute_patch(), "raw": result["messages"][-1].content, "messages": result["messages"]}


def _message_trace(messages: list) -> str:
    """Every message with its type, tool calls and content."""
    lines = []
    for msg in messages:
        lines.append(f"=== {getattr(msg, 'type', '?').upper()} ===\n")
        for tc in getattr(msg, "tool_calls", None) or []:
            name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "?")
            args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {})
            lines.append(f"[tool_call] {name}({json.dumps(args, default=str)[:500]})\n")
        content = getattr(msg, "content", "")
        if content:
            lines.append(f"{content}\n")
        lines.append("\n")
    return "".join(lines)


def _evaluate(instance: dict, patch: str, eval_mode: str) -> tuple[dict | None, str | None, str | None]:
    """Score ``patch``: ``(report, None, None)``, or ``(None, error, stage)`` when evaluation fails."""
    f2p, p2p = task.instance_tests(instance)
    if eval_mode == "singularity":
        try:
            return run_tests_singularity(instance, patch, f2p, p2p), None, None
        except Exception as e:
            return None, f"{type(e).__name__}: {e}", "singularity_eval"
    test_patch = instance.get("test_patch") or ""
    if test_patch:
        error = apply_test_patch(test_patch)
        if error:
            return None, error, "apply_test_patch"
    return run_tests_local(f2p, p2p), None, None


def run_one(instance: dict, workdir_root: Path, out_dir: Path, eval_mode: str = "local") -> dict:
    """Clone, solve and score one instance; writes its patch, prediction and message trace under ``out_dir``."""
    iid = instance["instance_id"]
    summary, out = task.solve_in_checkout(
        instance,
        workdir_root,
        _set_repo_dir,
        lambda: solve(instance["problem_statement"], iid, instance.get("hints_text") or None),
    )
    if out is None:
        return summary
    patch = out["patch"] or ""
    summary["patch_chars"] = len(patch)
    summary["tool_calls"] = sum(1 for m in out["messages"] if getattr(m, "type", None) == "tool")
    summary.update(normalize(langchain_telemetry(out.get("messages") or [])))
    task.write_artifacts(out_dir, iid, patch, predictions_entry(iid, patch), _message_trace(out["messages"]))
    if eval_mode == "none":
        return {**summary, **task.eval_fields(eval_mode, None)}
    start = time.time()
    report, error, stage = _evaluate(instance, patch, eval_mode)
    if report is None:
        return {**summary, "error": error, "stage": stage}
    return {**summary, **task.eval_fields(eval_mode, report, eval_s=round(time.time() - start, 1))}


def _run_instances(
    instances: list[dict],
    out_dir: Path | None = None,
    workdir_root: Path | None = None,
    eval_mode: str = "local",
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
    logger.info(
        "\nFor leaderboard-comparable scoring, run the official Docker harness:\n"
        "    python -m swebench.harness.run_evaluation \\\n"
        "        --predictions_path %s \\\n"
        "        --max_workers 4 --run_id mas-promptbench_single_swe",
        out_dir / "predictions.jsonl",
    )


def run_batch(
    subset: str = "test",
    limit: int | None = None,
    offset: int = 0,
    only: list[str] | None = None,
    workdir_root: Path | None = None,
    out_dir: Path | None = None,
    eval_mode: str = "local",
    keep_workdirs: bool = False,
) -> None:
    """Solve and score a Verified slice; ``eval_mode`` is ``local`` (host pytest), ``singularity`` or ``none``."""
    instances = load_instances(subset, limit, offset, only)
    task.log_loaded(instances)
    _run_instances(instances, out_dir, workdir_root, eval_mode, keep_workdirs)


def main(argv: list[str] | None = None) -> int:
    return task.cli_main(
        argv,
        description="Single-topology agent on SWE-bench Verified.",
        run_batch=_run_instances,
        default_out_dir=DEFAULT_OUT_DIR,
        eval_modes=("local", "singularity", "none"),
    )


if __name__ == "__main__":
    raise SystemExit(main())
