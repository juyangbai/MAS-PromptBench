"""Sequential SWE-bench Verified runner (CrewAI): investigator -> planner -> patcher -> tester crew.

Each task sees the outputs of the earlier tasks as context and the issue brief;
the agents' backstories are the sequential role prompts. The agents edit one
clone of the instance repository; the patch is ``git diff HEAD`` of that clone.
"""

from __future__ import annotations

from pathlib import Path

from crewai import LLM, Agent, Crew, Process, Task
from crewai.tools import tool

from core import prompts, settings
from core.llm import crewai_llm
from core.tasks import swe as task
from core.tasks.swe import (  # noqa: F401  (runner API)
    clone_and_checkout,
    is_resolved,
    load_instances,
    run_tests_singularity,
)
from core.telemetry import crewai_telemetry, normalize
from core.thinking import strip_thinking  # noqa: F401  (runner API)

TOPOLOGY = "sequential"
MODEL_NAME = "mas-promptbench-sequential"
DEFAULT_WORKDIR_ROOT, DEFAULT_OUT_DIR = task.default_dirs(TOPOLOGY)
_STAGES = ("investigator", "planner", "patcher", "tester")

VLLM_BASE_URL = settings.base_url()
MODEL_ID = settings.model_id()

# CrewAI may run tools in threads that do not inherit the context, so the
# last bound checkout also serves as the default.
WORKDIR = task.Workdir(task.env_repo_dir(), sticky=True)
file_read = tool("file_read")(task.make_file_read(WORKDIR, task.FILE_READ_DOC))
str_replace = tool("str_replace")(
    task.make_str_replace(
        WORKDIR, task.STR_REPLACE_DOC_TARGETED_EM_DASH, not_found=task.NOT_FOUND_READ_FIRST, preview=True
    )
)
list_dir = tool("list_dir")(task.make_list_dir(WORKDIR, task.LIST_DIR_DOC))
search_repo = tool("search_repo")(task.make_search_repo(WORKDIR, task.SEARCH_REPO_DOC))
shell_exec = tool("shell_exec")(task.make_shell_exec(WORKDIR, task.SHELL_EXEC_DOC_TESTER))

_ensure_sif = task.ensure_sif


def _load_prompt(role: str) -> str:
    return prompts.role_prompt(TOPOLOGY, task.DATASET, role)


def _build_llm() -> LLM:
    return crewai_llm(model=MODEL_ID, base_url=VLLM_BASE_URL)


def _set_repo_dir(path: Path | str) -> None:
    """Bind the checkout the tools act on."""
    WORKDIR.bind(Path(path).resolve())


def format_task_brief(problem_statement: str, instance_id: str | None = None, hints_text: str | None = None) -> str:
    """The issue brief in every task description, at the bound checkout."""
    return task.issue_brief(
        problem_statement, instance_id, hints_text, checkout=task.checked_out_at(WORKDIR.get()), note=task.NO_TESTS_NOTE
    )


def compute_patch() -> str:
    """``git diff HEAD`` of the bound checkout."""
    return task.compute_patch(WORKDIR.get())


def predictions_entry(instance_id: str, patch: str, model_name: str = MODEL_NAME) -> dict:
    return task.predictions_entry(instance_id, patch, model_name)


def _agent(role: str, title: str, goal: str, tools: list, model: LLM) -> Agent:
    return Agent(
        role=title,
        goal=goal,
        backstory=_load_prompt(role),
        tools=tools,
        llm=model,
        verbose=False,
        allow_delegation=False,
    )


def build_crew(llm: LLM | None = None) -> Crew:
    """The four-agent sequential crew."""
    if llm is None:
        llm = _build_llm()
    investigator = _agent(
        "investigator",
        "SWE Investigator",
        "Explore the repo, localize the bug, and hand off a precise "
        "pointer (file path + line range + root-cause explanation). "
        "Do NOT write any code.",
        [file_read, list_dir, search_repo],
        llm,
    )
    planner = _agent(
        "planner",
        "SWE Planner",
        "Design the fix strategy from the Investigator's pointer: "
        "which lines change, what the new code should do, how it "
        "relates to the failing test. Do NOT write code.",
        [],
        llm,
    )
    # The patcher edits with str_replace only: file_write would overwrite whole files.
    patcher = _agent(
        "patcher",
        "SWE Patcher",
        "Write the code edit following the Planner's strategy. Use "
        "file_read to confirm the lines you plan to change; use "
        "str_replace(path, old, new) to make TARGETED edits. Keep "
        "the change minimal.",
        [file_read, str_replace],
        llm,
    )
    tester = _agent(
        "tester",
        "SWE Tester",
        "Sanity-check the patched workdir via shell_exec (syntax "
        "compile, import smoke tests, `git diff` inspection). The "
        "real FAIL_TO_PASS eval runs externally; your job is to "
        "catch obvious breakage.",
        [shell_exec, file_read],
        llm,
    )

    investigate_task = Task(
        description=(
            "Explore the repo to localize the reported bug. Use "
            "list_dir, search_repo, and file_read. Identify the "
            "specific file(s), function(s), and line range(s) where "
            "the fix must land. Do NOT edit anything.\n\n"
            "{task_brief}"
        ),
        expected_output=(
            "A pointer document: (1) the file(s) and line range(s) that "
            "need changing, (2) a short root-cause explanation, (3) "
            "any surrounding context a patcher would need to know."
        ),
        agent=investigator,
    )
    plan_task = Task(
        description=(
            "Given the Investigator's pointer, design the fix. Specify "
            "concretely: what lines change, what the new code should "
            "look like (semantically, not as a literal diff), and how "
            "the change resolves the failing test.\n\n"
            "{task_brief}"
        ),
        expected_output=(
            "A fix plan: (1) target location, (2) description of the "
            "replacement / insertion, (3) why it fixes the reported "
            "behavior."
        ),
        agent=planner,
        context=[investigate_task],
    )
    patch_task = Task(
        description=(
            "Execute the Planner's strategy. You MUST actually invoke "
            "the tools — do NOT describe tool calls in text or emit "
            "JSON blobs. Every str_replace + file_read should be a real "
            "tool invocation.\n\n"
            "Workflow:\n"
            "  1. CALL file_read on the target file to confirm the "
            "     exact lines (they may have shifted from the "
            "     Investigator's pointer).\n"
            "  2. CALL str_replace(path, old, new) to make a targeted "
            "     edit. Include enough surrounding context in `old` so "
            "     the match is unique.\n"
            "  3. If str_replace returns an error (not found / "
            "     ambiguous), CALL file_read again and retry with more "
            "     context.\n"
            "  4. Keep changes minimal. file_write is not available; "
            "     str_replace is the only edit tool.\n"
            "After the edit lands, produce a short text summary — do "
            "NOT re-emit the JSON arguments.\n\n"
            "{task_brief}"
        ),
        expected_output=(
            "A 1-3 sentence summary of the edit actually made "
            "(e.g., 'Replaced `cright[...] = 1` with `cright[...] = "
            "right` in astropy/modeling/separable.py'). Do NOT include "
            "JSON — just the outcome."
        ),
        agent=patcher,
        context=[investigate_task, plan_task],
    )
    test_task = Task(
        description=(
            "Sanity-check the patched workdir. Run shell_exec with a "
            "Python syntax check (e.g. `python -m py_compile <file>`) "
            "and, if feasible, a lightweight import smoke test. "
            "Inspect `git diff HEAD` to confirm the patch makes sense. "
            "Report any obvious breakage; the Patcher can re-read "
            "this, but you do NOT edit files yourself.\n\n"
            "{task_brief}"
        ),
        expected_output=(
            "A short sanity-check report ending with 'Patch looks wholesome' or 'Patch has problems: <summary>'."
        ),
        agent=tester,
        context=[investigate_task, plan_task, patch_task],
    )
    return Crew(
        agents=[investigator, planner, patcher, tester],
        tasks=[investigate_task, plan_task, patch_task, test_task],
        process=Process.sequential,
        verbose=False,
    )


def solve(instance: dict, eval_mode: str = "singularity") -> dict:
    """Run the crew on the instance checked out at the bound checkout.

    Returns ``{"patch", "resolved", "report", "by_stage", "telemetry"}``: the
    clone's patch, its evaluation (``resolved`` None and no report with
    ``eval_mode='none'``; False without a patch), every stage's output and
    token/call counts.
    """
    brief = format_task_brief(
        instance["problem_statement"], instance_id=instance.get("instance_id"), hints_text=instance.get("hints_text")
    )
    crew = build_crew()
    result = crew.kickoff(inputs={"task_brief": brief})
    try:
        stages = {role: result.tasks_output[i].raw for i, role in enumerate(_STAGES)}
    except (AttributeError, IndexError):
        stages = dict.fromkeys(_STAGES, "")
    out = {
        "patch": compute_patch(),
        "resolved": None if eval_mode == "none" else False,
        "report": None,
        "by_stage": stages,
        "telemetry": normalize(crewai_telemetry(crew, n_stages=len(stages))),
    }
    if eval_mode != "none" and out["patch"]:
        f2p, p2p = task.instance_tests(instance)
        out["report"] = run_tests_singularity(instance, out["patch"], f2p, p2p)
        out["resolved"] = is_resolved(out["report"])
    return out


def run_one(instance: dict, workdir_root: Path, out_dir: Path, eval_mode: str = "singularity") -> dict:
    """Clone, solve and score one instance; writes its patch, prediction and stage trace under ``out_dir``."""
    summary, out = task.solve_in_checkout(
        instance, workdir_root, _set_repo_dir, lambda: solve(instance, eval_mode=eval_mode)
    )
    if out is None:
        return summary
    summary.update(out.get("telemetry") or {})
    patch = out["patch"] or ""
    summary["patch_chars"] = len(patch)
    iid = instance["instance_id"]
    trace = task.sections((stage.upper(), text) for stage, text in (out.get("by_stage") or {}).items())
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
        description="Sequential-topology SWE-bench Verified agent (CrewAI).",
        run_batch=_run_instances,
        default_out_dir=DEFAULT_OUT_DIR,
        eval_modes=("singularity", "none"),
    )


if __name__ == "__main__":
    raise SystemExit(main())
