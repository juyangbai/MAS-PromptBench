"""Centralized topology specialized for SWE-bench Verified, AutoGen."""

# Config
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from topologies.output_contracts import append_output_contract_from_path
from typing import Sequence

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage
from autogen_agentchat.teams import SelectorGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Shared telemetry.
_TOPO_ROOT = str(Path(__file__).resolve().parents[4])
if _TOPO_ROOT not in sys.path:
    sys.path.insert(0, _TOPO_ROOT)
from topologies.telemetry import autogen_telemetry, normalize  # noqa: E402

# Infrastructure reuse from sequential/crewai/swe: clone + SIF eval + dataset
# loader + patch extraction. All aligned to single/swe's scorer.
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "sequential" / "crewai" / "swe"))
import crewai_swe as _crewai_swe  # noqa: E402
from crewai_swe import (  # noqa: E402
    clone_and_checkout,
    compute_patch,
    is_resolved,
    load_instances,
    run_tests_singularity,
)


VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://lai:8001/v1")
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3.5-9B")

_REPO_ROOT = Path(__file__).resolve().parents[4]
_PROMPTS_DIR = _REPO_ROOT / "configs" / "prompts" / "centralized" / "swe"

_SHELL_TIMEOUT_S = 60
_READ_CHAR_BUDGET = 20_000
_PROBLEM_CHAR_BUDGET = int(os.environ.get("SWE_PROBLEM_CHAR_BUDGET", "16000"))
_HINTS_CHAR_BUDGET = int(os.environ.get("SWE_HINTS_CHAR_BUDGET", "4000"))


def _load_prompt(role: str) -> str:
    return append_output_contract_from_path((_PROMPTS_DIR / f"{role}.txt").read_text().strip(), __file__, role)


def _set_repo_dir(path: Path | str) -> None:
    """Bind this task's repo workdir in the shared SWE helper module."""
    _crewai_swe._set_repo_dir(path)


# Tools (plain Python callables; AutoGen wraps them)
def _repo_path(path: str) -> Path:
    """Resolve `path` against the current REPO_DIR; refuse to escape."""
    repo = _crewai_swe._get_repo_dir()
    candidate = (repo / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
    candidate.relative_to(repo)  # raises ValueError if path escapes
    return candidate


def file_read(path: str, offset: int = 0, limit: int | None = None) -> str:
    """Read a file from the repository working directory.

    `offset` is a 0-indexed starting line; `limit` caps the number of
    lines. Output is truncated to ~20000 characters as a safety net.
    """
    try:
        p = _repo_path(path)
        content = p.read_text(errors="replace")
    except Exception as e:
        return f"ERROR: {e}"
    total = content.count("\n") + 1
    if offset or limit is not None:
        lines = content.splitlines(keepends=True)
        start = max(0, int(offset))
        end = start + int(limit) if limit is not None else len(lines)
        content = (
            f"[lines {start + 1}-{min(end, len(lines))} of {total}]\n"
            + "".join(lines[start:end])
        )
    if len(content) > _READ_CHAR_BUDGET:
        return content[:_READ_CHAR_BUDGET] + f"\n... [truncated, total {len(content)} chars]"
    return content


def str_replace(path: str, old: str, new: str) -> str:
    """Replace EXACTLY ONE occurrence of `old` with `new` in `path`.

    Narrow-anchor edit tool: `old` must match exactly once in the file,
    otherwise the tool errors (so no accidental catastrophic overwrite).
    Include enough surrounding context in `old` to make the match
    unique.
    """
    try:
        p = _repo_path(path)
        content = p.read_text(errors="replace")
    except Exception as e:
        return f"ERROR: {e}"

    count = content.count(old)
    if count == 0:
        return (
            f"ERROR: `old` not found in {path}. Check for whitespace, "
            "line endings, or tab/space mismatch. Call file_read to inspect."
        )
    if count > 1:
        return (
            f"ERROR: `old` appears {count} times in {path}; edit would be "
            "ambiguous. Add more surrounding lines to `old` so the match is unique."
        )
    new_content = content.replace(old, new, 1)
    try:
        p.write_text(new_content)
    except Exception as e:
        return f"ERROR: {e}"
    return f"replaced 1 occurrence in {path}"


def list_dir(path: str = ".") -> str:
    """List entries in a directory under the repository workdir."""
    try:
        p = _repo_path(path)
        if not p.is_dir():
            return f"ERROR: {path} is not a directory"
        entries = sorted(p.iterdir(), key=lambda e: (not e.is_dir(), e.name))
        repo = _crewai_swe._get_repo_dir()
        return "\n".join(
            f"{'d' if e.is_dir() else 'f'}  {e.relative_to(repo)}" for e in entries
        )
    except Exception as e:
        return f"ERROR: {e}"


def search_repo(pattern: str, path: str = ".", max_matches: int = 50) -> str:
    """grep-style regex search under the repository workdir."""
    try:
        target = _repo_path(path)
    except Exception as e:
        return f"ERROR: {e}"
    try:
        r = subprocess.run(
            ["grep", "-rn", "-E", pattern, str(target)],
            capture_output=True, text=True, timeout=_SHELL_TIMEOUT_S,
        )
    except Exception as e:
        return f"ERROR: {e}"
    if r.returncode not in (0, 1):
        return f"ERROR: grep exit {r.returncode}\n{r.stderr}"
    lines = r.stdout.splitlines()
    if not lines:
        return f"[no matches for {pattern!r}]"
    if len(lines) > max_matches:
        extra = len(lines) - max_matches
        lines = lines[:max_matches] + [f"... [+{extra} more]"]
    prefix = str(_crewai_swe._get_repo_dir()) + "/"
    return "\n".join(line.replace(prefix, "") for line in lines)


def shell_exec(command: str, timeout_s: int = _SHELL_TIMEOUT_S) -> str:
    """Run a shell command in the repository working directory."""
    try:
        r = subprocess.run(
            command, shell=True, cwd=str(_crewai_swe._get_repo_dir()),
            capture_output=True, text=True, timeout=timeout_s,
        )
        return f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}\nexit_code: {r.returncode}"
    except subprocess.TimeoutExpired:
        return f"ERROR: exceeded {timeout_s}s"
    except Exception as e:
        return f"ERROR: {e}"


# LLM
def _build_client() -> OpenAIChatCompletionClient:
    return OpenAIChatCompletionClient(
        model=MODEL_ID,
        base_url=VLLM_BASE_URL,
        api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        model_info={
            "vision": False,
            "function_calling": True,
            "json_output": True,
            "family": "qwen",
            "structured_output": False,
        },
        temperature=0.2,
        top_p=0.9,
        seed=0,
        # SWE stages need room for fenced code + test output summaries.
        max_tokens=4096,
        extra_body={
            "repetition_penalty": 1.05,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )


# Team
_MANAGER_TERMINATE_NUDGE = (
    "\n\nWhen you have a patch applied to the repo and are satisfied "
    "with it, emit a short summary of the changes and immediately "
    "follow with the literal string TERMINATE on its own line so the "
    "group-chat knows to stop. The final patch is extracted from the "
    "workdir via `git diff HEAD` — you do NOT need to re-print it."
)


def build_team() -> SelectorGroupChat:
    client = _build_client()

    manager = AssistantAgent(
        "manager",
        description="Coordinator for the SWE pipeline; plans, dispatches, synthesizes.",
        model_client=client,
        system_message=_load_prompt("manager") + _MANAGER_TERMINATE_NUDGE,
        # Manager has read-only inspection tools; it delegates edits to
        # patcher_worker and tests to tester_worker.
        tools=[file_read, list_dir, search_repo],
    )

    navigator_worker = AssistantAgent(
        "navigator_worker",
        description="Read-only repo exploration (list_dir, search_repo, file_read).",
        model_client=client,
        system_message=_load_prompt("navigator_worker"),
        tools=[file_read, list_dir, search_repo],
    )

    patcher_worker = AssistantAgent(
        "patcher_worker",
        description="Targeted file edits (file_read + str_replace).",
        model_client=client,
        system_message=(
            _load_prompt("patcher_worker")
            + "\n\nYou speak only to the manager. Always file_read the "
              "target file BEFORE str_replace; str_replace requires an "
              "exact-unique `old` substring. Include enough surrounding "
              "context in `old` to make the match unique. Do NOT emit "
              "TERMINATE — the manager decides when the task is done."
        ),
        tools=[file_read, str_replace],
    )

    tester_worker = AssistantAgent(
        "tester_worker",
        description="Shell-based sanity checks (shell_exec + file_read).",
        model_client=client,
        system_message=(
            _load_prompt("tester_worker")
            + "\n\nYou speak only to the manager. Do NOT emit TERMINATE."
        ),
        tools=[shell_exec, file_read],
    )

    # Force manager-routing: after any worker speaks, the manager MUST be
    # the next speaker (so workers never chain turns with each other).
    def _selector_func(messages: Sequence[BaseAgentEvent | BaseChatMessage]) -> str | None:
        if not messages:
            return manager.name
        if messages[-1].source != manager.name:
            return manager.name
        return None

    selector_prompt = (
        "You are coordinating a 4-agent team resolving a GitHub issue.\n"
        "Select the next agent to act.\n\n{roles}\n\n"
        "Conversation so far:\n{history}\n\n"
        "Pick exactly one agent from {participants}."
    )

    # SWE pipelines need more turns than the textual benchmarks (manager
    # will typically do 2-5 navigator cycles + 1-2 patcher cycles + 1-2
    # tester cycles).
    termination = TextMentionTermination("TERMINATE") | MaxMessageTermination(30)

    return SelectorGroupChat(
        [manager, navigator_worker, patcher_worker, tester_worker],
        model_client=client,
        termination_condition=termination,
        selector_prompt=selector_prompt,
        selector_func=_selector_func,
        allow_repeated_speaker=True,
    )


# Prompt scaffolding
def _truncate(text: str, cap: int, label: str) -> str:
    if len(text) <= cap:
        return text
    return text[:cap] + f"\n... [truncated {label}: {len(text)} -> {cap} chars]"


def format_task_brief(
    problem_statement: str,
    instance_id: str | None = None,
    hints_text: str | None = None,
) -> str:
    parts = []
    if instance_id:
        parts.append(f"INSTANCE: {instance_id}")
    parts.append(f"The repository is checked out at {_crewai_swe._get_repo_dir()} on the failing commit.")
    parts.append(
        "ISSUE:\n"
        + _truncate(problem_statement.strip(), _PROBLEM_CHAR_BUDGET, "problem_statement")
    )
    if hints_text:
        parts.append(
            "HINTS (from maintainers):\n"
            + _truncate(hints_text.strip(), _HINTS_CHAR_BUDGET, "hints_text")
        )
    parts.append(
        "Do NOT try to run the repo's own tests here — this workdir is only a "
        "source checkout; C extensions and test deps are NOT installed. Tests "
        "will be run separately in a prepared environment."
    )
    return "\n\n".join(parts)


# Orchestration
async def solve_async(instance: dict, eval_mode: str = "singularity") -> dict:
    """Run the centralized team on one SWE-bench instance.

    Assumes the workdir for this instance has already been cloned +
    checked out and `_set_repo_dir(...)` has been called.

    Returns:
        {
            "patch":    unified diff (git diff HEAD) or "",
            "resolved": bool (None if eval_mode='none'),
            "report":   scorer output dict (None if eval_mode='none'),
            "messages": list of {source, content} from every turn,
        }
    """
    team = build_team()
    brief = format_task_brief(
        instance["problem_statement"],
        instance_id=instance.get("instance_id"),
        hints_text=instance.get("hints_text"),
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

    patch = compute_patch()
    telem = normalize(autogen_telemetry(result))

    if eval_mode == "none" or not patch:
        return {
            "patch": patch,
            "resolved": None if eval_mode == "none" else False,
            "report": None,
            "messages": messages,
            "telemetry": telem,
        }

    f2p = instance["FAIL_TO_PASS"]
    p2p = instance["PASS_TO_PASS"]
    if isinstance(f2p, str):
        f2p = json.loads(f2p)
    if isinstance(p2p, str):
        p2p = json.loads(p2p)
    report = run_tests_singularity(instance, patch, f2p, p2p)
    return {
        "patch": patch,
        "resolved": is_resolved(report),
        "report": report,
        "messages": messages,
        "telemetry": telem,
    }


def solve(instance: dict, eval_mode: str = "singularity") -> dict:
    return asyncio.run(solve_async(instance, eval_mode))


# Predictions + batch runner
def predictions_entry(
    instance_id: str, patch: str,
    model_name: str = "mas-promptbench-centralized",
) -> dict:
    return {
        "instance_id": instance_id,
        "model_patch": patch,
        "model_name_or_path": model_name,
    }


def run_one(
    instance: dict,
    workdir_root: Path,
    out_dir: Path,
    eval_mode: str = "singularity",
) -> dict:
    """Clone + solve + score one SWE-bench Verified instance end-to-end."""
    iid = instance["instance_id"]
    repo = instance["repo"]
    base_commit = instance["base_commit"]
    summary: dict = {"instance_id": iid, "repo": repo, "base_commit": base_commit}
    workdir = workdir_root / iid

    t0 = time.time()
    err = clone_and_checkout(repo, base_commit, workdir)
    if err:
        summary["error"] = err
        summary["stage"] = "clone"
        return summary
    summary["clone_s"] = round(time.time() - t0, 1)

    _set_repo_dir(workdir)

    t0 = time.time()
    try:
        out = solve(instance, eval_mode=eval_mode)
    except Exception as e:
        summary["error"] = f"{type(e).__name__}: {e}"
        summary["stage"] = "solve"
        return summary
    summary["solve_s"] = round(time.time() - t0, 1)
    summary.update(out.get("telemetry") or {})

    patch = out["patch"] or ""
    summary["patch_chars"] = len(patch)
    summary["n_messages"] = len(out.get("messages") or [])

    (out_dir / "patches").mkdir(parents=True, exist_ok=True)
    (out_dir / "patches" / f"{iid}.diff").write_text(patch)
    with (out_dir / "predictions.jsonl").open("a") as f:
        f.write(json.dumps(predictions_entry(iid, patch)) + "\n")

    # Dump the group-chat message trace.
    (out_dir / "traces").mkdir(parents=True, exist_ok=True)
    with (out_dir / "traces" / f"{iid}.txt").open("w") as f:
        for m in out.get("messages") or []:
            src = m.get("source", "?")
            content = m.get("content", "")
            f.write(f"=== {src.upper()} ===\n{content}\n\n")

    if eval_mode == "none":
        summary["eval"] = "skipped"
        return summary

    report = out.get("report")
    if report is None:
        summary["eval_mode"] = eval_mode
        summary["resolved"] = False
        summary["f2p_rate"] = 0.0
        summary["p2p_rate"] = 0.0
        return summary

    summary["eval_mode"] = eval_mode
    summary["f2p_rate"] = report["f2p_rate"]
    summary["p2p_rate"] = report["p2p_rate"]
    summary["resolved"] = is_resolved(report)
    summary["f2p_failures"] = report["fail_to_pass"]["failure"]
    summary["p2p_failures"] = report["pass_to_pass"]["failure"]
    if report.get("error"):
        summary["eval_error"] = report["error"]
    return summary


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
    """Iterate Verified instances and run_one() each — matches single/swe's
    batch flow so the official harness post-processing works identically."""
    import shutil as _sh

    workdir_root = workdir_root or Path(
        f"{os.path.expanduser('~')}/swe_work_centralized"
    )
    out_dir = out_dir or (
        _REPO_ROOT / "results" / "swe_bench_centralized"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "predictions.jsonl").write_text("")
    results_path = out_dir / "results.jsonl"
    results_path.write_text("")

    instances = load_instances(subset, limit, offset, only)
    print(f"loaded {len(instances)} instance(s) from princeton-nlp/SWE-bench_Verified",
          file=sys.stderr)

    for i, inst in enumerate(instances, 1):
        print(
            f"\n[{i}/{len(instances)}] {inst['instance_id']}  "
            f"({inst['repo']}@{inst['base_commit'][:7]})",
            file=sys.stderr,
        )
        summary = run_one(inst, workdir_root, out_dir, eval_mode=eval_mode)
        with results_path.open("a") as f:
            f.write(json.dumps(summary) + "\n")
        print(f"  -> {json.dumps(summary)}", file=sys.stderr)

        if not keep_workdirs:
            _sh.rmtree(workdir_root / inst["instance_id"], ignore_errors=True)

    print(f"\ndone: predictions -> {out_dir / 'predictions.jsonl'}", file=sys.stderr)
    print(f"      results     -> {results_path}", file=sys.stderr)
    if eval_mode != "none":
        resolved = sum(
            1 for line in results_path.read_text().splitlines()
            if json.loads(line).get("resolved") is True
        )
        print(f"      resolved ({eval_mode}): {resolved}/{len(instances)}",
              file=sys.stderr)


# Demo / CLI
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Centralized-topology SWE-bench Verified agent (AutoGen)."
    )
    parser.add_argument("--subset", default="test")
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--only", action="append", default=None,
                        metavar="INSTANCE_ID")
    parser.add_argument(
        "--workdir-root",
        default=f"{os.path.expanduser('~')}/swe_work_centralized",
    )
    _default_out = str(_REPO_ROOT / "results" / "swe_bench_centralized")
    parser.add_argument("--out-dir", default=_default_out)
    parser.add_argument("--eval", dest="eval_mode", default="singularity",
                        choices=["singularity", "none"])
    parser.add_argument("--keep-workdirs", action="store_true")
    args = parser.parse_args()

    run_batch(
        subset=args.subset,
        limit=args.limit if not args.only else None,
        offset=args.offset,
        only=args.only,
        workdir_root=Path(args.workdir_root).expanduser(),
        out_dir=Path(args.out_dir).expanduser(),
        eval_mode=args.eval_mode,
        keep_workdirs=args.keep_workdirs,
    )
