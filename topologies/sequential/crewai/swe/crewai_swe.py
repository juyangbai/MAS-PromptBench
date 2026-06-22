"""Sequential topology specialized for SWE-bench Verified, in CrewAI."""

# Config
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from contextvars import ContextVar
from pathlib import Path

from topologies.output_contracts import append_output_contract_from_path

from crewai import LLM, Agent, Crew, Process, Task
from crewai.tools import tool

# Shared telemetry.
_TOPO_ROOT = str(Path(__file__).resolve().parents[4])
if _TOPO_ROOT not in sys.path:
    sys.path.insert(0, _TOPO_ROOT)
from topologies.telemetry import crewai_telemetry, normalize  # noqa: E402


VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://lai:8001/v1")
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3.5-9B")

# Legacy fallback repo workdir. Current per-instance runs set the ContextVar via
# _set_repo_dir(), which keeps concurrent_runner.py workers isolated.
REPO_DIR = Path(os.environ.get("SWE_REPO_DIR", ".")).resolve()
_REPO_DIR_VAR: ContextVar[Path | None] = ContextVar("_REPO_DIR_VAR", default=None)

SWE_SIF_DIR = Path(
    os.environ.get("SWE_SIF_DIR", f"{Path.home()}/containers/swe")
).resolve()
_SWEBENCH_IMAGE = "docker://swebench/sweb.eval.x86_64.{tag}:latest"
_SIF_PULL_TIMEOUT_S = 900
_SIF_EVAL_TIMEOUT_S = 1800

_SHELL_TIMEOUT_S = 60
_READ_CHAR_BUDGET = 20_000
_PROBLEM_CHAR_BUDGET = int(os.environ.get("SWE_PROBLEM_CHAR_BUDGET", "16000"))
_HINTS_CHAR_BUDGET = int(os.environ.get("SWE_HINTS_CHAR_BUDGET", "4000"))

_REPO_ROOT = Path(__file__).resolve().parents[4]
_PROMPTS_DIR = _REPO_ROOT / "configs" / "prompts" / "sequential" / "swe"


def _load_prompt(role: str) -> str:
    return append_output_contract_from_path((_PROMPTS_DIR / f"{role}.txt").read_text().strip(), __file__, role)


def _set_repo_dir(path: Path | str) -> None:
    """Bind this task's repo workdir so tool wrappers see the right checkout."""
    global REPO_DIR
    REPO_DIR = Path(path).resolve()
    _REPO_DIR_VAR.set(REPO_DIR)


def _get_repo_dir() -> Path:
    return _REPO_DIR_VAR.get() or REPO_DIR


# Tools
def _repo_path(path: str) -> Path:
    """Resolve `path` against REPO_DIR; refuse to escape the workdir."""
    repo = _get_repo_dir()
    candidate = (repo / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
    try:
        candidate.relative_to(repo)
    except ValueError as e:
        raise ValueError(f"path {path!r} escapes repo workdir {repo}") from e
    return candidate


@tool("file_read")
def file_read(path: str, offset: int = 0, limit: int | None = None) -> str:
    """Read a file from the repository working directory.

    Path is interpreted relative to the repo root. `offset` is a 0-indexed
    starting line number; `limit` caps the number of lines returned.
    Output truncated to ~20000 characters as a safety net.
    """
    try:
        p = _repo_path(path)
        content = p.read_text(errors="replace")
    except Exception as e:
        return f"ERROR: {e}"

    total_lines = content.count("\n") + 1
    if offset or limit is not None:
        lines = content.splitlines(keepends=True)
        start = max(0, int(offset))
        end = start + int(limit) if limit is not None else len(lines)
        sliced = "".join(lines[start:end])
        header = f"[lines {start + 1}-{min(end, len(lines))} of {total_lines}]\n"
        content = header + sliced

    if len(content) > _READ_CHAR_BUDGET:
        return content[:_READ_CHAR_BUDGET] + f"\n... [truncated, total {len(content)} chars]"
    return content


@tool("str_replace")
def str_replace(path: str, old: str, new: str) -> str:
    """Replace EXACTLY ONE occurrence of `old` with `new` in the file at `path`.

    This is a targeted-edit tool: it only rewrites the matched region,
    leaving the rest of the file untouched. Use it instead of `file_write`
    for bug fixes — `file_write` overwrites the ENTIRE file, which is
    almost never what you want when you're changing a few lines.

    Include enough surrounding context in `old` to uniquely identify the
    location. If `old` is not found, returns an error. If `old` appears
    more than once, returns an error listing the match count — add more
    context to disambiguate.

    Args:
        path: file path relative to the repository working directory.
        old:  the exact substring to replace (with enough surrounding
              lines to be unique).
        new:  the replacement substring. Use "" to delete `old`.

    Returns a success message with a short preview, or an ERROR string.
    """
    try:
        p = _repo_path(path)
        content = p.read_text(errors="replace")
    except Exception as e:
        return f"ERROR: {e}"

    count = content.count(old)
    if count == 0:
        # Give a hint: if a line starts with this text it's probably a
        # whitespace or newline mismatch.
        return (
            f"ERROR: `old` not found in {path}. "
            "Check for whitespace, line endings, or tab/space mismatch. "
            "If you need to inspect the file, call file_read first."
        )
    if count > 1:
        return (
            f"ERROR: `old` appears {count} times in {path}; edit would be "
            "ambiguous. Add more surrounding lines to `old` so the match "
            "is unique."
        )
    new_content = content.replace(old, new, 1)
    try:
        p.write_text(new_content)
    except Exception as e:
        return f"ERROR: {e}"
    preview_old = old if len(old) < 120 else old[:120] + "..."
    preview_new = new if len(new) < 120 else new[:120] + "..."
    return (
        f"replaced 1 occurrence in {path}\n"
        f"  old: {preview_old!r}\n"
        f"  new: {preview_new!r}"
    )


@tool("file_write")
def file_write(path: str, content: str) -> str:
    """Overwrite (or create) a file in the repository with `content`."""
    try:
        p = _repo_path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    except Exception as e:
        return f"ERROR: {e}"
    return f"wrote {len(content)} chars to {path}"


@tool("list_dir")
def list_dir(path: str = ".") -> str:
    """List entries in a directory under the repository workdir."""
    try:
        p = _repo_path(path)
        if not p.is_dir():
            return f"ERROR: {path} is not a directory"
        entries = sorted(p.iterdir(), key=lambda e: (not e.is_dir(), e.name))
        repo = _get_repo_dir()
        return "\n".join(
            f"{'d' if e.is_dir() else 'f'}  {e.relative_to(repo)}" for e in entries
        )
    except Exception as e:
        return f"ERROR: {e}"


@tool("search_repo")
def search_repo(pattern: str, path: str = ".", max_matches: int = 50) -> str:
    """grep-style regex search under the repository workdir."""
    try:
        target = _repo_path(path)
    except Exception as e:
        return f"ERROR: {e}"
    try:
        result = subprocess.run(
            ["grep", "-rn", "-E", pattern, str(target)],
            capture_output=True,
            text=True,
            timeout=_SHELL_TIMEOUT_S,
        )
    except Exception as e:
        return f"ERROR: {e}"
    if result.returncode not in (0, 1):
        return f"ERROR: grep exited {result.returncode}\n{result.stderr}"
    lines = result.stdout.splitlines()
    if not lines:
        return f"[no matches for {pattern!r} in {path}]"
    if len(lines) > max_matches:
        lines = lines[:max_matches] + [
            f"... [+{len(result.stdout.splitlines()) - max_matches} more]"
        ]
    prefix = str(_get_repo_dir()) + "/"
    return "\n".join(line.replace(prefix, "") for line in lines)


@tool("shell_exec")
def shell_exec(command: str, timeout_s: int = _SHELL_TIMEOUT_S) -> str:
    """Run a shell command in the repository working directory.

    Captures stdout, stderr, and exit code. Timeout defaults to 60 s.
    Useful for the Tester stage to run `python -c "import module"` sanity
    checks or `git diff` to inspect the patch.
    """
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=str(_get_repo_dir()),
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        return (
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}\n"
            f"exit_code: {result.returncode}"
        )
    except subprocess.TimeoutExpired:
        return f"ERROR: command exceeded {timeout_s}s timeout"
    except Exception as e:
        return f"ERROR: {e}"


# LLM
def _build_llm() -> LLM:
    return LLM(
        model=f"openai/{MODEL_ID}",
        base_url=VLLM_BASE_URL,
        api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        temperature=0.2,
        top_p=0.9,
        seed=0,
        max_tokens=4096,
        extra_body={
            "repetition_penalty": 1.05,
            "chat_template_kwargs": {"enable_thinking": False},
        },
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
    """Build a task brief with issue + optional hints. Used as the static
    portion of every Task.description (CrewAI's context= chain carries
    stage-to-stage hand-offs; this is the per-instance variable input)."""
    parts = []
    if instance_id:
        parts.append(f"INSTANCE: {instance_id}")
    parts.append(f"The repository is checked out at {_get_repo_dir()} on the failing commit.")
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


# Crew
def build_crew(llm: LLM | None = None) -> Crew:
    """Build the 4-stage investigator -> planner -> patcher -> tester pipeline."""
    if llm is None:
        llm = _build_llm()

    investigator = Agent(
        role="SWE Investigator",
        goal=(
            "Explore the repo, localize the bug, and hand off a precise "
            "pointer (file path + line range + root-cause explanation). "
            "Do NOT write any code."
        ),
        backstory=_load_prompt("investigator"),
        tools=[file_read, list_dir, search_repo],
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    planner = Agent(
        role="SWE Planner",
        goal=(
            "Design the fix strategy from the Investigator's pointer: "
            "which lines change, what the new code should do, how it "
            "relates to the failing test. Do NOT write code."
        ),
        backstory=_load_prompt("planner"),
        tools=[],
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    patcher = Agent(
        role="SWE Patcher",
        goal=(
            "Write the code edit following the Planner's strategy. Use "
            "file_read to confirm the lines you plan to change; use "
            "str_replace(path, old, new) to make TARGETED edits. Keep "
            "the change minimal."
        ),
        # Narrow tool surface: only str_replace for edits (NOT file_write).
        # file_write overwrites the entire file — a single bad call can
        # clobber 300+ lines if the model emits a fragment. str_replace
        # only rewrites the exact `old` region, making that failure mode
        # structurally impossible.
        backstory=_load_prompt("patcher"),
        tools=[file_read, str_replace],
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    tester = Agent(
        role="SWE Tester",
        goal=(
            "Sanity-check the patched workdir via shell_exec (syntax "
            "compile, import smoke tests, `git diff` inspection). The "
            "real FAIL_TO_PASS eval runs externally; your job is to "
            "catch obvious breakage."
        ),
        backstory=_load_prompt("tester"),
        tools=[shell_exec, file_read],
        llm=llm,
        verbose=False,
        allow_delegation=False,
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
            "A short sanity-check report ending with 'Patch looks "
            "wholesome' or 'Patch has problems: <summary>'."
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


# Patch + Parsing
def strip_thinking(text: str) -> str:
    """Cut everything up through the last </think> tag (Qwen3 convention)."""
    index = text.lower().rfind("</think>")
    if index >= 0:
        text = text[index + len("</think>"):]
    return text.strip()


def compute_patch() -> str:
    """Return the unified-diff patch for the current workdir state."""
    try:
        result = subprocess.run(
            ["git", "-C", str(_get_repo_dir()), "diff", "HEAD"],
            capture_output=True,
            text=True,
            timeout=_SHELL_TIMEOUT_S,
        )
    except Exception as e:
        return f"ERROR: {e}"
    if result.returncode != 0:
        return f"ERROR: git diff exited {result.returncode}\n{result.stderr}"
    return result.stdout


# Clone + Dataset
def clone_and_checkout(repo: str, base_commit: str, workdir: Path) -> str:
    """Clone https://github.com/{repo} into `workdir`; detach at `base_commit`."""
    import shutil

    url = f"https://github.com/{repo}.git"
    workdir.parent.mkdir(parents=True, exist_ok=True)
    if workdir.exists():
        shutil.rmtree(workdir)
    r = subprocess.run(
        ["git", "clone", "--quiet", "--no-tags", url, str(workdir)],
        capture_output=True, text=True, timeout=900,
    )
    if r.returncode != 0:
        return f"clone failed: {r.stderr.strip()}"
    r = subprocess.run(
        ["git", "-C", str(workdir), "checkout", "--quiet", "--detach", base_commit],
        capture_output=True, text=True, timeout=60,
    )
    if r.returncode != 0:
        return f"checkout failed: {r.stderr.strip()}"
    subprocess.run(
        ["git", "-C", str(workdir), "config", "advice.detachedHead", "false"],
        capture_output=True, text=True,
    )
    return ""


def load_instances(
    subset: str = "test",
    limit: int | None = None,
    offset: int = 0,
    only: list[str] | None = None,
) -> list[dict]:
    """Load rows from princeton-nlp/SWE-bench_Verified."""
    from datasets import load_dataset

    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split=subset)
    rows = list(ds)
    if only:
        wanted = set(only)
        rows = [r for r in rows if r["instance_id"] in wanted]
    rows = rows[offset:]
    if limit is not None:
        rows = rows[:limit]
    return rows


# Scoring (Singularity)
# Aligned to topologies/single/swe/langgraph_swe.py so sequential
# resolve rates are directly comparable.
_PASSING_VERDICTS = {"PASSED", "XFAIL"}

_SIF_EVAL_SCRIPT = r"""
set -eo pipefail
cat > /tmp/gitcfg <<EOF
[safe]
    directory = /testbed
EOF
export GIT_CONFIG_GLOBAL=/tmp/gitcfg

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate testbed

cd /testbed
if [ -s /tmp/test.patch ]; then
    if ! git apply --verbose --recount /tmp/test.patch; then
        echo "__TEST_PATCH_APPLY_FAILED__"
        exit 3
    fi
fi
if [ -s /tmp/model.patch ]; then
    if ! git apply --verbose --recount /tmp/model.patch; then
        echo "__APPLY_FAILED__"
        exit 2
    fi
fi

python -m pytest \
    -p no:cacheprovider \
    -v --tb=no --no-header \
    -o console_output_style=classic \
    "$@"
"""


def _ensure_sif(instance_id: str) -> Path:
    sif = SWE_SIF_DIR / f"{instance_id}.sif"
    if sif.exists():
        return sif
    SWE_SIF_DIR.mkdir(parents=True, exist_ok=True)
    tag = instance_id.replace("__", "_1776_")
    docker_ref = _SWEBENCH_IMAGE.format(tag=tag)
    print(f"[swe] pulling {docker_ref} -> {sif.name}", file=sys.stderr)
    result = subprocess.run(
        ["singularity", "pull", str(sif), docker_ref],
        capture_output=True, text=True, timeout=_SIF_PULL_TIMEOUT_S,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"singularity pull failed for {instance_id}: {result.stderr.strip()}"
        )
    return sif


def run_tests_singularity(
    instance: dict,
    patch: str,
    fail_to_pass: list[str],
    pass_to_pass: list[str],
    timeout_s: int = _SIF_EVAL_TIMEOUT_S,
) -> dict:
    """Evaluate a model patch inside the instance's per-repo SIF image."""
    import tempfile

    iid = instance["instance_id"]
    sif = _ensure_sif(iid)
    test_patch = instance.get("test_patch") or ""

    test_ids = list(fail_to_pass) + list(pass_to_pass)
    if not test_ids:
        return {
            "fail_to_pass": {"success": [], "failure": []},
            "pass_to_pass": {"success": [], "failure": []},
            "f2p_rate": 1.0, "p2p_rate": 1.0,
        }

    with tempfile.NamedTemporaryFile("w", suffix=".patch", delete=False) as f:
        f.write(patch or "")
        patch_path = f.name
    with tempfile.NamedTemporaryFile("w", suffix=".patch", delete=False) as f:
        f.write(test_patch)
        test_patch_path = f.name

    try:
        cmd = [
            "singularity", "exec",
            "--writable-tmpfs",
            "--bind", f"{patch_path}:/tmp/model.patch:ro",
            "--bind", f"{test_patch_path}:/tmp/test.patch:ro",
            str(sif),
            "bash", "-c", _SIF_EVAL_SCRIPT,
            "bash",
            *test_ids,
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return {
            "fail_to_pass": {"success": [], "failure": list(fail_to_pass)},
            "pass_to_pass": {"success": [], "failure": list(pass_to_pass)},
            "f2p_rate": 0.0, "p2p_rate": 0.0,
            "error": "timeout",
        }
    finally:
        for p in (patch_path, test_patch_path):
            try:
                os.unlink(p)
            except FileNotFoundError:
                pass

    combined = (result.stdout or "") + "\n" + (result.stderr or "")
    eval_log_dir = Path(os.environ.get("SWE_EVAL_LOG_DIR", "/tmp"))
    eval_log_dir.mkdir(parents=True, exist_ok=True)
    (eval_log_dir / f"eval_{iid}.log").write_text(combined)

    if "__TEST_PATCH_APPLY_FAILED__" in combined:
        return {
            "fail_to_pass": {"success": [], "failure": list(fail_to_pass)},
            "pass_to_pass": {"success": [], "failure": list(pass_to_pass)},
            "f2p_rate": 0.0, "p2p_rate": 0.0,
            "error": "test_patch apply failed",
            "stderr_tail": combined[-500:],
        }
    if "__APPLY_FAILED__" in combined:
        return {
            "fail_to_pass": {"success": [], "failure": list(fail_to_pass)},
            "pass_to_pass": {"success": [], "failure": list(pass_to_pass)},
            "f2p_rate": 0.0, "p2p_rate": 0.0,
            "error": "patch apply failed",
            "stderr_tail": combined[-500:],
        }

    line_re = re.compile(
        r"^(?P<nodeid>\S+?)\s+(?P<verdict>PASSED|FAILED|ERROR|XFAIL|XPASS|SKIPPED)\b"
    )
    status: dict[str, str] = {}
    for line in combined.splitlines():
        m = line_re.match(line.strip())
        if m:
            status[m.group("nodeid")] = m.group("verdict")
    for tid in test_ids:
        status.setdefault(tid, "not_run")

    def _bucket(ids: list[str]) -> dict:
        success, failure = [], []
        for tid in ids:
            if status.get(tid) in _PASSING_VERDICTS:
                success.append(tid)
            else:
                failure.append(tid)
        return {"success": success, "failure": failure}

    f2p = _bucket(fail_to_pass)
    p2p = _bucket(pass_to_pass)
    return {
        "fail_to_pass": f2p,
        "pass_to_pass": p2p,
        "f2p_rate": (len(f2p["success"]) / len(fail_to_pass)) if fail_to_pass else 1.0,
        "p2p_rate": (len(p2p["success"]) / len(pass_to_pass)) if pass_to_pass else 1.0,
    }


def is_resolved(report: dict) -> bool:
    """SWE-bench RESOLVED verdict: both rates must equal 1.0 (strict)."""
    return report.get("f2p_rate") == 1.0 and report.get("p2p_rate") == 1.0


# Orchestration
def solve(instance: dict, eval_mode: str = "singularity") -> dict:
    """Run the 4-stage sequential crew on one SWE-bench instance.

    The caller sets the repo dir before calling (via _set_repo_dir, typically
    in the demo flow below). Returns:

        {
            "patch":     the unified-diff produced (str),
            "resolved":  bool (or None if eval_mode='none'),
            "report":    scorer output dict,
            "by_stage":  {investigator, planner, patcher, tester} stage outputs,
        }
    """
    brief = format_task_brief(
        instance["problem_statement"],
        instance_id=instance.get("instance_id"),
        hints_text=instance.get("hints_text"),
    )
    crew = build_crew()
    result = crew.kickoff(inputs={"task_brief": brief})

    stages = {}
    try:
        stages["investigator"] = result.tasks_output[0].raw
        stages["planner"]      = result.tasks_output[1].raw
        stages["patcher"]      = result.tasks_output[2].raw
        stages["tester"]       = result.tasks_output[3].raw
    except (AttributeError, IndexError):
        stages = {"investigator": "", "planner": "", "patcher": "", "tester": ""}

    patch = compute_patch()

    telem = normalize(crewai_telemetry(crew, n_stages=len(stages)))
    if eval_mode == "none" or not patch:
        return {
            "patch": patch,
            "resolved": None if eval_mode == "none" else False,
            "report": None,
            "by_stage": stages,
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
        "by_stage": stages,
        "telemetry": telem,
    }


# Predictions + batch runner
def predictions_entry(
    instance_id: str, patch: str,
    model_name: str = "mas-promptbench-sequential",
) -> dict:
    """Build one line of the predictions JSONL consumed by the official harness.

    Schema: {"instance_id": str, "model_patch": str, "model_name_or_path": str}
    """
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
    """Clone + solve + score one SWE-bench Verified instance end-to-end.

    Side effects:
        - creates workdir_root / instance_id  (fresh repo checkout)
        - writes out_dir / patches / <iid>.diff
        - appends to out_dir / predictions.jsonl
        - writes out_dir / traces / <iid>.txt (4-stage crew output)
    """
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

    (out_dir / "patches").mkdir(parents=True, exist_ok=True)
    (out_dir / "patches" / f"{iid}.diff").write_text(patch)
    with (out_dir / "predictions.jsonl").open("a") as f:
        f.write(json.dumps(predictions_entry(iid, patch)) + "\n")

    # Dump the 4-stage crew output so we can inspect what happened.
    (out_dir / "traces").mkdir(parents=True, exist_ok=True)
    with (out_dir / "traces" / f"{iid}.txt").open("w") as f:
        for stage, content in (out.get("by_stage") or {}).items():
            f.write(f"=== {stage.upper()} ===\n{content}\n\n")

    if eval_mode == "none":
        summary["eval"] = "skipped"
        return summary

    report = out.get("report")
    if report is None:
        # `solve()` returns report=None when patch was empty; treat as unresolved.
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
    """Iterate Verified instances, run_one() each, write predictions.jsonl +
    results.jsonl. Matches single/swe's batch flow so Docker-harness post-
    processing works identically across topologies.
    """
    import shutil as _sh

    workdir_root = workdir_root or Path(
        f"{os.path.expanduser('~')}/swe_work_sequential"
    )
    out_dir = out_dir or (
        Path(__file__).resolve().parents[4] / "results" / "swe_bench_sequential"
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
    parser = argparse.ArgumentParser(
        description="Sequential-topology SWE-bench Verified agent (CrewAI).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  %(prog)s                                        # 2 instances, singularity eval\n"
            "  %(prog)s --limit 5 --eval singularity\n"
            "  %(prog)s --only astropy__astropy-12907\n"
            "  %(prog)s --limit 50 --eval none                 # collect patches only"
        ),
    )
    parser.add_argument("--subset", default="test",
                        help="HF split on Verified (default: test)")
    parser.add_argument("--limit", type=int, default=2,
                        help="max number of instances (default: 2)")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--only", action="append", default=None,
                        metavar="INSTANCE_ID",
                        help="run only these instance ids (repeatable)")
    parser.add_argument(
        "--workdir-root",
        default=f"{os.path.expanduser('~')}/swe_work_sequential",
    )
    _default_out = str(
        Path(__file__).resolve().parents[4] / "results" / "swe_bench_sequential"
    )
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
