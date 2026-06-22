"""Single-agent ReAct topology specialized for SWE-bench (Verified)."""

# Config
import json
import os
import re
import subprocess
import sys
import time
from contextvars import ContextVar
from pathlib import Path

from topologies.output_contracts import append_output_contract_from_path

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

# Shared telemetry.
_TOPO_ROOT = str(Path(__file__).resolve().parents[3])
if _TOPO_ROOT not in sys.path:
    sys.path.insert(0, _TOPO_ROOT)
from topologies.telemetry import langchain_telemetry, normalize  # noqa: E402


VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://lai:8001/v1")
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3.5-9B")

# Per-instance repo workdir. Store it in a ContextVar so concurrent_runner.py
# can safely evaluate multiple SWE instances in parallel within one process.
_REPO_DIR_VAR: ContextVar[Path] = ContextVar(
    "_REPO_DIR_VAR",
    default=Path(os.environ.get("SWE_REPO_DIR", ".")).resolve(),
)


def _get_repo_dir() -> Path:
    return _REPO_DIR_VAR.get()

# Singularity eval: per-instance SIFs, pulled on demand from docker://swebench.
SWE_SIF_DIR = Path(
    os.environ.get("SWE_SIF_DIR", f"{Path.home()}/containers/swe")
).resolve()
_SWEBENCH_IMAGE = "docker://swebench/sweb.eval.x86_64.{tag}:latest"
_SIF_PULL_TIMEOUT_S = 900          # 15 min max per pull
_SIF_EVAL_TIMEOUT_S = 1800         # 30 min max per instance test run

_SHELL_TIMEOUT_S = 60
_READ_CHAR_BUDGET = 20_000   # cap single file_read output to keep prompts small


# System Prompt
_REPO_ROOT = Path(__file__).resolve().parents[3]
_PROMPT_PATH = _REPO_ROOT / "configs" / "prompts" / "single" / "swe" / "solver.txt"
SYSTEM_PROMPT = append_output_contract_from_path(_PROMPT_PATH.read_text().strip(), __file__, _PROMPT_PATH.stem)


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


@tool
def file_read(path: str, offset: int = 0, limit: int | None = None) -> str:
    """Read a file from the repository working directory.

    Path is interpreted relative to the repo root. `offset` is a 0-indexed
    starting line number; `limit` caps the number of lines returned. Output
    is also truncated to ~20000 characters as a safety net; for large files,
    use search_repo first to locate the line range, then pass offset/limit.
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


@tool
def file_write(path: str, content: str) -> str:
    """Overwrite (or create) a file in the repository with `content`.

    Creates parent directories if needed.
    """
    try:
        p = _repo_path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    except Exception as e:
        return f"ERROR: {e}"
    return f"wrote {len(content)} chars to {path}"


@tool
def list_dir(path: str = ".") -> str:
    """List entries in a directory under the repository workdir."""
    try:
        p = _repo_path(path)
        if not p.is_dir():
            return f"ERROR: {path} is not a directory"
        entries = sorted(p.iterdir(), key=lambda e: (not e.is_dir(), e.name))
        return "\n".join(
            f"{'d' if e.is_dir() else 'f'}  {e.relative_to(_get_repo_dir())}" for e in entries
        )
    except Exception as e:
        return f"ERROR: {e}"


@tool
def search_repo(pattern: str, path: str = ".", max_matches: int = 50) -> str:
    """grep-style search under the repository workdir.

    Uses `grep -rn` with fixed-string matching disabled (so `pattern` is a
    regex). Returns up to `max_matches` lines as "file:line:content".
    """
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
    if result.returncode not in (0, 1):    # 1 = no matches, anything else = failure
        return f"ERROR: grep exited {result.returncode}\n{result.stderr}"
    lines = result.stdout.splitlines()
    if not lines:
        return f"[no matches for {pattern!r} in {path}]"
    if len(lines) > max_matches:
        lines = lines[:max_matches] + [f"... [+{len(result.stdout.splitlines()) - max_matches} more]"]
    # make paths relative for brevity
    prefix = str(_get_repo_dir()) + "/"
    return "\n".join(line.replace(prefix, "") for line in lines)


@tool
def shell_exec(command: str, timeout_s: int = _SHELL_TIMEOUT_S) -> str:
    """Run a shell command in the repository working directory.

    Captures stdout, stderr, and exit code. Timeout defaults to 60 s.
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


TOOLS = [file_read, file_write, list_dir, search_repo, shell_exec]


# Agent
# Cap prompt-side text so the initial message fits the model context even
# with pathological issue bodies or hints. Tool calls still add tokens on
# top; the vLLM max_model_len must be large enough for at least a few
# rounds of file_read output (recommend >=32k).
_PROBLEM_CHAR_BUDGET = int(os.environ.get("SWE_PROBLEM_CHAR_BUDGET", "16000"))
_HINTS_CHAR_BUDGET = int(os.environ.get("SWE_HINTS_CHAR_BUDGET", "4000"))


def _truncate(text: str, cap: int, label: str) -> str:
    """Return `text` trimmed to `cap` chars with a visible truncation note."""
    if len(text) <= cap:
        return text
    return text[:cap] + f"\n... [truncated {label}: {len(text)} -> {cap} chars]"


def format_prompt(
    problem_statement: str,
    instance_id: str | None = None,
    hints_text: str | None = None,
) -> str:
    """Build the user-facing prompt for one SWE-bench instance.

    Includes the issue body, optional hints (from the `hints_text` field of
    the dataset), and instructions on the tool workflow. Long issue bodies
    and hints are truncated via SWE_PROBLEM_CHAR_BUDGET / SWE_HINTS_CHAR_BUDGET.
    """
    parts = []
    if instance_id:
        parts.append(f"INSTANCE: {instance_id}")
    parts.append(f"The repository is checked out at {_get_repo_dir()} on the failing commit.")
    parts.append("ISSUE:\n" + _truncate(problem_statement.strip(), _PROBLEM_CHAR_BUDGET, "problem_statement"))
    if hints_text:
        parts.append("HINTS (from maintainers):\n" + _truncate(hints_text.strip(), _HINTS_CHAR_BUDGET, "hints_text"))
    parts.append(
        "Use the available tools (file_read, file_write, list_dir, search_repo, "
        "shell_exec) to investigate the codebase and apply a fix. Modify files "
        "in place with file_write.\n"
        "\n"
        "Do NOT try to run the repo's own code or tests here — this workdir is "
        "only a source checkout; C extensions and test deps are NOT installed. "
        "Tests will be run separately in a prepared environment. Focus on "
        "reading source files to understand the bug, then write a fix.\n"
        "\n"
        "When done, do NOT hand-write a diff — the harness computes the patch "
        "from the git state of the workdir."
    )
    return "\n\n".join(parts)


def build_agent():
    llm = ChatOpenAI(
        model=MODEL_ID,
        base_url=VLLM_BASE_URL,
        api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        # Greedy (temp=0) traps Qwen3.5-9B in read-more-code loops on SWE-bench
        # without ever committing to a patch. Light stochastic sampling with a
        # fixed seed keeps runs reproducible per seed while breaking those loops.
        temperature=0.2,
        top_p=0.9,
        seed=0,
        extra_body={
            "repetition_penalty": 1.05,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    return create_react_agent(model=llm, tools=TOOLS, prompt=SYSTEM_PROMPT)


# Output Parsing
def strip_thinking(text: str) -> str:
    """Cut everything up through the last </think> tag."""
    index = text.lower().rfind("</think>")
    if index >= 0:
        text = text[index + len("</think>"):]
    return text.strip()


def compute_patch() -> str:
    """Return the unified-diff patch for the current workdir state.

    Runs `git diff HEAD` inside REPO_DIR. An empty string means the agent
    made no changes to tracked files.
    """
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


def extract_answer(text: str) -> str | None:
    """For symmetry with other topologies — but the real 'answer' for SWE
    is the patch, computed separately via compute_patch()."""
    return None


# Scoring
# SWE-bench grading follows swebench.harness.grading:
#   - an instance is RESOLVED iff every FAIL_TO_PASS test passes AND every
#     PASS_TO_PASS test still passes after the patch is applied
#   - XFAIL counts as passing; tests not present in the run are silently
#     skipped by the official parser
# The official harness evaluates in a per-repo Docker image. What this file
# provides is a best-effort LOCAL runner via pytest against the current
# workdir; useful for development, not numerically equivalent to the
# leaderboard. For official scoring use:
#   from swebench.harness.run_evaluation import main
# (requires Docker, per-instance images, and the swebench pip package).


def _run_pytest(test_ids: list[str], timeout_s: int = 300) -> dict:
    """Run a list of pytest node ids inside REPO_DIR. Returns per-id status.

    test_ids are pytest-style node identifiers (path/to/test_file.py::TestClass::test_name).
    """
    if not test_ids:
        return {}
    cmd = [
        "python", "-m", "pytest",
        "-p", "no:cacheprovider",
        "--tb=no", "-v", "--no-header",
        "-o", "console_output_style=classic",
        *test_ids,
    ]
    try:
        result = subprocess.run(
            cmd,
            cwd=str(_get_repo_dir()),
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return {tid: "timeout" for tid in test_ids}

    # Parse classic console output: each line "path::name PASSED|FAILED|ERROR|XFAIL|SKIPPED"
    status: dict[str, str] = {}
    line_re = re.compile(r"^(?P<nodeid>\S+?)\s+(?P<verdict>PASSED|FAILED|ERROR|XFAIL|XPASS|SKIPPED)\b")
    for line in (result.stdout + "\n" + result.stderr).splitlines():
        m = line_re.match(line.strip())
        if m:
            status[m.group("nodeid")] = m.group("verdict")

    # For any test_id not observed, record as "not_run"
    for tid in test_ids:
        status.setdefault(tid, "not_run")
    return status


_PASSING_VERDICTS = {"PASSED", "XFAIL"}


def run_tests_local(
    fail_to_pass: list[str],
    pass_to_pass: list[str],
    timeout_s: int = 600,
) -> dict:
    """Run both test groups with pytest against the current REPO_DIR state.

    Returns:
        {
            "fail_to_pass": {"success": [...], "failure": [...]},
            "pass_to_pass": {"success": [...], "failure": [...]},
            "f2p_rate": float, "p2p_rate": float,
        }
    """
    f2p_status = _run_pytest(fail_to_pass, timeout_s=timeout_s)
    p2p_status = _run_pytest(pass_to_pass, timeout_s=timeout_s)

    def _bucket(status_map: dict) -> dict:
        success, failure = [], []
        for tid, verdict in status_map.items():
            (success if verdict in _PASSING_VERDICTS else failure).append(tid)
        return {"success": success, "failure": failure}

    f2p = _bucket(f2p_status)
    p2p = _bucket(p2p_status)

    return {
        "fail_to_pass": f2p,
        "pass_to_pass": p2p,
        "f2p_rate": (len(f2p["success"]) / len(fail_to_pass)) if fail_to_pass else 1.0,
        "p2p_rate": (len(p2p["success"]) / len(pass_to_pass)) if pass_to_pass else 1.0,
    }


def is_resolved(report: dict) -> bool:
    """SWE-bench RESOLVED verdict: both rates must equal 1.0 (strict).

    Matches swebench.harness.grading.ResolvedStatus.FULL.
    """
    return report.get("f2p_rate") == 1.0 and report.get("p2p_rate") == 1.0


def exact_match_score(report: dict) -> float:
    """1.0 iff the instance is resolved, else 0.0 (SWE-bench primary metric)."""
    return 1.0 if is_resolved(report) else 0.0


# ---- Singularity eval path --------------------------------------------------
# Runs the model patch + pytest inside the per-instance SIF image pulled from
# docker://swebench/sweb.eval.x86_64.<tag>. This matches the official Docker
# harness's environment (same image contents, same Python, same pinned deps)
# and sidesteps the "repo deps don't match my env" problem that plagues local
# pytest runs.
#
# The inline script carries:
#   1. a safe.directory workaround via GIT_CONFIG_GLOBAL (git 2.34.1 in these
#      images + host UID != 0 triggers git's dubious-ownership check)
#   2. conda activate testbed (the repo's installed env lives there, not base)
#   3. git apply test_patch  (SWE-bench stores test modifications separately;
#      these introduce the parametrize ids named in FAIL_TO_PASS)
#   4. git apply model patch
#   5. pytest with classic output so the verdict regex in run_tests_local parses

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
    """Return the SIF path for `instance_id`, pulling on first use.

    Cache lives under `SWE_SIF_DIR` (default `~/containers/swe/<iid>.sif`).
    Raises RuntimeError if the pull fails.
    """
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
    """Evaluate a model patch inside the instance's per-repo SIF image.

    Returns the same shape as `run_tests_local` so `is_resolved()` works
    unchanged. An empty `patch` runs the baseline tests (no changes applied).
    A `patch` that fails to apply yields f2p_rate=p2p_rate=0 with an
    "error: patch apply failed" field.
    """
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
            "bash",       # $0 for the inline script
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

    # Persist raw pytest output for debugging parse issues.
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

    # Parse classic pytest output: "path::name VERDICT ..."
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


def apply_test_patch(test_patch: str) -> str:
    """Apply the dataset's `test_patch` to the workdir.

    Each SWE-bench instance ships a `test_patch` that installs the evaluation
    tests (often NEW test files that didn't exist at base_commit). It must be
    applied AFTER the agent finishes and BEFORE running tests so (a) the agent
    is judged on hidden tests and (b) pytest can discover them.

    Returns an empty string on success, or a non-empty error message.
    """
    if not test_patch.strip():
        return ""
    try:
        result = subprocess.run(
            ["git", "-C", str(_get_repo_dir()), "apply", "--verbose", "--recount", "-"],
            input=test_patch,
            capture_output=True,
            text=True,
            timeout=_SHELL_TIMEOUT_S,
        )
    except Exception as e:
        return f"ERROR: {e}"
    if result.returncode != 0:
        return f"ERROR: git apply exited {result.returncode}\n{result.stderr}"
    return ""


def predictions_entry(instance_id: str, patch: str, model_name: str = "mas-promptbench-single") -> dict:
    """Build one line of the predictions JSONL consumed by the official harness.

    Schema expected by swebench.harness.run_evaluation:
        {"instance_id": str, "model_patch": str, "model_name_or_path": str}
    """
    return {
        "instance_id": instance_id,
        "model_patch": patch,
        "model_name_or_path": model_name,
    }


# Orchestration
def solve(
    problem_statement: str,
    instance_id: str | None = None,
    hints_text: str | None = None,
) -> dict:
    """Run the agent on one SWE-bench instance.

    Assumes REPO_DIR already points at a checkout of the target repo at
    `base_commit`. The caller is responsible for: clone, checkout, pip
    install, and resetting any prior state.

    Returns {'patch': str, 'raw': str, 'messages': list}.
    """
    agent = build_agent()
    result = agent.invoke(
        {"messages": [("user", format_prompt(problem_statement, instance_id, hints_text))]},
        config={"recursion_limit": 100},
    )
    for msg in result["messages"]:
        if msg.type == "ai" and isinstance(msg.content, str):
            msg.content = strip_thinking(msg.content)
    final = result["messages"][-1].content
    return {
        "patch": compute_patch(),
        "raw": final,
        "messages": result["messages"],
    }


# Dataset & Workspace
def _set_repo_dir(path: Path | str) -> None:
    """Re-bind this task's repo workdir so tool functions see the right checkout."""
    _REPO_DIR_VAR.set(Path(path).resolve())


def clone_and_checkout(repo: str, base_commit: str, workdir: Path) -> str:
    """Clone https://github.com/{repo} into `workdir` and detach at `base_commit`.

    Returns "" on success or a non-empty error string.
    """
    url = f"https://github.com/{repo}.git"
    workdir.parent.mkdir(parents=True, exist_ok=True)
    if workdir.exists():
        import shutil as _sh
        _sh.rmtree(workdir)
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
    """Load rows from princeton-nlp/SWE-bench_Verified. Requires `datasets`."""
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


# Batch Runner
def run_one(
    instance: dict,
    workdir_root: Path,
    out_dir: Path,
    eval_mode: str = "local",
) -> dict:
    """Solve one Verified instance end-to-end and write its artifacts.

    Side effects:
        - creates workdir_root / instance_id  (clone of repo at base_commit)
        - writes out_dir / patches / <iid>.diff
        - appends to out_dir / predictions.jsonl

    Returns a summary dict (timings, tool_calls, f2p/p2p rates if local_eval).
    """
    iid = instance["instance_id"]
    repo = instance["repo"]
    base_commit = instance["base_commit"]
    test_patch = instance.get("test_patch") or ""
    f2p = instance["FAIL_TO_PASS"]
    p2p = instance["PASS_TO_PASS"]
    if isinstance(f2p, str):
        f2p = json.loads(f2p)
    if isinstance(p2p, str):
        p2p = json.loads(p2p)

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
        out = solve(
            problem_statement=instance["problem_statement"],
            instance_id=iid,
            hints_text=instance.get("hints_text") or None,
        )
    except Exception as e:
        summary["error"] = f"{type(e).__name__}: {e}"
        summary["stage"] = "solve"
        return summary
    summary["solve_s"] = round(time.time() - t0, 1)

    patch = out["patch"] or ""
    summary["patch_chars"] = len(patch)
    summary["tool_calls"] = sum(
        1 for m in out["messages"] if getattr(m, "type", None) == "tool"
    )
    summary.update(normalize(langchain_telemetry(out.get("messages") or [])))

    (out_dir / "patches").mkdir(parents=True, exist_ok=True)
    (out_dir / "patches" / f"{iid}.diff").write_text(patch)
    with (out_dir / "predictions.jsonl").open("a") as f:
        f.write(json.dumps(predictions_entry(iid, patch)) + "\n")

    # Dump the full message trace so we can inspect what the agent did.
    (out_dir / "traces").mkdir(parents=True, exist_ok=True)
    with (out_dir / "traces" / f"{iid}.txt").open("w") as f:
        for msg in out["messages"]:
            mtype = getattr(msg, "type", "?")
            content = getattr(msg, "content", "")
            tcalls = getattr(msg, "tool_calls", None) or []
            f.write(f"=== {mtype.upper()} ===\n")
            if tcalls:
                for tc in tcalls:
                    name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "?")
                    args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {})
                    f.write(f"[tool_call] {name}({json.dumps(args, default=str)[:500]})\n")
            if content:
                f.write(f"{content}\n")
            f.write("\n")

    if eval_mode == "none":
        summary["eval"] = "skipped"
        return summary

    t0 = time.time()
    if eval_mode == "singularity":
        try:
            report = run_tests_singularity(instance, patch, f2p, p2p)
        except Exception as e:
            summary["error"] = f"{type(e).__name__}: {e}"
            summary["stage"] = "singularity_eval"
            return summary
    else:   # "local"
        if test_patch:
            err = apply_test_patch(test_patch)
            if err:
                summary["error"] = err
                summary["stage"] = "apply_test_patch"
                return summary
        report = run_tests_local(f2p, p2p)

    summary["eval_mode"] = eval_mode
    summary["eval_s"] = round(time.time() - t0, 1)
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
    eval_mode: str = "local",
    keep_workdirs: bool = False,
) -> None:
    """Iterate over a Verified slice, solve each instance, write predictions.

    eval_mode:
        "local"       pytest in the host env (fast, env-fragile)
        "singularity" pytest inside the per-instance SIF (authoritative, matches Docker harness)
        "none"        skip eval, just collect patches
    """
    import shutil as _sh

    workdir_root = workdir_root or Path(f"{os.path.expanduser('~')}/swe_work")
    out_dir = out_dir or (_REPO_ROOT / "results" / "swe_bench")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "predictions.jsonl").write_text("")   # truncate; append per instance
    results_path = out_dir / "results.jsonl"
    results_path.write_text("")

    instances = load_instances(subset, limit, offset, only)
    print(f"loaded {len(instances)} instance(s) from princeton-nlp/SWE-bench_Verified", file=sys.stderr)

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
        print(f"      resolved ({eval_mode}): {resolved}/{len(instances)}", file=sys.stderr)
    print(
        "\nFor leaderboard-comparable scoring, run the official Docker harness:\n"
        f"    python -m swebench.harness.run_evaluation \\\n"
        f"        --predictions_path {out_dir / 'predictions.jsonl'} \\\n"
        f"        --max_workers 4 --run_id mas-promptbench_single_swe",
        file=sys.stderr,
    )


# Demo
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Single-topology agent on SWE-bench Verified.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  %(prog)s                                        # 2 instances, local pytest eval\n"
            "  %(prog)s --limit 1 --eval singularity           # 1 instance, eval inside SIF\n"
            "  %(prog)s --only astropy__astropy-12907 --eval singularity\n"
            "  %(prog)s --limit 50 --eval none                 # collect patches only, score with Docker harness later"
        ),
    )
    parser.add_argument("--subset", default="test", help="HF split name on Verified (default: test)")
    parser.add_argument("--limit", type=int, default=2, help="max number of instances to run (default: 2)")
    parser.add_argument("--offset", type=int, default=0, help="skip the first N instances")
    parser.add_argument("--only", action="append", default=None, metavar="INSTANCE_ID",
                        help="run only these instance ids (repeatable)")
    parser.add_argument("--workdir-root", default=f"{os.path.expanduser('~')}/swe_work",
                        help="root dir for per-instance clones")
    parser.add_argument("--out-dir", default=str(_REPO_ROOT / "results" / "swe_bench"),
                        help="where predictions.jsonl and results.jsonl are written")
    parser.add_argument("--eval", dest="eval_mode", default="local",
                        choices=["local", "singularity", "none"],
                        help="eval backend: local pytest (default), per-instance SIF, or skip")
    parser.add_argument("--keep-workdirs", action="store_true",
                        help="don't delete per-instance clones after solving")
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
