"""Centralized topology specialized for SWE-bench Verified, LangGraph."""

# Config
from __future__ import annotations

import argparse
import asyncio  # noqa: F401
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from topologies.output_contracts import append_output_contract_from_path
from typing import Annotated, Optional

from typing_extensions import TypedDict

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, create_react_agent

# Shared telemetry.
_REPO_ROOT = Path(__file__).resolve().parents[4]
_TOPO_ROOT = str(_REPO_ROOT)
if _TOPO_ROOT not in sys.path:
    sys.path.insert(0, _TOPO_ROOT)
from topologies.telemetry import langchain_telemetry, normalize  # noqa: E402


VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://n12:8000/v1")
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3.5-9B")

_PROMPTS_DIR = _REPO_ROOT / "configs" / "prompts" / "centralized" / "swe"

# Per-instance repo workdir. Stored as a ContextVar so concurrent threads
# (when concurrent_runner.py drives many instances per shard) each see their
# own workdir. Module-globals would race: one thread would overwrite
# another's _set_repo_dir() and patches would be written against the wrong
# repo (yielding diffs that mention django files for an astropy issue).
from contextvars import ContextVar  # noqa: E402
_REPO_DIR_VAR: ContextVar[Path] = ContextVar(
    "_REPO_DIR_VAR",
    default=Path(os.environ.get("SWE_REPO_DIR", ".")).resolve(),
)

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

# Same cap as AutoGen sibling's MaxMessageTermination(30).
MAX_TURNS = 30


def _load_prompt(role: str) -> str:
    return append_output_contract_from_path((_PROMPTS_DIR / f"{role}.txt").read_text().strip(), __file__, role)


def _set_repo_dir(path: Path | str) -> None:
    """Re-bind the per-thread repo workdir via a ContextVar so concurrent
    instances do not race on a shared global."""
    _REPO_DIR_VAR.set(Path(path).resolve())


# Tools (LangChain @tool)
def _repo_path(path: str) -> Path:
    """Resolve `path` against the current thread's REPO_DIR; refuse to
    escape the workdir."""
    repo = _REPO_DIR_VAR.get()
    candidate = (
        (repo / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
    )
    try:
        candidate.relative_to(repo)
    except ValueError as e:
        raise ValueError(f"path {path!r} escapes repo workdir {repo}") from e
    return candidate


@tool("file_read")
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


@tool("str_replace")
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


@tool("list_dir")
def list_dir(path: str = ".") -> str:
    """List entries in a directory under the repository workdir."""
    try:
        p = _repo_path(path)
        if not p.is_dir():
            return f"ERROR: {path} is not a directory"
        entries = sorted(p.iterdir(), key=lambda e: (not e.is_dir(), e.name))
        return "\n".join(
            f"{'d' if e.is_dir() else 'f'}  {e.relative_to(_REPO_DIR_VAR.get())}" for e in entries
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
    prefix = str(_REPO_DIR_VAR.get()) + "/"
    return "\n".join(line.replace(prefix, "") for line in lines)


@tool("shell_exec")
def shell_exec(command: str, timeout_s: int = _SHELL_TIMEOUT_S) -> str:
    """Run a shell command in the repository working directory."""
    try:
        r = subprocess.run(
            command, shell=True, cwd=str(_REPO_DIR_VAR.get()),
            capture_output=True, text=True, timeout=timeout_s,
        )
        return f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}\nexit_code: {r.returncode}"
    except subprocess.TimeoutExpired:
        return f"ERROR: exceeded {timeout_s}s"
    except Exception as e:
        return f"ERROR: {e}"


# Delegation tools (routing markers)
# The manager "calls" these to hand the floor to a specific worker. The
# body echoes the instructions, producing a ToolMessage the worker can
# read as context. The router after `manager_tools` inspects the name
# to route to the right worker node.
@tool("delegate_to_navigator_worker")
def delegate_to_navigator_worker(instructions: str) -> str:
    """Hand the next turn to the navigator_worker. Use this when you need
    the navigator to explore the repo (list_dir / search_repo / file_read)
    and localize the bug or gather context.

    Args:
        instructions: what you want the navigator to look up this turn.
    """
    return instructions


@tool("delegate_to_patcher_worker")
def delegate_to_patcher_worker(instructions: str) -> str:
    """Hand the next turn to the patcher_worker. Use this when you know
    which file(s) and line(s) to edit; the patcher has file_read +
    str_replace (narrow-anchor edit — ambiguous matches are rejected).

    Args:
        instructions: what edit the patcher should make this turn.
    """
    return instructions


@tool("delegate_to_tester_worker")
def delegate_to_tester_worker(instructions: str) -> str:
    """Hand the next turn to the tester_worker. Use this when a patch
    has been applied and you want sanity checks (syntax compile, import
    smoke, git diff inspection) via shell_exec + file_read.

    Args:
        instructions: what check the tester should run this turn.
    """
    return instructions


DELEGATION_TOOLS = [
    delegate_to_navigator_worker,
    delegate_to_patcher_worker,
    delegate_to_tester_worker,
]
DELEGATION_NAMES = {t.name for t in DELEGATION_TOOLS}

# Per-worker tool subsets (match AutoGen sibling exactly).
NAVIGATOR_TOOLS = [file_read, list_dir, search_repo]
PATCHER_TOOLS = [file_read, str_replace]
TESTER_TOOLS = [shell_exec, file_read]

# Manager has read-only inspection tools + 3 delegation markers.
MANAGER_TOOLS = [file_read, list_dir, search_repo] + DELEGATION_TOOLS


# LLM
def _build_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=MODEL_ID,
        base_url=VLLM_BASE_URL,
        api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        temperature=0.2,
        top_p=0.9,
        seed=0,
        # SWE stages need room for fenced code + test output summaries.
        max_tokens=8192,
        extra_body={
            "repetition_penalty": 1.05,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )


# System-prompt nudges
_MANAGER_TERMINATE_NUDGE = (
    "\n\nWhen you have a patch applied to the repo and are satisfied "
    "with it, emit a short summary of the changes and immediately "
    "follow with the literal string TERMINATE on its own line so the "
    "group-chat knows to stop. The final patch is extracted from the "
    "workdir via `git diff HEAD` — you do NOT need to re-print it.\n\n"
    "Delegation: when you want a specific worker to act, call the "
    "matching delegate_to_<worker> tool with clear instructions "
    "(instead of merely addressing them in free-form text). The three "
    "workers are: navigator_worker, patcher_worker, tester_worker."
)

_PATCHER_NUDGE = (
    "\n\nYou speak only to the manager. Always file_read the target "
    "file BEFORE str_replace; str_replace requires an exact-unique "
    "`old` substring. Include enough surrounding context in `old` to "
    "make the match unique. Do NOT emit TERMINATE — the manager "
    "decides when the task is done."
)

_TESTER_NUDGE = (
    "\n\nYou speak only to the manager. Do NOT emit TERMINATE."
)


# State + nodes
class CentralizedState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    turn_count: int


def _tag_source(msg: BaseMessage, source: str) -> None:
    """Attach an AutoGen-style `source` field via additional_kwargs."""
    try:
        kw = dict(getattr(msg, "additional_kwargs", None) or {})
        kw["source"] = source
        msg.additional_kwargs = kw
    except Exception:
        pass


def _manager_system() -> str:
    return _load_prompt("manager") + _MANAGER_TERMINATE_NUDGE


def _manager_node(state: CentralizedState) -> dict:
    llm = _build_llm().bind_tools(MANAGER_TOOLS)
    sys_msg = SystemMessage(content=_manager_system())
    ai = llm.invoke([sys_msg] + state["messages"])
    # AutoGen messages carry a `.source` name; we mimic that on the
    # AIMessage via additional_kwargs for trace rendering parity.
    _tag_source(ai, "manager")
    return {"messages": [ai], "turn_count": int(state.get("turn_count", 0)) + 1}


_manager_tool_node = ToolNode(MANAGER_TOOLS)


def _route_from_manager(state: CentralizedState) -> str:
    msgs = state["messages"]
    if not msgs:
        return "manager"
    last = msgs[-1]
    if int(state.get("turn_count", 0)) >= MAX_TURNS:
        return END
    if isinstance(last, AIMessage):
        content = last.content or ""
        if isinstance(content, str) and "TERMINATE" in content:
            return END
        if getattr(last, "tool_calls", None):
            return "manager_tools"
    # No tool call, no TERMINATE — loop back and let the manager try again.
    return "manager"


def _route_from_manager_tools(state: CentralizedState) -> str:
    # Find the most recent AIMessage with tool_calls; its tool_calls tell
    # us whether any delegation was requested.
    for m in reversed(state["messages"]):
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            for tc in m.tool_calls:
                name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
                if name in DELEGATION_NAMES:
                    return name.removeprefix("delegate_to_")
            # Only real-tool calls were made; loop back to manager.
            return "manager"
    return "manager"


def _make_worker_node(name: str, tools: list, llm: ChatOpenAI, extra_prompt: str = ""):
    sys_prompt = _load_prompt(name) + extra_prompt
    agent = create_react_agent(model=llm, tools=tools, prompt=sys_prompt)

    def node(state: CentralizedState) -> dict:
        # create_react_agent returns {"messages": [full history incl. input]},
        # so we splice out only the new messages it appended.
        prior = list(state["messages"])
        result = agent.invoke(
            {"messages": prior},
            config={"recursion_limit": 30},
        )
        full = result["messages"]
        new_msgs = full[len(prior):]
        # Tag worker outputs with source for trace rendering.
        for m in new_msgs:
            if isinstance(m, AIMessage):
                _tag_source(m, name)
        # Each new AIMessage counts as one turn; tool-executions don't.
        n_turns = sum(1 for m in new_msgs if isinstance(m, AIMessage))
        return {"messages": new_msgs, "turn_count": int(state.get("turn_count", 0)) + n_turns}

    return node


def _build_graph(llm: Optional[ChatOpenAI] = None):
    if llm is None:
        llm = _build_llm()

    graph = StateGraph(CentralizedState)
    graph.add_node("manager", _manager_node)
    graph.add_node("manager_tools", _manager_tool_node)

    worker_specs = [
        ("navigator_worker", NAVIGATOR_TOOLS, ""),
        ("patcher_worker", PATCHER_TOOLS, _PATCHER_NUDGE),
        ("tester_worker", TESTER_TOOLS, _TESTER_NUDGE),
    ]
    for name, tools, extra in worker_specs:
        graph.add_node(name, _make_worker_node(name, tools, llm, extra))

    graph.add_edge(START, "manager")
    graph.add_conditional_edges(
        "manager",
        _route_from_manager,
        {
            "manager_tools": "manager_tools",
            "manager": "manager",
            END: END,
        },
    )
    graph.add_conditional_edges(
        "manager_tools",
        _route_from_manager_tools,
        {
            "navigator_worker": "navigator_worker",
            "patcher_worker": "patcher_worker",
            "tester_worker": "tester_worker",
            "manager": "manager",
        },
    )
    for name, _, _extra in worker_specs:
        graph.add_edge(name, "manager")

    return graph.compile(), ["manager"] + [n for n, _, _ in worker_specs]


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
    parts.append(f"The repository is checked out at {_REPO_DIR_VAR.get()} on the failing commit.")
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
            ["git", "-C", str(_REPO_DIR_VAR.get()), "diff", "HEAD"],
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
# Aligned to topologies/single/swe/langgraph_swe.py so centralized
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
def _communications_source(m: BaseMessage) -> str:
    kw = getattr(m, "additional_kwargs", None) or {}
    src = kw.get("source")
    if src:
        return src
    t = getattr(m, "type", None)
    return {"human": "user", "ai": "assistant", "tool": "tool"}.get(t, t or "?")


def _communications_to_record(m: BaseMessage) -> dict:
    content = getattr(m, "content", "") or ""
    if not isinstance(content, str):
        content = str(content)
    return {"source": _communications_source(m), "content": content}


def solve(instance: dict, eval_mode: str = "singularity") -> dict:
    """Run the centralized team on one SWE-bench instance.

    Assumes the workdir for this instance has already been cloned +
    checked out and `_set_repo_dir(...)` has been called.

    Returns:
        {
            "patch":    unified diff (git diff HEAD) or "",
            "resolved": bool (None if eval_mode='none'),
            "report":   scorer output dict (None if eval_mode='none'),
            "messages": list of {source, content} from every turn,
            "telemetry": normalized 5-key token/call counts,
        }
    """
    compiled, _ = _build_graph()
    brief = format_task_brief(
        instance["problem_statement"],
        instance_id=instance.get("instance_id"),
        hints_text=instance.get("hints_text"),
    )
    result = compiled.invoke(
        {"messages": [HumanMessage(content=brief)], "turn_count": 0},
        config={"recursion_limit": MAX_TURNS * 4},
    )
    msgs = result.get("messages") or []
    rendered = [_communications_to_record(m) for m in msgs]

    patch = compute_patch()
    telem = normalize(langchain_telemetry(msgs))

    if eval_mode == "none" or not patch:
        return {
            "patch": patch,
            "resolved": None if eval_mode == "none" else False,
            "report": None,
            "messages": rendered,
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
        "messages": rendered,
        "telemetry": telem,
    }


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
            f.write(f"=== {str(src).upper()} ===\n{content}\n\n")

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
        f"{os.path.expanduser('~')}/swe_work_centralized_langgraph"
    )
    out_dir = out_dir or (
        _REPO_ROOT / "results" / "swe_bench_centralized_langgraph"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "predictions.jsonl").write_text("")
    results_path = out_dir / "results.jsonl"
    results_path.write_text("")

    instances = load_instances(subset, limit, offset, only)
    print(
        f"loaded {len(instances)} instance(s) from princeton-nlp/SWE-bench_Verified",
        file=sys.stderr,
    )

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
        description="Centralized-topology SWE-bench Verified agent (LangGraph)."
    )
    parser.add_argument("--subset", default="test")
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--only", action="append", default=None,
                        metavar="INSTANCE_ID")
    parser.add_argument(
        "--workdir-root",
        default=f"{os.path.expanduser('~')}/swe_work_centralized_langgraph",
    )
    _default_out = str(_REPO_ROOT / "results" / "swe_bench_centralized_langgraph")
    parser.add_argument("--out-dir", default=_default_out)
    parser.add_argument("--eval", dest="eval_mode", default="singularity",
                        choices=["singularity", "none"])
    parser.add_argument("--skip-eval", dest="eval_mode", action="store_const",
                        const="none",
                        help="alias for --eval none")
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
