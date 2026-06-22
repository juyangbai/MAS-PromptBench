"""Decentralized debate topology specialized for SWE-bench Verified, LangGraph."""

# Config
from __future__ import annotations

import argparse
import json
import operator
import os
import re
import subprocess
import sys
import time
from contextvars import ContextVar
from pathlib import Path

from teamsizes.output_contracts import append_output_contract_from_path
from typing import Annotated, Any

from typing_extensions import TypedDict

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
)
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent

# Shared telemetry.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_TOPO_ROOT = str(_REPO_ROOT)
if _TOPO_ROOT not in sys.path:
    sys.path.insert(0, _TOPO_ROOT)
from topologies.telemetry import langchain_telemetry, normalize  # noqa: E402


VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://n12:8000/v1")
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3.5-9B")

N_AGENTS = int(os.environ.get("DECENTRALIZED_N_AGENTS", "4"))
N_ROUNDS = int(os.environ.get("DECENTRALIZED_N_ROUNDS", "2"))

# Match openai sibling's max_tool_loops=20. create_react_agent uses its own
# recursion_limit; give it enough headroom to cover the same tool-loop budget
# (roughly 3x max_tool_loops to account for alternating AI/Tool messages).
_MAX_TOOL_LOOPS = 20
_RECURSION_LIMIT = _MAX_TOOL_LOOPS * 3  # 60

_SHELL_TIMEOUT_S = 60
_READ_CHAR_BUDGET = 20_000
_PROBLEM_CHAR_BUDGET = int(os.environ.get("SWE_PROBLEM_CHAR_BUDGET", "16000"))
_HINTS_CHAR_BUDGET = int(os.environ.get("SWE_HINTS_CHAR_BUDGET", "4000"))

SWE_SIF_DIR = Path(
    os.environ.get("SWE_SIF_DIR", f"{Path.home()}/containers/swe")
).resolve()
_SWEBENCH_IMAGE = "docker://swebench/sweb.eval.x86_64.{tag}:latest"
_SIF_PULL_TIMEOUT_S = 900
_SIF_EVAL_TIMEOUT_S = 1800

_PROMPTS_DIR = _REPO_ROOT / "configs" / "prompts" / "decentralized" / "swe"


def _load_prompt(role: str) -> str:
    return append_output_contract_from_path((_PROMPTS_DIR / f"{role}.txt").read_text().strip(), __file__, role)


SYSTEM_PROMPT = _load_prompt("debater")


# Per-peer workdir tracking
# Each peer has its OWN clone of the repo — tools need to resolve paths
# against that peer's workdir, not a global one. ContextVar + per-turn
# bind keeps it simple and works under sequential peer execution inside
# the round_node (no threading concerns in this port).
_REPO_DIR_VAR: ContextVar[Path] = ContextVar("_REPO_DIR_VAR", default=Path("."))


def _get_repo_dir() -> Path:
    return _REPO_DIR_VAR.get()


def _repo_path(path: str) -> Path:
    repo = _get_repo_dir()
    candidate = (repo / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
    candidate.relative_to(repo)  # raises if escapes
    return candidate


# Tools (LangChain @tool — same 5 tools as openai sibling)
@tool
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


@tool
def str_replace(path: str, old: str, new: str) -> str:
    """Replace EXACTLY ONE occurrence of `old` with `new` in `path`.

    Include enough surrounding context in `old` to uniquely identify the
    location. Returns an error if `old` is not found or appears more
    than once.
    """
    try:
        p = _repo_path(path)
        content = p.read_text(errors="replace")
    except Exception as e:
        return f"ERROR: {e}"
    count = content.count(old)
    if count == 0:
        return (
            f"ERROR: `old` not found in {path}. Check whitespace / line "
            "endings / tab-space mismatch. Call file_read to inspect."
        )
    if count > 1:
        return (
            f"ERROR: `old` appears {count} times in {path}; add more "
            "surrounding context to make the match unique."
        )
    try:
        p.write_text(content.replace(old, new, 1))
    except Exception as e:
        return f"ERROR: {e}"
    return f"replaced 1 occurrence in {path}"


@tool
def list_dir(path: str = ".") -> str:
    """List entries in a directory under the repo workdir."""
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


@tool
def search_repo(pattern: str, path: str = ".", max_matches: int = 50) -> str:
    """grep-style regex search under the repo workdir."""
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
    prefix = str(_get_repo_dir()) + "/"
    return "\n".join(line.replace(prefix, "") for line in lines)


@tool
def shell_exec(command: str, timeout_s: int = _SHELL_TIMEOUT_S) -> str:
    """Run a shell command in the repo workdir (default 60s timeout)."""
    try:
        r = subprocess.run(
            command, shell=True, cwd=str(_get_repo_dir()),
            capture_output=True, text=True, timeout=timeout_s,
        )
        return f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}\nexit_code: {r.returncode}"
    except subprocess.TimeoutExpired:
        return f"ERROR: exceeded {timeout_s}s"
    except Exception as e:
        return f"ERROR: {e}"


TOOLS = [file_read, str_replace, list_dir, search_repo, shell_exec]


# LLM
def _build_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=MODEL_ID,
        base_url=VLLM_BASE_URL,
        api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        temperature=0.2,
        top_p=0.9,
        seed=0,
        # SWE edits can be verbose; give headroom.
        max_tokens=8192,
        extra_body={
            "repetition_penalty": 1.05,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )


def _build_agent():
    """Build a react agent carrying the shared 5-tool surface. Each peer
    invokes this agent with its OWN message history + its OWN bound
    `_REPO_DIR_VAR` context, so a single agent object is reused safely
    across peers + rounds."""
    return create_react_agent(model=_build_llm(), tools=TOOLS, prompt=SYSTEM_PROMPT)


# Prompt scaffolding (aligned to openai sibling)
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
    parts.append(f"The repository is checked out on the failing commit at your peer-local workdir.")
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


# Peer-injection template (aligned to openai sibling)
def _peer_injection(others_final: list[BaseMessage], brief: str) -> HumanMessage:
    body = [
        "These are the final summaries from other peer agents in the "
        "previous round (each peer worked in its OWN repo clone — their "
        "file changes are NOT visible in yours):"
    ]
    for i, m in enumerate(others_final):
        content = getattr(m, "content", "") or ""
        if not isinstance(content, str):
            content = str(content)
        body.append(f"\nPeer {i + 1}:\n```\n{content}\n```")
    body.append(
        "\nCompare their approach with yours. If a peer found a better fix "
        "location or caught a regression, REVISE your edits in your own "
        "workdir. Use str_replace to apply the revised fix; use shell_exec "
        "+ `git diff HEAD` to inspect the current state of your workdir.\n\n"
        "Original brief:\n" + brief
    )
    return HumanMessage(content="\n".join(body))


# State + round node
class DebateState(TypedDict, total=False):
    contexts: list[list[BaseMessage]]       # per-peer message histories (sans system)
    round_finals: list[list[BaseMessage]]   # per-round final AIMessages (one per peer)
    round: int
    brief: str
    peer_workdirs: list[Path]


def _last_ai(msgs: list[BaseMessage]) -> BaseMessage | None:
    """Return the last AIMessage with non-empty content (no tool_calls)."""
    for m in reversed(msgs):
        if isinstance(m, AIMessage) and (m.content or "") and not getattr(m, "tool_calls", None):
            return m
    # Fallback: any AIMessage
    for m in reversed(msgs):
        if isinstance(m, AIMessage):
            return m
    return None


def _invoke_peer_with_recovery(
    agent, ctx: list[BaseMessage],
) -> list[BaseMessage]:
    """Invoke the react agent with BadRequestError recovery, matching the
    openai sibling's `_chat_with_tools` 400-retry block.

    vLLM sometimes rejects a follow-up request when the prior assistant
    turn's tool_call `arguments` contain an unterminated JSON string
    (truncated at max_tokens, unescaped quotes, etc). Strip trailing
    tool-result + poisoned assistant messages and retry once with a
    user nudge so the peer can keep working.
    """
    try:
        result = agent.invoke(
            {"messages": ctx},
            config={"recursion_limit": _RECURSION_LIMIT},
        )
        return result["messages"]
    except Exception as e:
        # Narrow the recovery to LangChain/OpenAI BadRequest-like errors.
        # `langchain_openai` surfaces vLLM 400s via openai.BadRequestError.
        # We string-match on the type name so we never import openai here.
        etype = type(e).__name__
        is_bad_request = (
            "BadRequest" in etype
            or "BadRequest" in str(e)[:120]
            or "400" in str(e)[:20]
        )
        if not is_bad_request:
            raise

    # Recovery path: pop trailing tool messages + poisoned ai message, append nudge.
    repaired = list(ctx)
    # Remove trailing tool-role messages.
    while repaired and getattr(repaired[-1], "type", None) == "tool":
        repaired.pop()
    # Remove the poisoned assistant message.
    if repaired and getattr(repaired[-1], "type", None) == "ai":
        repaired.pop()
    repaired.append(HumanMessage(content=(
        "The previous tool call produced an invalid request. "
        "Continue with a different approach (do not repeat the same call)."
    )))
    try:
        result = agent.invoke(
            {"messages": repaired},
            config={"recursion_limit": _RECURSION_LIMIT},
        )
        return result["messages"]
    except Exception as e2:
        # Give up — publish a synthetic error AIMessage so best-of-N can
        # proceed (the peer's workdir may still carry partial edits).
        repaired.append(AIMessage(
            content=f"ERROR: peer crashed on retry: {type(e2).__name__}: {e2}"
        ))
        return repaired


def _round_node(state: DebateState) -> dict:
    """Run one debate round: iterate ALL N peers sequentially (see module
    docstring for why sequential vs threaded), each in its own ContextVar
    binding so tools land in its OWN workdir."""
    agent = _build_agent()
    r = int(state.get("round", 0))
    brief = state["brief"]
    peer_workdirs: list[Path] = state["peer_workdirs"]
    contexts = [list(c) for c in state["contexts"]]
    prev_finals = state.get("round_finals") or []

    this_round_finals: list[BaseMessage] = []

    for i, workdir in enumerate(peer_workdirs):
        ctx = contexts[i]
        if r > 0 and prev_finals:
            others = [prev_finals[-1][j] for j in range(len(contexts)) if j != i]
            ctx = ctx + [_peer_injection(others, brief)]

        # Bind this peer's workdir for the duration of the agent invocation
        # so the 5 @tool bodies resolve paths against peer_k/ and not a
        # sibling's workdir.
        token = _REPO_DIR_VAR.set(Path(workdir))
        try:
            try:
                new_msgs = _invoke_peer_with_recovery(agent, ctx)
            except Exception as e:
                new_msgs = ctx + [AIMessage(
                    content=f"ERROR: peer {i} round {r} crashed: {type(e).__name__}: {e}"
                )]
        finally:
            _REPO_DIR_VAR.reset(token)

        contexts[i] = new_msgs
        final = _last_ai(new_msgs) or AIMessage(content="")
        this_round_finals.append(final)

    return {
        "contexts": contexts,
        "round_finals": prev_finals + [this_round_finals],
        "round": r + 1,
    }


def _route(state: DebateState) -> str:
    return END if int(state.get("round", 0)) >= N_ROUNDS else "round"


def _build_graph():
    g = StateGraph(DebateState)
    g.add_node("round", _round_node)
    g.add_edge(START, "round")
    g.add_conditional_edges("round", _route, {"round": "round", END: END})
    return g.compile()


# Patch + Parsing
def strip_thinking(text: str) -> str:
    """Cut everything up through the last </think> tag (Qwen3 convention)."""
    index = text.lower().rfind("</think>")
    if index >= 0:
        text = text[index + len("</think>"):]
    return text.strip()


def compute_patch(repo_dir: Path) -> str:
    """Return the unified-diff patch for the given workdir's current state."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_dir), "diff", "HEAD"],
            capture_output=True, text=True, timeout=_SHELL_TIMEOUT_S,
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


# Scoring (aligned to single/swe / independent/swe)
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


# Best-of-N over peer workdirs
def best_of_n(
    peer_workdirs: list[Path],
    instance: dict,
    eval_mode: str = "singularity",
) -> tuple[int | None, list[dict]]:
    """Run compute_patch() + SIF eval on each peer's workdir; pick the
    first resolved peer (lowest index), else max f2p_rate × p2p_rate."""
    f2p = instance["FAIL_TO_PASS"]
    p2p = instance["PASS_TO_PASS"]
    if isinstance(f2p, str):
        f2p = json.loads(f2p)
    if isinstance(p2p, str):
        p2p = json.loads(p2p)

    # Compute patches sequentially (each peer has its OWN workdir, so
    # there's no shared-state race — `git diff HEAD` is fast and local).
    scored: list[dict] = []
    for i, workdir in enumerate(peer_workdirs):
        scored.append({
            "peer": i, "workdir": str(workdir),
            "patch": compute_patch(Path(workdir)),
            "report": None, "resolved": False, "score": 0.0,
        })

    # Run Singularity evals sequentially — see module docstring on the
    # sequential trade-off. (OpenAI sibling parallelizes SIF evals via
    # threads; this port keeps them simple. Each eval spawns its own
    # container, no shared Python state, so it would parallelize safely
    # if wall-time becomes a concern later.)
    if eval_mode != "none":
        for rec in scored:
            if not rec["patch"]:
                continue
            try:
                report = run_tests_singularity(instance, rec["patch"], f2p, p2p)
            except Exception as e:
                rec["report"] = {"error": f"{type(e).__name__}: {e}"}
                continue
            rec["report"] = report
            rec["resolved"] = is_resolved(report)
            rec["score"] = report.get("f2p_rate", 0.0) * report.get("p2p_rate", 0.0)

    perfect = [s for s in scored if s["resolved"]]
    if perfect:
        winner = min(perfect, key=lambda s: s["peer"])
    elif any(s["patch"] for s in scored):
        winner = max(scored, key=lambda s: (s["score"], -s["peer"]))
    else:
        return None, scored
    return winner["peer"], scored


# Orchestration
def _init_contexts(n: int, brief: str) -> list[list[BaseMessage]]:
    """Each peer's initial history = [HumanMessage(brief)]. System prompt is
    injected by create_react_agent via its `prompt=` arg, not embedded here.
    """
    return [[HumanMessage(content=brief)] for _ in range(n)]


def solve(
    instance: dict,
    peer_workdirs: list[Path],
    eval_mode: str = "singularity",
) -> dict:
    """Run N peers × R rounds. Each peer edits ITS OWN workdir — caller is
    responsible for cloning N separate copies before calling solve()."""
    brief = format_task_brief(
        instance["problem_statement"],
        instance_id=instance.get("instance_id"),
        hints_text=instance.get("hints_text"),
    )
    compiled = _build_graph()
    init_state: DebateState = {
        "contexts": _init_contexts(N_AGENTS, brief),
        "round_finals": [],
        "round": 0,
        "brief": brief,
        "peer_workdirs": list(peer_workdirs),
    }
    # LangGraph's own per-graph recursion budget — the N_ROUNDS loop plus
    # the internal react sub-graph invocations. Give it plenty of headroom.
    result = compiled.invoke(init_state, config={"recursion_limit": 200})
    contexts = result.get("contexts") or []

    winner_idx, scored = best_of_n(peer_workdirs, instance, eval_mode=eval_mode)
    winner = scored[winner_idx] if winner_idx is not None else None

    # Aggregate telemetry across all peers' full message histories.
    flat_msgs: list[BaseMessage] = []
    for ctx in contexts:
        flat_msgs.extend(ctx)

    return {
        "patch": (winner or {}).get("patch") or "",
        "resolved": (winner or {}).get("resolved") if eval_mode != "none" else None,
        "winner": winner_idx,
        "per_peer": scored,
        "all_contexts": contexts,
        "telemetry": normalize(langchain_telemetry(flat_msgs)),
    }


# Predictions + batch runner
def predictions_entry(
    instance_id: str, patch: str,
    model_name: str = "mas-promptbench-decentralized-langgraph",
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
    """Clone N peer workdirs, run the debate, score via best-of-N, write
    artifacts. Matches openai sibling's run_one pattern but with
    sequential clones (threading removed along with the rest of the
    concurrency — see module docstring)."""
    iid = instance["instance_id"]
    summary: dict = {
        "instance_id": iid,
        "repo": instance["repo"],
        "base_commit": instance["base_commit"],
        "n_peers": N_AGENTS,
        "n_rounds": N_ROUNDS,
    }
    root = workdir_root / iid

    # Clone N fresh copies — one per peer. Sequential (simpler; matches
    # the sequential peer-execution trade-off in this port).
    peer_workdirs: list[Path] = [root / f"peer_{i}" for i in range(N_AGENTS)]
    t0 = time.time()
    for i, workdir in enumerate(peer_workdirs):
        err = clone_and_checkout(
            instance["repo"], instance["base_commit"], workdir
        )
        if err:
            summary["error"] = err
            summary["stage"] = f"clone/peer_{i}"
            summary["clone_s"] = round(time.time() - t0, 1)
            return summary
    summary["clone_s"] = round(time.time() - t0, 1)

    t0 = time.time()
    try:
        out = solve(instance, peer_workdirs, eval_mode=eval_mode)
    except Exception as e:
        summary["error"] = f"{type(e).__name__}: {e}"
        summary["stage"] = "solve"
        summary["solve_s"] = round(time.time() - t0, 1)
        return summary
    summary["solve_s"] = round(time.time() - t0, 1)
    summary.update(out.get("telemetry") or {})

    patch = out.get("patch") or ""
    summary["patch_chars"] = len(patch)
    summary["winner"] = out.get("winner")

    (out_dir / "patches").mkdir(parents=True, exist_ok=True)
    (out_dir / "patches" / f"{iid}.diff").write_text(patch)
    with (out_dir / "predictions.jsonl").open("a") as f:
        f.write(json.dumps(predictions_entry(iid, patch)) + "\n")

    # Trace: per-peer patch + score.
    (out_dir / "traces").mkdir(parents=True, exist_ok=True)
    with (out_dir / "traces" / f"{iid}.txt").open("w") as f:
        f.write(f"winner: peer {out.get('winner')}  "
                f"resolved={out.get('resolved')}\n\n")
        for s in out.get("per_peer") or []:
            report = s.get("report") or {}
            f.write(
                f"=== peer {s['peer']} ===\n"
                f"  patch_chars={len(s.get('patch') or '')}\n"
                f"  f2p_rate={report.get('f2p_rate')}  p2p_rate={report.get('p2p_rate')}\n"
                f"  resolved={s.get('resolved')}  score={s.get('score')}\n\n"
            )

    per_peer_rates = []
    for s in out.get("per_peer") or []:
        r = s.get("report") or {}
        per_peer_rates.append({
            "peer": s["peer"],
            "patch_chars": len(s.get("patch") or ""),
            "f2p_rate": r.get("f2p_rate"),
            "p2p_rate": r.get("p2p_rate"),
            "resolved": s.get("resolved"),
        })
    summary["per_peer"] = per_peer_rates

    if eval_mode == "none":
        summary["eval"] = "skipped"
        return summary

    winner_id = out.get("winner")
    winner = next(
        (s for s in out.get("per_peer") or [] if s.get("peer") == winner_id),
        None,
    )
    wreport = (winner or {}).get("report") or {}
    summary["eval_mode"] = eval_mode
    summary["f2p_rate"] = wreport.get("f2p_rate", 0.0)
    summary["p2p_rate"] = wreport.get("p2p_rate", 0.0)
    summary["resolved"] = out.get("resolved") or False
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
    """Iterate Verified instances and run_one() each. N peer clones per
    instance → disk footprint ≈ N × single. Default keep_workdirs=False
    removes each instance's peer clones after it's scored."""
    import shutil as _sh

    workdir_root = workdir_root or Path(
        f"{os.path.expanduser('~')}/swe_work_decentralized_langgraph"
    )
    out_dir = out_dir or (_REPO_ROOT / "results" / "swe_bench_decentralized_langgraph")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "predictions.jsonl").write_text("")
    results_path = out_dir / "results.jsonl"
    results_path.write_text("")

    instances = load_instances(subset, limit, offset, only)
    print(f"loaded {len(instances)} instance(s) from princeton-nlp/SWE-bench_Verified "
          f"(N={N_AGENTS}, R={N_ROUNDS})", file=sys.stderr)

    for i, inst in enumerate(instances, 1):
        print(
            f"\n[{i}/{len(instances)}] {inst['instance_id']}  "
            f"({inst['repo']}@{inst['base_commit'][:7]})",
            file=sys.stderr,
        )
        summary = run_one(inst, workdir_root, out_dir, eval_mode=eval_mode)
        with results_path.open("a") as f:
            f.write(json.dumps(summary) + "\n")
        compact = {k: v for k, v in summary.items() if k != "per_peer"}
        print(f"  -> {json.dumps(compact)}", file=sys.stderr)

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
        description="Decentralized-topology SWE-bench Verified agent (LangGraph debate)."
    )
    parser.add_argument("--subset", default="test")
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--only", action="append", default=None,
                        metavar="INSTANCE_ID")
    parser.add_argument(
        "--workdir-root",
        default=f"{os.path.expanduser('~')}/swe_work_decentralized_langgraph",
    )
    _default_out = str(
        _REPO_ROOT / "results" / "swe_bench_decentralized_langgraph"
    )
    parser.add_argument("--out-dir", default=_default_out)
    parser.add_argument("--eval", dest="eval_mode", default="singularity",
                        choices=["singularity", "none"])
    parser.add_argument("--skip-eval", dest="skip_eval", action="store_true",
                        help="shorthand for --eval none")
    parser.add_argument("--keep-workdirs", action="store_true")
    args = parser.parse_args()

    eval_mode = "none" if args.skip_eval else args.eval_mode

    run_batch(
        subset=args.subset,
        limit=args.limit if not args.only else None,
        offset=args.offset,
        only=args.only,
        workdir_root=Path(args.workdir_root).expanduser(),
        out_dir=Path(args.out_dir).expanduser(),
        eval_mode=eval_mode,
        keep_workdirs=args.keep_workdirs,
    )
