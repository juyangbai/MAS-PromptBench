"""Independent topology specialized for SWE-bench Verified."""

# Config
from __future__ import annotations

import asyncio
import contextvars
import json
import operator
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from teamsizes.output_contracts import append_output_contract_from_path
from typing import Annotated

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.constants import END, START
from langgraph.graph.state import StateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.types import Send

# Shared telemetry.
_TOPO_ROOT = str(Path(__file__).resolve().parents[3])
if _TOPO_ROOT not in sys.path:
    sys.path.insert(0, _TOPO_ROOT)
from topologies.telemetry import langchain_ensemble_telemetry, normalize  # noqa: E402
from typing_extensions import TypedDict


VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://lai:8001/v1")
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3.5-9B")

# Number of parallel replicas. Seeds are 0 .. N_AGENTS-1. Defaults to 4 but
# SWE-bench ensembles are expensive (clone + ~3 min solve + eval per replica);
# consider 2 while iterating.
N_AGENTS = int(os.environ.get("INDEPENDENT_N_AGENTS", "4"))

# Workdir root: each replica clones into WORKDIR_ROOT/<instance_id>_a<k>/.
WORKDIR_ROOT = Path(
    os.environ.get("SWE_WORKDIR_ROOT", f"{Path.home()}/swe_work_independent")
).resolve()

# Singularity eval cache (shared across replicas; each instance_id has ONE SIF).
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

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PROMPT_PATH = (
    _REPO_ROOT / "configs" / "prompts" / "independent" / "swe" / "patcher.txt"
)
SYSTEM_PROMPT = append_output_contract_from_path(_PROMPT_PATH.read_text().strip(), __file__, _PROMPT_PATH.stem)


# Per-replica REPO_DIR
# Each replica's async task sets this to its own cloned workdir before
# invoking the agent. The tools (below) read it on every call, so parallel
# replicas don't clobber each other's file writes.
_REPO_DIR: contextvars.ContextVar[Path] = contextvars.ContextVar("_REPO_DIR")


def _repo_dir() -> Path:
    try:
        return _REPO_DIR.get()
    except LookupError as e:
        raise RuntimeError(
            "REPO_DIR not set in this context; set it before invoking the agent"
        ) from e


# Tools
def _repo_path(path: str) -> Path:
    """Resolve `path` against the current replica's workdir; refuse escapes."""
    repo = _repo_dir()
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


@tool
def file_write(path: str, content: str) -> str:
    """Overwrite (or create) a file in the repository with `content`."""
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
        repo = _repo_dir()
        entries = sorted(p.iterdir(), key=lambda e: (not e.is_dir(), e.name))
        return "\n".join(
            f"{'d' if e.is_dir() else 'f'}  {e.relative_to(repo)}" for e in entries
        )
    except Exception as e:
        return f"ERROR: {e}"


@tool
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
    prefix = str(_repo_dir()) + "/"
    return "\n".join(line.replace(prefix, "") for line in lines)


@tool
def shell_exec(command: str, timeout_s: int = _SHELL_TIMEOUT_S) -> str:
    """Run a shell command in the repository working directory."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=str(_repo_dir()),
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
def _truncate(text: str, cap: int, label: str) -> str:
    if len(text) <= cap:
        return text
    return text[:cap] + f"\n... [truncated {label}: {len(text)} -> {cap} chars]"


def format_prompt(
    problem_statement: str,
    repo_dir: Path,
    instance_id: str | None = None,
    hints_text: str | None = None,
) -> str:
    """Build the user-facing prompt for one SWE-bench instance."""
    parts = []
    if instance_id:
        parts.append(f"INSTANCE: {instance_id}")
    parts.append(f"The repository is checked out at {repo_dir} on the failing commit.")
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


def _build_one_agent(seed: int):
    """Build one replica's react agent, seeded differently from siblings."""
    llm = ChatOpenAI(
        model=MODEL_ID,
        base_url=VLLM_BASE_URL,
        api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        max_tokens=8192,
        # Greedy (temp=0) traps Qwen3.5-9B in read-more-code loops on
        # SWE-bench without ever committing. Light stochastic sampling
        # with fixed per-replica seed keeps results reproducible per seed.
        temperature=0.2,
        top_p=0.9,
        seed=seed,
        extra_body={
            "repetition_penalty": 1.05,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    return create_react_agent(model=llm, tools=TOOLS, prompt=SYSTEM_PROMPT)


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


# Scoring
# Aligned to topologies/single/swe/langgraph_swe.py so ensemble
# resolve rates are directly comparable with single-topology numbers.
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


def exact_match_score(report: dict) -> float:
    """1.0 iff the instance is resolved, else 0.0."""
    return 1.0 if is_resolved(report) else 0.0


# Aggregation
def best_of_n(
    answers: list[dict],
    instance: dict,
    eval_mode: str = "singularity",
) -> tuple[dict | None, list[dict]]:
    """Score each candidate patch and return (winner, scored_list).

    Selection order for winner:
      1. First candidate (lowest agent_id) with `resolved==True`.
      2. Else candidate with highest f2p_rate * p2p_rate (first on tie).
      3. None iff no replica produced a non-empty patch.

    The returned `scored_list` carries per-replica {report, resolved,
    score} fields so callers can populate reporting structures without
    re-running the (expensive) Singularity eval. When
    eval_mode='none', scored_list is empty.
    """
    valid = [a for a in answers if a.get("patch")]
    if not valid:
        return None, []

    if eval_mode == "none":
        valid.sort(key=lambda a: a["agent_id"])
        return valid[0], []

    f2p = instance["FAIL_TO_PASS"]
    p2p = instance["PASS_TO_PASS"]
    if isinstance(f2p, str):
        f2p = json.loads(f2p)
    if isinstance(p2p, str):
        p2p = json.loads(p2p)

    scored = []
    for a in valid:
        if eval_mode == "singularity":
            report = run_tests_singularity(instance, a["patch"], f2p, p2p)
        else:
            raise ValueError(f"unknown eval_mode {eval_mode!r}")
        scored.append({
            **a,
            "report": report,
            "resolved": is_resolved(report),
            "score": report.get("f2p_rate", 0.0) * report.get("p2p_rate", 0.0),
        })

    scored.sort(key=lambda s: s["agent_id"])
    perfect = [s for s in scored if s["resolved"]]
    if perfect:
        return perfect[0], scored
    return max(scored, key=lambda s: (s["score"], -s["agent_id"])), scored


# Graph
class State(TypedDict):
    instance: dict
    answers: Annotated[list[dict], operator.add]


class AgentInput(TypedDict):
    agent_id: int
    seed: int
    instance: dict


async def _run_replica(inp: AgentInput) -> dict:
    """Clone, set per-replica REPO_DIR, run agent, compute patch."""
    instance = inp["instance"]
    iid = instance["instance_id"]
    workdir = WORKDIR_ROOT / f"{iid}_a{inp['agent_id']}"

    clone_start = time.time()
    err = clone_and_checkout(instance["repo"], instance["base_commit"], workdir)
    clone_s = time.time() - clone_start
    if err:
        return {"answers": [{
            "agent_id": inp["agent_id"],
            "seed": inp["seed"],
            "patch": "",
            "raw": None,
            "messages": [],
            "error": err,
            "clone_s": clone_s,
            "solve_s": 0.0,
            "workdir": str(workdir),
        }]}

    token = _REPO_DIR.set(workdir)
    try:
        agent = _build_one_agent(seed=inp["seed"])
        prompt = format_prompt(
            instance["problem_statement"],
            repo_dir=workdir,
            instance_id=iid,
            hints_text=instance.get("hints_text"),
        )

        solve_start = time.time()
        try:
            result = await agent.ainvoke(
                {"messages": [("user", prompt)]},
                config={"recursion_limit": 100},
            )
        except Exception as e:
            return {"answers": [{
                "agent_id": inp["agent_id"],
                "seed": inp["seed"],
                "patch": "",
                "raw": None,
                "messages": [],
                "error": f"{type(e).__name__}: {e}",
                "clone_s": clone_s,
                "solve_s": time.time() - solve_start,
                "workdir": str(workdir),
            }]}
        solve_s = time.time() - solve_start

        for msg in result["messages"]:
            if msg.type == "ai" and isinstance(msg.content, str):
                msg.content = strip_thinking(msg.content)
        final = result["messages"][-1].content if result["messages"] else ""
        patch = compute_patch(workdir)

        return {"answers": [{
            "agent_id": inp["agent_id"],
            "seed": inp["seed"],
            "patch": patch,
            "raw": final,
            "messages": result["messages"],
            "clone_s": round(clone_s, 1),
            "solve_s": round(solve_s, 1),
            "workdir": str(workdir),
        }]}
    finally:
        _REPO_DIR.reset(token)


def _fan_out(state: State) -> list[Send]:
    return [
        Send(
            f"agent_{i}",
            {"agent_id": i, "seed": i, "instance": state["instance"]},
        )
        for i in range(N_AGENTS)
    ]


def build_graph() -> StateGraph:
    graph = StateGraph(State)
    for i in range(N_AGENTS):
        graph.add_node(f"agent_{i}", _run_replica)
    graph.add_conditional_edges(START, _fan_out)
    graph.add_edge([f"agent_{i}" for i in range(N_AGENTS)], END)
    return graph


# Orchestration
def solve(instance: dict, eval_mode: str = "singularity") -> dict:
    """Run the ensemble on one SWE-bench instance.

    Returns:
        {
            "patch":     winning patch (str) or None,
            "resolved":  bool (or None if eval_mode='none'),
            "winner":    agent_id of the selected candidate,
            "per_agent": list of {agent_id, seed, patch, clone_s, solve_s,
                                  report?, resolved?, score?, error?},
        }
    """
    compiled = build_graph().compile()
    result = asyncio.run(
        compiled.ainvoke({"instance": instance, "answers": []})
    )
    per_agent = sorted(result["answers"], key=lambda a: a["agent_id"])
    winner, scored = best_of_n(per_agent, instance, eval_mode=eval_mode)

    # Merge best_of_n's per-replica scores back into per_agent (avoids
    # re-running the expensive Singularity eval a second time — each
    # replica is scored exactly ONCE, inside best_of_n).
    if eval_mode != "none":
        scored_by_id = {s["agent_id"]: s for s in scored}
        for a in per_agent:
            s = scored_by_id.get(a["agent_id"])
            if s is None:
                # Replica produced no patch; best_of_n skipped it.
                a["report"] = None
                a["resolved"] = False
                a["score"] = 0.0
            else:
                a["report"] = s["report"]
                a["resolved"] = s["resolved"]
                a["score"] = s["score"]

    return {
        "patch": (winner or {}).get("patch"),
        "resolved": (winner or {}).get("resolved"),
        "winner": (winner or {}).get("agent_id"),
        "per_agent": per_agent,
    }


# Demo
def _print_summary(per_agent: list[dict], winner: int | None, resolved) -> None:
    print(f"\n=== Ensemble ({N_AGENTS} replicas) ===")
    for a in per_agent:
        pr = ""
        if a.get("report"):
            pr = f"f2p={a['report'].get('f2p_rate'):.2f} p2p={a['report'].get('p2p_rate'):.2f}"
        err = f" err={a['error']!r}" if a.get("error") else ""
        print(
            f"  agent_{a['agent_id']} (seed {a['seed']}): "
            f"patch={len(a.get('patch') or '')} chars  "
            f"clone={a.get('clone_s')}s  solve={a.get('solve_s')}s  "
            f"{pr}{err}"
        )
    if winner is not None:
        print(f"=== best-of-N winner: agent_{winner}  resolved={resolved} ===")


# Predictions + batch runner
def predictions_entry(
    instance_id: str, patch: str,
    model_name: str = "mas-promptbench-independent",
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
    """Run the N-agent ensemble on one instance and write artifacts.

    Per-agent clones live under `workdir_root / <iid>_a<k>` (managed
    inside `_run_replica`). We rebind the module's WORKDIR_ROOT so that
    each replica sees the caller's chosen root.

    Writes:
        - out_dir / patches / <iid>.diff     (winner's patch)
        - out_dir / predictions.jsonl        (one line per instance)
        - out_dir / traces / <iid>.txt       (per-agent summary + winner)
    """
    global WORKDIR_ROOT
    WORKDIR_ROOT = Path(workdir_root).resolve()

    iid = instance["instance_id"]
    summary: dict = {
        "instance_id": iid,
        "repo": instance["repo"],
        "base_commit": instance["base_commit"],
    }

    t0 = time.time()
    try:
        out = solve(instance, eval_mode=eval_mode)
    except Exception as e:
        summary["error"] = f"{type(e).__name__}: {e}"
        summary["stage"] = "solve"
        return summary
    summary["solve_s"] = round(time.time() - t0, 1)

    patch = out.get("patch") or ""
    summary["patch_chars"] = len(patch)
    summary["winner"] = out.get("winner")
    summary["n_agents"] = N_AGENTS
    summary.update(normalize(langchain_ensemble_telemetry(out.get("per_agent") or [])))

    (out_dir / "patches").mkdir(parents=True, exist_ok=True)
    (out_dir / "patches" / f"{iid}.diff").write_text(patch)
    with (out_dir / "predictions.jsonl").open("a") as f:
        f.write(json.dumps(predictions_entry(iid, patch)) + "\n")

    # Per-agent trace: clone/solve timings + score + error, plus winner highlight.
    (out_dir / "traces").mkdir(parents=True, exist_ok=True)
    with (out_dir / "traces" / f"{iid}.txt").open("w") as f:
        f.write(f"winner: agent_{out.get('winner')}  "
                f"resolved={out.get('resolved')}\n\n")
        for a in out.get("per_agent") or []:
            report = a.get("report") or {}
            f.write(
                f"=== agent_{a['agent_id']} seed={a.get('seed')} ===\n"
                f"  patch_chars={len(a.get('patch') or '')}\n"
                f"  clone_s={a.get('clone_s')}  solve_s={a.get('solve_s')}\n"
                f"  f2p_rate={report.get('f2p_rate')}  p2p_rate={report.get('p2p_rate')}\n"
                f"  resolved={a.get('resolved')}  error={a.get('error')!r}\n\n"
            )

    # Aggregate per-agent pass rates from the winner selection step.
    per_agent_rates = []
    for a in out.get("per_agent") or []:
        r = a.get("report") or {}
        per_agent_rates.append({
            "agent_id": a["agent_id"],
            "seed": a.get("seed"),
            "patch_chars": len(a.get("patch") or ""),
            "f2p_rate": r.get("f2p_rate"),
            "p2p_rate": r.get("p2p_rate"),
            "resolved": a.get("resolved"),
            "error": a.get("error"),
        })
    summary["per_agent"] = per_agent_rates

    if eval_mode == "none":
        summary["eval"] = "skipped"
        return summary

    # Winner-level resolved / rates go at the top level for easy aggregation.
    winner_id = out.get("winner")
    winner = next(
        (a for a in out.get("per_agent") or [] if a.get("agent_id") == winner_id),
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
    """Iterate Verified instances, run_one() each — matches single/swe's
    batch flow. Multi-agent: per-instance disk footprint ≈ N × single."""
    import shutil as _sh

    workdir_root = workdir_root or Path(
        f"{os.path.expanduser('~')}/swe_work_independent"
    )
    _repo_root = Path(__file__).resolve().parents[3]
    out_dir = out_dir or (_repo_root / "results" / "swe_bench_independent")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "predictions.jsonl").write_text("")
    results_path = out_dir / "results.jsonl"
    results_path.write_text("")

    instances = load_instances(subset, limit, offset, only)
    print(f"loaded {len(instances)} instance(s) from princeton-nlp/SWE-bench_Verified "
          f"(N={N_AGENTS})", file=sys.stderr)

    for i, inst in enumerate(instances, 1):
        print(
            f"\n[{i}/{len(instances)}] {inst['instance_id']}  "
            f"({inst['repo']}@{inst['base_commit'][:7]})",
            file=sys.stderr,
        )
        summary = run_one(inst, workdir_root, out_dir, eval_mode=eval_mode)
        with results_path.open("a") as f:
            f.write(json.dumps(summary) + "\n")
        # Cap the printed summary: per_agent list can balloon logs.
        compact = {k: v for k, v in summary.items() if k != "per_agent"}
        print(f"  -> {json.dumps(compact)}", file=sys.stderr)

        if not keep_workdirs:
            # Each replica has its own workdir (<iid>_a<k>); remove all.
            for k in range(N_AGENTS):
                _sh.rmtree(workdir_root / f"{inst['instance_id']}_a{k}",
                           ignore_errors=True)

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
        description="Independent-topology SWE-bench Verified agent (ensemble)."
    )
    parser.add_argument("--subset", default="test")
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--only", action="append", default=None,
                        metavar="INSTANCE_ID")
    parser.add_argument(
        "--workdir-root",
        default=f"{os.path.expanduser('~')}/swe_work_independent",
    )
    _default_out = str(
        Path(__file__).resolve().parents[3] / "results" / "swe_bench_independent"
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
