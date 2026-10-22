"""Code-generation tasks (LCB, APPS): code extraction, the python_exec tool, guarded calls, selection and records.

A dataset module supplies its test runner ``run(code, tests, timeout_s=...)``,
which returns ``{"pass", "total", "pass_rate", "details"}``; the scoring and
demo helpers here take it as ``run``.
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import tempfile
from collections.abc import Callable
from functools import partial
from pathlib import Path

from core import batch, voting

TestRunner = Callable[..., dict]

# Appended to the centralized manager prompts.
TERMINATE_NUDGE = (
    "\n\nWhen you emit the final fenced ```python``` code block "
    "containing your chosen solution, immediately follow it with the "
    "literal string TERMINATE on its own line so the group-chat knows "
    "to stop."
)


def agents_input(problem: str, starter_code: str | None = None) -> str:
    """Task text of an OpenAI Agents SDK debater (LCB and APPS)."""
    return (
        f"{problem}\n\nStarter code:\n{starter_code or ''}\n\nReturn the complete solution in one fenced python block."
    )


# Code extraction
_PY_FENCE_RE = re.compile(r"```(?:python|py)\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_BARE_FENCE_RE = re.compile(r"```\s*\n(.*?)```", re.DOTALL)


def extract_code(text: str) -> str | None:
    """The last non-empty fenced block that parses as Python, or None.

    Labelled ```python blocks win over bare fences; empty, JSON-looking and prose
    blocks are skipped, so a trailing empty fence does not hide an earlier program.
    """
    raw = text or ""
    for blocks in (_PY_FENCE_RE.findall(raw), _BARE_FENCE_RE.findall(raw)):
        for block in reversed(blocks):
            candidate = block.strip()
            if _is_python_code(candidate):
                return candidate
    return None


def _is_python_code(candidate: str) -> bool:
    if not candidate or candidate.lstrip().startswith(("{", "[")):
        return False
    try:
        ast.parse(candidate)
    except SyntaxError:
        return False
    return True


def extract_code_before_terminate(text: str) -> str | None:
    """:func:`extract_code` after removing the group chat's ``TERMINATE`` marker."""
    return extract_code(re.sub(r"\bTERMINATE\b", "", text))


# python_exec tool. A tool's docstring is the description the model sees, so the
# runners' three wordings are kept verbatim, indentation included.
_PYTHON_EXEC_ARGS = (
    "    Args:\n"
    "        code: the Python source to run.\n"
    "        stdin: optional stdin fed to the subprocess.\n"
    "        timeout_s: hard wall-clock limit (seconds). Defaults to 10.\n\n"
    "    Returns a single string with stdout, stderr, and exit code.\n"
    "    "
)
PYTHON_EXEC_DOC = (
    "Execute a Python code snippet in a fresh subprocess and return the\n    captured output.\n\n" + _PYTHON_EXEC_ARGS
)
PYTHON_EXEC_DOC_SHORT = "Execute a Python code snippet in a fresh subprocess.\n\n" + _PYTHON_EXEC_ARGS
PYTHON_EXEC_DOC_SANDBOXED = "Execute a Python code snippet in a sandboxed subprocess.\n\n" + _PYTHON_EXEC_ARGS
EXEC_TIMEOUT_S = 10


def execute_python(code: str, stdin: str = "", timeout_s: int = EXEC_TIMEOUT_S, char_budget: int | None = None) -> str:
    """Run ``code`` with ``python -c`` and report stdout, stderr and the exit code.

    With ``char_budget`` each stream is cut to that many characters.
    """

    def clip(text: str) -> str:
        if char_budget is None or len(text) <= char_budget:
            return text
        return text[:char_budget] + "\n...<truncated tool output>..."

    try:
        result = subprocess.run(["python", "-c", code], input=stdin, capture_output=True, text=True, timeout=timeout_s)
        return f"stdout:\n{clip(result.stdout)}\nstderr:\n{clip(result.stderr)}\nexit_code: {result.returncode}"
    except subprocess.TimeoutExpired:
        return f"ERROR: code exceeded {timeout_s}s timeout"
    except Exception as e:
        return f"ERROR: {e}"


def make_python_exec(doc: str, *, default_timeout_s: int = EXEC_TIMEOUT_S, char_budget: int | None = None):
    """A ``python_exec(code, stdin, timeout_s)`` function documented by ``doc``, ready for a framework's tool wrapper."""

    def python_exec(code: str, stdin: str = "", timeout_s: int = default_timeout_s) -> str:
        return execute_python(code, stdin, timeout_s, char_budget)

    python_exec.__doc__ = doc
    return python_exec


# Guarded call: a submission's function called in a fresh subprocess under the
# reference harnesses' reliability_guard and memory limits.
# argv: [1]=max_memory_bytes (0: unlimited), [2]=fn_name, [3]=args_json, [4]=outfile;
# stdin: the submission source.
GUARDED_CALL_WORKER = """
import os, sys, json, platform, resource, shutil, subprocess, builtins, faulthandler


def _reliability_guard(max_memory_bytes):
    if max_memory_bytes:
        resource.setrlimit(resource.RLIMIT_AS, (max_memory_bytes, max_memory_bytes))
        resource.setrlimit(resource.RLIMIT_DATA, (max_memory_bytes, max_memory_bytes))
        if platform.uname().system != "Darwin":
            resource.setrlimit(resource.RLIMIT_STACK, (max_memory_bytes, max_memory_bytes))
    faulthandler.disable()
    builtins.exit = None
    builtins.quit = None
    os.environ["OMP_NUM_THREADS"] = "1"
    for _name in (
        "kill","system","putenv","remove","removedirs","rmdir","fchdir","setuid",
        "fork","forkpty","killpg","rename","renames","truncate","replace","unlink",
        "fchmod","fchown","chmod","chown","chroot","lchflags","lchmod","lchown","chdir",
    ):
        if hasattr(os, _name):
            try:
                setattr(os, _name, None)
            except Exception:
                pass
    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None
    subprocess.Popen = None
    for _mod in ("ipdb", "joblib", "resource", "psutil", "tkinter"):
        sys.modules[_mod] = None


_max_mem = int(sys.argv[1])
_fn = sys.argv[2]
_args = json.loads(sys.argv[3])
_outfile = sys.argv[4]
_code = sys.stdin.read()

_reliability_guard(_max_mem)

_out = None
try:
    _ns = {}
    exec(_code, _ns)
    if "Solution" in _ns:
        _target = getattr(_ns["Solution"](), _fn)
    elif _fn in _ns:
        _target = _ns[_fn]
    else:
        _out = {"ok": False, "error": "neither Solution." + _fn + " nor " + _fn + " defined"}
    if _out is None:
        _r = _target(*_args)
        if isinstance(_r, tuple):
            _r = list(_r)
        _out = {"ok": True, "result": _r}
except BaseException as _e:
    _out = {"ok": False, "error": type(_e).__name__ + ": " + str(_e)}

with open(_outfile, "w") as _f:
    _f.write(json.dumps(_out, default=list))
"""


def guarded_call(code: str, fn_name: str, args: list, timeout_s: int, memory_bytes: int) -> dict:
    """Call ``fn_name(*args)`` of ``code`` (``Solution().fn_name`` when defined) in a guarded subprocess.

    Returns ``{"ok": True, "result": value}`` or ``{"ok": False, "error": ...}`` (plus
    ``"stderr"`` when the worker left no result).
    """
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        outfile = f.name
    command = ["python", "-c", GUARDED_CALL_WORKER, str(memory_bytes), fn_name, json.dumps(args), outfile]
    try:
        proc = subprocess.run(command, input=code, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        os.unlink(outfile)
        return {"ok": False, "error": "timeout"}
    except Exception as e:
        os.unlink(outfile)
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    try:
        with open(outfile) as f:
            payload = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError, OSError):
        return {
            "ok": False,
            "error": f"worker produced no parseable result (exit={proc.returncode})",
            "stderr": (proc.stderr or "").strip(),
        }
    finally:
        try:
            os.unlink(outfile)
        except FileNotFoundError:
            pass
    if not payload.get("ok"):
        return {"ok": False, "error": payload.get("error", "worker error")}
    return {"ok": True, "result": payload["result"]}


def exact_match_score(pass_rate: float) -> float:
    """1.0 iff every test passes (LCB pass@1, APPS strict accuracy)."""
    return 1.0 if pass_rate == 1.0 else 0.0


# Selection over several agents' programs
def select_program(codes: list[str | None]) -> int:
    """Index of the submitted program: the vote over the whitespace-normalized programs (:mod:`core.voting`)."""
    return voting.majority_text(codes)


def score_selected_peer(per_peer: list[dict], winner: int, code: str | None, tests, *, run: TestRunner) -> None:
    """Attach the test report of the selected peer's program (``run`` with its default timeout), in place.

    The other peers get ``pass_rate`` / ``resolved`` / ``report`` set to None.
    """
    for entry in per_peer:
        entry.update({"pass_rate": None, "resolved": None, "report": None})
    report = run(code, tests) if code else None
    pass_rate = report["pass_rate"] if report else 0.0
    per_peer[winner].update({"pass_rate": pass_rate, "resolved": pass_rate == 1.0, "report": report})


# Batch records
def unscored(total: int) -> dict:
    """The test summary of a row without a program."""
    return {"pass": 0, "total": total, "pass_rate": 0.0, "details": []}


def scores_of(code: str | None, scored: dict) -> dict:
    """Record fields pass / total / pass_rate / em of a test summary."""
    em = exact_match_score(scored["pass_rate"]) if code else 0.0
    return {"pass": scored["pass"], "total": scored["total"], "pass_rate": scored["pass_rate"], "em": em}


def selection_scores(code: str | None, winner, pass_rate: float) -> dict:
    """Record fields winner / pass_rate / em of an ensemble's submitted program."""
    return {"winner": winner, "pass_rate": pass_rate, "em": exact_match_score(pass_rate) if code else 0.0}


def winner_pass_rate(out: dict) -> float:
    """Pass rate of the selected peer of a debate (0.0 without code or winner)."""
    peers, winner = out.get("per_peer") or [], out.get("winner")
    if out.get("code") and winner is not None and 0 <= winner < len(peers):
        return peers[winner].get("pass_rate") or 0.0
    return 0.0


def compact_replicas(per_agent: list[dict]) -> list[dict]:
    """The per-replica record field."""
    return [{"agent_id": a["agent_id"], "seed": a.get("seed"), "code": a.get("code")} for a in per_agent]


def compact_peers(per_peer: list[dict]) -> list[dict]:
    """The per-peer record field."""
    return [
        {
            "peer": p.get("peer"),
            "has_code": bool(p.get("code")),
            "pass_rate": p.get("pass_rate", 0.0),
            "resolved": p.get("resolved", False),
        }
        for p in per_peer
    ]


def summarize(per_instance: list[dict]) -> dict:
    """Batch scores: n, extracted programs, summed and mean EM (overall, on extracted rows, per difficulty)."""
    n = len(per_instance)
    n_extracted = sum(1 for rec in per_instance if rec["predicted_code"])
    em_sum = sum((rec["em"] for rec in per_instance), 0.0)
    by_difficulty: dict[str, list[float]] = {}
    for rec in per_instance:
        by_difficulty.setdefault(rec["difficulty"] or "unk", []).append(rec["em"])
    return {
        "n": n,
        "n_extracted": n_extracted,
        "em_sum": em_sum,
        "em": (em_sum / n) if n else 0.0,
        "extracted_em": (em_sum / n_extracted) if n_extracted else 0.0,
        "by_difficulty": {d: {"n": len(v), "em": sum(v) / len(v)} for d, v in by_difficulty.items()},
    }


def progress_line(index: int, total: int, rec: dict, done: list[dict], *, id_width: int) -> str:
    """One verbose progress line after ``rec`` (``done`` holds the records so far)."""
    em = rec["em"]
    mark = "✓" if em == 1.0 else ("?" if rec["predicted_code"] is None else "✗")
    if "pass" in rec:
        detail = f"pass={rec['pass']}/{rec['total']}"
    else:
        detail = f"winner={rec['winner']}  pass_rate={rec['pass_rate']:.2f}"
    running_em = sum(r["em"] for r in done) / len(done)
    return (
        f"[{index + 1:>3}/{total}] {rec['id'][:id_width]:<{id_width}} {mark}  em={em:.0f}  {detail}  "
        f"EM={running_em:.3f} lat={rec['latency_s']:.1f}s"
    )


def banner(label: str, summary: dict, *, metric: str, difficulty_width: int) -> str:
    """End-of-batch report."""
    lines = [
        f"\n=== {label} batch complete ===",
        f"  n={summary['n']}  n_extracted={summary['n_extracted']}",
        f"  {metric} EM={summary['em']:.3f}  (on extracted only: {summary['extracted_em']:.3f})\n",
    ]
    for d, v in summary["by_difficulty"].items():
        lines.append(f"    {d:>{difficulty_width}s}: n={v['n']:3d}  EM={v['em']:.3f}")
    lines.append(f"  total_s={summary['total_s']}\n")
    return "\n".join(lines)


def run_batch(
    instances: list[dict],
    row: Callable[[int, dict], dict],
    *,
    out_path: Path | None,
    verbose: bool,
    label: str,
    metric: str,
    id_width: int,
    difficulty_width: int,
) -> dict:
    """:func:`core.batch.run_batch` with the code-task summary, progress lines and report."""
    return batch.run_batch(
        instances,
        row,
        summarize=summarize,
        out_path=out_path,
        verbose=verbose,
        progress=partial(progress_line, id_width=id_width),
        banner=partial(banner, label, metric=metric, difficulty_width=difficulty_width),
    )


# Demo
TWO_SUM_STARTER = (
    "from typing import List\n\n"
    "class Solution:\n"
    "    def twoSum(self, nums: List[int], target: int) -> List[int]:\n"
    "        "
)


def print_demo_code(code: str | None, tests, *, run: TestRunner, metric: str) -> None:
    """Print a demo program and its test results."""
    print(f"\n=== Extracted code ===\n{code}\n")
    if not code:
        return
    scored = run(code, tests)
    print(
        f"=== Tests: {scored['pass']}/{scored['total']}   pass_rate: {scored['pass_rate']:.2f}   "
        f"{metric}: {exact_match_score(scored['pass_rate']):.2f} ==="
    )
    for d in scored["details"]:
        if not d.get("ok"):
            print(
                f"    [FAIL {d.get('mode', '?')}] test {d['test']}: expected={d.get('expected')!r}  "
                f"actual={d.get('actual')!r}  err={d.get('error') or d.get('stderr')!r}"
            )
