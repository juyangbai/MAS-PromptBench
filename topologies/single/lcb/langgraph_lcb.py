"""Single-agent ReAct topology specialized for LiveCodeBench."""

# Config
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import zlib
from decimal import Decimal, InvalidOperation
from pathlib import Path

from topologies.output_contracts import append_output_contract_from_path
from topologies.code_extract import extract_python_code

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


# System Prompt
_REPO_ROOT = Path(__file__).resolve().parents[3]
_PROMPT_PATH = _REPO_ROOT / "configs" / "prompts" / "single" / "lcb" / "solver.txt"
SYSTEM_PROMPT = append_output_contract_from_path(_PROMPT_PATH.read_text().strip(), __file__, _PROMPT_PATH.stem)


# Tools
_EXEC_TIMEOUT_S = 10


@tool
def python_exec(code: str, stdin: str = "", timeout_s: int = _EXEC_TIMEOUT_S) -> str:
    """Execute a Python code snippet in a fresh subprocess and return the
    captured output.

    Args:
        code: the Python source to run.
        stdin: optional stdin fed to the subprocess.
        timeout_s: hard wall-clock limit (seconds). Defaults to 10.

    Returns a single string with stdout, stderr, and exit code.
    """
    try:
        result = subprocess.run(
            ["python", "-c", code],
            input=stdin,
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
        return f"ERROR: code exceeded {timeout_s}s timeout"
    except Exception as e:
        return f"ERROR: {e}"


# Agent
# User-message format templates — verbatim from LiveCodeBench's official
# lcb_runner/prompts/code_generation.py. Public test cases are intentionally
# NOT included in the prompt (matching LCB behavior).

_FORMAT_STDIN = (
    "### Format: Read the inputs from stdin solve the problem and write "
    "the answer to stdout (do not directly test on the sample inputs). "
    "Enclose your code within delimiters as follows. Ensure that when the "
    "python program runs, it reads the inputs, runs the algorithm and writes "
    "output to STDOUT.\n"
    "```python\n"
    "# YOUR CODE HERE\n"
    "```"
)

_FORMAT_FUNCTIONAL = (
    "### Format: You will use the following starter code to write the "
    "solution to the problem and enclose your code within delimiters.\n"
    "```python\n"
    "{starter_code}\n"
    "```"
)


def format_prompt(problem: str, starter_code: str | None = None) -> str:
    """Build the user-facing prompt for one LCB problem.

    Follows the official LCB prompt format. Presence of `starter_code`
    selects functional / LeetCode mode; absence selects stdin mode.
    """
    if starter_code:
        suffix = _FORMAT_FUNCTIONAL.format(starter_code=starter_code.rstrip())
    else:
        suffix = _FORMAT_STDIN
    return f"{problem}\n\n{suffix}"


def build_agent():
    llm = ChatOpenAI(
        model=MODEL_ID,
        base_url=VLLM_BASE_URL,
        api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        temperature=0.2,
        top_p=0.9,
        seed=0,
        max_tokens=2048,
        extra_body={
            "repetition_penalty": 1.05,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    return create_react_agent(model=llm, tools=[python_exec], prompt=SYSTEM_PROMPT)


# Output Parsing
# Grabs fenced code blocks: ```python\n...\n``` (also accepts ```py or bare ```).
_CODE_BLOCK_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def strip_thinking(text: str) -> str:
    """Cut everything up through the last </think> tag."""
    index = text.lower().rfind("</think>")
    if index >= 0:
        text = text[index + len("</think>"):]
    return text.strip()


def extract_code(text: str) -> str | None:
    """Return the LAST fenced code block in `text`, or None.

    Takes the last block because models often revise their code in earlier
    turns before settling on the final version.
    """
    return extract_python_code(text)


def extract_answer(text: str) -> str | None:
    """Alias for extract_code — kept to match the other scorers' interface."""
    return extract_code(text)


# Scoring
# Matches the two evaluation paths used by LiveCodeBench's official
# lcb_runner/evaluation/testing_util.py:
#   - stdin tests: child process, feed stdin, compare stdout (line-by-line
#     with decimal tolerance).
#   - functional tests: fresh subprocess with LCB's reliability_guard and
#     resource limits applied, then exec + call fn, compare return value.
#
# The functional path runs in a subprocess (instead of in-process) so the
# reliability_guard monkey-patches and setrlimit caps do not leak into the
# harness. Guard contents are ported verbatim from LCB testing_util.py.


def _compare_stdout(actual: str, expected: str) -> bool:
    """Line-by-line comparison. Exact match first; fall back to per-token
    decimal comparison on numeric-looking lines (matches official behavior).
    """
    if actual.strip() == expected.strip():
        return True
    actual_lines = actual.splitlines()
    expected_lines = expected.splitlines()
    if len(actual_lines) != len(expected_lines):
        return False
    for a, e in zip(actual_lines, expected_lines):
        if a.strip() == e.strip():
            continue
        try:
            a_parts = [Decimal(x) for x in a.split()]
            e_parts = [Decimal(x) for x in e.split()]
        except (InvalidOperation, ValueError):
            return False
        if a_parts != e_parts:
            return False
    return True


def _run_stdin_test(code: str, tc: dict, timeout_s: int) -> dict:
    """Run `code` as a subprocess with tc['input'] as stdin; compare stdout."""
    try:
        result = subprocess.run(
            ["python", "-c", code],
            input=tc.get("input", ""),
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timeout"}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}

    expected = tc.get("output", "")
    actual = result.stdout
    ok = _compare_stdout(actual, expected) and result.returncode == 0
    return {
        "ok": ok,
        "expected": expected.strip(),
        "actual": actual.strip(),
        "stderr": result.stderr.strip(),
        "exit_code": result.returncode,
    }


def _parse_maybe_json(value):
    """Accept either a raw Python object or a JSON string and return the object."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


# Worker script run inside each functional test's subprocess.
# Ported from LCB's testing_util.py `reliability_guard` + `run_test` logic.
# argv: [1]=max_memory_bytes (0 => unlimited), [2]=fn_name, [3]=args_json, [4]=outfile
# stdin: user's submission source
_FUNCTIONAL_WORKER = """
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

# Per-test memory cap. Default 4 GB, matching LCB's recommended setting.
# Set LCB_FUNCTIONAL_MEMORY_BYTES=0 to disable.
_FUNCTIONAL_MEMORY_BYTES = int(os.environ.get("LCB_FUNCTIONAL_MEMORY_BYTES", str(4 * 1024 ** 3)))


def _run_functional_test(code: str, tc: dict, timeout_s: int) -> dict:
    """Run `code` in a fresh subprocess with LCB's reliability_guard + setrlimit.

    Matches LCB's official behavior (reliability_guard + resource limits +
    call the target function) while keeping the guards isolated inside the
    child process so they do not leak into the harness.
    """
    fn_name = tc.get("fn_name") or tc.get("func_name")
    if not fn_name:
        return {"ok": False, "error": "functional test missing fn_name"}

    args = _parse_maybe_json(tc.get("input", "[]"))
    if not isinstance(args, (list, tuple)):
        args = [args]
    expected = _parse_maybe_json(tc.get("output"))

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        outfile = f.name

    try:
        proc = subprocess.run(
            [
                "python", "-c", _FUNCTIONAL_WORKER,
                str(_FUNCTIONAL_MEMORY_BYTES),
                fn_name,
                json.dumps(args),
                outfile,
            ],
            input=code,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
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
    actual = payload["result"]
    return {"ok": actual == expected, "expected": expected, "actual": actual}


def run_tests(code: str, tests: list[dict], timeout_s: int = 5) -> dict:
    """Run `code` against a list of LCB-style tests (stdin or functional).

    Each test dict should look like one of:
        {'input': '5\\n', 'output': '15', 'testtype': 'stdin'}
        {'input': '[[2,7,11,15], 9]', 'output': '[0,1]', 'fn_name': 'twoSum', 'testtype': 'functional'}

    Dispatches per-test based on 'testtype' (or 'fn_name' presence as fallback).
    Returns a summary dict with aggregate pass rate and per-test details.
    """
    if not tests:
        return {"pass": 0, "total": 0, "pass_rate": 0.0, "details": []}

    passed = 0
    details = []
    for i, tc in enumerate(tests):
        testtype = tc.get("testtype")
        fn_name = tc.get("fn_name") or tc.get("func_name")
        is_functional = testtype == "functional" or (testtype is None and fn_name is not None)

        if is_functional:
            r = _run_functional_test(code, tc, timeout_s)
            r["mode"] = "functional"
        else:
            r = _run_stdin_test(code, tc, timeout_s)
            r["mode"] = "stdin"

        passed += int(r.get("ok", False))
        details.append({"test": i, **r})

    return {
        "pass": passed,
        "total": len(tests),
        "pass_rate": passed / len(tests),
        "details": details,
    }


def exact_match_score(pass_rate: float) -> float:
    """LCB pass@1 convention: 1.0 iff all tests pass, else 0.0."""
    return 1.0 if pass_rate == 1.0 else 0.0


# Orchestration
def solve(problem: str, starter_code: str | None = None, agent=None) -> dict:
    """Run the agent on one coding problem.

    Pass `starter_code` (a LeetCode-style class/method stub) to run in
    functional mode; omit it for stdin mode. Strips Qwen3's <think>...</think>
    reasoning from every AI message. Optional `agent` kwarg lets callers
    reuse one agent across a batch.

    Returns {'code': str | None, 'raw': str, 'messages': list}.
    """
    if agent is None:
        agent = build_agent()
    result = agent.invoke(
        {"messages": [("user", format_prompt(problem, starter_code))]},
        config={"recursion_limit": 25},
    )
    for msg in result["messages"]:
        if msg.type == "ai" and isinstance(msg.content, str):
            msg.content = strip_thinking(msg.content)
    final = result["messages"][-1].content
    return {
        "code": extract_code(final),
        "raw": final,
        "messages": result["messages"],
    }


# Dataset loader
_HF_DATASET = "livecodebench/code_generation_lite"
_HF_SPLIT = "test"


def _decode_private_tests(blob: str) -> list[dict]:
    """LCB stores private test cases as base64(zlib(pickle(json_str))).
    The pickled payload is a JSON string that deserializes to a list of
    {input, output, testtype, fn_name?} test dicts. Decode through all
    three layers, falling back to plainer encodings for robustness."""
    if not blob:
        return []
    import pickle
    try:
        decompressed = zlib.decompress(base64.b64decode(blob))
    except Exception:
        return []
    # Most common: pickle wraps a JSON string.
    try:
        payload = pickle.loads(decompressed)
        if isinstance(payload, str):
            return json.loads(payload)
        if isinstance(payload, list):
            return payload
    except Exception:
        pass
    # Fallback: direct JSON bytes (older LCB dumps).
    try:
        return json.loads(decompressed)
    except Exception:
        return []


def load_instances(
    limit: int | None = None,
    offset: int = 0,
    only: list[str] | None = None,
    difficulty: str | None = None,
) -> list[dict]:
    """Load LCB rows from HuggingFaceH4's `livecodebench/code_generation_lite`.

    Row shape returned:
        {id, problem, starter_code, tests, difficulty, platform, raw}

    `difficulty` filter (easy/medium/hard) subsets the eval to one tier —
    useful because LCB pass@1 varies by ~30pp across tiers.
    """
    from datasets import load_dataset

    ds = load_dataset(_HF_DATASET, split=_HF_SPLIT, trust_remote_code=True)
    rows: list[dict] = []
    for row in ds:
        rid = row.get("question_id") or ""
        if only is not None and rid not in set(only):
            continue
        if difficulty is not None and row.get("difficulty") != difficulty:
            continue
        problem = (row.get("question_content") or "").strip()
        if not problem:
            continue
        tests = _decode_private_tests(row.get("private_test_cases") or "")
        if not tests:
            continue
        rows.append({
            "id": rid,
            "problem": problem,
            "starter_code": (row.get("starter_code") or "").rstrip(),
            "tests": tests,
            "difficulty": row.get("difficulty"),
            "platform": row.get("platform"),
            "raw": {k: row.get(k) for k in (
                "question_title", "question_id", "contest_id",
                "difficulty", "platform", "contest_date",
            )},
        })
    rows = rows[offset:]
    if limit is not None:
        rows = rows[:limit]
    return rows


# Batch eval
def run_batch(
    instances: list[dict],
    out_path: Path | None = None,
    verbose: bool = True,
    per_test_timeout_s: int = 6,
) -> dict:
    """Run `solve()` on every problem, score with `run_tests()`, write
    per-instance predictions to JSONL. `em` is LCB pass@1 (1.0 iff all
    tests pass, else 0.0) — standard LCB metric.
    """
    agent = build_agent()  # build once; reuse across the batch
    per_instance: list[dict] = []
    n = len(instances)
    em_sum = 0.0
    n_extracted = 0
    by_diff: dict[str, list[float]] = {}
    start = time.time()

    out_f = None
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_f = open(out_path, "w")

    try:
        for i, inst in enumerate(instances):
            t0 = time.time()
            try:
                out = solve(
                    inst["problem"],
                    starter_code=inst.get("starter_code") or None,
                    agent=agent,
                )
                error = None
            except Exception as e:
                out = {"code": None, "raw": "", "messages": []}
                error = f"{type(e).__name__}: {e}"
            latency_s = time.time() - t0

            code = out["code"]
            if code:
                n_extracted += 1
                scored = run_tests(code, inst["tests"], timeout_s=per_test_timeout_s)
            else:
                scored = {
                    "pass": 0,
                    "total": len(inst["tests"]),
                    "pass_rate": 0.0,
                    "details": [],
                }
            em = exact_match_score(scored["pass_rate"]) if code else 0.0
            em_sum += em
            by_diff.setdefault(inst.get("difficulty") or "unk", []).append(em)

            telem = normalize(langchain_telemetry(out.get("messages") or []))
            rec = {
                "id": inst["id"],
                "problem": inst["problem"][:400],
                "starter_code": inst.get("starter_code") or "",
                "predicted_code": code,
                "pass": scored["pass"],
                "total": scored["total"],
                "pass_rate": scored["pass_rate"],
                "em": em,
                "difficulty": inst.get("difficulty"),
                "platform": inst.get("platform"),
                "raw": (out.get("raw") or "")[:2000],
                "latency_s": round(latency_s, 2),
                **telem,
                "error": error,
            }
            per_instance.append(rec)
            if out_f is not None:
                out_f.write(json.dumps(rec) + "\n")
                out_f.flush()

            if verbose:
                running_em = em_sum / (i + 1)
                mark = "✓" if em == 1.0 else ("?" if code is None else "✗")
                print(
                    f"[{i + 1:>3}/{n}] {inst['id'][:22]:<22} {mark}  "
                    f"em={em:.0f}  pass={scored['pass']}/{scored['total']}  "
                    f"EM={running_em:.3f} lat={latency_s:.1f}s",
                    flush=True,
                )
    finally:
        if out_f is not None:
            out_f.close()

    elapsed = time.time() - start
    summary = {
        "n": n,
        "n_extracted": n_extracted,
        "em_sum": em_sum,
        "em": (em_sum / n) if n else 0.0,
        "extracted_em": (em_sum / n_extracted) if n_extracted else 0.0,
        "by_difficulty": {
            d: {"n": len(v), "em": sum(v) / len(v)} for d, v in by_diff.items()
        },
        "total_s": round(elapsed, 1),
        "per_instance": per_instance,
    }
    if verbose:
        print(
            f"\n=== single/LCB batch complete ===\n"
            f"  n={summary['n']}  n_extracted={summary['n_extracted']}\n"
            f"  pass@1 EM={summary['em']:.3f}  "
            f"(on extracted only: {summary['extracted_em']:.3f})\n"
        )
        for d, v in summary["by_difficulty"].items():
            print(f"    {d:>6s}: n={v['n']:3d}  EM={v['em']:.3f}")
        print(f"  total_s={summary['total_s']}\n")
    return summary


# Demo
def _print_scoring(scored: dict) -> None:
    print(
        f"=== Tests: {scored['pass']}/{scored['total']}   "
        f"pass_rate: {scored['pass_rate']:.2f}   "
        f"pass@1: {exact_match_score(scored['pass_rate']):.2f} ==="
    )
    for d in scored["details"]:
        if not d.get("ok"):
            print(
                f"    [FAIL {d['mode']}] test {d['test']}: "
                f"expected={d.get('expected')!r}  "
                f"actual={d.get('actual')!r}  "
                f"err={d.get('error') or d.get('stderr')!r}"
            )


def _canned_demo() -> None:
    # --- Stdin-mode problem (AtCoder/Codeforces style) ---
    stdin_problem = (
        "Read a single integer n from standard input (1 <= n <= 1000) and "
        "print the sum 1 + 2 + ... + n on one line."
    )
    stdin_tests = [
        {"input": "5\n",   "output": "15",   "testtype": "stdin"},
        {"input": "1\n",   "output": "1",    "testtype": "stdin"},
        {"input": "10\n",  "output": "55",   "testtype": "stdin"},
        {"input": "100\n", "output": "5050", "testtype": "stdin"},
    ]

    print("\n========== STDIN MODE ==========")
    out = solve(stdin_problem)
    print(f"\n=== Extracted code ===\n{out['code']}\n")
    if out["code"]:
        _print_scoring(run_tests(out["code"], stdin_tests))

    # --- Functional-mode problem (LeetCode style) ---
    functional_problem = (
        "Given an array of integers `nums` and an integer `target`, return "
        "the indices of the two numbers such that they add up to `target`. "
        "Assume each input has exactly one solution; the same element may "
        "not be used twice. Return the answer as a list [i, j] with i < j."
    )
    functional_tests = [
        {"fn_name": "twoSum", "input": "[[2,7,11,15], 9]", "output": "[0,1]", "testtype": "functional"},
        {"fn_name": "twoSum", "input": "[[3,2,4], 6]",     "output": "[1,2]", "testtype": "functional"},
        {"fn_name": "twoSum", "input": "[[3,3], 6]",       "output": "[0,1]", "testtype": "functional"},
    ]

    # LCB provides the starter code for LeetCode problems; the model fills in
    # the method body. The test dict carries `fn_name` so the scorer knows
    # which method to call.
    functional_starter = (
        "from typing import List\n"
        "\n"
        "class Solution:\n"
        "    def twoSum(self, nums: List[int], target: int) -> List[int]:\n"
        "        "
    )

    print("\n========== FUNCTIONAL MODE ==========")
    out = solve(functional_problem, starter_code=functional_starter)
    print(f"\n=== Extracted code ===\n{out['code']}\n")
    if out["code"]:
        _print_scoring(run_tests(out["code"], functional_tests))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Single-topology LCB runner (LangGraph, LiveCodeBench)."
    )
    parser.add_argument(
        "--batch", action="store_true",
        help="Run the real LCB eval (else: two canned demos).",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument(
        "--difficulty", type=str, default=None, choices=("easy", "medium", "hard"),
        help="Filter to one LCB difficulty tier.",
    )
    args = parser.parse_args()

    if not args.batch:
        _canned_demo()
        sys.exit(0)

    print(f"loading LCB from {_HF_DATASET} [{_HF_SPLIT}] ...")
    instances = load_instances(
        limit=args.limit, offset=args.offset,
        only=args.only, difficulty=args.difficulty,
    )
    if not instances:
        print("no instances loaded (check --limit/--offset/--only/--difficulty)",
              file=sys.stderr)
        sys.exit(1)
    print(f"  loaded {len(instances)} instance(s)")
    out_path = Path(args.out) if args.out else None
    run_batch(instances, out_path=out_path)
    if out_path:
        print(f"  predictions written to {out_path}")
