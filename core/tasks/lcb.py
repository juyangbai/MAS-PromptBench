"""LiveCodeBench: data, prompts, test execution, selection and batch records.

The benchmark slice is livecodebench/code_generation_lite [test]. A submission is the
last fenced Python block of the model output; it is run against the problem's
private tests with LCB's two evaluation paths (stdin programs compared line by line
with decimal tolerance, functional calls in a guarded subprocess) and scores pass@1
(1.0 iff every test passes). The helpers shared with APPS live in :mod:`core.code_tasks`.
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import zlib
from collections.abc import Callable
from decimal import Decimal, InvalidOperation
from pathlib import Path

from core import code_tasks

DATASET = "lcb"
HF_DATASET = "livecodebench/code_generation_lite"
HF_SPLIT = "test"
SOURCE = f"LCB from {HF_DATASET} [{HF_SPLIT}]"
DIFFICULTIES = ("easy", "medium", "hard")
PLATFORMS = ("codeforces", "leetcode", "atcoder")

# Per-test timeout of a batch run, and of run_tests when not given.
BATCH_TEST_TIMEOUT_S = 6
TEST_TIMEOUT_S = 5


# Command line
def add_arguments(parser) -> None:
    """``--difficulty``: one LCB tier."""
    parser.add_argument("--difficulty", type=str, default=None, choices=DIFFICULTIES, help="filter to one LCB tier")


def add_platform_arguments(parser) -> None:
    """``--difficulty`` and ``--platform``."""
    add_arguments(parser)
    parser.add_argument("--platform", type=str, default=None, choices=PLATFORMS, help="filter to one LCB platform")


# Data
def decode_private_tests(blob: str) -> list[dict]:
    """Private tests stored as base64(zlib(pickle(json string))); plain JSON bytes are accepted too."""
    if not blob:
        return []
    import pickle

    try:
        decompressed = zlib.decompress(base64.b64decode(blob))
    except Exception:
        return []
    try:
        payload = pickle.loads(decompressed)
        if isinstance(payload, str):
            return json.loads(payload)
        if isinstance(payload, list):
            return payload
    except Exception:
        pass
    try:
        return json.loads(decompressed)
    except Exception:
        return []


def load_instances(
    limit: int | None = None,
    offset: int = 0,
    only: list[str] | None = None,
    difficulty: str | None = None,
    platform: str | None = None,
) -> list[dict]:
    """Rows with private tests as ``{id, problem, starter_code, tests, difficulty, platform, raw}``."""
    from datasets import load_dataset

    rows: list[dict] = []
    for row in load_dataset(HF_DATASET, split=HF_SPLIT, trust_remote_code=True):
        rid = row.get("question_id") or ""
        if only is not None and rid not in set(only):
            continue
        if difficulty is not None and row.get("difficulty") != difficulty:
            continue
        if platform is not None and row.get("platform") != platform:
            continue
        problem = (row.get("question_content") or "").strip()
        if not problem:
            continue
        tests = decode_private_tests(row.get("private_test_cases") or "")
        if not tests:
            continue
        rows.append(
            {
                "id": rid,
                "problem": problem,
                "starter_code": (row.get("starter_code") or "").rstrip(),
                "tests": tests,
                "difficulty": row.get("difficulty"),
                "platform": row.get("platform"),
                "raw": {
                    k: row.get(k)
                    for k in ("question_title", "question_id", "contest_id", "difficulty", "platform", "contest_date")
                },
            }
        )
    rows = rows[offset:]
    return rows if limit is None else rows[:limit]


# User message: LCB's official code-generation format (public tests are not shown).
FORMAT_STDIN = (
    "### Format: Read the inputs from stdin solve the problem and write "
    "the answer to stdout (do not directly test on the sample inputs). "
    "Enclose your code within delimiters as follows. Ensure that when the "
    "python program runs, it reads the inputs, runs the algorithm and writes "
    "output to STDOUT.\n"
    "```python\n"
    "# YOUR CODE HERE\n"
    "```"
)
# The decentralized debaters' shorter stdin format.
FORMAT_STDIN_SHORT = (
    "### Format: Read the inputs from stdin solve the problem and write "
    "the answer to stdout (do not directly test on the sample inputs). "
    "Enclose your code within delimiters as follows.\n"
    "```python\n# YOUR CODE HERE\n```"
)
FORMAT_FUNCTIONAL = (
    "### Format: You will use the following starter code to write the "
    "solution to the problem and enclose your code within delimiters.\n"
    "```python\n"
    "{starter_code}\n"
    "```"
)

# Closing instruction of the decentralized peer-review message.
PEER_REVISION_NOTE = (
    "\nCompare their code and approach against your own. Revise your "
    "submission ONLY if a peer handles an edge case you missed or "
    "runs in better complexity. Re-emit your FINAL code in a single "
    "fenced ```python``` block at the end.\n\n"
    "Original problem:\n"
)


def format_prompt(
    problem: str,
    starter_code: str | None = None,
    *,
    stdin_format: str = FORMAT_STDIN,
) -> str:
    """The problem plus the functional format (with ``starter_code``) or ``stdin_format``."""
    if not starter_code:
        return f"{problem}\n\n{stdin_format}"
    return f"{problem}\n\n" + FORMAT_FUNCTIONAL.replace("{starter_code}", starter_code.rstrip())


# Scoring: LCB's lcb_runner/evaluation/testing_util.py evaluation paths. Kept as
# ported so pass@1 stays comparable with prior work.
FUNCTIONAL_MEMORY_BYTES = int(os.environ.get("LCB_FUNCTIONAL_MEMORY_BYTES", str(4 * 1024**3)))  # 0: no cap


def compare_stdout(actual: str, expected: str) -> bool:
    """Exact match after stripping, else line by line with per-token decimal comparison."""
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
    try:
        result = subprocess.run(
            ["python", "-c", code], input=tc.get("input", ""), capture_output=True, text=True, timeout=timeout_s
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timeout"}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    expected = tc.get("output", "")
    actual = result.stdout
    return {
        "ok": compare_stdout(actual, expected) and result.returncode == 0,
        "expected": expected.strip(),
        "actual": actual.strip(),
        "stderr": result.stderr.strip(),
        "exit_code": result.returncode,
    }


def _parse_maybe_json(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _run_functional_test(code: str, tc: dict, timeout_s: int) -> dict:
    fn_name = tc.get("fn_name") or tc.get("func_name")
    if not fn_name:
        return {"ok": False, "error": "functional test missing fn_name"}
    args = _parse_maybe_json(tc.get("input", "[]"))
    if not isinstance(args, (list, tuple)):
        args = [args]
    expected = _parse_maybe_json(tc.get("output"))
    called = code_tasks.guarded_call(code, fn_name, args, timeout_s, FUNCTIONAL_MEMORY_BYTES)
    if not called["ok"]:
        return called
    return {"ok": called["result"] == expected, "expected": expected, "actual": called["result"]}


def run_tests(code: str, tests: list[dict], timeout_s: int = TEST_TIMEOUT_S) -> dict:
    """Run ``code`` on LCB tests (``testtype`` "stdin" or "functional"; a test with ``fn_name`` and no type is functional).

    Returns ``{"pass", "total", "pass_rate", "details"}``.
    """
    if not tests:
        return {"pass": 0, "total": 0, "pass_rate": 0.0, "details": []}
    passed = 0
    details = []
    for i, tc in enumerate(tests):
        testtype = tc.get("testtype")
        fn_name = tc.get("fn_name") or tc.get("func_name")
        if testtype == "functional" or (testtype is None and fn_name is not None):
            r = _run_functional_test(code, tc, timeout_s)
            r["mode"] = "functional"
        else:
            r = _run_stdin_test(code, tc, timeout_s)
            r["mode"] = "stdin"
        passed += int(r.get("ok", False))
        details.append({"test": i, **r})
    return {"pass": passed, "total": len(tests), "pass_rate": passed / len(tests), "details": details}


# Scoring an ensemble's selected program
def score_selected_peer(per_peer: list[dict], winner: int, code: str | None, tests) -> None:
    """:func:`core.code_tasks.score_selected_peer` on LCB tests."""
    code_tasks.score_selected_peer(per_peer, winner, code, tests, run=run_tests)


# Batch records
def test_scores(code: str | None, tests: list[dict], timeout_s: int) -> dict:
    """Record fields pass / total / pass_rate / em of ``code`` on ``tests`` (zero without code)."""
    scored = run_tests(code, tests, timeout_s=timeout_s) if code else code_tasks.unscored(len(tests))
    return code_tasks.scores_of(code, scored)


def record(inst: dict, code: str | None, scores: dict, **fields) -> dict:
    """Per-instance record: id, problem, starter code, program, ``scores``, difficulty, platform, then ``fields``."""
    return {
        "id": inst["id"],
        "problem": inst["problem"][:400],
        "starter_code": inst.get("starter_code") or "",
        "predicted_code": code,
        **scores,
        "difficulty": inst.get("difficulty"),
        "platform": inst.get("platform"),
        **fields,
    }


def run_batch(
    instances: list[dict],
    row: Callable[[int, dict], dict],
    *,
    out_path: Path | None = None,
    verbose: bool = True,
    label: str = "LCB",
) -> dict:
    """:func:`core.code_tasks.run_batch` with the LCB report (pass@1)."""
    return code_tasks.run_batch(
        instances,
        row,
        out_path=out_path,
        verbose=verbose,
        label=label,
        metric="pass@1",
        id_width=22,
        difficulty_width=6,
    )


# Demo
STDIN_DEMO_PROBLEM = (
    "Read a single integer n from standard input (1 <= n <= 1000) and print the sum 1 + 2 + ... + n on one line."
)
STDIN_DEMO_TESTS = [
    {"input": "5\n", "output": "15", "testtype": "stdin"},
    {"input": "1\n", "output": "1", "testtype": "stdin"},
    {"input": "10\n", "output": "55", "testtype": "stdin"},
    {"input": "100\n", "output": "5050", "testtype": "stdin"},
]
FUNCTIONAL_DEMO_PROBLEM = (
    "Given an array of integers `nums` and an integer `target`, return "
    "the indices of the two numbers such that they add up to `target`. "
    "Assume each input has exactly one solution; the same element may "
    "not be used twice. Return the answer as a list [i, j] with i < j."
)
FUNCTIONAL_DEMO_TESTS = [
    {"fn_name": "twoSum", "input": "[[2,7,11,15], 9]", "output": "[0,1]", "testtype": "functional"},
    {"fn_name": "twoSum", "input": "[[3,2,4], 6]", "output": "[1,2]", "testtype": "functional"},
    {"fn_name": "twoSum", "input": "[[3,3], 6]", "output": "[0,1]", "testtype": "functional"},
]
# (mode, problem, starter code, tests)
DEMOS = (
    ("STDIN", STDIN_DEMO_PROBLEM, None, STDIN_DEMO_TESTS),
    ("FUNCTIONAL", FUNCTIONAL_DEMO_PROBLEM, code_tasks.TWO_SUM_STARTER, FUNCTIONAL_DEMO_TESTS),
)


def print_demo_code(code: str | None, tests) -> None:
    """Print a demo program and its LCB test results."""
    code_tasks.print_demo_code(code, tests, run=run_tests, metric="pass@1")
