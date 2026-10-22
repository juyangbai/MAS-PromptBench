"""APPS: data, prompts, test execution, selection and batch records.

The benchmark slice is codeparrot/apps [test] with at most 20 tests per problem. A
submission is the last fenced Python block of the model output; it is run against
the problem's tests with the comparison cascade of APPS' reference evaluation
(standard-input programs, or call-based functions in a guarded subprocess) and
scores strict accuracy (1.0 iff every test passes). The helpers shared with LCB
live in :mod:`core.code_tasks`.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
from collections.abc import Callable
from pathlib import Path

import numpy as np

from core import code_tasks

DATASET = "apps"
HF_DATASET = "codeparrot/apps"
HF_SPLIT = "test"
SOURCE = f"APPS from {HF_DATASET} [{HF_SPLIT}]"
DIFFICULTIES = ("introductory", "interview", "competition")
MAX_TESTS_PER_ROW = 20

# APPS' reference per-test timeout (signal.SIGALRM), the default of every test run.
TEST_TIMEOUT_S = 4


# Command line
def add_arguments(parser) -> None:
    """``--difficulty`` and ``--max-tests-per-row``."""
    parser.add_argument(
        "--difficulty", type=str, default=None, choices=DIFFICULTIES, help="filter to one APPS difficulty tier"
    )
    parser.add_argument(
        "--max-tests-per-row",
        type=int,
        default=MAX_TESTS_PER_ROW,
        help=f"cap tests per APPS row (default: {MAX_TESTS_PER_ROW}; a value <= 0 keeps all tests)",
    )


# Data
def parse_input_output(blob: str) -> dict | None:
    """The row's ``input_output`` JSON, or None when it is missing, malformed or empty."""
    if not blob:
        return None
    try:
        io = json.loads(blob)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(io, dict):
        return None
    if not io.get("inputs") or not io.get("outputs"):
        return None
    return io


def load_instances(
    limit: int | None = None,
    offset: int = 0,
    only: list[str] | None = None,
    difficulty: str | None = None,
    max_tests_per_row: int | None = MAX_TESTS_PER_ROW,
) -> list[dict]:
    """Rows with tests as ``{id, problem, starter_code, input_output, difficulty, raw}``.

    ``input_output`` keeps the first ``max_tests_per_row`` tests (all of them for None or a value <= 0).
    """
    from datasets import load_dataset

    rows: list[dict] = []
    for row in load_dataset(HF_DATASET, split=HF_SPLIT, trust_remote_code=True):
        rid = str(row.get("problem_id"))
        if only is not None and rid not in set(only):
            continue
        if difficulty is not None and row.get("difficulty") != difficulty:
            continue
        problem = (row.get("question") or "").strip()
        if not problem:
            continue
        io = parse_input_output(row.get("input_output") or "")
        if not io:
            continue
        if max_tests_per_row is not None and max_tests_per_row > 0:
            io = {
                "inputs": io["inputs"][:max_tests_per_row],
                "outputs": io["outputs"][:max_tests_per_row],
                **({"fn_name": io["fn_name"]} if io.get("fn_name") else {}),
            }
        rows.append(
            {
                "id": rid,
                "problem": problem,
                "starter_code": (row.get("starter_code") or "").rstrip(),
                "input_output": io,
                "difficulty": row.get("difficulty"),
                "raw": {k: row.get(k) for k in ("problem_id", "difficulty", "url")},
            }
        )
    rows = rows[offset:]
    return rows if limit is None else rows[:limit]


# User message: the APPS paper's "QUESTION: ... ANSWER:" scaffold with a mode directive.
FORMAT_STDIN_DIRECTIVE = "Use Standard Input format."
FORMAT_CALL_BASED_DIRECTIVE = "Use Call-Based format."

# Closing instruction of the decentralized peer-review message.
PEER_REVISION_NOTE = (
    "\nCompare their code and approach. Revise your submission ONLY if "
    "a peer handles an edge case you missed or runs in better "
    "complexity. Re-emit your final code in a SINGLE fenced ```python``` "
    "block at the end.\n\nOriginal problem:\n"
)


def format_prompt(problem: str, starter_code: str | None = None) -> str:
    """The question, the starter code and call-based directive (or the standard-input directive), and the answer cue."""
    parts = [f"QUESTION:\n{problem}"]
    if starter_code:
        parts.append(f"```python\n{starter_code.rstrip()}\n```")
        parts.append(FORMAT_CALL_BASED_DIRECTIVE)
    else:
        parts.append(FORMAT_STDIN_DIRECTIVE)
    parts.append("Enclose your final solution in a ```python``` code block.")
    parts.append("ANSWER:")
    return "\n\n".join(parts)


# Scoring: the comparison cascade of APPS' eval/testing_util.py (stripped string,
# list equality, numeric allclose, set equality, rounded-float set). Kept as ported
# so strict accuracy stays comparable with prior work.
CALL_BASED_MEMORY_BYTES = int(os.environ.get("APPS_CALL_BASED_MEMORY_BYTES", str(4 * 1024**3)))  # 0: no cap


def _stripped_string_compare(a: str, b: str) -> bool:
    return a.strip() == b.strip()


def _try_numeric_allclose(a, b) -> bool:
    try:
        a_arr = np.asarray(a, dtype=float).flatten()
        b_arr = np.asarray(b, dtype=float).flatten()
    except (ValueError, TypeError):
        return False
    if a_arr.shape != b_arr.shape:
        return False
    return bool(np.allclose(a_arr, b_arr, rtol=1e-5, atol=1e-6))


def _try_set_equal(a, b) -> bool:
    try:
        return set(a) == set(b)
    except TypeError:
        return False


def _try_rounded_float_set(a, b, precision: int = 3) -> bool:
    try:
        a_round = {round(float(x), precision) for x in a}
        b_round = {round(float(x), precision) for x in b}
    except (TypeError, ValueError):
        return False
    return a_round == b_round


def call_based_compare(actual, expected) -> bool:
    """Cascading comparison of one call-based test's return value."""
    if isinstance(actual, tuple):
        actual = list(actual)
    if actual == expected:
        return True
    if _try_numeric_allclose(actual, expected):
        return True
    if _try_set_equal(actual, expected):
        return True
    if _try_rounded_float_set(actual, expected):
        return True
    return False


def stdout_compare(actual_str: str, expected_str: str) -> bool:
    """Cascading comparison of one standard-input test's captured stdout."""
    if _stripped_string_compare(actual_str, expected_str):
        return True
    a_lines = [line.rstrip() for line in actual_str.strip().splitlines()]
    e_lines = [line.rstrip() for line in expected_str.strip().splitlines()]
    if a_lines == e_lines:
        return True
    # numeric per-token tolerance over the concatenated token stream
    try:
        a_nums = [float(t) for line in a_lines for t in line.split()]
        e_nums = [float(t) for line in e_lines for t in line.split()]
    except ValueError:
        return False
    if len(a_nums) != len(e_nums):
        return False
    return bool(np.allclose(a_nums, e_nums, rtol=1e-5, atol=1e-6))


def _parse_maybe_literal(value):
    """A raw object, or a JSON / Python-literal string parsed (the string itself when neither parses)."""
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        pass
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def _run_stdin_test(code: str, stdin: str, expected: str, timeout_s: int) -> dict:
    try:
        result = subprocess.run(["python", "-c", code], input=stdin, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timeout", "mode": "stdin"}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}", "mode": "stdin"}
    actual = result.stdout
    return {
        "ok": stdout_compare(actual, expected) and result.returncode == 0,
        "expected": expected.strip() if isinstance(expected, str) else expected,
        "actual": actual.strip(),
        "stderr": result.stderr.strip(),
        "exit_code": result.returncode,
        "mode": "stdin",
    }


def _run_call_based_test(code: str, fn_name: str, raw_args, raw_expected, timeout_s: int) -> dict:
    args = _parse_maybe_literal(raw_args)
    if not isinstance(args, (list, tuple)):
        args = [args]
    expected = _parse_maybe_literal(raw_expected)
    called = code_tasks.guarded_call(code, fn_name, args, timeout_s, CALL_BASED_MEMORY_BYTES)
    if not called["ok"]:
        return {**called, "mode": "call_based"}
    actual = called["result"]
    return {"ok": call_based_compare(actual, expected), "expected": expected, "actual": actual, "mode": "call_based"}


def run_tests(code: str, input_output: dict, timeout_s: int = TEST_TIMEOUT_S) -> dict:
    """Run ``code`` on an APPS ``{"inputs", "outputs", "fn_name"?}`` dict (``fn_name`` selects call-based mode).

    Returns ``{"pass", "total", "pass_rate", "details"}`` (plus ``"error"`` when inputs and outputs differ in length).
    """
    inputs = input_output.get("inputs", []) or []
    outputs = input_output.get("outputs", []) or []
    fn_name = input_output.get("fn_name")
    if len(inputs) != len(outputs):
        return {"pass": 0, "total": 0, "pass_rate": 0.0, "details": [], "error": "inputs/outputs length mismatch"}
    if not inputs:
        return {"pass": 0, "total": 0, "pass_rate": 0.0, "details": []}
    passed = 0
    details = []
    for i, (inp, exp) in enumerate(zip(inputs, outputs)):
        if fn_name:
            r = _run_call_based_test(code, fn_name, inp, exp, timeout_s)
        else:
            stdin_str = inp if isinstance(inp, str) else str(inp)
            exp_str = exp if isinstance(exp, str) else str(exp)
            r = _run_stdin_test(code, stdin_str, exp_str, timeout_s)
        passed += int(r.get("ok", False))
        details.append({"test": i, **r})
    return {"pass": passed, "total": len(inputs), "pass_rate": passed / len(inputs), "details": details}


# Scoring an ensemble's selected program
def score_selected_peer(per_peer: list[dict], winner: int, code: str | None, input_output: dict) -> None:
    """:func:`core.code_tasks.score_selected_peer` on APPS tests."""
    code_tasks.score_selected_peer(per_peer, winner, code, input_output, run=run_tests)


# Batch records
def test_scores(code: str | None, input_output: dict, timeout_s: int) -> dict:
    """Record fields pass / total / pass_rate / em of ``code`` on the tests (zero without code)."""
    if code:
        scored = run_tests(code, input_output, timeout_s=timeout_s)
    else:
        scored = code_tasks.unscored(len(input_output.get("inputs", [])))
    return code_tasks.scores_of(code, scored)


def record(inst: dict, code: str | None, scores: dict, **fields) -> dict:
    """Per-instance record: id, problem, starter code, program, ``scores``, difficulty, then ``fields``."""
    return {
        "id": inst["id"],
        "problem": inst["problem"][:400],
        "starter_code": inst.get("starter_code") or "",
        "predicted_code": code,
        **scores,
        "difficulty": inst.get("difficulty"),
        **fields,
    }


def run_batch(
    instances: list[dict],
    row: Callable[[int, dict], dict],
    *,
    out_path: Path | None = None,
    verbose: bool = True,
    label: str = "APPS",
) -> dict:
    """:func:`core.code_tasks.run_batch` with the APPS report (strict accuracy)."""
    return code_tasks.run_batch(
        instances,
        row,
        out_path=out_path,
        verbose=verbose,
        label=label,
        metric="strict_acc",
        id_width=10,
        difficulty_width=14,
    )


# Demo
STDIN_DEMO_PROBLEM = (
    "Read a single integer n (1 <= n <= 1000) from standard input and print n squared on a single line."
)
STDIN_DEMO_TESTS = {"inputs": ["5\n", "1\n", "10\n", "23\n"], "outputs": ["25", "1", "100", "529"]}
CALL_BASED_DEMO_PROBLEM = (
    "Given a list of integers `nums` and an integer `target`, return the "
    "indices of the two numbers in `nums` that add up to `target`. Assume "
    "exactly one solution exists and the same element may not be used "
    "twice. Return the answer as a list [i, j] with i < j."
)
CALL_BASED_DEMO_TESTS = {
    "fn_name": "twoSum",
    "inputs": [[[2, 7, 11, 15], 9], [[3, 2, 4], 6], [[3, 3], 6]],
    "outputs": [[0, 1], [1, 2], [0, 1]],
}
# (mode, problem, starter code, tests)
DEMOS = (
    ("STANDARD INPUT", STDIN_DEMO_PROBLEM, None, STDIN_DEMO_TESTS),
    ("CALL-BASED", CALL_BASED_DEMO_PROBLEM, code_tasks.TWO_SUM_STARTER, CALL_BASED_DEMO_TESTS),
)


def print_demo_code(code: str | None, input_output: dict) -> None:
    """Print a demo program and its APPS test results."""
    code_tasks.print_demo_code(code, input_output, run=run_tests, metric="strict_acc")
