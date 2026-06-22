"""APPS loader and lightweight scorer for real-runner GEPA."""
from __future__ import annotations

import ast
import json
import random
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import dspy

from real_runner_gepa.datasets.split_utils import train_val_split_excluding_real_eval


HF_DATASET = "codeparrot/apps"
HF_SPLIT = "test"
DEFAULT_DIFFICULTY = "interview"
MAX_TESTS_PER_ROW = 20
COMPILE_TEST_LIMIT = 3
CODE_BLOCK_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def strip_thinking(text: str | None) -> str:
    text = text or ""
    idx = text.lower().rfind("</think>")
    if idx >= 0:
        text = text[idx + len("</think>"):]
    return text.strip()


def extract_code(text: str | None) -> str | None:
    matches = CODE_BLOCK_RE.findall(strip_thinking(text))
    return matches[-1].strip() if matches else None


def _parse_input_output(blob: str) -> dict | None:
    if not blob:
        return None
    try:
        io = json.loads(blob)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(io, dict) or not io.get("inputs") or not io.get("outputs"):
        return None
    return io


def load_all(difficulty: str = DEFAULT_DIFFICULTY) -> list[dspy.Example]:
    from datasets import load_dataset

    ds = load_dataset(HF_DATASET, split=HF_SPLIT)
    rows: list[dspy.Example] = []
    for raw in ds:
        if difficulty and raw.get("difficulty") != difficulty:
            continue
        rid = str(raw.get("problem_id"))
        problem = (raw.get("question") or "").strip()
        io = _parse_input_output(raw.get("input_output") or "")
        if not rid or not problem or not io:
            continue
        io = {
            "inputs": io["inputs"][:MAX_TESTS_PER_ROW],
            "outputs": io["outputs"][:MAX_TESTS_PER_ROW],
            **({"fn_name": io["fn_name"]} if io.get("fn_name") else {}),
        }
        starter_code = (raw.get("starter_code") or "").rstrip()
        instance = {
            "id": rid,
            "problem": problem,
            "starter_code": starter_code,
            "input_output": io,
            "difficulty": raw.get("difficulty"),
            "raw": dict(raw),
        }
        rows.append(
            dspy.Example(
                id=rid,
                task_instance=instance,
                problem=problem,
                starter_code=starter_code,
                input_output=io,
                answer=f"<{len(io['inputs'])} hidden tests>",
            ).with_inputs("task_instance")
        )
    return rows


def train_val_split(
    examples: list[dspy.Example],
    train_size: int,
    val_size: int,
    seed: int = 0,
    offset: int = 0,
) -> tuple[list[dspy.Example], list[dspy.Example]]:
    return train_val_split_excluding_real_eval("apps", examples, train_size, val_size, seed, offset)


def _parse_maybe_literal(value: Any) -> Any:
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


def _exec_via_tempfile(script_text: str, stdin: str | None, timeout_s: int = 5):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(script_text)
        path = f.name
    try:
        return subprocess.run(
            ["python", path],
            input=stdin,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return None
    finally:
        Path(path).unlink(missing_ok=True)


def _stdout_compare(actual: str, expected: str) -> bool:
    if actual.strip() == expected.strip():
        return True
    a_lines = [line.rstrip() for line in actual.strip().splitlines()]
    e_lines = [line.rstrip() for line in expected.strip().splitlines()]
    return a_lines == e_lines


CALL_DRIVER = """
import json, sys
{code}
_payload = json.loads(sys.stdin.read())
raw_args = _payload["args"]
fn_name = _payload["fn_name"]
try:
    args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
except Exception:
    import ast
    args = ast.literal_eval(raw_args) if isinstance(raw_args, str) else raw_args
fn = locals().get(fn_name) or globals().get(fn_name)
if fn is None:
    for v in list(globals().values()):
        if isinstance(v, type) and getattr(v, fn_name, None) is not None:
            fn = lambda *a, _v=v, _m=fn_name: getattr(_v(), _m)(*a)
            break
out = fn(*args) if isinstance(args, list) else fn(args)
print(json.dumps(out, default=str))
"""


def _run_call_test(code: str, fn_name: str, raw_args: Any, expected: Any) -> dict:
    payload = json.dumps({"args": raw_args, "fn_name": fn_name})
    result = _exec_via_tempfile(CALL_DRIVER.format(code=code), payload)
    if result is None:
        return {"ok": False, "expected": str(expected)[-120:], "actual": "(timeout)", "mode": "call_based"}
    try:
        actual = json.loads(result.stdout.strip())
    except Exception:
        actual = result.stdout.strip()
    exp = _parse_maybe_literal(expected)
    return {"ok": actual == exp, "expected": str(exp)[-120:], "actual": str(actual)[-120:], "mode": "call_based"}


def _run_stdin_test(code: str, stdin: str, expected: str) -> dict:
    result = _exec_via_tempfile(code, stdin)
    if result is None:
        return {"ok": False, "expected": expected[-120:], "actual": "(timeout)", "mode": "stdin"}
    return {
        "ok": _stdout_compare(result.stdout or "", str(expected)),
        "expected": str(expected)[-120:],
        "actual": (result.stdout or "")[-120:],
        "mode": "stdin",
    }


def run_tests(code: str | None, io: dict | None, test_limit: int = COMPILE_TEST_LIMIT) -> dict:
    inputs = (io or {}).get("inputs") or []
    outputs = (io or {}).get("outputs") or []
    total = min(len(inputs), len(outputs), test_limit)
    if not code or total <= 0:
        return {"pass": 0, "total": total, "pass_rate": 0.0, "details": []}
    details = []
    passed = 0
    fn_name = (io or {}).get("fn_name")
    for raw_input, expected in zip(inputs[:total], outputs[:total]):
        detail = (
            _run_call_test(code, fn_name, raw_input, expected)
            if fn_name
            else _run_stdin_test(code, str(raw_input), str(expected))
        )
        passed += int(detail["ok"])
        details.append(detail)
    return {"pass": passed, "total": total, "pass_rate": passed / max(total, 1), "details": details}


def exact_match_score(pass_rate: float) -> float:
    return 1.0 if pass_rate == 1.0 else 0.0


def metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
    raw = getattr(prediction, "code", None) or getattr(prediction, "answer", None) or str(prediction)
    code = extract_code(raw)
    summary = run_tests(code, getattr(example, "input_output", None))
    score = exact_match_score(summary["pass_rate"])
    if score:
        feedback = f"Correct. All {summary['total']} capped APPS tests passed."
    elif code is None:
        feedback = "Format failure: no final fenced ```python``` solution block was extracted."
    else:
        feedback = f"Incorrect APPS solution: passed {summary['pass']}/{summary['total']} capped tests."
    return dspy.Prediction(score=score, feedback=feedback)
