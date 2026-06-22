"""LiveCodeBench loader and lightweight scorer for real-runner GEPA."""
from __future__ import annotations

import base64
import json
import pickle
import random
import re
import subprocess
import tempfile
import zlib
from pathlib import Path
from typing import Any

import dspy

from real_runner_gepa.datasets.split_utils import train_val_split_excluding_real_eval


HF_DATASET = "livecodebench/code_generation_lite"
HF_SPLIT = "test"
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


def _decode_private_tests(blob: str) -> list[dict]:
    if not blob:
        return []
    try:
        decompressed = zlib.decompress(base64.b64decode(blob))
    except Exception:
        return []
    try:
        payload = pickle.loads(decompressed)
        return json.loads(payload) if isinstance(payload, str) else payload
    except Exception:
        try:
            return json.loads(decompressed)
        except Exception:
            return []


def load_all() -> list[dspy.Example]:
    from datasets import load_dataset

    ds = load_dataset(HF_DATASET, split=HF_SPLIT)
    rows: list[dspy.Example] = []
    for raw in ds:
        rid = raw.get("question_id") or ""
        problem = (raw.get("question_content") or "").strip()
        tests = _decode_private_tests(raw.get("private_test_cases") or "")
        if not rid or not problem or not tests:
            continue
        starter_code = (raw.get("starter_code") or "").rstrip()
        instance = {
            "id": str(rid),
            "problem": problem,
            "starter_code": starter_code,
            "tests": tests,
            "difficulty": raw.get("difficulty"),
            "raw": dict(raw),
        }
        rows.append(
            dspy.Example(
                id=str(rid),
                task_instance=instance,
                problem=problem,
                starter_code=starter_code,
                tests=tests,
                answer=f"<{len(tests)} hidden tests>",
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
    return train_val_split_excluding_real_eval("lcb", examples, train_size, val_size, seed, offset)


def _exec_via_tempfile(script_text: str, stdin: str | None, timeout_s: int = 5):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(script_text)
        path = f.name
    try:
        return subprocess.run(["python", path], input=stdin, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return None
    finally:
        Path(path).unlink(missing_ok=True)


def _normalize_stdout(s: str) -> str:
    lines = [line.rstrip() for line in (s or "").splitlines()]
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


FUNCTIONAL_DRIVER = """
import json, sys
{code}
_payload = json.loads(sys.stdin.read())
raw_inputs = _payload["inputs"]
fn_name = _payload["fn_name"]
try:
    inputs = json.loads(raw_inputs) if isinstance(raw_inputs, str) else raw_inputs
except Exception:
    import ast
    inputs = ast.literal_eval(raw_inputs) if isinstance(raw_inputs, str) else raw_inputs
fn = locals().get(fn_name) or globals().get(fn_name)
out = fn(*inputs) if isinstance(inputs, list) else fn(**inputs)
print(json.dumps(out, default=str))
"""


def _run_stdin_test(code: str, tc: dict) -> dict:
    expected = _normalize_stdout(tc.get("output", ""))
    result = _exec_via_tempfile(code, tc.get("input", ""))
    actual = "(timeout)" if result is None else _normalize_stdout(result.stdout)
    return {"ok": actual == expected, "expected": expected[-120:], "actual": actual[-120:], "mode": "stdin"}


def _run_functional_test(code: str, tc: dict) -> dict:
    payload = json.dumps({"inputs": tc.get("input", "[]"), "fn_name": tc.get("fn_name") or tc.get("func_name") or ""})
    result = _exec_via_tempfile(FUNCTIONAL_DRIVER.format(code=code), payload)
    expected = (tc.get("output", "") or "").strip()
    actual = "(timeout)" if result is None else (result.stdout or "").strip()
    try:
        ok = json.loads(actual) == json.loads(expected)
    except Exception:
        ok = actual == expected
    return {"ok": ok, "expected": expected[-120:], "actual": actual[-120:], "mode": "functional"}


def run_tests(code: str | None, tests: list[dict] | None, test_limit: int = COMPILE_TEST_LIMIT) -> dict:
    tests = (tests or [])[:test_limit]
    if not code or not tests:
        return {"pass": 0, "total": len(tests), "pass_rate": 0.0, "details": []}
    details = []
    passed = 0
    for tc in tests:
        is_func = tc.get("testtype") == "functional" or tc.get("fn_name") or tc.get("func_name")
        detail = _run_functional_test(code, tc) if is_func else _run_stdin_test(code, tc)
        passed += int(detail["ok"])
        details.append(detail)
    return {"pass": passed, "total": len(tests), "pass_rate": passed / max(len(tests), 1), "details": details}


def exact_match_score(pass_rate: float) -> float:
    return 1.0 if pass_rate == 1.0 else 0.0


def metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
    raw = getattr(prediction, "code", None) or getattr(prediction, "answer", None) or str(prediction)
    code = extract_code(raw)
    summary = run_tests(code, getattr(example, "tests", None))
    score = exact_match_score(summary["pass_rate"])
    if score:
        feedback = f"Correct. All {summary['total']} capped LCB tests passed."
    elif code is None:
        feedback = "Format failure: no final fenced ```python``` solution block was extracted."
    else:
        feedback = f"Incorrect LCB solution: passed {summary['pass']}/{summary['total']} capped tests."
    return dspy.Prediction(score=score, feedback=feedback)
