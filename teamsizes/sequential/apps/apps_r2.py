"""Sequential topology specialized for APPS, implemented in LangGraph."""

# Config
from __future__ import annotations

import argparse
import ast
import json
import operator
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from teamsizes.output_contracts import append_output_contract_from_path
from topologies.code_extract import extract_python_code
from typing import Annotated

import numpy as np
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent
from typing_extensions import TypedDict

# Shared telemetry.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_TOPO_ROOT = str(_REPO_ROOT)
if _TOPO_ROOT not in sys.path:
    sys.path.insert(0, _TOPO_ROOT)
from topologies.telemetry import langchain_telemetry, normalize  # noqa: E402


VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://n12:8000/v1")
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3.5-9B")

_PROMPTS_DIR = _REPO_ROOT / "configs" / "prompts" / "sequential" / "apps"


def _load_prompt(role: str) -> str:
    return append_output_contract_from_path((_PROMPTS_DIR / f"{role}.txt").read_text().strip(), __file__, role)


# Tools
_EXEC_TIMEOUT_S = 10
# APPS reference timeout is 4 s per test case (signal.SIGALRM).
_APPS_TEST_TIMEOUT_S = 4


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


EXEC_TOOLS = [python_exec]


# LLM
def _build_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=MODEL_ID,
        base_url=VLLM_BASE_URL,
        api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        temperature=0.2,
        top_p=0.9,
        seed=0,
        max_tokens=4096,
        extra_body={
            "repetition_penalty": 1.05,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )


# Prompt scaffolding
_FORMAT_STDIN_DIRECTIVE = "Use Standard Input format."
_FORMAT_CALL_BASED_DIRECTIVE = "Use Call-Based format."


def format_prompt(problem: str, starter_code: str | None = None) -> str:
    """Build the user-facing prompt for one APPS problem."""
    parts = [f"QUESTION:\n{problem}"]
    if starter_code:
        # Escape braces so str.format() downstream doesn't choke on Python
        # dict literals or type hints in the starter.
        starter_safe = starter_code.replace("{", "{{").replace("}", "}}")
        parts.append(f"```python\n{starter_safe.rstrip()}\n```")
        parts.append(_FORMAT_CALL_BASED_DIRECTIVE)
    else:
        parts.append(_FORMAT_STDIN_DIRECTIVE)
    parts.append("Enclose your final solution in a ```python``` code block.")
    parts.append("ANSWER:")
    return "\n\n".join(parts)


def _escape_braces(s: str) -> str:
    return s.replace("{", "{{").replace("}", "}}")


# Per-stage task descriptions (same as CrewAI Task.description)
_TASK_DESCRIPTIONS = {
    "analyzer": (
        "Read the programming problem below. Pick a correct "
        "algorithmic approach: name the data structures, target "
        "complexity, and the edge cases that will bite a naive "
        "implementation. Do NOT write code — the Coder handles that.\n\n"
        "PROBLEM:\n{problem_prompt}"
    ),
    "coder": (
        "Implement the Analyzer's plan in Python. Follow the I/O "
        "format specified in the problem (stdin or starter_code "
        "call-based). Emit the complete Python program inside a "
        "fenced ```python ... ``` block.\n\n"
        "PROBLEM:\n{problem_prompt}"
    ),
    "tester": (
        "Construct 3-5 targeted test cases for the Coder's program, "
        "including at least one edge case. Run each case via "
        "python_exec. Report which cases pass or fail with the "
        "stdout/stderr evidence. Do NOT modify the program.\n\n"
        "PROBLEM:\n{problem_prompt}"
    ),
    "debugger": (
        "If the Tester reported failures, fix the Coder's program "
        "to address them (use python_exec to verify your fix). If "
        "the Tester reported all-pass, pass the Coder's program "
        "through unchanged. Your final output MUST contain the "
        "final Python program inside a fenced ```python ... ``` "
        "block (the extractor takes the last one).\n\n"
        "PROBLEM:\n{problem_prompt}"
    ),
}


# StateGraph scaffolding (sequential 4-stage pipeline)
def _merge_dict(a: dict | None, b: dict | None) -> dict:
    out = dict(a or {})
    out.update(b or {})
    return out


class SequentialState(TypedDict, total=False):
    inputs: dict
    by_stage: Annotated[dict, _merge_dict]
    messages: Annotated[list, operator.add]


def _format_user(
    template: str, inputs: dict, by_stage: dict, prior_roles: list[str]
) -> str:
    body = template.format(**inputs)
    for r in prior_roles:
        body += f"\n\n--- PRIOR STAGE: {r} ---\n{by_stage.get(r, '')}"
    return body


def _make_tool_node(role, sys_prompt, tools, llm, template, prior_roles):
    agent = create_react_agent(model=llm, tools=tools, prompt=sys_prompt)

    def node(state: SequentialState) -> dict:
        user = _format_user(
            template, state["inputs"], state.get("by_stage") or {}, prior_roles
        )
        res = agent.invoke(
            {"messages": [("user", user)]},
            config={"recursion_limit": 50},
        )
        raw = next(
            (
                m.content
                for m in reversed(res["messages"])
                if getattr(m, "type", None) == "ai" and getattr(m, "content", "")
            ),
            "",
        )
        ai_msgs = [m for m in res["messages"] if getattr(m, "type", None) == "ai"]
        return {"by_stage": {role: raw}, "messages": ai_msgs}

    return node


def _make_plain_node(role, sys_prompt, llm, template, prior_roles):
    def node(state: SequentialState) -> dict:
        user = _format_user(
            template, state["inputs"], state.get("by_stage") or {}, prior_roles
        )
        ai = llm.invoke(
            [SystemMessage(content=sys_prompt), HumanMessage(content=user)]
        )
        return {"by_stage": {role: ai.content or ""}, "messages": [ai]}

    return node


def _build_graph(llm: ChatOpenAI):
    """Build the 4-stage analyzer -> coder -> tester -> debugger pipeline."""
    stages = [
        (
            "coder",
            _load_prompt("coder"),
            [],
            _TASK_DESCRIPTIONS["coder"],
        ),
        (
            "debugger",
            _load_prompt("debugger"),
            EXEC_TOOLS,
            _TASK_DESCRIPTIONS["debugger"],
        ),
    ]

    graph = StateGraph(SequentialState)
    prior: list[str] = []
    for role, sys_p, tools, tmpl in stages:
        node_fn = (
            _make_tool_node(role, sys_p, tools, llm, tmpl, list(prior))
            if tools
            else _make_plain_node(role, sys_p, llm, tmpl, list(prior))
        )
        graph.add_node(role, node_fn)
        prior.append(role)

    graph.add_edge(START, stages[0][0])
    for a, b in zip(stages, stages[1:]):
        graph.add_edge(a[0], b[0])
    graph.add_edge(stages[-1][0], END)

    return graph.compile(), [s[0] for s in stages]


# Output Parsing
_CODE_BLOCK_RE = re.compile(
    r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE
)


def extract_code(text: str) -> str | None:
    """Return the last fenced block that parses as Python, or None."""
    return extract_python_code(text)
# Scoring
# Aligned to topologies/single/apps/langgraph_apps.py so sequential
# strict-accuracy is directly comparable with single/independent numbers.
# Comparison cascade matches APPS' eval/testing_util.py:
#   1. stripped string match
#   2. list equality
#   3. numeric allclose (rtol=1e-5, atol=1e-6)
#   4. set equality (unordered)
#   5. rounded-float set (3 decimals)


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


def _call_based_compare(actual, expected) -> bool:
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


def _stdout_compare(actual_str: str, expected_str: str) -> bool:
    if _stripped_string_compare(actual_str, expected_str):
        return True

    a_lines = [line.rstrip() for line in actual_str.strip().splitlines()]
    e_lines = [line.rstrip() for line in expected_str.strip().splitlines()]
    if a_lines == e_lines:
        return True

    try:
        a_nums = [float(t) for line in a_lines for t in line.split()]
        e_nums = [float(t) for line in e_lines for t in line.split()]
    except ValueError:
        return False
    if len(a_nums) != len(e_nums):
        return False
    return bool(np.allclose(a_nums, e_nums, rtol=1e-5, atol=1e-6))


def _parse_maybe_literal(value):
    """Accept raw Python objects, JSON strings, or Python-literal strings."""
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
        result = subprocess.run(
            ["python", "-c", code],
            input=stdin,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timeout", "mode": "stdin"}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}", "mode": "stdin"}

    actual = result.stdout
    ok = _stdout_compare(actual, expected) and result.returncode == 0
    return {
        "ok": ok,
        "expected": expected.strip() if isinstance(expected, str) else expected,
        "actual": actual.strip(),
        "stderr": result.stderr.strip(),
        "exit_code": result.returncode,
        "mode": "stdin",
    }


# Worker script for call-based tests. Runs in a fresh subprocess with APPS'
# reliability_guard (disables os.kill/system/fork/..., shutil.rmtree, etc.)
# and optional resource.setrlimit caps. Guard ported verbatim from APPS'
# testing_util.py. Communicates result back via a temp JSON file.
# argv: [1]=max_memory_bytes (0=>unlimited), [2]=fn_name, [3]=args_json,
#       [4]=outfile; stdin: user's submission source.
_CALL_BASED_WORKER = """
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

# Per-test memory cap. Default 4 GB, matching APPS' recommended setting.
# Set APPS_CALL_BASED_MEMORY_BYTES=0 to disable.
_CALL_BASED_MEMORY_BYTES = int(
    os.environ.get("APPS_CALL_BASED_MEMORY_BYTES", str(4 * 1024 ** 3))
)


def _run_call_based_test(
    code: str, fn_name: str, raw_args, raw_expected, timeout_s: int
) -> dict:
    args = _parse_maybe_literal(raw_args)
    if not isinstance(args, (list, tuple)):
        args = [args]
    expected = _parse_maybe_literal(raw_expected)

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        outfile = f.name

    try:
        proc = subprocess.run(
            [
                "python", "-c", _CALL_BASED_WORKER,
                str(_CALL_BASED_MEMORY_BYTES),
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
        return {"ok": False, "error": "timeout", "mode": "call_based"}
    except Exception as e:
        os.unlink(outfile)
        return {"ok": False, "error": f"{type(e).__name__}: {e}", "mode": "call_based"}

    try:
        with open(outfile) as f:
            payload = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError, OSError):
        return {
            "ok": False,
            "error": f"worker produced no parseable result (exit={proc.returncode})",
            "stderr": (proc.stderr or "").strip(),
            "mode": "call_based",
        }
    finally:
        try:
            os.unlink(outfile)
        except FileNotFoundError:
            pass

    if not payload.get("ok"):
        return {
            "ok": False,
            "error": payload.get("error", "worker error"),
            "mode": "call_based",
        }
    actual = payload["result"]
    return {
        "ok": _call_based_compare(actual, expected),
        "expected": expected,
        "actual": actual,
        "mode": "call_based",
    }


def run_tests(
    code: str,
    input_output: dict,
    timeout_s: int = _APPS_TEST_TIMEOUT_S,
) -> dict:
    """Run `code` against an APPS-style input_output dict.

    Expected shape (matches the raw dataset field):
        {
            "inputs":  [<stdin_str> or <arg_list>, ...],
            "outputs": [<expected_stdout> or <expected_return>, ...],
            "fn_name": <str>          # optional; presence => call-based mode
        }
    """
    inputs = input_output.get("inputs", []) or []
    outputs = input_output.get("outputs", []) or []
    fn_name = input_output.get("fn_name")

    if len(inputs) != len(outputs):
        return {
            "pass": 0,
            "total": 0,
            "pass_rate": 0.0,
            "details": [],
            "error": "inputs/outputs length mismatch",
        }
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

    return {
        "pass": passed,
        "total": len(inputs),
        "pass_rate": passed / len(inputs),
        "details": details,
    }


def exact_match_score(pass_rate: float) -> float:
    """APPS strict accuracy: 1.0 iff all tests pass, else 0.0."""
    return 1.0 if pass_rate == 1.0 else 0.0


# Orchestration
def solve(problem: str, starter_code: str | None = None) -> dict:
    """Run the 4-stage sequential graph on one APPS problem.

    Returns:
        {
            "code":      final Python program (str) or None,
            "raw":       debugger's final text,
            "by_stage":  {analyzer, coder, tester, debugger} -> each stage's output,
            "telemetry": normalized 5-key token/call counts,
        }
    """
    llm = _build_llm()
    compiled, roles = _build_graph(llm)
    problem_prompt = _escape_braces(format_prompt(problem, starter_code))
    result = compiled.invoke(
        {"inputs": {"problem_prompt": problem_prompt}, "by_stage": {}, "messages": []}
    )

    stages_out = result.get("by_stage") or {}
    final = stages_out.get(roles[-1], "")

    # Prefer the debugger's emission; fall back to the coder's if empty.
    code = extract_code(final)
    if code is None:
        code = extract_code(stages_out.get("coder", ""))

    return {
        "code": code,
        "raw": final,
        "by_stage": stages_out,
        "telemetry": normalize(langchain_telemetry(result.get("messages") or [])),
    }


# Dataset loader
_HF_DATASET = "codeparrot/apps"
_HF_SPLIT = "test"


def _parse_input_output(blob: str) -> dict | None:
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
    max_tests_per_row: int | None = 20,
) -> list[dict]:
    """Load APPS test rows — same IDs as single/apps for parity."""
    from datasets import load_dataset

    ds = load_dataset(_HF_DATASET, split=_HF_SPLIT, trust_remote_code=True)
    rows: list[dict] = []
    for row in ds:
        rid = str(row.get("problem_id"))
        if only is not None and rid not in set(only):
            continue
        if difficulty is not None and row.get("difficulty") != difficulty:
            continue
        problem = (row.get("question") or "").strip()
        if not problem:
            continue
        io = _parse_input_output(row.get("input_output") or "")
        if not io:
            continue
        if max_tests_per_row is not None and max_tests_per_row > 0:
            io = {
                "inputs":  io["inputs"][:max_tests_per_row],
                "outputs": io["outputs"][:max_tests_per_row],
                **({"fn_name": io["fn_name"]} if io.get("fn_name") else {}),
            }
        rows.append({
            "id": rid,
            "problem": problem,
            "starter_code": (row.get("starter_code") or "").rstrip(),
            "input_output": io,
            "difficulty": row.get("difficulty"),
            "raw": {k: row.get(k) for k in ("problem_id", "difficulty", "url")},
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
    per_test_timeout_s: int = _APPS_TEST_TIMEOUT_S,
) -> dict:
    """Run `solve()` on every problem, score with `run_tests()`, write JSONL."""
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
                out = solve(inst["problem"], starter_code=inst.get("starter_code") or None)
                error = None
            except Exception as e:
                out = {"code": None, "raw": "", "by_stage": {}}
                error = f"{type(e).__name__}: {e}"
            latency_s = time.time() - t0

            code = out["code"]
            if code:
                n_extracted += 1
                scored = run_tests(code, inst["input_output"], timeout_s=per_test_timeout_s)
            else:
                scored = {
                    "pass": 0,
                    "total": len(inst["input_output"].get("inputs", [])),
                    "pass_rate": 0.0, "details": [],
                }
            em = exact_match_score(scored["pass_rate"]) if code else 0.0
            em_sum += em
            by_diff.setdefault(inst.get("difficulty") or "unk", []).append(em)

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
                "by_stage": out.get("by_stage") or {},
                "latency_s": round(latency_s, 2),
                **(out.get("telemetry") or {}),
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
                    f"[{i + 1:>3}/{n}] {inst['id'][:10]:<10} {mark}  "
                    f"em={em:.0f}  pass={scored['pass']}/{scored['total']}  "
                    f"EM={running_em:.3f} lat={latency_s:.1f}s",
                    flush=True,
                )
    finally:
        if out_f is not None:
            out_f.close()

    elapsed = time.time() - start
    summary = {
        "n": n, "n_extracted": n_extracted,
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
            f"\n=== sequential/APPS batch complete ===\n"
            f"  n={summary['n']}  n_extracted={summary['n_extracted']}\n"
            f"  strict_acc EM={summary['em']:.3f}  "
            f"(on extracted only: {summary['extracted_em']:.3f})\n"
        )
        for d, v in summary["by_difficulty"].items():
            print(f"    {d:>14s}: n={v['n']:3d}  EM={v['em']:.3f}")
        print(f"  total_s={summary['total_s']}\n")
    return summary


# Demo
def _print_scoring(scored: dict) -> None:
    print(
        f"=== Tests: {scored['pass']}/{scored['total']}   "
        f"pass_rate: {scored['pass_rate']:.2f}   "
        f"strict_acc: {exact_match_score(scored['pass_rate']):.2f} ==="
    )
    for d in scored["details"]:
        if not d.get("ok"):
            print(
                f"    [FAIL {d.get('mode', '?')}] test {d['test']}: "
                f"expected={d.get('expected')!r}  "
                f"actual={d.get('actual')!r}  "
                f"err={d.get('error') or d.get('stderr')!r}"
            )


def _canned_demo() -> None:
    stdin_problem = (
        "Read a single integer n (1 <= n <= 1000) from standard input and "
        "print n squared on a single line."
    )
    stdin_io = {
        "inputs":  ["5\n",  "1\n", "10\n",  "23\n"],
        "outputs": ["25",   "1",   "100",   "529"],
    }

    print("========== STANDARD INPUT MODE ==========")
    out = solve(stdin_problem)
    print(f"\n=== Analyzer (excerpt) ===\n{out['by_stage'].get('analyzer', '')[:400]}...")
    print(f"\n=== Coder (excerpt) ===\n{out['by_stage'].get('coder', '')[:400]}...")
    print(f"\n=== Tester (excerpt) ===\n{out['by_stage'].get('tester', '')[:400]}...")
    print(f"\n=== Debugger (excerpt) ===\n{out['by_stage'].get('debugger', '')[:400]}...")
    if out["code"]:
        print(f"\n=== Final code ===\n{out['code']}\n")
        _print_scoring(run_tests(out["code"], stdin_io))
    else:
        print("\n=== No code extracted ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sequential-topology APPS runner (LangGraph 4-stage)."
    )
    parser.add_argument(
        "--batch", action="store_true",
        help="Run the real APPS eval (else: one canned demo).",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument(
        "--difficulty", type=str, default=None,
        choices=("introductory", "interview", "competition"),
    )
    parser.add_argument("--max-tests-per-row", type=int, default=20)
    args = parser.parse_args()

    if not args.batch:
        _canned_demo()
        sys.exit(0)

    max_tests = None if args.max_tests_per_row < 0 else args.max_tests_per_row
    print(f"loading APPS from {_HF_DATASET} [{_HF_SPLIT}] ...")
    instances = load_instances(
        limit=args.limit, offset=args.offset,
        only=args.only, difficulty=args.difficulty,
        max_tests_per_row=max_tests,
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
