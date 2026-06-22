"""Decentralized debate topology specialized for APPS, OpenAI SDK."""

# Config
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from topologies.output_contracts import append_output_contract_from_path
from topologies.code_extract import extract_python_code

import numpy as np
from openai import OpenAI

# Shared telemetry.
_TOPO_ROOT = str(Path(__file__).resolve().parents[4])
if _TOPO_ROOT not in sys.path:
    sys.path.insert(0, _TOPO_ROOT)
from topologies.telemetry import (  # noqa: E402
    openai_sdk_telemetry, openai_sdk_accumulate, normalize,
)


_TELEM_ACC: dict = {
    "prompt_tokens": 0, "completion_tokens": 0,
    "total_tokens": 0, "n_llm_calls": 0, "n_tool_calls": 0,
}


def _reset_telem_acc() -> None:
    for k in _TELEM_ACC:
        _TELEM_ACC[k] = 0


VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://lai:8001/v1")
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3.5-9B")

N_AGENTS = int(os.environ.get("DECENTRALIZED_N_AGENTS", "4"))
N_ROUNDS = int(os.environ.get("DECENTRALIZED_N_ROUNDS", "2"))

_APPS_TEST_TIMEOUT_S = 4
_EXEC_TIMEOUT_S = 10

_REPO_ROOT = Path(__file__).resolve().parents[4]
_PROMPTS_DIR = _REPO_ROOT / "configs" / "prompts" / "decentralized" / "apps"


def _load_prompt(role: str) -> str:
    return append_output_contract_from_path((_PROMPTS_DIR / f"{role}.txt").read_text().strip(), __file__, role)


SYSTEM_PROMPT = _load_prompt("debater")


# Tool
def python_exec(code: str, stdin: str = "", timeout_s: int = _EXEC_TIMEOUT_S) -> str:
    try:
        result = subprocess.run(
            ["python", "-c", code],
            input=stdin, capture_output=True, text=True, timeout=timeout_s,
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


_PYTHON_EXEC_SCHEMA = {
    "type": "function",
    "function": {
        "name": "python_exec",
        "description": "Execute Python code in a sandboxed subprocess.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string"},
                "stdin": {"type": "string"},
                "timeout_s": {"type": "integer"},
            },
            "required": ["code"],
        },
    },
}


def _dispatch_tool(name: str, arguments: dict) -> str:
    if name == "python_exec":
        return python_exec(
            arguments.get("code", ""),
            arguments.get("stdin", ""),
            arguments.get("timeout_s", _EXEC_TIMEOUT_S),
        )
    return f"ERROR: unknown tool {name!r}"


# Client
def _build_client() -> OpenAI:
    return OpenAI(base_url=VLLM_BASE_URL, api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"))


def _completion_kwargs() -> dict:
    return {
        "model": MODEL_ID,
        "temperature": 0.2,
        "top_p": 0.9,
        "seed": 0,
        "max_tokens": 2048,
        "extra_body": {
            "repetition_penalty": 1.05,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    }


def _chat_with_tools(client: OpenAI, messages: list[dict], max_tool_loops: int = 6) -> dict:
    kwargs = _completion_kwargs()
    kwargs["tools"] = [_PYTHON_EXEC_SCHEMA]
    kwargs["tool_choice"] = "auto"
    dump: dict = {}
    for _ in range(max_tool_loops):
        resp = client.chat.completions.create(messages=messages, **kwargs)
        openai_sdk_accumulate(_TELEM_ACC, resp)
        msg = resp.choices[0].message
        dump = msg.model_dump() if hasattr(msg, "model_dump") else dict(msg)
        tool_calls = dump.get("tool_calls") or []
        messages.append(dump)
        if not tool_calls:
            return dump
        for tc in tool_calls:
            fn = tc.get("function", {}) if isinstance(tc, dict) else {}
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            result = _dispatch_tool(fn.get("name", ""), args)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.get("id"),
                "content": result,
            })
    return dump


# Prompt scaffolding (APPS format)
_FORMAT_STDIN_DIRECTIVE = "Use Standard Input format."
_FORMAT_CALL_BASED_DIRECTIVE = "Use Call-Based format."


def format_prompt(problem: str, starter_code: str | None = None) -> str:
    parts = [f"QUESTION:\n{problem}"]
    if starter_code:
        parts.append(f"```python\n{starter_code.rstrip()}\n```")
        parts.append(_FORMAT_CALL_BASED_DIRECTIVE)
    else:
        parts.append(_FORMAT_STDIN_DIRECTIVE)
    parts.append("Enclose your final solution in a ```python``` code block.")
    parts.append("ANSWER:")
    return "\n\n".join(parts)


# Debate loop
def _peer_injection(others_final: list[dict], prompt: str) -> dict:
    body = ["These are the final solutions from other peer agents in the previous round:"]
    for i, m in enumerate(others_final):
        body.append(f"\nPeer {i + 1}:\n```\n{m.get('content') or ''}\n```")
    body.append(
        "\nCompare their code and approach. Revise your submission ONLY if "
        "a peer handles an edge case you missed or runs in better "
        "complexity. Re-emit your final code in a SINGLE fenced ```python``` "
        "block at the end.\n\nOriginal problem:\n" + prompt
    )
    return {"role": "user", "content": "\n".join(body)}


def run_debate(problem: str, starter_code: str | None = None) -> list[list[dict]]:
    client = _build_client()
    user_prompt = format_prompt(problem, starter_code)
    contexts: list[list[dict]] = [
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        for _ in range(N_AGENTS)
    ]
    round_finals: list[list[dict]] = []
    for r in range(N_ROUNDS):
        this_round: list[dict] = []
        for i, ctx in enumerate(contexts):
            if r > 0:
                others = [round_finals[r - 1][j] for j in range(N_AGENTS) if j != i]
                ctx.append(_peer_injection(others, user_prompt))
            final_msg = _chat_with_tools(client, ctx)
            this_round.append(final_msg)
        round_finals.append(this_round)
    return contexts


# Output parsing
_CODE_BLOCK_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_code(text: str) -> str | None:
    """Return the last fenced block that parses as Python, or None."""
    return extract_python_code(text)
# Scoring (aligned to single/apps)
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
            ["python", "-c", code], input=stdin,
            capture_output=True, text=True, timeout=timeout_s,
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

_CALL_BASED_MEMORY_BYTES = int(os.environ.get("APPS_CALL_BASED_MEMORY_BYTES", str(4 * 1024 ** 3)))


def _run_call_based_test(code, fn_name, raw_args, raw_expected, timeout_s):
    args = _parse_maybe_literal(raw_args)
    if not isinstance(args, (list, tuple)):
        args = [args]
    expected = _parse_maybe_literal(raw_expected)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        outfile = f.name
    try:
        proc = subprocess.run(
            ["python", "-c", _CALL_BASED_WORKER, str(_CALL_BASED_MEMORY_BYTES),
             fn_name, json.dumps(args), outfile],
            input=code, capture_output=True, text=True, timeout=timeout_s,
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
        return {"ok": False,
                "error": f"worker produced no parseable result (exit={proc.returncode})",
                "stderr": (proc.stderr or "").strip(), "mode": "call_based"}
    finally:
        try:
            os.unlink(outfile)
        except FileNotFoundError:
            pass
    if not payload.get("ok"):
        return {"ok": False, "error": payload.get("error", "worker error"), "mode": "call_based"}
    actual = payload["result"]
    return {"ok": _call_based_compare(actual, expected),
            "expected": expected, "actual": actual, "mode": "call_based"}


def run_tests(code: str, input_output: dict, timeout_s: int = _APPS_TEST_TIMEOUT_S) -> dict:
    inputs = input_output.get("inputs", []) or []
    outputs = input_output.get("outputs", []) or []
    fn_name = input_output.get("fn_name")
    if len(inputs) != len(outputs):
        return {"pass": 0, "total": 0, "pass_rate": 0.0, "details": [],
                "error": "inputs/outputs length mismatch"}
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
    return {"pass": passed, "total": len(inputs),
            "pass_rate": passed / len(inputs), "details": details}


def exact_match_score(pass_rate: float) -> float:
    return 1.0 if pass_rate == 1.0 else 0.0


# Aggregation (best-of-N)
def best_of_n(
    per_peer_codes: list[str | None],
    input_output: dict,
    timeout_s: int = _APPS_TEST_TIMEOUT_S,
) -> tuple[int | None, list[dict]]:
    scored = []
    for i, code in enumerate(per_peer_codes):
        if not code:
            scored.append({"peer": i, "code": code, "pass_rate": 0.0,
                           "resolved": False, "report": None})
            continue
        report = run_tests(code, input_output, timeout_s=timeout_s)
        scored.append({
            "peer": i, "code": code, "pass_rate": report["pass_rate"],
            "resolved": report["pass_rate"] == 1.0, "report": report,
        })
    perfect = [s for s in scored if s["resolved"]]
    if perfect:
        winner = min(perfect, key=lambda s: s["peer"])
    elif any(s["code"] for s in scored):
        winner = max(scored, key=lambda s: (s["pass_rate"], -s["peer"]))
    else:
        return None, scored
    return winner["peer"], scored


# Orchestration
def solve(
    problem: str,
    starter_code: str | None = None,
    input_output: dict | None = None,
) -> dict:
    _reset_telem_acc()
    contexts = run_debate(problem, starter_code=starter_code)
    per_peer = []
    for i, ctx in enumerate(contexts):
        final = ctx[-1].get("content") or ""
        code = extract_code(final)
        per_peer.append({"peer": i, "code": code, "raw": final})

    telem_src = dict(_TELEM_ACC)
    if telem_src["n_llm_calls"] == 0:
        telem_src = openai_sdk_telemetry(contexts)
    telem = normalize(telem_src)
    if input_output:
        winner_idx, scored = best_of_n([p["code"] for p in per_peer], input_output)
        return {
            "code": per_peer[winner_idx]["code"] if winner_idx is not None else None,
            "winner": winner_idx,
            "per_peer": scored,
            "all_contexts": contexts,
            "telemetry": telem,
        }
    return {
        "code": per_peer[0]["code"],
        "winner": 0,
        "per_peer": per_peer,
        "all_contexts": contexts,
        "telemetry": telem,
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
    """Run `solve()` on every problem with best-of-N scoring across peers."""
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
                    input_output=inst["input_output"],
                )
                error = None
            except Exception as e:
                out = {"code": None, "winner": None, "per_peer": [], "all_contexts": []}
                error = f"{type(e).__name__}: {e}"
            latency_s = time.time() - t0

            code = out.get("code")
            pass_rate = 0.0
            if code:
                n_extracted += 1
                pp = out.get("per_peer") or []
                winner = out.get("winner")
                if winner is not None and 0 <= winner < len(pp):
                    pass_rate = pp[winner].get("pass_rate", 0.0)
                em = exact_match_score(pass_rate)
            else:
                em = 0.0
            em_sum += em
            by_diff.setdefault(inst.get("difficulty") or "unk", []).append(em)

            compact_per_peer = [
                {"peer": p.get("peer"), "has_code": bool(p.get("code")),
                 "pass_rate": p.get("pass_rate", 0.0),
                 "resolved": p.get("resolved", False)}
                for p in out.get("per_peer") or []
            ]
            rec = {
                "id": inst["id"],
                "problem": inst["problem"][:400],
                "starter_code": inst.get("starter_code") or "",
                "predicted_code": code,
                "winner": out.get("winner"),
                "pass_rate": pass_rate,
                "em": em,
                "difficulty": inst.get("difficulty"),
                "per_peer": compact_per_peer,
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
                    f"em={em:.0f}  winner={out.get('winner')}  "
                    f"pass_rate={pass_rate:.2f}  EM={running_em:.3f} "
                    f"lat={latency_s:.1f}s",
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
            f"\n=== decentralized/APPS batch complete "
            f"(N={N_AGENTS}, R={N_ROUNDS}) ===\n"
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


def _canned_demo() -> None:
    stdin_problem = (
        "Read a single integer n (1 <= n <= 1000) from standard input and "
        "print n squared on a single line."
    )
    stdin_io = {
        "inputs":  ["5\n",  "1\n", "10\n",  "23\n"],
        "outputs": ["25",   "1",   "100",   "529"],
    }
    print("\n========== STANDARD INPUT MODE ==========")
    out = solve(stdin_problem, input_output=stdin_io)
    print(f"\n=== Winner: peer {out['winner']} ===")
    if out["code"]:
        _print_scoring(run_tests(out["code"], stdin_io))

    functional_problem = (
        "Given a list of integers `nums` and an integer `target`, return the "
        "indices of the two numbers in `nums` that add up to `target`. Assume "
        "exactly one solution exists and the same element may not be used "
        "twice. Return the answer as a list [i, j] with i < j."
    )
    functional_starter = (
        "from typing import List\n\nclass Solution:\n    def twoSum(self, nums: List[int], target: int) -> List[int]:\n        "
    )
    functional_io = {
        "fn_name": "twoSum",
        "inputs":  [[[2, 7, 11, 15], 9], [[3, 2, 4], 6], [[3, 3], 6]],
        "outputs": [[0, 1],              [1, 2],         [0, 1]],
    }
    print("\n========== CALL-BASED MODE ==========")
    out = solve(functional_problem, starter_code=functional_starter, input_output=functional_io)
    print(f"\n=== Winner: peer {out['winner']} ===")
    if out["code"]:
        _print_scoring(run_tests(out["code"], functional_io))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Decentralized-topology APPS runner (OpenAI SDK debate)."
    )
    parser.add_argument(
        "--batch", action="store_true",
        help="Run the real APPS eval (else: two canned demos).",
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
    print(
        f"loading APPS from {_HF_DATASET} [{_HF_SPLIT}] "
        f"(N={N_AGENTS}, R={N_ROUNDS}) ..."
    )
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
