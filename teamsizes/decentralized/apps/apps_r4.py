"""Decentralized debate topology specialized for APPS, LangGraph."""

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

from teamsizes.output_contracts import append_output_contract_from_path
from topologies.code_extract import extract_python_code
from typing import Any, Optional

from typing_extensions import TypedDict

import numpy as np
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

# Match openai sibling's max_tool_loops=6. create_react_agent uses its own
# recursion_limit; give it enough headroom to cover the same tool-loop
# budget (roughly 3x max_tool_loops to account for alternating AI/Tool
# messages and the final AI turn).
_MAX_TOOL_LOOPS = 6
_RECURSION_LIMIT = _MAX_TOOL_LOOPS * 3

_APPS_TEST_TIMEOUT_S = 4
_EXEC_TIMEOUT_S = 10

_PROMPTS_DIR = _REPO_ROOT / "configs" / "prompts" / "decentralized" / "apps"


def _load_prompt(role: str) -> str:
    return append_output_contract_from_path((_PROMPTS_DIR / f"{role}.txt").read_text().strip(), __file__, role)


SYSTEM_PROMPT = _load_prompt("debater")


# Tool (LangChain @tool)
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


# LLM / Agent
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


def _build_agent():
    """One react agent, reused across all peers + rounds. Each peer keeps
    its own message history; the agent is stateless."""
    return create_react_agent(
        model=_build_llm(), tools=[python_exec], prompt=SYSTEM_PROMPT
    )


# Prompt scaffolding (APPS format; aligned to openai sibling)
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


# Peer injection (aligned template to openai sibling)
def _peer_injection(others_final: list[BaseMessage], prompt: str) -> HumanMessage:
    body = ["These are the final solutions from other peer agents in the previous round:"]
    for i, m in enumerate(others_final):
        content = getattr(m, "content", "") or ""
        if not isinstance(content, str):
            content = str(content)
        body.append(f"\nPeer {i + 1}:\n```\n{content}\n```")
    body.append(
        "\nCompare their code and approach. Revise your submission ONLY if "
        "a peer handles an edge case you missed or runs in better "
        "complexity. Re-emit your final code in a SINGLE fenced ```python``` "
        "block at the end.\n\nOriginal problem:\n" + prompt
    )
    return HumanMessage(content="\n".join(body))


# State + round node
class DebateState(TypedDict, total=False):
    contexts: list[list[BaseMessage]]       # per-peer message histories (sans system)
    round_finals: list[list[BaseMessage]]   # per-round final AIMessages (one per peer)
    round: int
    prompt: str


def _last_ai(msgs: list[BaseMessage]) -> BaseMessage | None:
    """Return the last AIMessage with non-empty content (no tool_calls)."""
    for m in reversed(msgs):
        if isinstance(m, AIMessage) and (m.content or "") and not getattr(m, "tool_calls", None):
            return m
    # Fallback: any AIMessage.
    for m in reversed(msgs):
        if isinstance(m, AIMessage):
            return m
    return None


def _round_node(state: DebateState) -> dict:
    agent = _build_agent()
    r = int(state.get("round", 0))
    prompt = state["prompt"]
    contexts = [list(c) for c in state["contexts"]]
    prev_finals = state.get("round_finals") or []

    this_round_finals: list[BaseMessage] = []

    for i in range(len(contexts)):
        ctx = contexts[i]
        if r > 0 and prev_finals:
            others = [prev_finals[-1][j] for j in range(len(contexts)) if j != i]
            ctx = ctx + [_peer_injection(others, prompt)]

        result = agent.invoke(
            {"messages": ctx},
            config={"recursion_limit": _RECURSION_LIMIT},
        )
        # result["messages"] includes the input ctx + all new AI/Tool messages.
        contexts[i] = result["messages"]

        final = _last_ai(contexts[i]) or AIMessage(content="")
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


# Output parsing (aligned to openai sibling)
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
def _init_contexts(n: int, user_prompt: str) -> list[list[BaseMessage]]:
    """Each peer's initial history = [HumanMessage(user_prompt)]. System
    prompt is injected by create_react_agent via its `prompt=` arg, not
    embedded here.
    """
    return [[HumanMessage(content=user_prompt)] for _ in range(n)]


def solve(
    problem: str,
    starter_code: str | None = None,
    input_output: dict | None = None,
) -> dict:
    compiled = _build_graph()
    user_prompt = format_prompt(problem, starter_code)
    init_state: DebateState = {
        "contexts": _init_contexts(N_AGENTS, user_prompt),
        "round_finals": [],
        "round": 0,
        "prompt": user_prompt,
    }
    result = compiled.invoke(init_state)
    contexts = result.get("contexts") or []

    per_peer = []
    for i, ctx in enumerate(contexts):
        final_msg = _last_ai(ctx)
        final = getattr(final_msg, "content", "") or "" if final_msg else ""
        code = extract_code(final if isinstance(final, str) else "")
        if code is None:
            # Fallback: scan earlier messages for a fenced code block.
            for m in reversed(ctx):
                content = getattr(m, "content", "") or ""
                c = extract_code(content if isinstance(content, str) else "")
                if c:
                    code = c
                    break
        per_peer.append({"peer": i, "code": code, "raw": final})

    # Aggregate telemetry across all peers' full histories.
    flat_msgs: list[BaseMessage] = []
    for ctx in contexts:
        flat_msgs.extend(ctx)
    telem = normalize(langchain_telemetry(flat_msgs))

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
                mark = "OK" if em == 1.0 else ("?" if code is None else "NO")
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
        description="Decentralized-topology APPS runner (LangGraph debate)."
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
    _default_out = (
        _REPO_ROOT / "results" / "apps_decentralized_langgraph" / "predictions.jsonl"
    )
    out_path = Path(args.out) if args.out else _default_out
    print(f"  loaded {len(instances)} instance(s)")
    run_batch(instances, out_path=out_path)
    print(f"  predictions written to {out_path}")
