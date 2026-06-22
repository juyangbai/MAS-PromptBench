"""Independent topology specialized for LiveCodeBench."""

# Config
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import operator
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

# Number of parallel replicas. Seeds are 0 .. N_AGENTS-1.
N_AGENTS = int(os.environ.get("INDEPENDENT_N_AGENTS", "4"))

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PROMPT_PATH = (
    _REPO_ROOT / "configs" / "prompts" / "independent" / "lcb" / "coder.txt"
)

SYSTEM_PROMPT = append_output_contract_from_path(_PROMPT_PATH.read_text().strip(), __file__, _PROMPT_PATH.stem)


# Tools
_EXEC_TIMEOUT_S = int(os.environ.get("LCB_TOOL_TIMEOUT_S", "10"))
_TOOL_OUTPUT_CHAR_BUDGET = int(os.environ.get("LCB_TOOL_OUTPUT_CHAR_BUDGET", "4000"))
_RECURSION_LIMIT = int(os.environ.get("LCB_INDEPENDENT_RECURSION_LIMIT", "20"))
PER_ROW_TIMEOUT_S = int(os.environ.get("LCB_INDEPENDENT_ROW_TIMEOUT_S", "180"))


def _clip_tool_text(text: str) -> str:
    if len(text) <= _TOOL_OUTPUT_CHAR_BUDGET:
        return text
    return text[:_TOOL_OUTPUT_CHAR_BUDGET] + "\n...<truncated tool output>..."


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
            f"stdout:\n{_clip_tool_text(result.stdout)}\n"
            f"stderr:\n{_clip_tool_text(result.stderr)}\n"
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
    """Build the user-facing prompt for one LCB problem."""
    if starter_code:
        suffix = _FORMAT_FUNCTIONAL.format(starter_code=starter_code.rstrip())
    else:
        suffix = _FORMAT_STDIN
    return f"{problem}\n\n{suffix}"


def _build_one_agent(seed: int):
    """Build one replica's react agent, seeded differently from its siblings."""
    llm = ChatOpenAI(
        model=MODEL_ID,
        base_url=VLLM_BASE_URL,
        api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        max_tokens=4096,
        # Greedy (temp=0) would collapse all N replicas to the same code,
        # defeating the ensemble.
        temperature=0.2,
        top_p=0.9,
        seed=seed,
        extra_body={
            "repetition_penalty": 1.05,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    return create_react_agent(model=llm, tools=[python_exec], prompt=SYSTEM_PROMPT)


# Output Parsing
_CODE_BLOCK_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def strip_thinking(text: str) -> str:
    """Cut everything up through the last </think> tag (Qwen3 convention)."""
    index = text.lower().rfind("</think>")
    if index >= 0:
        text = text[index + len("</think>"):]
    return text.strip()


def extract_code(text: str) -> str | None:
    """Return the LAST fenced code block in `text`, or None."""
    return extract_python_code(text)


# Scoring
# Aligned to topologies/single/lcb/langgraph_lcb.py so ensemble
# pass@1 numbers remain directly comparable with single-topology numbers.
# Two evaluation paths mirror LCB's official testing_util.py:
#   - stdin tests: child process, feed stdin, compare stdout (line-by-line
#     with decimal tolerance).
#   - functional tests: fresh subprocess with LCB's reliability_guard and
#     resource limits applied, then exec + call fn, compare return value.


def _compare_stdout(actual: str, expected: str) -> bool:
    """Line-by-line comparison. Exact match first; fall back to per-token
    decimal comparison on numeric-looking lines."""
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
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


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

_FUNCTIONAL_MEMORY_BYTES = int(os.environ.get("LCB_FUNCTIONAL_MEMORY_BYTES", str(4 * 1024 ** 3)))


def _run_functional_test(code: str, tc: dict, timeout_s: int) -> dict:
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
    """Run `code` against LCB-style tests; returns a summary with pass rate."""
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


# Aggregation
def best_of_n(
    answers: list[dict],
    tests: list[dict],
    timeout_s: int = 5,
) -> dict | None:
    """Score each candidate and return the best one.

    Selection order:
      1. First candidate (lowest agent_id) with pass_rate == 1.0.
      2. Else candidate with highest pass_rate (first on tie).
      3. None iff no candidate has extractable code.
    """
    valid = [a for a in answers if a.get("code")]
    if not valid:
        return None

    scored = []
    for a in valid:
        r = run_tests(a["code"], tests, timeout_s=timeout_s)
        scored.append({**a, "pass_rate": r["pass_rate"], "test_detail": r})

    # Stable sort by agent_id to preserve first-occurrence tie-break.
    scored.sort(key=lambda s: s["agent_id"])
    perfect = [s for s in scored if s["pass_rate"] == 1.0]
    if perfect:
        return perfect[0]
    return max(scored, key=lambda s: (s["pass_rate"], -s["agent_id"]))


# Graph
class State(TypedDict):
    problem: str
    starter_code: str | None
    prompt: str
    answers: Annotated[list[dict], operator.add]


class AgentInput(TypedDict):
    agent_id: int
    seed: int
    prompt: str


async def _run_replica(inp: AgentInput) -> dict:
    """Run one replica's react agent and return its extracted code."""
    agent = _build_one_agent(seed=inp["seed"])
    result = await agent.ainvoke(
        {"messages": [("user", inp["prompt"])]},
        config={"recursion_limit": _RECURSION_LIMIT},
    )
    for msg in result["messages"]:
        if msg.type == "ai" and isinstance(msg.content, str):
            msg.content = strip_thinking(msg.content)
    final = result["messages"][-1].content
    return {
        "answers": [
            {
                "agent_id": inp["agent_id"],
                "seed": inp["seed"],
                "code": extract_code(final),
                "raw": final,
                "messages": result["messages"],
            }
        ]
    }


def _fan_out(state: State) -> list[Send]:
    return [
        Send(
            f"agent_{i}",
            {"agent_id": i, "seed": i, "prompt": state["prompt"]},
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
def solve(
    problem: str,
    starter_code: str | None = None,
    tests: list[dict] | None = None,
    timeout_s: int = 5,
) -> dict:
    """Run the ensemble on one coding problem.

    If `tests` is provided, the N candidates are scored and best-of-N is
    applied. If omitted, the raw per-agent candidates are returned without
    aggregation (useful for collecting predictions + scoring later).

    Returns:
        {
            "code":      best-of-N code (str) or None,
            "pass_rate": float in [0, 1] (if scored), else None,
            "winner":    agent_id of the selected candidate (if scored),
            "per_agent": list of {agent_id, seed, code, raw, messages,
                                  pass_rate?, test_detail?},
        }
    """
    compiled = build_graph().compile()
    prompt = format_prompt(problem, starter_code)
    async def _run():
        return await asyncio.wait_for(
            compiled.ainvoke({
                "problem": problem,
                "starter_code": starter_code,
                "prompt": prompt,
                "answers": [],
            }),
            timeout=PER_ROW_TIMEOUT_S,
        )

    result = asyncio.run(_run())
    per_agent = sorted(result["answers"], key=lambda a: a["agent_id"])

    if tests is None:
        return {
            "code": None,
            "pass_rate": None,
            "winner": None,
            "per_agent": per_agent,
        }

    winner = best_of_n(per_agent, tests, timeout_s=timeout_s)
    # Annotate per_agent with pass_rate for reporting symmetry with the
    # winner (each row then carries its own test_detail).
    ids_scored = {}
    if winner is not None:
        # Re-run tests on every replica (cheap) so per_agent can show
        # pass_rate for all of them, not just the winner. We already ran
        # the scorer inside best_of_n; duplicating here is intentional to
        # keep that function's signature simple.
        for a in per_agent:
            if a.get("code") is None:
                a["pass_rate"] = 0.0
                a["test_detail"] = None
            else:
                r = run_tests(a["code"], tests, timeout_s=timeout_s)
                a["pass_rate"] = r["pass_rate"]
                a["test_detail"] = r
            ids_scored[a["agent_id"]] = a["pass_rate"]

    return {
        "code": (winner or {}).get("code"),
        "pass_rate": (winner or {}).get("pass_rate"),
        "winner": (winner or {}).get("agent_id"),
        "per_agent": per_agent,
    }


# Dataset loader
_HF_DATASET = "livecodebench/code_generation_lite"
_HF_SPLIT = "test"


def _decode_private_tests(blob: str) -> list[dict]:
    """LCB stores private test cases as base64(zlib(pickle(json_str)))."""
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
) -> list[dict]:
    """Load LCB rows — same schema/IDs as single/lcb for cross-topology parity."""
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
    _propagate_errors: bool = False,
) -> dict:
    """Run `solve()` on every problem with best-of-N scoring, write
    per-instance predictions to JSONL."""
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
                    tests=inst["tests"],
                    timeout_s=per_test_timeout_s,
                )
                error = None
            except Exception as e:
                if _propagate_errors:
                    raise
                out = {"code": None, "pass_rate": None, "winner": None, "per_agent": []}
                error = f"{type(e).__name__}: {e}"
            latency_s = time.time() - t0

            code = out["code"]
            pass_rate = out.get("pass_rate") or 0.0
            if code:
                n_extracted += 1
                em = exact_match_score(pass_rate)
            else:
                em = 0.0
            em_sum += em
            by_diff.setdefault(inst.get("difficulty") or "unk", []).append(em)

            compact_per_agent = [
                {"agent_id": a["agent_id"], "seed": a.get("seed"),
                 "code": a.get("code"), "pass_rate": a.get("pass_rate", 0.0)}
                for a in out.get("per_agent") or []
            ]
            telem = normalize(langchain_ensemble_telemetry(out.get("per_agent") or []))
            rec = {
                "id": inst["id"],
                "problem": inst["problem"][:400],
                "starter_code": inst.get("starter_code") or "",
                "predicted_code": code,
                "winner": out.get("winner"),
                "pass_rate": pass_rate,
                "em": em,
                "difficulty": inst.get("difficulty"),
                "platform": inst.get("platform"),
                "per_agent": compact_per_agent,
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
            f"\n=== independent/LCB batch complete (N={N_AGENTS}) ===\n"
            f"  n={summary['n']}  n_extracted={summary['n_extracted']}\n"
            f"  pass@1 EM={summary['em']:.3f}  "
            f"(on extracted only: {summary['extracted_em']:.3f})\n"
        )
        for d, v in summary["by_difficulty"].items():
            print(f"    {d:>6s}: n={v['n']:3d}  EM={v['em']:.3f}")
        print(f"  total_s={summary['total_s']}\n")
    return summary


# Demo
def _print_scoring(per_agent: list[dict], winner: int | None, pass_rate: float | None) -> None:
    print(f"=== Ensemble ({N_AGENTS} replicas) ===")
    for a in per_agent:
        pr = a.get("pass_rate")
        pr_str = f"{pr:.2f}" if pr is not None else "—"
        print(
            f"  agent_{a['agent_id']} (seed {a['seed']}): "
            f"code={'yes' if a.get('code') else 'NO'}  "
            f"pass_rate={pr_str}"
        )
    if winner is not None:
        print(
            f"=== best-of-N winner: agent_{winner}  "
            f"pass_rate: {pass_rate:.2f}   pass@1: "
            f"{exact_match_score(pass_rate):.2f} ==="
        )


def _canned_demo() -> None:
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
    out = solve(stdin_problem, tests=stdin_tests)
    _print_scoring(out["per_agent"], out["winner"], out["pass_rate"])
    if out["code"]:
        print(f"\n=== Winner's code ===\n{out['code']}\n")

def run_one(instance: dict, out_dir: Path | None = None) -> dict:
    """Single-instance entrypoint for `concurrent_runner.py`.

    Calls `run_batch([instance], _propagate_errors=True)` so any transient
    exception (APIConnectionError, TimeoutError, BadRequestError "Unterminated
    string", etc.) bubbles up to the runner's retry-with-backoff wrapper
    instead of being swallowed into an `error` field on a "successful" row.
    """
    summary = run_batch([instance], out_path=None, verbose=False, _propagate_errors=True)
    return summary["per_instance"][0]



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Independent-topology LCB runner (LangGraph best-of-N)."
    )
    parser.add_argument(
        "--batch", action="store_true",
        help="Run the real LCB eval (else: one canned demo).",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument(
        "--difficulty", type=str, default=None, choices=("easy", "medium", "hard"),
    )
    args = parser.parse_args()

    if not args.batch:
        _canned_demo()
        sys.exit(0)

    print(
        f"loading LCB from {_HF_DATASET} [{_HF_SPLIT}] (N_AGENTS={N_AGENTS}) ..."
    )
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
