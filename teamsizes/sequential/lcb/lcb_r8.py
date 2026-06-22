"""Sequential topology specialized for LiveCodeBench, implemented in LangGraph."""

# Config
from __future__ import annotations

import argparse
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

from teamsizes.output_contracts import append_output_contract_from_path
from topologies.code_extract import extract_python_code
from typing import Annotated

from typing_extensions import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
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

_PROMPTS_DIR = _REPO_ROOT / "configs" / "prompts" / "sequential" / "lcb"


def _load_prompt(role: str) -> str:
    return append_output_contract_from_path((_PROMPTS_DIR / f"{role}.txt").read_text().strip(), __file__, role)


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


PYTHON_EXEC_TOOLS = [python_exec]


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
# User-message format templates — verbatim from LiveCodeBench's official
# lcb_runner/prompts/code_generation.py. Public test cases are intentionally
# NOT included (matching LCB behavior).

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
        # Escape starter_code braces so str.format() doesn't choke.
        starter_safe = starter_code.replace("{", "{{").replace("}", "}}")
        suffix = _FORMAT_FUNCTIONAL.replace("{starter_code}", starter_safe.rstrip())
    else:
        suffix = _FORMAT_STDIN
    return f"{problem}\n\n{suffix}"


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
    'requirements_parser': (
        'Parse the problem to extract MODE (stdin or call-based), INPUT_FORMAT, OUTPUT_FORMAT, CONSTRAINTS, OBVIOUS_EDGE_CASES.\n\nPROBLEM:\n{problem_prompt}'
    ),
    'algorithm_designer': (
        'Propose an algorithm in pseudocode with complexity analysis.\n\nPROBLEM:\n{problem_prompt}'
    ),
    'edge_case_thinker': (
        'After the coder drafts code, enumerate 5-10 specific edge cases.\n\nPROBLEM:\n{problem_prompt}'
    ),
    'code_reviewer': (
        "Review the debugger's output and emit the FINAL fenced ```python``` block.\n\nPROBLEM:\n{problem_prompt}"
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
            'requirements_parser',
            _load_prompt('requirements_parser'),
            [],
            _TASK_DESCRIPTIONS['requirements_parser'],
        ),
        (
            'analyzer',
            _load_prompt('analyzer'),
            [],
            _TASK_DESCRIPTIONS['analyzer'],
        ),
        (
            'algorithm_designer',
            _load_prompt('algorithm_designer'),
            [],
            _TASK_DESCRIPTIONS['algorithm_designer'],
        ),
        (
            'coder',
            _load_prompt('coder'),
            [],
            _TASK_DESCRIPTIONS['coder'],
        ),
        (
            'edge_case_thinker',
            _load_prompt('edge_case_thinker'),
            [],
            _TASK_DESCRIPTIONS['edge_case_thinker'],
        ),
        (
            'tester',
            _load_prompt('tester'),
            PYTHON_EXEC_TOOLS,
            _TASK_DESCRIPTIONS['tester'],
        ),
        (
            'debugger',
            _load_prompt('debugger'),
            PYTHON_EXEC_TOOLS,
            _TASK_DESCRIPTIONS['debugger'],
        ),
        (
            'code_reviewer',
            _load_prompt('code_reviewer'),
            [],
            _TASK_DESCRIPTIONS['code_reviewer'],
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
    """Return the LAST fenced code block in `text`, or None.

    Takes the last block so the debugger's final (fixed) program is
    returned rather than an earlier draft.
    """
    return extract_python_code(text)


# Scoring
# Aligned to topologies/single/lcb/langgraph_lcb.py and the CrewAI
# sibling so sequential pass@1 is directly comparable with single/independent
# numbers. Two evaluation paths mirror LCB's official testing_util.py:
#   - stdin tests: child process, feed stdin, compare stdout
#   - functional tests: fresh subprocess with reliability_guard + rlimits


def _compare_stdout(actual: str, expected: str) -> bool:
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

_FUNCTIONAL_MEMORY_BYTES = int(
    os.environ.get("LCB_FUNCTIONAL_MEMORY_BYTES", str(4 * 1024 ** 3))
)


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


# Orchestration
def solve(problem: str, starter_code: str | None = None) -> dict:
    """Run the 4-stage sequential graph on one LCB problem.

    Returns:
        {
            "code":      final Python program (str) or None,
            "raw":       debugger's full output text,
            "by_stage":  {analyzer, coder, tester, debugger} -> each stage's output,
            "telemetry": normalized 5-key token/call counts,
        }
    """
    llm = _build_llm()
    compiled, roles = _build_graph(llm)
    problem_prompt = format_prompt(problem, starter_code)
    result = compiled.invoke(
        {"inputs": {"problem_prompt": problem_prompt}, "by_stage": {}, "messages": []}
    )

    stages = result.get("by_stage") or {}
    final = stages.get(roles[-1], "")

    # Try the debugger's output first; fall back to the coder's in case
    # the debugger emitted prose without re-fencing the code.
    code = extract_code(stages.get("debugger", "") or final)
    if code is None:
        code = extract_code(stages.get("coder", ""))

    return {
        "code": code,
        "raw": final,
        "by_stage": stages,
        "telemetry": normalize(langchain_telemetry(result.get("messages") or [])),
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
    platform: str | None = None,
) -> list[dict]:
    """Load LCB rows — same schema/IDs as single/lcb for parity."""
    from datasets import load_dataset

    ds = load_dataset(_HF_DATASET, split=_HF_SPLIT, trust_remote_code=True)
    rows: list[dict] = []
    for row in ds:
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
                )
                error = None
            except Exception as e:
                out = {"code": None, "raw": "", "by_stage": {}}
                error = f"{type(e).__name__}: {e}"
            latency_s = time.time() - t0

            code = out["code"]
            if code:
                n_extracted += 1
                scored = run_tests(code, inst["tests"], timeout_s=per_test_timeout_s)
            else:
                scored = {"pass": 0, "total": len(inst["tests"]), "pass_rate": 0.0, "details": []}
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
                "platform": inst.get("platform"),
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
                mark = "+" if em == 1.0 else ("?" if code is None else "-")
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
            f"\n=== sequential/LCB batch complete ===\n"
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
                f"    [FAIL {d.get('mode', '?')}] test {d['test']}: "
                f"expected={d.get('expected')!r}  "
                f"actual={d.get('actual')!r}  "
                f"err={d.get('error') or d.get('stderr')!r}"
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

    print("========== STDIN MODE ==========")
    out = solve(stdin_problem)
    print(f"\n=== Analyzer (excerpt) ===\n{out['by_stage'].get('analyzer', '')[:400]}...")
    print(f"\n=== Coder (excerpt) ===\n{out['by_stage'].get('coder', '')[:400]}...")
    print(f"\n=== Tester (excerpt) ===\n{out['by_stage'].get('tester', '')[:400]}...")
    print(f"\n=== Debugger (excerpt) ===\n{out['by_stage'].get('debugger', '')[:400]}...")
    if out["code"]:
        print(f"\n=== Final code ===\n{out['code']}\n")
        _print_scoring(run_tests(out["code"], stdin_tests))
    else:
        print("\n=== No code extracted ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sequential-topology LCB runner (LangGraph 4-stage)."
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
    parser.add_argument(
        "--platform", type=str, default=None,
        choices=("codeforces", "leetcode", "atcoder"),
    )
    args = parser.parse_args()

    if not args.batch:
        _canned_demo()
        sys.exit(0)

    print(f"loading LCB from {_HF_DATASET} [{_HF_SPLIT}] ...")
    instances = load_instances(
        limit=args.limit, offset=args.offset,
        only=args.only, difficulty=args.difficulty,
        platform=args.platform,
    )
    if not instances:
        print("no instances loaded (check --limit/--offset/--only/--difficulty)",
              file=sys.stderr)
        sys.exit(1)
    print(f"  loaded {len(instances)} instance(s)")
    _default_out = (
        _REPO_ROOT / "results" / "lcb_sequential_r8" / "predictions.jsonl"
    )
    out_path = Path(args.out) if args.out else _default_out
    run_batch(instances, out_path=out_path)
    print(f"  predictions written to {out_path}")
