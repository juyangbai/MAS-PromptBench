"""Centralized topology specialized for APPS, LangGraph."""

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
from typing import Annotated, Optional

import numpy as np
from typing_extensions import TypedDict

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, create_react_agent

_REPO_ROOT = Path(__file__).resolve().parents[3]

# Shared telemetry.
_TOPO_ROOT = str(_REPO_ROOT)
if _TOPO_ROOT not in sys.path:
    sys.path.insert(0, _TOPO_ROOT)
from topologies.telemetry import langchain_telemetry, normalize  # noqa: E402


VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://n12:8000/v1")
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3.5-9B")

_APPS_TEST_TIMEOUT_S = 4  # APPS reference default

_PROMPTS_DIR = _REPO_ROOT / "configs" / "prompts" / "centralized" / "apps"

# Same cap as AutoGen sibling's MaxMessageTermination(26).
MAX_TURNS = 26


def _load_prompt(role: str) -> str:
    return append_output_contract_from_path((_PROMPTS_DIR / f"{role}.txt").read_text().strip(), __file__, role)


# Tools
_EXEC_TIMEOUT_S = 10


@tool
def python_exec(code: str, stdin: str = "", timeout_s: int = _EXEC_TIMEOUT_S) -> str:
    """Execute a Python code snippet in a fresh subprocess.

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


# Delegation tools (routing markers)
# The manager "calls" these to hand the floor to a specific worker. The
# body echoes the instructions, producing a ToolMessage the worker can
# read as context. The router after `manager_tools` inspects the name
# to route to the right worker node.
@tool("delegate_to_analyzer_worker")
def delegate_to_analyzer_worker(instructions: str) -> str:
    """Hand the next turn to the analyzer_worker. Use this when you need
    an algorithmic approach, complexity analysis, or edge-case enumeration
    for the current problem.

    Args:
        instructions: what you want the analyzer to produce this turn.
    """
    return instructions


@tool("delegate_to_coder_worker")
def delegate_to_coder_worker(instructions: str) -> str:
    """Hand the next turn to the coder_worker. Use this when you want a
    Python implementation written to match a specification.

    Args:
        instructions: what you want the coder to implement this turn.
    """
    return instructions


@tool("delegate_to_tester_worker")
def delegate_to_tester_worker(instructions: str) -> str:
    """Hand the next turn to the tester_worker. Use this when you want
    candidate code run against specific test inputs and the results
    reported back.

    Args:
        instructions: which code to run against which inputs this turn.
    """
    return instructions




@tool("delegate_to_design_worker")
def delegate_to_design_worker(instructions: str) -> str:
    """Hand the next turn to the design_worker.

    Args:
        instructions: what you want the design_worker to do this turn.
    """
    return instructions

@tool("delegate_to_edge_case_worker")
def delegate_to_edge_case_worker(instructions: str) -> str:
    """Hand the next turn to the edge_case_worker.

    Args:
        instructions: what you want the edge_case_worker to do this turn.
    """
    return instructions

@tool("delegate_to_debug_worker")
def delegate_to_debug_worker(instructions: str) -> str:
    """Hand the next turn to the debug_worker.

    Args:
        instructions: what you want the debug_worker to do this turn.
    """
    return instructions

@tool("delegate_to_review_worker")
def delegate_to_review_worker(instructions: str) -> str:
    """Hand the next turn to the review_worker.

    Args:
        instructions: what you want the review_worker to do this turn.
    """
    return instructions




@tool("delegate_to_complexity_analyzer_worker")
def delegate_to_complexity_analyzer_worker(instructions: str) -> str:
    """Hand the next turn to the complexity_analyzer_worker.

    Args:
        instructions: what you want the complexity_analyzer_worker to do this turn.
    """
    return instructions

@tool("delegate_to_optimizer_worker")
def delegate_to_optimizer_worker(instructions: str) -> str:
    """Hand the next turn to the optimizer_worker.

    Args:
        instructions: what you want the optimizer_worker to do this turn.
    """
    return instructions


DELEGATION_TOOLS = [
    delegate_to_analyzer_worker,
    delegate_to_coder_worker,
    delegate_to_tester_worker,
    delegate_to_design_worker,
    delegate_to_edge_case_worker,
    delegate_to_debug_worker,
    delegate_to_review_worker,
    delegate_to_complexity_analyzer_worker,
    delegate_to_optimizer_worker,
]
DELEGATION_NAMES = {t.name for t in DELEGATION_TOOLS}

MANAGER_TOOLS = [python_exec] + DELEGATION_TOOLS


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


# Team
_MANAGER_TERMINATE_NUDGE = (
    "\n\nWhen you emit the final fenced ```python``` code block "
    "containing your chosen solution, immediately follow it with the "
    "literal string TERMINATE on its own line so the group-chat knows "
    "to stop.\n\n"
    "Delegation: when you want a specific worker to act, call the "
    "matching delegate_to_<worker> tool with clear instructions "
    "(instead of merely addressing them in free-form text). The three "
    "workers are: analyzer_worker, coder_worker, tester_worker."
)


# Prompt scaffolding
# APPS prompt convention from the APPS paper / lm-evaluation-harness:
# "QUESTION: ... ANSWER:" scaffold with an explicit mode directive.
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


# Output parsing
_CODE_BLOCK_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_code(text: str) -> str | None:
    """Return the last fenced block that parses as Python, or None."""
    text = re.sub(r"\bTERMINATE\b", "", text)
    return extract_python_code(text)
# Scoring (aligned to single/apps / APPS testing_util.py)
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
    """Cascading comparison for one call-based test's return value."""
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
    """Cascading comparison for one stdin-mode test's captured stdout."""
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
    """Accept raw objects, JSON strings, or Python-literal strings."""
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
# reliability_guard (disables destructive os/shutil/subprocess calls) and
# optional setrlimit caps. Ported verbatim from APPS testing_util.py.
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
        return {"ok": False, "error": payload.get("error", "worker error"), "mode": "call_based"}
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
            "fn_name": <str>          # optional; presence -> call-based mode
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


# State + nodes
class CentralizedState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    turn_count: int


def _tag_source(msg: BaseMessage, source: str) -> None:
    """Attach an AutoGen-style `source` field via additional_kwargs."""
    try:
        kw = dict(getattr(msg, "additional_kwargs", None) or {})
        kw["source"] = source
        msg.additional_kwargs = kw
    except Exception:
        pass


def _manager_system() -> str:
    return _load_prompt("manager_r10") + _MANAGER_TERMINATE_NUDGE


def _manager_node(state: CentralizedState) -> dict:
    llm = _build_llm().bind_tools(MANAGER_TOOLS)
    sys_msg = SystemMessage(content=_manager_system())
    ai = llm.invoke([sys_msg] + state["messages"])
    # AutoGen messages carry a `.source` name; we mimic that on the
    # AIMessage via additional_kwargs for trace rendering parity.
    _tag_source(ai, "manager")
    return {"messages": [ai], "turn_count": int(state.get("turn_count", 0)) + 1}


_manager_tool_node = ToolNode(MANAGER_TOOLS)


def _route_from_manager(state: CentralizedState) -> str:
    msgs = state["messages"]
    if not msgs:
        return "manager"
    last = msgs[-1]
    if int(state.get("turn_count", 0)) >= MAX_TURNS:
        return END
    if isinstance(last, AIMessage):
        content = last.content or ""
        if isinstance(content, str) and "TERMINATE" in content:
            return END
        if getattr(last, "tool_calls", None):
            return "manager_tools"
    # No tool call, no TERMINATE — loop back and let the manager try again.
    return "manager"


def _route_from_manager_tools(state: CentralizedState) -> str:
    # Find the most recent AIMessage with tool_calls; its tool_calls tell
    # us whether any delegation was requested.
    for m in reversed(state["messages"]):
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            for tc in m.tool_calls:
                name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
                if name in DELEGATION_NAMES:
                    return name.removeprefix("delegate_to_")
            # Only real-tool calls were made; loop back to manager.
            return "manager"
    return "manager"


def _make_worker_node(name: str, tools: list, llm: ChatOpenAI):
    sys_prompt = _load_prompt(name)
    agent = create_react_agent(model=llm, tools=tools, prompt=sys_prompt)

    def node(state: CentralizedState) -> dict:
        # create_react_agent returns {"messages": [full history incl. input]},
        # so we splice out only the new messages it appended.
        prior = list(state["messages"])
        result = agent.invoke(
            {"messages": prior},
            config={"recursion_limit": 30},
        )
        full = result["messages"]
        new_msgs = full[len(prior):]
        # Tag worker outputs with source for trace rendering.
        for m in new_msgs:
            if isinstance(m, AIMessage):
                _tag_source(m, name)
        # Each new AIMessage counts as one turn; tool-executions don't.
        n_turns = sum(1 for m in new_msgs if isinstance(m, AIMessage))
        return {"messages": new_msgs, "turn_count": int(state.get("turn_count", 0)) + n_turns}

    return node


def _build_graph(llm: Optional[ChatOpenAI] = None):
    if llm is None:
        llm = _build_llm()

    graph = StateGraph(CentralizedState)
    graph.add_node("manager", _manager_node)
    graph.add_node("manager_tools", _manager_tool_node)

    worker_specs = [
        ("analyzer_worker", [python_exec]),
        ("coder_worker", [python_exec]),
        ("tester_worker", [python_exec]),
        ('design_worker', [python_exec]),
        ('edge_case_worker', [python_exec]),
        ('debug_worker', [python_exec]),
        ('review_worker', [python_exec]),
        ('complexity_analyzer_worker', [python_exec]),
        ('optimizer_worker', [python_exec]),
    ]
    for name, tools in worker_specs:
        graph.add_node(name, _make_worker_node(name, tools, llm))

    graph.add_edge(START, "manager")
    graph.add_conditional_edges(
        "manager",
        _route_from_manager,
        {
            "manager_tools": "manager_tools",
            "manager": "manager",
            END: END,
        },
    )
    graph.add_conditional_edges(
        "manager_tools",
        _route_from_manager_tools,
        {
            "analyzer_worker": "analyzer_worker",
            "coder_worker": "coder_worker",
            "tester_worker": "tester_worker",
            "manager": "manager",
            "design_worker": "design_worker",
            "edge_case_worker": "edge_case_worker",
            "debug_worker": "debug_worker",
            "review_worker": "review_worker",
            "complexity_analyzer_worker": "complexity_analyzer_worker",
            "optimizer_worker": "optimizer_worker",
        },
    )
    for name, _ in worker_specs:
        graph.add_edge(name, "manager")

    return graph.compile(), ["manager"] + [n for n, _ in worker_specs]


# Orchestration
def _communications_source(m: BaseMessage) -> str:
    kw = getattr(m, "additional_kwargs", None) or {}
    src = kw.get("source")
    if src:
        return src
    t = getattr(m, "type", None)
    return {"human": "user", "ai": "assistant", "tool": "tool"}.get(t, t or "?")


def _communications_to_record(m: BaseMessage) -> dict:
    content = getattr(m, "content", "") or ""
    if not isinstance(content, str):
        content = str(content)
    return {"source": _communications_source(m), "content": content}


def solve(problem: str, starter_code: str | None = None) -> dict:
    """Run the centralized team on one APPS problem.

    Returns:
        {
            "code":      inner content of the last fenced ```python``` block or None,
            "raw":       manager's last message content,
            "messages":  list of {source, content} from every turn,
            "telemetry": normalized 5-key token/call counts,
        }
    """
    compiled, _ = _build_graph()
    task = format_prompt(problem, starter_code)
    result = compiled.invoke(
        {"messages": [HumanMessage(content=task)], "turn_count": 0},
        config={"recursion_limit": MAX_TURNS * 4},
    )
    msgs = result.get("messages") or []
    rendered = [_communications_to_record(m) for m in msgs]

    manager_msgs = [r for r in rendered if r["source"] == "manager"]
    final = manager_msgs[-1]["content"] if manager_msgs else ""
    # Primary: extract from manager's final message.
    code = extract_code(final)
    # Fallback: scan the full history in reverse for the most recent
    # valid fenced code block (e.g. from coder_worker).
    if code is None:
        for r in reversed(rendered):
            c = extract_code(r["content"] or "")
            if c is not None:
                code = c
                break
    return {
        "code": code,
        "raw": final,
        "messages": rendered,
        "telemetry": normalize(langchain_telemetry(msgs)),
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
                out = {"code": None, "raw": "", "messages": []}
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
                "n_messages": len(out.get("messages") or []),
                "raw": (out.get("raw") or "")[:2000],
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
            f"\n=== centralized/APPS batch complete ===\n"
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
                f"    [FAIL {d['mode']}] test {d['test']}: "
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

    print("\n========== STANDARD INPUT MODE ==========")
    out = solve(stdin_problem)
    print(f"\n=== Extracted code ===\n{out['code']}\n")
    if out["code"]:
        _print_scoring(run_tests(out["code"], stdin_io))
    print(f"=== {len(out['messages'])} messages across the group chat ===")

    # --- Call-based-mode problem ---
    functional_problem = (
        "Given a list of integers `nums` and an integer `target`, return the "
        "indices of the two numbers in `nums` that add up to `target`. Assume "
        "exactly one solution exists and the same element may not be used "
        "twice. Return the answer as a list [i, j] with i < j."
    )
    functional_starter = (
        "from typing import List\n"
        "\n"
        "class Solution:\n"
        "    def twoSum(self, nums: List[int], target: int) -> List[int]:\n"
        "        "
    )
    # APPS-native parallel-list form (inputs = arg-lists, outputs = return values).
    functional_io = {
        "fn_name": "twoSum",
        "inputs":  [[[2, 7, 11, 15], 9], [[3, 2, 4], 6], [[3, 3], 6]],
        "outputs": [[0, 1],              [1, 2],         [0, 1]],
    }

    print("\n========== CALL-BASED MODE ==========")
    out = solve(functional_problem, starter_code=functional_starter)
    print(f"\n=== Extracted code ===\n{out['code']}\n")
    if out["code"]:
        _print_scoring(run_tests(out["code"], functional_io))
    print(f"=== {len(out['messages'])} messages across the group chat ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Centralized-topology APPS runner (LangGraph manager/worker)."
    )
    parser.add_argument(
        "--batch", action="store_true",
        help="Run the real APPS eval (else: two canned demos).",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    _default_out = str(
        Path(__file__).resolve().parents[3] / "results" / "apps_centralized_r10"
        / "predictions.jsonl"
    )
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
    out_path = Path(args.out) if args.out else Path(_default_out)
    run_batch(instances, out_path=out_path)
    print(f"  predictions written to {out_path}")
