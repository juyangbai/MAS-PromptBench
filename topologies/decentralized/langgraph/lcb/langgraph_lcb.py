"""Decentralized debate topology specialized for LiveCodeBench, LangGraph."""

# Config
from __future__ import annotations

import argparse
import base64
import json
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
from typing import Optional

from typing_extensions import TypedDict

import langchain_core.tools  # noqa: F401  # ensure module import before @tool use
from langchain_core.messages import (  # noqa: E402
    AIMessage,
    BaseMessage,
    HumanMessage,
)
from langchain_openai import ChatOpenAI  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402
from langgraph.prebuilt import create_react_agent  # noqa: E402

# Shared telemetry.
_REPO_ROOT = Path(__file__).resolve().parents[4]
_TOPO_ROOT = str(_REPO_ROOT)
if _TOPO_ROOT not in sys.path:
    sys.path.insert(0, _TOPO_ROOT)
from topologies.telemetry import langchain_telemetry, normalize  # noqa: E402
from communications.communication_formats import format_handoff  # noqa: E402


VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://n12:8000/v1")
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3.5-9B")
COMMUNICATION_FORMAT = "freeform"

N_AGENTS = int(os.environ.get("DECENTRALIZED_N_AGENTS", "4"))
N_ROUNDS = int(os.environ.get("DECENTRALIZED_N_ROUNDS", "2"))

_EXEC_TIMEOUT_S = 10

# Match openai sibling's max_tool_loops=6. create_react_agent uses its own
# recursion_limit; give it enough headroom to cover the same tool-loop budget
# (roughly 3x max_tool_loops to account for alternating AI/Tool messages).
_MAX_TOOL_LOOPS = 6
_RECURSION_LIMIT = _MAX_TOOL_LOOPS * 3

_PROMPTS_DIR = _REPO_ROOT / "configs" / "prompts" / "decentralized" / "lcb"


def _load_prompt(role: str) -> str:
    return append_output_contract_from_path((_PROMPTS_DIR / f"{role}.txt").read_text().strip(), __file__, role)


SYSTEM_PROMPT = _load_prompt("debater")


# Tool (LangChain @tool)
@langchain_core.tools.tool
def python_exec(code: str, stdin: str = "", timeout_s: int = _EXEC_TIMEOUT_S) -> str:
    """Execute a Python code snippet in a sandboxed subprocess.

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


TOOLS = [python_exec]


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


def _build_agent():
    """One react agent, reused across all peers + rounds. Each peer keeps
    its own message history; the agent is stateless."""
    return create_react_agent(model=_build_llm(), tools=TOOLS, prompt=SYSTEM_PROMPT)


# Prompt scaffolding (LCB format blocks)
_FORMAT_STDIN = (
    "### Format: Read the inputs from stdin solve the problem and write "
    "the answer to stdout (do not directly test on the sample inputs). "
    "Enclose your code within delimiters as follows.\n"
    "```python\n# YOUR CODE HERE\n```"
)

_FORMAT_FUNCTIONAL = (
    "### Format: You will use the following starter code to write the "
    "solution to the problem and enclose your code within delimiters.\n"
    "```python\n{starter_code}\n```"
)


def format_prompt(problem: str, starter_code: str | None = None) -> str:
    if starter_code:
        suffix = _FORMAT_FUNCTIONAL.format(starter_code=starter_code.rstrip())
    else:
        suffix = _FORMAT_STDIN
    return f"{problem}\n\n{suffix}"


# Peer injection (aligned template to openai sibling)
def _peer_injection(others_final: list[BaseMessage], prompt: str) -> HumanMessage:
    body = ["These are the final solutions from other peer agents in the previous round:"]
    for i, m in enumerate(others_final):
        content = getattr(m, "content", "") or ""
        if not isinstance(content, str):
            content = str(content)
        if COMMUNICATION_FORMAT == "freeform":
            body.append(f"\nPeer {i + 1}:\n```\n{content}\n```")
        else:
            rendered = format_handoff(
                f"peer_{i + 1}",
                content,
                fmt=COMMUNICATION_FORMAT,
                dataset="lcb",
                topology="decentralized",
                next_action="Use this previous-round peer report when deciding whether to revise.",
                payload={"handoff": "peer_previous_round", "peer": i + 1},
            )
            body.append(f"\nPeer {i + 1}:\n{rendered}")
    body.append(
        "\nCompare their code and approach against your own. Revise your "
        "submission ONLY if a peer handles an edge case you missed or "
        "runs in better complexity. Re-emit your FINAL code in a single "
        "fenced ```python``` block at the end.\n\n"
        "Original problem:\n" + prompt
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
    # Fallback: any AIMessage
    for m in reversed(msgs):
        if isinstance(m, AIMessage):
            return m
    return None


def _json_safe_tool_args(args) -> dict:
    if not isinstance(args, dict):
        return {}
    try:
        safe = json.loads(json.dumps(args, default=str))
    except Exception:
        return {str(k): str(v) for k, v in args.items()}
    return safe if isinstance(safe, dict) else {}


def _message_content_text(msg: BaseMessage) -> str:
    content = getattr(msg, "content", "") or ""
    return content if isinstance(content, str) else str(content)


def _openai_safe_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Replay only normalized tool-call metadata to the next peer request."""
    safe: list[BaseMessage] = []
    for msg in messages:
        if not isinstance(msg, AIMessage):
            safe.append(msg)
            continue
        tool_calls = []
        for idx, tc in enumerate(getattr(msg, "tool_calls", None) or []):
            name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
            if not name:
                continue
            args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {})
            call_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
            tool_calls.append(
                {
                    "name": str(name),
                    "args": _json_safe_tool_args(args),
                    "id": str(call_id or f"call_{idx}"),
                }
            )
        safe.append(
            AIMessage(
                content=_message_content_text(msg),
                tool_calls=tool_calls,
            )
        )
    return safe


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

        ctx = _openai_safe_messages(ctx)
        result = agent.invoke(
            {"messages": ctx},
            config={"recursion_limit": _RECURSION_LIMIT},
        )
        # result["messages"] includes the input ctx + all new AI/Tool messages.
        contexts[i] = _openai_safe_messages(result["messages"])

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


# Output parsing
_CODE_BLOCK_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_code(text: str) -> str | None:
    return extract_python_code(text)


# Scoring (aligned to single/lcb)
def _compare_stdout(actual: str, expected: str) -> bool:
    if actual.strip() == expected.strip():
        return True
    a_lines = actual.splitlines()
    e_lines = expected.splitlines()
    if len(a_lines) != len(e_lines):
        return False
    for a, e in zip(a_lines, e_lines):
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
            ["python", "-c", code], input=tc.get("input", ""),
            capture_output=True, text=True, timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timeout"}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    expected = tc.get("output", "")
    actual = result.stdout
    ok = _compare_stdout(actual, expected) and result.returncode == 0
    return {"ok": ok, "expected": expected.strip(), "actual": actual.strip(),
            "stderr": result.stderr.strip(), "exit_code": result.returncode}


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
            ["python", "-c", _FUNCTIONAL_WORKER, str(_FUNCTIONAL_MEMORY_BYTES),
             fn_name, json.dumps(args), outfile],
            input=code, capture_output=True, text=True, timeout=timeout_s,
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
        return {"ok": False,
                "error": f"worker produced no parseable result (exit={proc.returncode})",
                "stderr": (proc.stderr or "").strip()}
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
    return {"pass": passed, "total": len(tests), "pass_rate": passed / len(tests),
            "details": details}


def exact_match_score(pass_rate: float) -> float:
    return 1.0 if pass_rate == 1.0 else 0.0


# Aggregation (best-of-N, same semantics as openai sibling)
def best_of_n(
    per_peer_codes: list[str | None],
    tests: list[dict],
    timeout_s: int = 5,
) -> tuple[int | None, list[dict]]:
    """Score each peer's round-R code; return (winner_idx, scored_list)."""
    scored = []
    for i, code in enumerate(per_peer_codes):
        if not code:
            scored.append({"peer": i, "code": code, "pass_rate": 0.0,
                           "resolved": False, "report": None})
            continue
        report = run_tests(code, tests, timeout_s=timeout_s)
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
    """Each peer's initial history = [HumanMessage(user_prompt)]. System prompt
    is injected by create_react_agent via its `prompt=` arg, not embedded here.
    """
    return [[HumanMessage(content=user_prompt)] for _ in range(n)]


def solve(
    problem: str,
    starter_code: str | None = None,
    tests: list[dict] | None = None,
) -> dict:
    """Run N-peer x R-round debate. If `tests` is provided, run best-of-N
    selection across round-R codes; otherwise return peer-0's code as
    the submission."""
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
        code = extract_code(final)
        if code is None:
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

    if tests:
        winner_idx, scored = best_of_n([p["code"] for p in per_peer], tests)
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
                    tests=inst["tests"],
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
                "platform": inst.get("platform"),
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
                mark = "PASS" if em == 1.0 else ("?" if code is None else "FAIL")
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
            f"\n=== decentralized/LCB batch complete "
            f"(N={N_AGENTS}, R={N_ROUNDS}) ===\n"
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
    print(f"\n=== Winner: peer {out['winner']} ===\n{out['code']}\n")
    if out["code"]:
        _print_scoring(run_tests(out["code"], stdin_tests))

    functional_problem = (
        "Given an array of integers `nums` and an integer `target`, return "
        "the indices of the two numbers such that they add up to `target`. "
        "Assume each input has exactly one solution; the same element may "
        "not be used twice. Return the answer as a list [i, j] with i < j."
    )
    functional_tests = [
        {"fn_name": "twoSum", "input": "[[2,7,11,15], 9]", "output": "[0,1]", "testtype": "functional"},
        {"fn_name": "twoSum", "input": "[[3,2,4], 6]",     "output": "[1,2]", "testtype": "functional"},
        {"fn_name": "twoSum", "input": "[[3,3], 6]",       "output": "[0,1]", "testtype": "functional"},
    ]
    functional_starter = (
        "from typing import List\n\nclass Solution:\n    def twoSum(self, nums: List[int], target: int) -> List[int]:\n        "
    )

    print("\n========== FUNCTIONAL MODE ==========")
    out = solve(functional_problem, starter_code=functional_starter, tests=functional_tests)
    print(f"\n=== Winner: peer {out['winner']} ===\n{out['code']}\n")
    if out["code"]:
        _print_scoring(run_tests(out["code"], functional_tests))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Decentralized-topology LCB runner (LangGraph debate)."
    )
    parser.add_argument(
        "--batch", action="store_true",
        help="Run the real LCB eval (else: two canned demos).",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    _default_out = str(
        Path(__file__).resolve().parents[4] / "results" / "lcb_decentralized_langgraph" / "predictions.jsonl"
    )
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

    print(
        f"loading LCB from {_HF_DATASET} [{_HF_SPLIT}] "
        f"(N={N_AGENTS}, R={N_ROUNDS}) ..."
    )
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
    out_path = Path(args.out) if args.out else Path(_default_out)
    run_batch(instances, out_path=out_path)
    if out_path:
        print(f"  predictions written to {out_path}")
