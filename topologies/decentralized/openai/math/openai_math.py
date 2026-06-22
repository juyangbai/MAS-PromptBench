"""Decentralized debate topology specialized for competition MATH, OpenAI SDK."""

# Config
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import re
import signal
import sys
import time
from pathlib import Path

from topologies.output_contracts import append_output_contract_from_path

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


# Stall safeguards
# Per-row wall-clock cap using SIGALRM when running on the main thread. The
# concurrent runner may execute run_batch([row]) inside a ThreadPool worker,
# where Python disallows signal handlers, so the guard becomes a no-op there.
# On symbolic MATH problems (matrices, trig) the calculator tool returns ERROR
# for non-numeric expressions and peers loop retrying variants.
PER_ROW_TIMEOUT_S = 120
_MAX_TOOL_LOOPS = 4  # was 6


class _RowTimeout(Exception):
    """Raised when a single row exceeds PER_ROW_TIMEOUT_S."""


def _row_timeout_handler(signum, frame):
    raise _RowTimeout(f"row exceeded {PER_ROW_TIMEOUT_S}s")


@contextlib.contextmanager
def _row_timeout_guard(seconds: int):
    """Install SIGALRM for `seconds`; uninstall on exit regardless of
    outcome so timeouts in one row don't bleed into the next."""
    import threading
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    old = signal.signal(signal.SIGALRM, _row_timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://lai:8001/v1")
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3.5-9B")

N_AGENTS = int(os.environ.get("DECENTRALIZED_N_AGENTS", "4"))
N_ROUNDS = int(os.environ.get("DECENTRALIZED_N_ROUNDS", "2"))

_REPO_ROOT = Path(__file__).resolve().parents[4]
_PROMPTS_DIR = _REPO_ROOT / "configs" / "prompts" / "decentralized_openai" / "math"


def _load_prompt(role: str) -> str:
    return append_output_contract_from_path((_PROMPTS_DIR / f"{role}.txt").read_text().strip(), __file__, role)


SYSTEM_PROMPT = _load_prompt("debater")


# Tool
def calculator(expression: str) -> str:
    allowed = {
        k: getattr(math, k)
        for k in (
            "sqrt", "log", "log10", "log2", "exp",
            "sin", "cos", "tan", "asin", "acos", "atan",
            "floor", "ceil", "pow", "pi", "e",
        )
    }
    allowed["__builtins__"] = {}
    try:
        return str(eval(expression, allowed))
    except Exception as e:
        return f"ERROR: {e}"


def _sympy_locals(variable: str):
    import sympy as sp

    x = sp.symbols(variable or "x")
    names = {
        variable or "x": x,
        "pi": sp.pi,
        "e": sp.E,
        "E": sp.E,
        "sqrt": sp.sqrt,
        "log": sp.log,
        "exp": sp.exp,
        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "sec": sp.sec,
        "asin": sp.asin,
        "acos": sp.acos,
        "atan": sp.atan,
        "abs": sp.Abs,
    }
    return x, names


def _parse_sympy_expr(expr: str, variable: str):
    import sympy as sp
    from sympy.parsing.sympy_parser import (
        convert_xor,
        implicit_multiplication_application,
        parse_expr,
        standard_transformations,
    )

    _, local_dict = _sympy_locals(variable)
    global_dict = {
        "__builtins__": {},
        "Integer": sp.Integer,
        "Float": sp.Float,
        "Rational": sp.Rational,
    }
    transformations = standard_transformations + (
        implicit_multiplication_application,
        convert_xor,
    )
    return parse_expr(
        str(expr or "").strip(),
        local_dict=local_dict,
        global_dict=global_dict,
        transformations=transformations,
    )


def _parse_domain(domain: str, variable: str) -> tuple[float, float]:
    import sympy as sp

    text = (domain or "").strip().replace(" ", "")
    lower = text.lower()

    def as_float(expr: str) -> float:
        return float(sp.N(_parse_sympy_expr(expr, variable)))

    m = re.match(r"^[\(\[](.+),(.+)[\)\]]$", text)
    if m:
        return as_float(m.group(1)), as_float(m.group(2))

    pieces = lower.split("<")
    if len(pieces) == 3 and pieces[1] == (variable or "x").lower():
        return as_float(pieces[0]), as_float(pieces[2])

    if "pi/2" in lower and ("positive" in lower or "0" in lower):
        return 0.0, float(sp.N(sp.pi / 2))
    if "positive" in lower:
        return 0.0, float(sp.N(2 * sp.pi))
    return float(sp.N(-2 * sp.pi)), float(sp.N(2 * sp.pi))


def solve_equation(
    lhs: str,
    rhs: str,
    variable: str = "x",
    domain: str = "real",
) -> str:
    """Numerically solve one-variable equations and simplify candidates.

    This intentionally avoids heavyweight symbolic `solve()` because trig
    equations can hang. We scan a bounded real domain, refine sign changes by
    bisection, and return exact-looking candidates via `nsimplify`.
    """
    import sympy as sp

    var = variable or "x"
    x, _ = _sympy_locals(var)
    try:
        left = _parse_sympy_expr(lhs, var)
        right = _parse_sympy_expr(rhs, var)
        lo, hi = _parse_domain(domain, var)
    except Exception as e:
        return f"ERROR: could not parse equation/domain: {e}"

    if not math.isfinite(lo) or not math.isfinite(hi) or lo >= hi:
        return f"ERROR: invalid finite domain ({lo}, {hi})"

    f_expr = left - right
    try:
        f = sp.lambdify(x, f_expr, "math")
    except Exception as e:
        return f"ERROR: could not build numeric function: {e}"

    def val(t: float) -> float | None:
        try:
            y = float(f(t))
        except Exception:
            return None
        return y if math.isfinite(y) else None

    eps = max(1e-8, (hi - lo) * 1e-8)
    grid_n = 1600
    pts = [lo + eps + (hi - lo - 2 * eps) * i / grid_n for i in range(grid_n + 1)]
    roots: list[float] = []
    prev_t: float | None = None
    prev_y: float | None = None

    def add_root(root: float) -> None:
        if root <= lo or root >= hi:
            return
        y = val(root)
        if y is None or abs(y) > 1e-5:
            return
        if all(abs(root - r) > 1e-6 for r in roots):
            roots.append(root)

    for t in pts:
        y = val(t)
        if y is None or abs(y) > 1e8:
            prev_t = None
            prev_y = None
            continue
        if abs(y) < 1e-7:
            add_root(t)
        if prev_t is not None and prev_y is not None and y * prev_y < 0:
            a, b = prev_t, t
            fa, fb = prev_y, y
            for _ in range(80):
                mid = (a + b) / 2
                fm = val(mid)
                if fm is None or abs(fm) > 1e8:
                    break
                if abs(fm) < 1e-12:
                    a = b = mid
                    break
                if fa * fm <= 0:
                    b, fb = mid, fm
                else:
                    a, fa = mid, fm
            add_root((a + b) / 2)
        prev_t = t
        prev_y = y

    roots.sort()
    if not roots:
        return (
            "No roots found in the requested finite domain. "
            f"Equation parsed as ({sp.sstr(left)}) = ({sp.sstr(right)})."
        )

    entries = []
    for root in roots[:10]:
        exact = sp.nsimplify(root, [sp.pi])
        entries.append(f"{sp.sstr(exact)} ~= {root:.12g}")
    suffix = "" if len(roots) <= 10 else f"; {len(roots) - 10} more roots omitted"
    return (
        f"Roots for ({sp.sstr(left)}) = ({sp.sstr(right)}) in ({lo:.12g}, {hi:.12g}): "
        + "; ".join(entries)
        + suffix
    )


_CALCULATOR_SCHEMA = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": (
            "Evaluate a concrete numeric Python expression. Use only for "
            "arithmetic or numeric checks, not equations or symbolic variables."
        ),
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    },
}


_SOLVE_EQUATION_SCHEMA = {
    "type": "function",
    "function": {
        "name": "solve_equation",
        "description": (
            "Solve a one-variable algebraic or trigonometric equation over a "
            "finite real domain. Use plain SymPy/Python syntax, not LaTeX. "
            "Examples: lhs='tan(2*x) + tan(3*x)', rhs='sec(3*x)', "
            "variable='x', domain='(0, pi/2)'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "lhs": {"type": "string"},
                "rhs": {"type": "string"},
                "variable": {"type": "string"},
                "domain": {"type": "string"},
            },
            "required": ["lhs", "rhs", "variable", "domain"],
        },
    },
}


def _dispatch_tool(name: str, arguments: dict) -> str:
    if name == "calculator":
        return calculator(arguments.get("expression", ""))
    if name == "solve_equation":
        return solve_equation(
            arguments.get("lhs", ""),
            arguments.get("rhs", ""),
            arguments.get("variable", "x"),
            arguments.get("domain", "real"),
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


def _chat_with_tools(client: OpenAI, messages: list[dict], max_tool_loops: int = _MAX_TOOL_LOOPS) -> dict:
    kwargs = _completion_kwargs()
    kwargs["tools"] = [_CALCULATOR_SCHEMA, _SOLVE_EQUATION_SCHEMA]
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
            _TELEM_ACC["n_tool_calls"] += 1
            messages.append({
                "role": "tool",
                "tool_call_id": tc.get("id"),
                "content": result,
            })
    return dump


# Debate loop
def _peer_injection(others_final: list[dict], problem: str) -> dict:
    body = ["These are the final solutions from other peer agents in the previous round:"]
    for i, m in enumerate(others_final):
        body.append(f"\nPeer {i + 1}:\n```\n{m.get('content') or ''}\n```")
    body.append(
        "\nCompare their derivations and final boxed answers against your own. "
        "Revise your answer ONLY if a peer catches an error in your work or "
        "presents concretely stronger reasoning. Re-emit your final answer "
        "inside \\boxed{...} at the end.\n\nOriginal problem:\n" + problem
    )
    return {"role": "user", "content": "\n".join(body)}


def run_debate(problem: str) -> list[list[dict]]:
    client = _build_client()
    contexts: list[list[dict]] = [
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": problem},
        ]
        for _ in range(N_AGENTS)
    ]
    round_finals: list[list[dict]] = []
    for r in range(N_ROUNDS):
        this_round: list[dict] = []
        for i, ctx in enumerate(contexts):
            if r > 0:
                others = [round_finals[r - 1][j] for j in range(N_AGENTS) if j != i]
                ctx.append(_peer_injection(others, problem))
            final_msg = _chat_with_tools(client, ctx)
            this_round.append(final_msg)
        round_finals.append(this_round)
    return contexts


# Output parsing
def extract_boxed(text: str) -> str | None:
    """Return the inner content of the LAST \boxed{...} in the text.
    Handles nested braces via brace counting. Aligned to
    single/math's extractor."""
    marker = r"\boxed{"
    idx = text.rfind(marker)
    if idx < 0:
        return None
    start = idx + len(marker) - 1
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1:i]
    return None


def extract_answer(text: str) -> str | None:
    return extract_boxed(text)


# Scoring (aligned port of Hendrycks MATH is_equiv)
def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a, b = substr[0], substr[1]
                if b != "{":
                    if len(substr) > 2:
                        new_str += "{" + a + "}{" + b + "}" + substr[2:]
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        new_str += "{" + a + "}" + b + substr[2:]
                    else:
                        new_str += "{" + a + "}" + b
    return new_str


def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        return "\\frac{" + str(a) + "}{" + str(b) + "}"
    except:
        return string


def _remove_right_units(string):
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    return string


def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def _strip_string(string):
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = _remove_right_units(string)
    string = string.replace("\\%", "")
    string = string.replace("\\%", "")
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]
    string = _fix_sqrt(string)
    string = string.replace(" ", "")
    string = _fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    string = _fix_a_slash_b(string)
    return string


def is_equiv(str1, str2, verbose: bool = False) -> bool:
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False
    try:
        ss1 = _strip_string(str1)
        ss2 = _strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except Exception:
        return str1 == str2


def exact_match_score(pred: str, gold: str) -> float:
    return float(is_equiv(pred, gold))


# Aggregation
def equiv_majority(answers: list[str]) -> str | None:
    """Majority over Hendrycks-equivalence buckets — aligned to
    independent/math's aggregator."""
    valid = [a for a in answers if a]
    if not valid:
        return None
    buckets: list[list[str]] = []
    for a in valid:
        for b in buckets:
            if is_equiv(a, b[0]):
                b.append(a)
                break
        else:
            buckets.append([a])
    best = max(buckets, key=len)
    return best[0]


# Orchestration
def solve(problem: str) -> dict:
    _reset_telem_acc()
    with _row_timeout_guard(PER_ROW_TIMEOUT_S):
        contexts = run_debate(problem)
    per_peer = []
    answers = []
    for i, ctx in enumerate(contexts):
        final = ctx[-1].get("content") or ""
        ans = extract_answer(final)
        per_peer.append({"peer": i, "answer": ans, "raw": final})
        if ans is not None:
            answers.append(ans)
    telem = dict(_TELEM_ACC)
    if telem["n_llm_calls"] == 0:
        telem = openai_sdk_telemetry(contexts)
    return {
        "answer": equiv_majority(answers),
        "per_peer": per_peer,
        "all_contexts": contexts,
        "telemetry": normalize(telem),
    }


# Dataset loader
# qwedsacf/competition_math filtered to Precalculus / Level 5 (312 rows).
# Gold is extracted from the LAST \\boxed{...} in the `solution` column.
# IDs are stable MD5 of problem text for cross-topology parity.
_HF_DATASET = "qwedsacf/competition_math"
_HF_SPLIT = "train"
_SUBJECT = "Precalculus"
_LEVEL = "Level 5"


def load_instances(
    limit: int | None = None,
    offset: int = 0,
    only: list[str] | None = None,
) -> list[dict]:
    """Load the Precalculus / Level-5 subset of qwedsacf/competition_math."""
    import hashlib

    from datasets import load_dataset

    ds = load_dataset(_HF_DATASET)[_HF_SPLIT]
    rows: list[dict] = []
    for row in ds:
        if row.get("type") != _SUBJECT:
            continue
        if row.get("level") != _LEVEL:
            continue
        problem = (row.get("problem") or "").strip()
        solution = (row.get("solution") or "").strip()
        if not problem or not solution:
            continue
        gold = extract_boxed(solution)
        if gold is None:
            continue
        rid = "math_" + hashlib.md5(problem.encode("utf-8")).hexdigest()[:10]
        if only is not None and rid not in set(only):
            continue
        rows.append({
            "id": rid,
            "problem": problem,
            "answer": gold,
            "subject": row.get("type"),
            "level": row.get("level"),
            "raw": {
                "problem": problem,
                "solution": solution,
                "type": row.get("type"),
                "level": row.get("level"),
            },
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
) -> dict:
    """Run `solve()` on every problem, compare debate boxed answer vs gold
    via Hendrycks `is_equiv`, return aggregate summary + optionally write
    per-instance predictions to JSONL.
    """
    per_instance: list[dict] = []
    n = len(instances)
    em_sum = 0.0
    n_extracted = 0
    start = time.time()

    out_f = None
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_f = open(out_path, "w")

    try:
        for i, inst in enumerate(instances):
            t0 = time.time()
            try:
                out = solve(inst["problem"])
                error = None
            except Exception as e:
                out = {"answer": None, "per_peer": [], "all_contexts": []}
                error = f"{type(e).__name__}: {e}"
            latency_s = time.time() - t0

            pred = out["answer"]
            gold = inst["answer"]
            if pred is not None:
                n_extracted += 1
                em = exact_match_score(pred, gold)
            else:
                em = 0.0
            em_sum += em

            compact_per_peer = [
                {"peer": p["peer"], "answer": p["answer"],
                 "raw": (p.get("raw") or "")[:2000]}
                for p in out.get("per_peer") or []
            ]
            rec_out = {
                "id": inst["id"],
                "problem": inst["problem"],
                "gold_answer": gold,
                "predicted_answer": pred,
                "em": em,
                "subject": inst.get("subject"),
                "level": inst.get("level"),
                "per_peer": compact_per_peer,
                "latency_s": round(latency_s, 2),
                **(out.get("telemetry") or {}),
                "error": error,
            }
            per_instance.append(rec_out)
            if out_f is not None:
                out_f.write(json.dumps(rec_out) + "\n")
                out_f.flush()

            if verbose:
                running_em = em_sum / (i + 1)
                mark = "✓" if em == 1.0 else ("?" if pred is None else "✗")
                pred_disp = (pred or "-")[:30]
                gold_disp = (gold or "-")[:30]
                print(
                    f"[{i + 1:>3}/{n}] {inst['id'][:30]:<30} {mark}  "
                    f"em={em:.0f}  pred={pred_disp!r} gold={gold_disp!r}  "
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
        "total_s": round(elapsed, 1),
        "per_instance": per_instance,
    }
    if verbose:
        print(
            f"\n=== decentralized/MATH-500 batch complete "
            f"(N={N_AGENTS}, R={N_ROUNDS}) ===\n"
            f"  n={summary['n']}  n_extracted={summary['n_extracted']}\n"
            f"  EM={summary['em']:.3f}  (on extracted only: {summary['extracted_em']:.3f})\n"
            f"  total_s={summary['total_s']}\n"
        )
    return summary


# Demo
def _canned_demo() -> None:
    problem = (
        "Compute the value of $\\frac{7!}{5!}$. "
        "Put your final answer inside \\boxed{}."
    )
    expected = "42"
    out = solve(problem)
    print(f"\n=== Debate: N={N_AGENTS} peers × R={N_ROUNDS} rounds ===")
    for p in out["per_peer"]:
        ans = p["answer"] if p["answer"] is not None else "(none)"
        print(f"  peer {p['peer']}: boxed={ans!r}")
    print(f"\n=== Majority-vote final answer: {out['answer']!r}  (expected: {expected!r}) ===")
    if out["answer"] is not None:
        em = exact_match_score(out["answer"], expected)
        print(f"=== EM (Hendrycks is_equiv): {em:.2f} ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Decentralized-topology MATH runner (OpenAI SDK debate)."
    )
    parser.add_argument(
        "--batch", action="store_true",
        help="Run the real MATH-500 eval (else: one canned demo).",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()

    if not args.batch:
        _canned_demo()
        sys.exit(0)

    print(
        f"loading MATH-500 from {_HF_DATASET} [{_HF_SPLIT}] "
        f"(N={N_AGENTS}, R={N_ROUNDS}) ..."
    )
    instances = load_instances(
        limit=args.limit, offset=args.offset, only=args.only,
    )
    if not instances:
        print("no instances loaded (check --limit/--offset/--only)", file=sys.stderr)
        sys.exit(1)
    print(f"  loaded {len(instances)} instance(s)")
    out_path = Path(args.out) if args.out else None
    run_batch(instances, out_path=out_path)
    if out_path:
        print(f"  predictions written to {out_path}")
