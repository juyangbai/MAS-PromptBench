"""Single-agent ReAct topology specialized for MATH (competition math).

Competition-style math problems. Models are expected to reason step by step
and emit their final answer inside \\boxed{...}. The scorer extracts the
last boxed answer and compares after light LaTeX normalization.

The system prompt is loaded from configs/prompts/single/math/solver.txt.
"""

# Config
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

from topologies.output_contracts import append_output_contract_from_path

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

# Shared telemetry.
_TOPO_ROOT = str(Path(__file__).resolve().parents[3])
if _TOPO_ROOT not in sys.path:
    sys.path.insert(0, _TOPO_ROOT)
from topologies.telemetry import langchain_telemetry, normalize  # noqa: E402


VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://lai:8001/v1")
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3.5-9B")


# System Prompt
_REPO_ROOT = Path(__file__).resolve().parents[3]
_PROMPT_PATH = _REPO_ROOT / "configs" / "prompts" / "single" / "math" / "solver.txt"

# The generated solver.txt encourages `\boxed{...}` but doesn't strictly
# require it on the final line — append a tight format nudge so the
# scorer's `extract_boxed` reliably finds the answer (Qwen3.5-9B
# sometimes emits "the answer is 42" without the box). Same pattern as
# single/hotpotqa's short-form nudge.
_OUTPUT_FORMAT_NUDGE = (
    "\n\nFINAL OUTPUT FORMAT:\n"
    "After all reasoning, end with a single line containing the final "
    "answer wrapped in \\boxed{...}. The scorer extracts the LAST "
    "\\boxed{} in your output and compares against gold via Hendrycks' "
    "LaTeX-normalizing equivalence. Examples of acceptable final lines:\n"
    "  \\boxed{42}\n"
    "  \\boxed{\\frac{1}{2}}\n"
    "  \\boxed{\\sqrt{2}}\n"
    "Do NOT include prose on the boxed line. If multiple candidate "
    "answers emerged during reasoning, commit to one and box ONLY the "
    "final answer."
)

SYSTEM_PROMPT = append_output_contract_from_path(_PROMPT_PATH.read_text().strip() + _OUTPUT_FORMAT_NUDGE, __file__, _PROMPT_PATH.stem)


# Tools
@tool
def calculator(expression: str) -> str:
    """Evaluate a numeric Python expression (arithmetic + math functions).

    Supports +, -, *, /, **, parentheses, and math functions (sqrt, log, log10,
    log2, exp, sin, cos, tan, asin, acos, atan, floor, ceil, pow, pi, e).
    Example: calculator("(4/3) * pi * 2**3")
    """
    import math

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


# Agent
def format_prompt(problem: str) -> str:
    """Build the user-facing prompt for one MATH problem."""
    return problem


def build_agent():
    llm = ChatOpenAI(
        model=MODEL_ID,
        base_url=VLLM_BASE_URL,
        api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        temperature=0.2,
        top_p=0.9,
        seed=0,
        max_tokens=2048,
        extra_body={
            "repetition_penalty": 1.05,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    return create_react_agent(model=llm, tools=[calculator], prompt=SYSTEM_PROMPT)


# Output Parsing
def strip_thinking(text: str) -> str:
    """Cut everything up through the last </think> tag."""
    index = text.lower().rfind("</think>")
    if index >= 0:
        text = text[index + len("</think>"):]
    return text.strip()


def extract_boxed(text: str) -> str | None:
    """Return the inner content of the LAST \\boxed{...} in the text.

    Handles nested braces (e.g., \\boxed{\\frac{1}{2}}) via brace counting.
    """
    marker = r"\boxed{"
    idx = text.rfind(marker)
    if idx < 0:
        return None
    # point at the opening brace of \boxed{
    start = idx + len(marker) - 1
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : i]
    return None   # unbalanced


def extract_answer(text: str) -> str | None:
    """Return the model's boxed final answer, else None."""
    return extract_boxed(text)


# Scoring
# Verbatim port of Hendrycks' math_equivalence.py (NeurIPS 2021).
# Source:  https://github.com/hendrycks/math/blob/main/modeling/math_equivalence.py
# This is the community-standard MATH scorer used by lm-evaluation-harness,
# OpenAI PRM800K, and most published MATH results. Do not modify — keep
# aligned so numbers are comparable to prior work.


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
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string


def _remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
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
    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{."
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # strip LHS of single-variable assignment "k = ..."
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} / \frac{1}{2}. Also handles \frac1{72}.
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # X/Y --> \frac{X}{Y} in simple integer cases
    string = _fix_a_slash_b(string)

    return string


def is_equiv(str1, str2, verbose: bool = False) -> bool:
    """Hendrycks MATH equivalence (official). True if the two strings are
    equivalent after normalization."""
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
    """Float wrapper over is_equiv, for uniform interface with other scorers."""
    return float(is_equiv(pred, gold))


# Orchestration
def solve(problem: str, agent=None) -> dict:
    """Run the agent on one MATH problem.

    Strips Qwen3's <think>...</think> reasoning from every AI message.
    Optional `agent` param lets callers reuse one agent across a batch.

    Returns {'answer': str | None, 'raw': str, 'messages': list}.
    """
    if agent is None:
        agent = build_agent()
    result = agent.invoke(
        {"messages": [("user", format_prompt(problem))]},
        config={"recursion_limit": 25},
    )
    for msg in result["messages"]:
        if msg.type == "ai" and isinstance(msg.content, str):
            msg.content = strip_thinking(msg.content)
    final = result["messages"][-1].content
    return {
        "answer": extract_answer(final),
        "raw": final,
        "messages": result["messages"],
    }


# Dataset loader
# Switched from HuggingFaceH4/MATH-500 (500 rows, `answer` field provided)
# to qwedsacf/competition_math (12500 rows, gold must be extracted from
# `solution`'s last \boxed{...}). We filter to Precalculus / Level 5 —
# 312 rows — so all topologies run on the same hard-math slice.
_HF_DATASET = "qwedsacf/competition_math"
_HF_SPLIT = "train"
_SUBJECT = "Precalculus"
_LEVEL = "Level 5"


def load_instances(
    limit: int | None = None,
    offset: int = 0,
    only: list[str] | None = None,
) -> list[dict]:
    """Load the Precalculus / Level-5 subset of qwedsacf/competition_math.

    The dataset has no `answer` field — gold is extracted from the LAST
    `\\boxed{...}` in the `solution` column. `id` is a stable MD5 of the
    problem text, so IDs line up across topologies for per-row parity.

    Returns list of dicts:
        {id, problem, answer, subject, level, raw}
    """
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
    """Run `solve()` on every problem, compare boxed answer vs gold via
    Hendrycks `is_equiv`, return aggregate summary + optionally write
    per-instance predictions to JSONL.
    """
    agent = build_agent()  # build once; reuse across the batch
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
                out = solve(inst["problem"], agent=agent)
                error = None
            except Exception as e:
                out = {"answer": None, "raw": "", "messages": []}
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

            telem = normalize(langchain_telemetry(out.get("messages") or []))
            rec_out = {
                "id": inst["id"],
                "problem": inst["problem"],
                "gold_answer": gold,
                "predicted_answer": pred,
                "em": em,
                "subject": inst.get("subject"),
                "level": inst.get("level"),
                "raw": out.get("raw") or "",
                "latency_s": round(latency_s, 2),
                **telem,
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
            f"\n=== MATH-500 batch complete ===\n"
            f"  n={summary['n']}  n_extracted={summary['n_extracted']}\n"
            f"  EM={summary['em']:.3f}  (on extracted only: {summary['extracted_em']:.3f})\n"
            f"  total_s={summary['total_s']}\n"
        )
    return summary


# Demo
def _canned_demo() -> None:
    problem = "Compute the value of $\\frac{7!}{5!}$. Put your final answer inside \\boxed{}."
    expected = "42"
    out = solve(problem)
    print(f"\n=== Extracted boxed answer: {out['answer']!r}  (expected: {expected!r}) ===")
    if out["answer"] is not None:
        em = exact_match_score(out["answer"], expected)
        print(f"=== EM: {em:.2f} ===\n")
    print("=== Full message trace ===")
    for msg in out["messages"]:
        msg.pretty_print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Single-topology MATH runner (LangGraph, MATH-500 subset)."
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

    print(f"loading MATH-500 from {_HF_DATASET} [{_HF_SPLIT}] ...")
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
