"""Independent topology specialized for MATH (competition math)."""

# Config
from __future__ import annotations

import argparse
import asyncio
import json
import operator
import os
import sys
import time
from pathlib import Path

from teamsizes.output_contracts import append_output_contract_from_path
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
N_AGENTS = int(os.environ.get("INDEPENDENT_N_AGENTS", "2"))

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PROMPT_PATH = (
    _REPO_ROOT / "configs" / "prompts" / "independent" / "math" / "solver.txt"
)

# Same \boxed{...} format nudge as single/math — without it, Qwen3.5-9B
# sometimes emits "the answer is 42" without the box, breaking extraction.
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


def _build_one_agent(seed: int):
    """Build one replica's react agent, seeded differently from its siblings."""
    llm = ChatOpenAI(
        model=MODEL_ID,
        base_url=VLLM_BASE_URL,
        api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        max_tokens=4096,
        # Greedy (temp=0) would collapse all N replicas to the same
        # trajectory, defeating the ensemble.
        temperature=0.2,
        top_p=0.9,
        seed=seed,
        extra_body={
            "repetition_penalty": 1.05,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    return create_react_agent(model=llm, tools=[calculator], prompt=SYSTEM_PROMPT)


# Output Parsing
def strip_thinking(text: str) -> str:
    """Cut everything up through the last </think> tag (Qwen3 convention)."""
    index = text.lower().rfind("</think>")
    if index >= 0:
        text = text[index + len("</think>"):]
    return text.strip()


def extract_boxed(text: str) -> str | None:
    """Return the inner content of the LAST \\boxed{...} (brace-counted)."""
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
                return text[start + 1 : i]
    return None


def extract_answer(text: str) -> str | None:
    """Return the model's boxed final answer, else None."""
    return extract_boxed(text)


# Scoring
# Verbatim Hendrycks MATH equivalence from math_equivalence.py, aligned
# to topologies/single/math/langgraph_math.py so ensemble numbers remain
# directly comparable with single-topology numbers.
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
def majority_vote(answers: list[dict]) -> str | None:
    """Majority vote over per-replica boxed answers, using Hendrycks
    equivalence to bucket.

    Two replicas' answers go in the same bucket iff `is_equiv(a, b)` —
    so `\\frac{1}{2}` and `0.5` are treated as one vote even though their
    surface strings differ. The returned string is the RAW text of the
    first replica in the winning bucket, preserving LaTeX so downstream
    `is_equiv(ensemble, gold)` still works. Ties broken by first
    occurrence (lowest agent index) among equally-common buckets.
    """
    valid = [a for a in answers if a.get("answer") is not None]
    if not valid:
        return None

    buckets: list[list[dict]] = []
    for a in valid:
        for b in buckets:
            if is_equiv(a["answer"], b[0]["answer"]):
                b.append(a)
                break
        else:
            buckets.append([a])

    best = max(buckets, key=len)
    return best[0]["answer"]


# Graph
class State(TypedDict):
    problem: str
    prompt: str
    answers: Annotated[list[dict], operator.add]


class AgentInput(TypedDict):
    agent_id: int
    seed: int
    prompt: str


# Stall safeguards
# Per-row wall-clock cap; on symbolic problems (matrix inversion, trig
# identities) the calculator tool returns ERROR for non-numeric inputs
# and the model loops retrying variants. With 4 concurrent agents each
# retrying, a single row can burn 15+ min of GPU time. Timeout + lower
# recursion_limit cap worst case from both angles.
PER_ROW_TIMEOUT_S = 120
_RECURSION_LIMIT = 15


async def _run_replica(inp: AgentInput) -> dict:
    """Run one replica's react agent and return its extracted boxed answer."""
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
                "answer": extract_answer(final),
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
def solve(problem: str) -> dict:
    """Run the ensemble on one MATH problem.

    Returns:
        {
            "answer":    equivalence-majority boxed answer (raw LaTeX) or None,
            "per_agent": list of {agent_id, seed, answer, raw, messages},
            "buckets":   list of [raw_answer, count] for each equivalence bucket,
        }
    """
    compiled = build_graph().compile()
    prompt = format_prompt(problem)

    async def _run():
        return await asyncio.wait_for(
            compiled.ainvoke({"problem": problem, "prompt": prompt, "answers": []}),
            timeout=PER_ROW_TIMEOUT_S,
        )

    result = asyncio.run(_run())
    per_agent = sorted(result["answers"], key=lambda a: a["agent_id"])

    # Re-derive bucket sizes for reporting. Matches majority_vote's
    # bucketing logic.
    valid = [a for a in per_agent if a.get("answer") is not None]
    buckets: list[list[dict]] = []
    for a in valid:
        for b in buckets:
            if is_equiv(a["answer"], b[0]["answer"]):
                b.append(a)
                break
        else:
            buckets.append([a])
    bucket_report = [[b[0]["answer"], len(b)] for b in buckets]

    return {
        "answer": majority_vote(per_agent),
        "per_agent": per_agent,
        "buckets": bucket_report,
    }


# Dataset loader
# qwedsacf/competition_math filtered to Precalculus / Level 5 (312 rows).
# Gold is extracted from the LAST \\boxed{...} in the `solution` column.
# IDs are stable MD5 of problem text so they match single/math's IDs
# row-for-row for cross-topology parity.
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
    _propagate_errors: bool = False,
) -> dict:
    """Run `solve()` on every problem, compare ensemble boxed answer vs gold
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
                if _propagate_errors:
                    raise
                out = {"answer": None, "per_agent": [], "buckets": []}
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

            compact_per_agent = [
                {"agent_id": a["agent_id"], "seed": a["seed"],
                 "answer": a["answer"], "raw": a.get("raw", "")}
                for a in out.get("per_agent") or []
            ]
            telem = normalize(langchain_ensemble_telemetry(out.get("per_agent") or []))
            rec_out = {
                "id": inst["id"],
                "problem": inst["problem"],
                "gold_answer": gold,
                "predicted_answer": pred,
                "em": em,
                "buckets": out.get("buckets") or [],
                "per_agent": compact_per_agent,
                "subject": inst.get("subject"),
                "level": inst.get("level"),
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
            f"\n=== independent/MATH-500 batch complete (N={N_AGENTS}) ===\n"
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
    print(f"\n=== Ensemble ({N_AGENTS} replicas) buckets: {out['buckets']}")
    print(f"=== Majority-vote boxed answer: {out['answer']!r}  (expected: {expected!r}) ===")
    if out["answer"] is not None:
        em = exact_match_score(out["answer"], expected)
        print(f"=== EM: {em:.2f} ===\n")
    for a in out["per_agent"]:
        print(f"--- agent_{a['agent_id']} (seed {a['seed']}) -> {a['answer']!r} ---")

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
        description="Independent-topology MATH runner (LangGraph ensemble)."
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
        f"(N_AGENTS={N_AGENTS}) ..."
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
