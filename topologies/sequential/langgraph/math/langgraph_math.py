"""Sequential topology specialized for MATH (competition math), LangGraph."""

# Config
from __future__ import annotations

import argparse
import json
import operator
import os
import re
import sys
import time
from pathlib import Path

from topologies.output_contracts import append_output_contract_from_path
from typing import Annotated

from typing_extensions import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent

# Shared telemetry.
_REPO_ROOT = Path(__file__).resolve().parents[4]
_TOPO_ROOT = str(_REPO_ROOT)
if _TOPO_ROOT not in sys.path:
    sys.path.insert(0, _TOPO_ROOT)
from topologies.telemetry import langchain_telemetry, normalize  # noqa: E402


VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://n12:8000/v1")
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3.5-9B")

_PROMPTS_DIR = _REPO_ROOT / "configs" / "prompts" / "sequential" / "math"


def _load_prompt(role: str) -> str:
    return append_output_contract_from_path((_PROMPTS_DIR / f"{role}.txt").read_text().strip(), __file__, role)


# Tools
@tool
def calculator(expression: str) -> str:
    """Evaluate a numeric Python expression (arithmetic + math functions).

    Supports +, -, *, /, **, parentheses, and math functions (sqrt, log,
    log10, log2, exp, sin, cos, tan, asin, acos, atan, floor, ceil, pow,
    pi, e). Example: calculator("(4/3) * 3.14159 * 2**3")
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


# Input-side escaping
def _escape_braces(s: str) -> str:
    return s.replace("{", "{{").replace("}", "}}")


# Per-stage task descriptions (same as CrewAI Task.description)
# NB: literal `{` / `}` in the templates (e.g. inside `\\boxed{...}`) are
# doubled to `{{` / `}}` so `str.format(**inputs)` leaves them intact.
_TASK_DESCRIPTIONS = {
    "decomposer": (
        "Decompose the problem below into a short ordered list of "
        "computational sub-steps. Number each step. Do NOT compute "
        "the result — that is the next stage's job. Do NOT emit a "
        "final answer.\n\n"
        "PROBLEM:\n{problem}"
    ),
    "computer": (
        "Execute each sub-step from the Decomposer's list, in order. "
        "For each step, call the calculator tool on the numeric "
        "expression and report the result. End with your best "
        "current answer. Do NOT emit the final \\boxed{{...}} yet — "
        "that is the Verifier's job.\n\n"
        "PROBLEM:\n{problem}"
    ),
    "checker": (
        "Re-derive the final answer via an ALTERNATIVE path — either "
        "a symbolic simplification, a different decomposition, or a "
        "sanity-check identity. Use the calculator to verify your "
        "alternative. Report whether your result agrees with the "
        "Computer's; do NOT emit the final \\boxed{{...}} yet.\n\n"
        "PROBLEM:\n{problem}"
    ),
    "verifier": (
        "Reconcile the Computer's and Checker's results. If they "
        "agree, confirm. If they disagree, use the calculator to "
        "resolve (re-run the critical arithmetic). Your output "
        "MUST end with the final answer inside \\boxed{{...}}.\n\n"
        "PROBLEM:\n{problem}"
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
    """Build the 4-stage decomposer -> computer -> checker -> verifier pipeline."""
    stages = [
        (
            "decomposer",
            _load_prompt("decomposer"),
            [],
            _TASK_DESCRIPTIONS["decomposer"],
        ),
        (
            "computer",
            _load_prompt("computer"),
            [calculator],
            _TASK_DESCRIPTIONS["computer"],
        ),
        (
            "checker",
            _load_prompt("checker"),
            [calculator],
            _TASK_DESCRIPTIONS["checker"],
        ),
        (
            "verifier",
            _load_prompt("verifier"),
            [calculator],
            _TASK_DESCRIPTIONS["verifier"],
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
# Verbatim Hendrycks MATH equivalence from math_equivalence.py, byte-
# identical to topologies/single/math/langgraph_math.py so sequential
# EM is directly comparable with single/independent numbers.
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


# Orchestration
def solve(problem: str) -> dict:
    """Run the 4-stage sequential graph on one MATH problem.

    Returns:
        {
            "answer":   boxed LaTeX (str) or None,
            "raw":      verifier's full output text,
            "by_stage": {decomposer, computer, checker, verifier} -> text,
            "telemetry": normalized 5-key token/call counts,
        }
    """
    llm = _build_llm()
    compiled, roles = _build_graph(llm)
    problem_escaped = _escape_braces(problem)
    result = compiled.invoke(
        {"inputs": {"problem": problem_escaped}, "by_stage": {}, "messages": []}
    )

    stages_out = result.get("by_stage") or {}
    final = stages_out.get(roles[-1], "")

    return {
        "answer": extract_answer(final),
        "raw": final,
        "by_stage": stages_out,
        "telemetry": normalize(langchain_telemetry(result.get("messages") or [])),
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


# Batch runner
def run_one(instance: dict, out_dir: Path) -> dict:
    """Run the 4-stage graph on one MATH row; dump stage trace."""
    rid = instance["id"]
    summary: dict = {
        "id": rid,
        "subject": instance.get("subject"),
        "level": instance.get("level"),
    }

    # Assign a stable numeric index for the trace filename. If the instance
    # carries one, use it; else fall back to the id hash.
    idx = instance.get("idx")
    if idx is None:
        import hashlib
        idx = int(hashlib.md5(rid.encode("utf-8")).hexdigest(), 16) % 10000

    t0 = time.time()
    try:
        out = solve(instance["problem"])
        error = None
    except Exception as e:
        out = {"answer": None, "raw": "", "by_stage": {}, "telemetry": {}}
        error = f"{type(e).__name__}: {e}"
    latency_s = time.time() - t0

    pred = out.get("answer")
    gold = instance["answer"]
    em = exact_match_score(pred, gold) if pred is not None else 0.0

    summary["gold_answer"] = gold
    summary["predicted_answer"] = pred
    summary["em"] = em
    summary["latency_s"] = round(latency_s, 2)
    summary["error"] = error
    summary.update(out.get("telemetry") or {})

    trace_path = out_dir / "traces" / f"{idx:04d}.txt"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("w") as f:
        for stage, content in (out.get("by_stage") or {}).items():
            f.write(f"=== {stage.upper()} ===\n{content}\n\n")

    summary["by_stage"] = out.get("by_stage") or {}
    summary["raw"] = out.get("raw") or ""
    return summary


# Batch eval
def run_batch(
    instances: list[dict],
    out_path: Path | None = None,
    out_dir: Path | None = None,
    verbose: bool = True,
) -> dict:
    """Run `solve()` on every problem, compare boxed answer vs gold via
    Hendrycks `is_equiv`, return aggregate summary + optionally write
    per-instance predictions to JSONL.
    """
    _default_root = Path(__file__).resolve().parents[4]
    out_dir = out_dir or (_default_root / "results" / "math_sequential_langgraph")
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

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
            # Give each instance a stable numeric idx for trace filenames.
            inst_with_idx = dict(inst)
            inst_with_idx.setdefault("idx", i)
            summary = run_one(inst_with_idx, out_dir)

            pred = summary.get("predicted_answer")
            em = summary.get("em") or 0.0
            if pred is not None:
                n_extracted += 1
            em_sum += em

            rec_out = {
                "id": inst["id"],
                "problem": inst["problem"],
                "gold_answer": summary.get("gold_answer"),
                "predicted_answer": pred,
                "em": em,
                "subject": inst.get("subject"),
                "level": inst.get("level"),
                "by_stage": summary.get("by_stage") or {},
                "raw": summary.get("raw") or "",
                "latency_s": summary.get("latency_s"),
                "error": summary.get("error"),
            }
            for k in (
                "prompt_tokens", "completion_tokens", "total_tokens",
                "n_calls", "n_tool_calls",
            ):
                if k in summary:
                    rec_out[k] = summary[k]
            per_instance.append(rec_out)
            if out_f is not None:
                out_f.write(json.dumps(rec_out) + "\n")
                out_f.flush()

            if verbose:
                running_em = em_sum / (i + 1)
                mark = "✓" if em == 1.0 else ("?" if pred is None else "✗")
                pred_disp = (pred or "-")[:30]
                gold_disp = (summary.get("gold_answer") or "-")[:30]
                print(
                    f"[{i + 1:>3}/{n}] {inst['id'][:30]:<30} {mark}  "
                    f"em={em:.0f}  pred={pred_disp!r} gold={gold_disp!r}  "
                    f"EM={running_em:.3f} lat={summary.get('latency_s')}s",
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
            f"\n=== sequential/MATH-500 batch complete ===\n"
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
    print(f"\n=== Decomposer (excerpt) ===\n{out['by_stage'].get('decomposer', '')[:400]}...")
    print(f"\n=== Computer (excerpt) ===\n{out['by_stage'].get('computer', '')[:400]}...")
    print(f"\n=== Checker (excerpt) ===\n{out['by_stage'].get('checker', '')[:400]}...")
    print(f"\n=== Verifier (excerpt) ===\n{out['by_stage'].get('verifier', '')[:400]}...")
    print(f"\n=== Extracted boxed answer: {out['answer']!r}  (expected: {expected!r}) ===")
    if out["answer"] is not None:
        em = exact_match_score(out["answer"], expected)
        print(f"=== EM (is_equiv): {em:.2f} ===")


# CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sequential-topology MATH runner (LangGraph 4-stage)."
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
