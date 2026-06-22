"""Sequential topology specialized for MATH (competition math), in CrewAI."""

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

from crewai import LLM, Agent, Crew, Process, Task
from crewai.tools import tool

# Shared telemetry.
_TOPO_ROOT = str(Path(__file__).resolve().parents[4])
if _TOPO_ROOT not in sys.path:
    sys.path.insert(0, _TOPO_ROOT)
from topologies.telemetry import crewai_telemetry, normalize  # noqa: E402


VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://lai:8001/v1")
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3.5-9B")

_REPO_ROOT = Path(__file__).resolve().parents[4]
_PROMPTS_DIR = _REPO_ROOT / "configs" / "prompts" / "sequential" / "math"


def _load_prompt(role: str) -> str:
    return append_output_contract_from_path((_PROMPTS_DIR / f"{role}.txt").read_text().strip(), __file__, role)


# Tools
@tool("calculator")
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
def _build_llm() -> LLM:
    return LLM(
        model=f"openai/{MODEL_ID}",
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


# Crew
def build_crew(llm: LLM | None = None) -> Crew:
    """Build the 3-stage decomposer -> computer -> verifier pipeline."""
    if llm is None:
        llm = _build_llm()

    decomposer = Agent(
        role="Math Decomposer",
        goal=(
            "Break the math problem into a short ordered list of "
            "computational sub-steps. Do NOT compute anything yet."
        ),
        backstory=_load_prompt("decomposer"),
        tools=[],  # no tools — planning only
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    computer = Agent(
        role="Math Computer",
        goal=(
            "Execute each sub-step in order, using the calculator for "
            "arithmetic. Report the numeric result of each step."
        ),
        backstory=_load_prompt("computer"),
        tools=[calculator],
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    checker = Agent(
        role="Math Checker",
        goal=(
            "Re-derive the final quantity via an alternative path to "
            "cross-check the computer. Do NOT emit the final \\boxed{...}."
        ),
        backstory=_load_prompt("checker"),
        tools=[calculator],
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    verifier = Agent(
        role="Math Verifier",
        goal=(
            "Reconcile the computer's and checker's results; emit the "
            "final answer inside \\boxed{...}."
        ),
        backstory=_load_prompt("verifier"),
        tools=[calculator],
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    decompose_task = Task(
        description=(
            "Decompose the problem below into a short ordered list of "
            "computational sub-steps. Number each step. Do NOT compute "
            "the result — that is the next stage's job. Do NOT emit a "
            "final answer.\n\n"
            "PROBLEM:\n{problem}"
        ),
        expected_output=(
            "A numbered list of 3-8 computational sub-steps describing "
            "what the next stage must compute."
        ),
        agent=decomposer,
    )

    compute_task = Task(
        description=(
            "Execute each sub-step from the Decomposer's list, in order. "
            "For each step, call the calculator tool on the numeric "
            "expression and report the result. End with your best "
            "current answer. Do NOT emit the final \\boxed{...} yet — "
            "that is the Verifier's job.\n\n"
            "PROBLEM:\n{problem}"
        ),
        expected_output=(
            "For each numbered step: the expression evaluated + the "
            "result. A final line stating your best-answer value "
            "derived from the last step."
        ),
        agent=computer,
        context=[decompose_task],
    )

    check_task = Task(
        description=(
            "Re-derive the final answer via an ALTERNATIVE path — either "
            "a symbolic simplification, a different decomposition, or a "
            "sanity-check identity. Use the calculator to verify your "
            "alternative. Report whether your result agrees with the "
            "Computer's; do NOT emit the final \\boxed{{...}} yet.\n\n"
            "PROBLEM:\n{problem}"
        ),
        expected_output=(
            "A short alternative derivation + a line stating 'Agrees "
            "with Computer' or 'Disagrees: Computer said X, Checker "
            "got Y'."
        ),
        agent=checker,
        context=[decompose_task, compute_task],
    )

    verify_task = Task(
        description=(
            "Reconcile the Computer's and Checker's results. If they "
            "agree, confirm. If they disagree, use the calculator to "
            "resolve (re-run the critical arithmetic). Your output "
            "MUST end with the final answer inside \\boxed{{...}}.\n\n"
            "PROBLEM:\n{problem}"
        ),
        expected_output=(
            "A short reconciliation note ending with the final boxed "
            "LaTeX answer (\\boxed{{...}})."
        ),
        agent=verifier,
        context=[decompose_task, compute_task, check_task],
    )

    return Crew(
        agents=[decomposer, computer, checker, verifier],
        tasks=[decompose_task, compute_task, check_task, verify_task],
        process=Process.sequential,
        verbose=False,
    )


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
    """Run the 3-stage sequential crew on one MATH problem.

    Returns:
        {
            "answer":   boxed LaTeX (str) or None,
            "raw":      verifier's full output text,
            "by_stage": {decomposer, computer, verifier} -> each stage's output,
        }
    """
    crew = build_crew()
    result = crew.kickoff(inputs={"problem": problem})

    final = result.raw
    stages = {}
    try:
        stages["decomposer"] = result.tasks_output[0].raw
        stages["computer"]   = result.tasks_output[1].raw
        stages["checker"]    = result.tasks_output[2].raw
        stages["verifier"]   = result.tasks_output[3].raw
    except (AttributeError, IndexError):
        stages = {"decomposer": "", "computer": "", "checker": "", "verifier": final}

    return {
        "answer": extract_answer(final),
        "raw": final,
        "by_stage": stages,
        "telemetry": normalize(crewai_telemetry(crew, n_stages=len(stages))),
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
    """Run `solve()` on every problem, compare boxed answer vs gold via
    Hendrycks `is_equiv`, return aggregate summary + optionally write
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
                out = {"answer": None, "raw": "", "by_stage": {}}
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

            rec_out = {
                "id": inst["id"],
                "problem": inst["problem"],
                "gold_answer": gold,
                "predicted_answer": pred,
                "em": em,
                "subject": inst.get("subject"),
                "level": inst.get("level"),
                "by_stage": out.get("by_stage") or {},
                "raw": out.get("raw") or "",
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
    print(f"\n=== Decomposer (excerpt) ===\n{out['by_stage']['decomposer'][:400]}...")
    print(f"\n=== Computer (excerpt) ===\n{out['by_stage']['computer'][:400]}...")
    print(f"\n=== Checker (excerpt) ===\n{out['by_stage']['checker'][:400]}...")
    print(f"\n=== Verifier (excerpt) ===\n{out['by_stage']['verifier'][:400]}...")
    print(f"\n=== Extracted boxed answer: {out['answer']!r}  (expected: {expected!r}) ===")
    if out["answer"] is not None:
        em = exact_match_score(out["answer"], expected)
        print(f"=== EM (is_equiv): {em:.2f} ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sequential-topology MATH runner (CrewAI 4-stage)."
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
