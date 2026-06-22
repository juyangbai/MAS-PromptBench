"""Sequential topology specialized for GPQA-Diamond, implemented in CrewAI."""

# Config
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
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
_PROMPTS_DIR = _REPO_ROOT / "configs" / "prompts" / "sequential" / "gpqa"


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
    """CrewAI routes completions through litellm; `openai/<model>` + api_base
    points it at our local vLLM OpenAI-compatible endpoint."""
    return LLM(
        model=f"openai/{MODEL_ID}",
        base_url=VLLM_BASE_URL,
        api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        # House default (matches single/independent).
        temperature=0.2,
        top_p=0.9,
        seed=0,
        max_tokens=2048,
        additional_drop_params=[],
        extra_body={
            "repetition_penalty": 1.05,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )


# Crew
def build_crew(llm: LLM | None = None) -> Crew:
    """Build the 3-stage analyzer -> solver -> verifier pipeline."""
    if llm is None:
        llm = _build_llm()

    analyzer = Agent(
        role="GPQA Analyzer",
        goal=(
            "Identify the scientific principles relevant to the MCQ and "
            "enumerate how each option would be derived."
        ),
        backstory=_load_prompt("analyzer"),
        tools=[calculator],
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    solver = Agent(
        role="GPQA Solver",
        goal=(
            "Apply the Analyzer's principles to choose one option and emit "
            "the final letter."
        ),
        backstory=_load_prompt("solver"),
        tools=[calculator],
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    critic = Agent(
        role="GPQA Critic",
        goal=(
            "Challenge the Solver's pick. For each rejected option, "
            "explain the strongest argument that WOULD defend it. Do "
            "NOT commit to a new final letter."
        ),
        backstory=_load_prompt("critic"),
        tools=[calculator],
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    verifier = Agent(
        role="GPQA Verifier",
        goal=(
            "Reconcile the Solver's pick with the Critic's challenges; "
            "emit the final letter."
        ),
        backstory=_load_prompt("verifier"),
        tools=[calculator],
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    analyze_task = Task(
        description=(
            "Analyze the multiple-choice question below. Enumerate the "
            "scientific principles at play and describe, option by option "
            "(A, B, C, D), how each candidate answer would be derived. "
            "Do NOT commit to a final letter.\n\n"
            "QUESTION:\n{question}"
        ),
        expected_output=(
            "A structured analysis: a short 'Principles' section followed "
            "by one numbered paragraph per option (A/B/C/D) explaining "
            "each option's derivation."
        ),
        agent=analyzer,
    )

    solve_task = Task(
        description=(
            "Using the Analyzer's principles, select the single correct "
            "option and emit your reasoning + final letter. Your output "
            "MUST end with a line matching 'Answer: X' where X is one of "
            "A, B, C, D.\n\n"
            "QUESTION:\n{question}"
        ),
        expected_output=(
            "Concise reasoning that applies the Analyzer's principles, "
            "followed by a final line of the form 'Answer: X'."
        ),
        agent=solver,
        context=[analyze_task],
    )

    critic_task = Task(
        description=(
            "Challenge the Solver's pick. Given the Analyzer's principles "
            "and the Solver's tentative letter + reasoning, identify any "
            "concrete errors in the Solver's logic and — for each "
            "REJECTED option — describe the strongest argument that "
            "would have defended it. Do NOT declare a new final letter.\n\n"
            "QUESTION:\n{question}"
        ),
        expected_output=(
            "A structured critique: 'Errors in solver's reasoning' "
            "section (may be empty if none), followed by 'Defense of "
            "rejected options' with a short note per option."
        ),
        agent=critic,
        context=[analyze_task, solve_task],
    )

    verify_task = Task(
        description=(
            "Reconcile the Solver's pick with the Critic's challenges. "
            "If the Critic exposed a concrete error, override; otherwise "
            "confirm. Your output MUST end with a line matching "
            "'Final answer: X' where X is one of A, B, C, D.\n\n"
            "QUESTION:\n{question}"
        ),
        expected_output=(
            "A short reconciliation note (confirm or override, with "
            "reason) ending with 'Final answer: X'."
        ),
        agent=verifier,
        context=[analyze_task, solve_task, critic_task],
    )

    return Crew(
        agents=[analyzer, solver, critic, verifier],
        tasks=[analyze_task, solve_task, critic_task, verify_task],
        process=Process.sequential,
        verbose=False,
    )


# Output Parsing
_LETTERS = ["A", "B", "C", "D"]

# Strip markdown `**bold**` / `*italic*` / backticks before matching — the
# 9B frequently emits "**Answer:** B" which broke the bare regexes.
_MARKDOWN_STRIP_RE = re.compile(r"[*_`]+")
# Primary: "Final answer: X" / "Answer: X" (what the verifier is asked for).
_ANSWER_RE = re.compile(
    r"\b(?:final\s+)?answer\b\s*[:\s]*\(?([A-D])\)?",
    re.IGNORECASE,
)
# Fallback: "Option X" / "choice X" / "Option: X" / "correct option is C".
_OPTION_RE = re.compile(
    r"\b(?:option|choice)\b\s*(?:is)?\s*[:\s]*\(?([A-D])\)?",
    re.IGNORECASE,
)
# Fallback: bare [A-D] on its own line near the end.
_BARE_LETTER_RE = re.compile(
    r"(?:^|\n)\s*\(?([A-D])\)?\s*(?:[.\n]|$)", re.MULTILINE
)


def extract_answer(text: str) -> str | None:
    """Return the MCQ letter from the verifier's final output.

    Matches the 3-pattern cascade + markdown stripping used by
    single/independent/centralized/decentralized gpqa so extracted
    letters are comparable across topologies.
    """
    cleaned = _MARKDOWN_STRIP_RE.sub("", text)
    for pattern in (_ANSWER_RE, _OPTION_RE, _BARE_LETTER_RE):
        matches = pattern.findall(cleaned)
        if matches:
            return matches[-1].upper()
    return None


# Format helpers
def format_mcq(question: str, choices: list[str]) -> str:
    """Build the user-facing MCQ string (4 choices, labeled A-D)."""
    assert len(choices) == 4, "GPQA expects exactly 4 choices."
    body = "\n".join(f"{_LETTERS[i]}) {choices[i]}" for i in range(4))
    return f"{question}\n\n{body}"


# Orchestration
def solve(question: str, choices: list[str]) -> dict:
    """Run the 3-stage sequential crew on one GPQA-style MCQ.

    Returns:
        {
            "answer":  final letter A/B/C/D or None,
            "raw":     verifier's final output text,
            "by_stage": {analyzer, solver, verifier} -> each stage's output,
        }
    """
    crew = build_crew()
    mcq = format_mcq(question, choices)
    result = crew.kickoff(inputs={"question": mcq})

    # CrewAI returns a CrewOutput whose `.raw` is the LAST task's output.
    # Per-stage outputs are in `.tasks_output[].raw`.
    final = result.raw
    stages = {}
    try:
        stages["analyzer"] = result.tasks_output[0].raw
        stages["solver"]   = result.tasks_output[1].raw
        stages["critic"]   = result.tasks_output[2].raw
        stages["verifier"] = result.tasks_output[3].raw
    except (AttributeError, IndexError):
        stages = {"analyzer": "", "solver": "", "critic": "", "verifier": final}

    return {
        "answer": extract_answer(final),
        "raw": final,
        "by_stage": stages,
        "telemetry": normalize(crewai_telemetry(crew, n_stages=len(stages))),
    }


# Dataset loader (aligned to single/gpqa + independent/gpqa)
_HF_DATASET = "Idavidrein/gpqa"
_HF_CONFIG = "gpqa_diamond"
_HF_SPLIT = "train"


def _stable_row_id(row: dict, fallback_idx: int) -> str:
    """Stable id for a GPQA row — hash of question text. Matches
    single/gpqa + independent/gpqa so per-row comparisons line up
    across topologies."""
    q = (row.get("Question") or "").strip()
    if q:
        return "gpqa_" + hashlib.md5(q.encode("utf-8")).hexdigest()[:10]
    return f"gpqa_idx_{fallback_idx}"


def load_instances(
    limit: int | None = None,
    offset: int = 0,
    only: list[str] | None = None,
    shuffle_seed: int = 0,
) -> list[dict]:
    """Load GPQA-Diamond rows from HuggingFace with 4 choices shuffled
    DETERMINISTICALLY per row (`Random(f"{shuffle_seed}|{row_id}")`).
    Same algorithm + same default seed as single/gpqa + independent/gpqa
    — identical `shuffle_seed` produces identical choice orderings, so
    `correct_letter` matches for each row id across topologies.
    """
    from datasets import load_dataset

    ds = load_dataset(_HF_DATASET, _HF_CONFIG)[_HF_SPLIT]
    rows: list[dict] = []
    for i, row in enumerate(ds):
        rid = _stable_row_id(row, i)
        if only is not None and rid not in set(only):
            continue
        correct = (row.get("Correct Answer") or "").strip()
        incorrects = [(row.get(f"Incorrect Answer {k}") or "").strip() for k in (1, 2, 3)]
        if not correct or any(not x for x in incorrects):
            continue
        four = [correct, *incorrects]
        rng = random.Random(f"{shuffle_seed}|{rid}")
        indices = list(range(4))
        rng.shuffle(indices)
        shuffled = [four[j] for j in indices]
        correct_slot = indices.index(0)
        rows.append({
            "id": rid,
            "question": (row.get("Question") or "").strip(),
            "choices": shuffled,
            "correct_letter": _LETTERS[correct_slot],
            "raw": dict(row),
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
    """Run the 4-stage CrewAI sequential pipeline on every instance,
    compare verifier-emitted letter vs gold, return aggregate summary +
    optionally write per-instance predictions to JSONL.

    Per-instance record shape:
        {id, question, choices, correct_letter,
         predicted_letter, correct,
         by_stage: {analyzer, solver, critic, verifier} (string excerpts),
         latency_s, error}
    """
    per_instance: list[dict] = []
    n = len(instances)
    n_correct = 0
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
                out = solve(inst["question"], inst["choices"])
                error = None
            except Exception as e:
                out = {"answer": None, "raw": "", "by_stage": {}}
                error = f"{type(e).__name__}: {e}"
            latency_s = time.time() - t0

            pred = out["answer"]
            gold = inst["correct_letter"]
            is_correct = pred is not None and pred == gold
            if pred is not None:
                n_extracted += 1
            if is_correct:
                n_correct += 1

            # Keep only short excerpts of each stage in the predictions
            # JSONL — full stage text is available via re-run if needed.
            by_stage = out.get("by_stage") or {}
            excerpts = {k: (v or "")[:800] for k, v in by_stage.items()}
            rec = {
                "id": inst["id"],
                "question": inst["question"],
                "choices": inst["choices"],
                "correct_letter": gold,
                "predicted_letter": pred,
                "correct": is_correct,
                "raw": out.get("raw") or "",
                "by_stage": excerpts,
                "latency_s": round(latency_s, 2),
                **(out.get("telemetry") or {}),
                "error": error,
            }
            per_instance.append(rec)
            if out_f is not None:
                out_f.write(json.dumps(rec) + "\n")
                out_f.flush()

            if verbose:
                running_acc = n_correct / (i + 1)
                mark = "✓" if is_correct else ("?" if pred is None else "✗")
                print(
                    f"[{i + 1:>3}/{n}] {inst['id']} {mark}  "
                    f"pred={pred or '-'}  gold={gold}  "
                    f"acc={running_acc:.3f}  lat={latency_s:.1f}s",
                    flush=True,
                )
    finally:
        if out_f is not None:
            out_f.close()

    elapsed = time.time() - start
    summary = {
        "n": n,
        "n_extracted": n_extracted,
        "n_correct": n_correct,
        "accuracy": (n_correct / n) if n else 0.0,
        "extracted_acc": (n_correct / n_extracted) if n_extracted else 0.0,
        "total_s": round(elapsed, 1),
        "per_instance": per_instance,
    }
    if verbose:
        print(
            f"\n=== sequential/GPQA-Diamond batch complete ===\n"
            f"  n={summary['n']}  n_extracted={summary['n_extracted']}  "
            f"n_correct={summary['n_correct']}\n"
            f"  accuracy={summary['accuracy']:.3f}  "
            f"extracted_acc={summary['extracted_acc']:.3f}  "
            f"total_s={summary['total_s']}\n"
        )
    return summary


# Demo
def _canned_demo() -> None:
    question = (
        "A circular wire loop of radius R carries a steady current I. "
        "What is the magnitude of the magnetic field at the geometric center "
        "of the loop? (mu_0 is the vacuum permeability.)"
    )
    choices = [
        "mu_0 * I / (2 * R)",
        "mu_0 * I / (4 * pi * R)",
        "mu_0 * I / R",
        "mu_0 * I / (pi * R)",
    ]
    out = solve(question, choices)
    print(f"\n=== Analyzer (excerpt) ===\n{out['by_stage']['analyzer'][:400]}...")
    print(f"\n=== Solver (excerpt) ===\n{out['by_stage']['solver'][:400]}...")
    print(f"\n=== Critic (excerpt) ===\n{out['by_stage']['critic'][:400]}...")
    print(f"\n=== Verifier (excerpt) ===\n{out['by_stage']['verifier'][:400]}...")
    print(f"\n=== Extracted answer: {out['answer']}  (expected: A) ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sequential-topology GPQA-Diamond runner (CrewAI)."
    )
    parser.add_argument(
        "--batch", action="store_true",
        help="Run the real GPQA-Diamond eval (else: one canned MCQ demo).",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument(
        "--shuffle-seed", type=int, default=0,
        help="Per-row choice-shuffling seed (default 0 — matches "
             "single/gpqa + independent/gpqa for cross-topology parity).",
    )
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()

    if not args.batch:
        _canned_demo()
        sys.exit(0)

    print(f"loading GPQA-Diamond from {_HF_DATASET} [{_HF_CONFIG}/{_HF_SPLIT}] ...")
    instances = load_instances(
        limit=args.limit, offset=args.offset,
        shuffle_seed=args.shuffle_seed, only=args.only,
    )
    if not instances:
        print("no instances loaded (check --limit/--offset/--only)", file=sys.stderr)
        sys.exit(1)
    print(f"  loaded {len(instances)} instance(s)")
    out_path = Path(args.out) if args.out else None
    run_batch(instances, out_path=out_path)
    if out_path:
        print(f"  predictions written to {out_path}")
