"""Sequential topology specialized for GPQA-Diamond, implemented in LangGraph."""

# Config
from __future__ import annotations

import argparse
import hashlib
import json
import operator
import os
import random
import re
import sys
import time
from pathlib import Path

from teamsizes.output_contracts import append_output_contract_from_path
from typing import Annotated

from typing_extensions import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage  # noqa: F401
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent

_REPO_ROOT = Path(__file__).resolve().parents[3]

# Shared telemetry.
_TOPO_ROOT = str(_REPO_ROOT)
if _TOPO_ROOT not in sys.path:
    sys.path.insert(0, _TOPO_ROOT)
from topologies.telemetry import langchain_telemetry, normalize  # noqa: E402


VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://n12:8000/v1")
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3.5-9B")

_PROMPTS_DIR = _REPO_ROOT / "configs" / "prompts" / "sequential" / "gpqa"


def _load_prompt(role: str) -> str:
    return append_output_contract_from_path((_PROMPTS_DIR / f"{role}.txt").read_text().strip(), __file__, role)


# Tools (LangChain)
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


CALC_TOOLS = [calculator]


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


# Per-stage task descriptions (same as CrewAI Task.description)
_TASK_DESCRIPTIONS = {
    "analyzer": (
        "Analyze the multiple-choice question below. Enumerate the "
        "scientific principles at play and describe, option by option "
        "(A, B, C, D), how each candidate answer would be derived. "
        "Do NOT commit to a final letter.\n\n"
        "QUESTION:\n{question}"
    ),
    "solver": (
        "Using the Analyzer's principles, select the single correct "
        "option and emit your reasoning + final letter. Your output "
        "MUST end with a line matching 'Answer: X' where X is one of "
        "A, B, C, D.\n\n"
        "QUESTION:\n{question}"
    ),
    "critic": (
        "Challenge the Solver's pick. Given the Analyzer's principles "
        "and the Solver's tentative letter + reasoning, identify any "
        "concrete errors in the Solver's logic and — for each "
        "REJECTED option — describe the strongest argument that "
        "would have defended it. Do NOT declare a new final letter.\n\n"
        "QUESTION:\n{question}"
    ),
    "verifier": (
        "Reconcile the Solver's pick with the Critic's challenges. "
        "If the Critic exposed a concrete error, override; otherwise "
        "confirm. Your output MUST end with a line matching "
        "'Final answer: X' where X is one of A, B, C, D.\n\n"
        "QUESTION:\n{question}"
    ),
    'question_clarifier': (
        'Restate the MCQ in your own words. Output: RESTATEMENT, KEY_TERMS (defined briefly), TYPE.\n\nQUESTION:\n{question}'
    ),
    'domain_expert': (
        'Provide BACKGROUND (3-5 lines of relevant facts/principles needed to answer).\n\nQUESTION:\n{question}'
    ),
    'alternative_explorer': (
        'For each of the OTHER 3 letter choices the solver did not pick, write 1-2 lines on why it is incorrect (or arguments for it). Conclude: KEEP <letter> or SWITCH TO <letter>.\n\nQUESTION:\n{question}'
    ),
    'consistency_checker': (
        "Cross-check the critic's reasoning against the original question + background facts. Output: CONSISTENT or INCONSISTENT (with the contradiction).\n\nQUESTION:\n{question}"
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


def _build_graph(llm: ChatOpenAI):
    """Build the 4-stage analyzer -> solver -> critic -> verifier pipeline.
    ALL 4 stages have access to the calculator tool."""
    stages = [
        (
            'question_clarifier',
            _load_prompt('question_clarifier'),
            CALC_TOOLS,
            _TASK_DESCRIPTIONS['question_clarifier'],
        ),
        (
            'analyzer',
            _load_prompt('analyzer'),
            CALC_TOOLS,
            _TASK_DESCRIPTIONS['analyzer'],
        ),
        (
            'domain_expert',
            _load_prompt('domain_expert'),
            CALC_TOOLS,
            _TASK_DESCRIPTIONS['domain_expert'],
        ),
        (
            'solver',
            _load_prompt('solver'),
            CALC_TOOLS,
            _TASK_DESCRIPTIONS['solver'],
        ),
        (
            'alternative_explorer',
            _load_prompt('alternative_explorer'),
            CALC_TOOLS,
            _TASK_DESCRIPTIONS['alternative_explorer'],
        ),
        (
            'critic',
            _load_prompt('critic'),
            CALC_TOOLS,
            _TASK_DESCRIPTIONS['critic'],
        ),
        (
            'consistency_checker',
            _load_prompt('consistency_checker'),
            CALC_TOOLS,
            _TASK_DESCRIPTIONS['consistency_checker'],
        ),
        (
            'verifier',
            _load_prompt('verifier'),
            CALC_TOOLS,
            _TASK_DESCRIPTIONS['verifier'],
        ),
    
    ]

    graph = StateGraph(SequentialState)
    prior: list[str] = []
    for role, sys_p, tools, tmpl in stages:
        node_fn = _make_tool_node(role, sys_p, tools, llm, tmpl, list(prior))
        graph.add_node(role, node_fn)
        prior.append(role)

    graph.add_edge(START, stages[0][0])
    for a, b in zip(stages, stages[1:]):
        graph.add_edge(a[0], b[0])
    graph.add_edge(stages[-1][0], END)

    return graph.compile(), [s[0] for s in stages]


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
    """Run the 4-stage sequential graph on one GPQA-style MCQ.

    Returns:
        {
            "answer":    final letter A/B/C/D or None,
            "raw":       verifier's final output text,
            "by_stage":  {analyzer, solver, critic, verifier} -> each stage's output,
            "telemetry": normalized 5-key token/call counts,
        }
    """
    llm = _build_llm()
    compiled, roles = _build_graph(llm)
    mcq = format_mcq(question, choices)
    result = compiled.invoke(
        {"inputs": {"question": mcq}, "by_stage": {}, "messages": []}
    )

    stages_out = result.get("by_stage") or {}
    final = stages_out.get(roles[-1], "")

    return {
        "answer": extract_answer(final),
        "raw": final,
        "by_stage": stages_out,
        "telemetry": normalize(langchain_telemetry(result.get("messages") or [])),
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
    """Run the 4-stage LangGraph sequential pipeline on every instance,
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
    print(f"\n=== Analyzer (excerpt) ===\n{out['by_stage'].get('analyzer', '')[:400]}...")
    print(f"\n=== Solver (excerpt) ===\n{out['by_stage'].get('solver', '')[:400]}...")
    print(f"\n=== Critic (excerpt) ===\n{out['by_stage'].get('critic', '')[:400]}...")
    print(f"\n=== Verifier (excerpt) ===\n{out['by_stage'].get('verifier', '')[:400]}...")
    print(f"\n=== Extracted answer: {out['answer']}  (expected: A) ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sequential-topology GPQA-Diamond runner (LangGraph)."
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
    _default_out = (
        _REPO_ROOT / "results" / "gpqa_sequential_r8" / "predictions.jsonl"
    )
    out_path = Path(args.out) if args.out else _default_out
    run_batch(instances, out_path=out_path)
    print(f"  predictions written to {out_path}")
