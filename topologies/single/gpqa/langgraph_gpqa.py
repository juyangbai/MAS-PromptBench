"""Single-agent ReAct topology specialized for GPQA-Diamond."""

# Config
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

from topologies.output_contracts import append_output_contract_from_path

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

# Shared telemetry helper (tokens + rounds). Relative import via the
# top-level `topologies` package; `topologies/__init__.py` exists.
_TOPO_ROOT = str(Path(__file__).resolve().parents[3])
if _TOPO_ROOT not in sys.path:
    sys.path.insert(0, _TOPO_ROOT)
from topologies.telemetry import langchain_telemetry, normalize  # noqa: E402


VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://lai:8001/v1")
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3.5-9B")


# System Prompt
_REPO_ROOT = Path(__file__).resolve().parents[3]
_PROMPT_PATH = _REPO_ROOT / "configs" / "prompts" / "single" / "gpqa" / "solver.txt"
SYSTEM_PROMPT = append_output_contract_from_path(_PROMPT_PATH.read_text().strip(), __file__, _PROMPT_PATH.stem)


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
def format_prompt(question: str, choices: list[str]) -> str:
    """Build the user-facing MCQ prompt. Expects exactly 4 choices."""
    assert len(choices) == 4, "GPQA expects exactly 4 choices."
    letters = ["A", "B", "C", "D"]
    body = "\n".join(f"{letters[i]}) {choices[i]}" for i in range(4))
    return f"{question}\n\n{body}"


def build_agent():
    """Build a ReAct agent with house-default sampling.

    Sampling matches the convention used by sequential/independent/
    centralized/decentralized on Qwen3.5-9B (temp=0.2, top_p=0.9, seed=0,
    repetition_penalty=1.05, enable_thinking=False). Greedy (temp=0) is
    empirically broken on this model; house default is required for
    comparability across topologies.
    """
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
# 3-pattern cascade aligned with sequential/crewai/gpqa, independent/gpqa,
# centralized/autogen/gpqa, decentralized/openai/gpqa. Single was the
# original outlier with only `_ANSWER_RE`; promoted here so real eval
# numbers are comparable across topologies.
#
# `_MARKDOWN_STRIP_RE` pre-cleans `**bold**` and `*italic*` wrappers
# before matching — Qwen3.5-9B frequently emits `"**Answer:** B"` or
# `"correct option is **A**"`, which broke the original strict regex.
_MARKDOWN_STRIP_RE = re.compile(r"[*_`]+")
_ANSWER_RE = re.compile(
    r"\b(?:final\s+)?answer\b\s*[:\s]*\(?([A-D])\)?",
    re.IGNORECASE,
)
_OPTION_RE = re.compile(
    r"\b(?:option|choice)\b\s*(?:is)?\s*[:\s]*\(?([A-D])\)?",
    re.IGNORECASE,
)
_BARE_LETTER_RE = re.compile(
    r"(?:^|\n)\s*\(?([A-D])\)?\s*(?:[.\n]|$)",
    re.MULTILINE,
)


def strip_thinking(text: str) -> str:
    """Cut everything up through the last </think> tag (Qwen3 convention)."""
    index = text.lower().rfind("</think>")
    if index >= 0:
        text = text[index + len("</think>"):]
    return text.strip()


def extract_answer(text: str) -> str | None:
    """Return the MCQ letter from `text`, using a 3-pattern cascade.

    Strips markdown emphasis first so `"**Answer:** B"` matches the same
    as `"Answer: B"`. Cascade order:
      1. strict `Answer: X` / `Final answer: X`
      2. `option X` / `choice X`
      3. bare A-D on its own line
    The LAST match per pattern wins (models often revise earlier letters
    during chain-of-thought).
    """
    cleaned = _MARKDOWN_STRIP_RE.sub("", text)
    for pattern in (_ANSWER_RE, _OPTION_RE, _BARE_LETTER_RE):
        matches = pattern.findall(cleaned)
        if matches:
            return matches[-1].upper()
    return None


# Orchestration
def solve(question: str, choices: list[str], agent=None) -> dict:
    """Run the agent on one GPQA-style MCQ.

    Strips Qwen3's <think>...</think> reasoning from every AI message so both
    the returned `raw` string and the `messages` list are clean.

    Optional `agent` param lets callers reuse a pre-built agent across a
    batch to avoid rebuild-per-instance cost.

    Returns {'answer': 'A'|'B'|'C'|'D'|None, 'raw': str, 'messages': list}.
    """
    if agent is None:
        agent = build_agent()
    prompt = format_prompt(question, choices)
    result = agent.invoke(
        {"messages": [("user", prompt)]},
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
_LETTERS = ["A", "B", "C", "D"]
_HF_DATASET = "Idavidrein/gpqa"
_HF_CONFIG = "gpqa_diamond"
_HF_SPLIT = "train"  # GPQA-Diamond is a single 198-row split


def _stable_row_id(row: dict, fallback_idx: int) -> str:
    """Return a stable id for a GPQA row. The dataset has no canonical id
    field, so we hash the question text. Fallback to the row index."""
    q = (row.get("Question") or "").strip()
    if q:
        import hashlib
        return "gpqa_" + hashlib.md5(q.encode("utf-8")).hexdigest()[:10]
    return f"gpqa_idx_{fallback_idx}"


def load_instances(
    limit: int | None = None,
    offset: int = 0,
    only: list[str] | None = None,
    shuffle_seed: int = 0,
) -> list[dict]:
    """Load GPQA-Diamond rows from HuggingFace and emit topology-ready
    instances with the 4 choices DETERMINISTICALLY shuffled per row.

    The raw HF row has ``"Correct Answer"`` + ``"Incorrect Answer 1..3"``
    as separate fields — if we fed them in that order, the correct letter
    would always be A. We shuffle with a per-row seed (derived from
    `shuffle_seed` + the row's stable id) so runs are reproducible.

    Returns list of dicts:
        {
            "id":             stable row id,
            "question":       raw question text,
            "choices":        shuffled list of 4 choice strings,
            "correct_letter": "A"|"B"|"C"|"D" — which shuffled slot holds
                              the correct answer,
            "raw":            the raw HF row (for auditing),
        }
    """
    from datasets import load_dataset

    ds = load_dataset(_HF_DATASET, _HF_CONFIG)[_HF_SPLIT]
    rows: list[dict] = []
    for i, row in enumerate(ds):
        rid = _stable_row_id(row, i)
        if only is not None and rid not in set(only):
            continue

        correct = (row.get("Correct Answer") or "").strip()
        incorrects = [
            (row.get(f"Incorrect Answer {k}") or "").strip() for k in (1, 2, 3)
        ]
        if not correct or any(not x for x in incorrects):
            # Skip malformed rows rather than silently mis-labeling.
            continue

        four = [correct, *incorrects]
        rng = random.Random(f"{shuffle_seed}|{rid}")
        indices = list(range(4))
        rng.shuffle(indices)
        shuffled = [four[j] for j in indices]
        correct_slot = indices.index(0)  # where did `four[0]` (correct) land
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
    """Run `solve()` on every instance, compare predicted letter vs gold,
    and return an aggregate summary. Optionally writes per-instance
    predictions to `out_path` (JSONL).

    Returns:
        {
            "n":              total instances attempted,
            "n_extracted":    instances where an A/B/C/D was extracted,
            "n_correct":      predictions matching the gold letter,
            "accuracy":       n_correct / n (strict),
            "extracted_acc":  n_correct / n_extracted (excl. non-extractions),
            "per_instance":   list of per-instance dicts,
        }
    """
    agent = build_agent()  # build once; reuse across the batch
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
                out = solve(inst["question"], inst["choices"], agent=agent)
                error = None
            except Exception as e:
                out = {"answer": None, "raw": "", "messages": []}
                error = f"{type(e).__name__}: {e}"
            latency_s = time.time() - t0

            pred = out["answer"]
            gold = inst["correct_letter"]
            is_correct = pred is not None and pred == gold
            if pred is not None:
                n_extracted += 1
            if is_correct:
                n_correct += 1

            telem = normalize(langchain_telemetry(out.get("messages") or []))
            rec = {
                "id": inst["id"],
                "question": inst["question"],
                "choices": inst["choices"],
                "correct_letter": gold,
                "predicted_letter": pred,
                "correct": is_correct,
                "raw": out["raw"],
                "latency_s": round(latency_s, 2),
                **telem,
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
            f"\n=== GPQA-Diamond batch complete ===\n"
            f"  n={summary['n']}  n_extracted={summary['n_extracted']}  "
            f"n_correct={summary['n_correct']}\n"
            f"  accuracy={summary['accuracy']:.3f}  "
            f"extracted_acc={summary['extracted_acc']:.3f}  "
            f"total_s={summary['total_s']}\n"
        )
    return summary


# CLI
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
    print(f"\n=== Extracted answer: {out['answer']} (expected: A) ===\n")
    print("=== Full message trace ===")
    for msg in out["messages"]:
        msg.pretty_print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Single-topology GPQA-Diamond runner (LangGraph)."
    )
    parser.add_argument(
        "--batch", action="store_true",
        help="Run the real GPQA-Diamond eval (else: one canned MCQ demo).",
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="Max rows to run (batch mode).")
    parser.add_argument("--offset", type=int, default=0,
                        help="Row offset into the split (batch mode).")
    parser.add_argument("--shuffle-seed", type=int, default=0,
                        help="Seed for per-row choice shuffling (batch mode).")
    parser.add_argument("--out", type=str, default=None,
                        help="Per-instance JSONL output path (batch mode).")
    parser.add_argument("--only", nargs="*", default=None,
                        help="Restrict to specific instance ids (batch mode).")
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
    summary = run_batch(instances, out_path=out_path)
    if out_path:
        print(f"  predictions written to {out_path}")
