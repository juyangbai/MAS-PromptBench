"""Competition MATH: data, answer extraction, scoring and batch records.

The benchmark slice is the Precalculus / Level 5 subset of qwedsacf/competition_math
(312 problems). The gold answer is the last ``\\boxed{...}`` of the reference
solution; predictions are compared with Hendrycks' MATH equivalence.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable
from functools import partial
from pathlib import Path

from core import batch

logger = logging.getLogger(__name__)

DATASET = "math"
HF_DATASET = "qwedsacf/competition_math"
HF_SPLIT = "train"
SUBJECT = "Precalculus"
LEVEL = "Level 5"
SOURCE = f"MATH from {HF_DATASET} [{HF_SPLIT}], {SUBJECT} / {LEVEL}"
LABEL = f"MATH {SUBJECT} {LEVEL}"  # the slice's name in the runners' batch reports

# Appended to the single and independent solver prompts.
OUTPUT_FORMAT_NUDGE = (
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

# Appended to the centralized manager prompt.
TERMINATE_NUDGE = (
    "\n\nWhen you emit the final \\boxed{...} answer, immediately follow "
    "it with the literal string TERMINATE on its own line so the group-"
    "chat knows to stop."
)

DEMO_PROBLEM = "Compute the value of $\\frac{7!}{5!}$. Put your final answer inside \\boxed{}."
DEMO_ANSWER = "42"


# Data
def load_instances(limit: int | None = None, offset: int = 0, only: list[str] | None = None) -> list[dict]:
    """Load the Precalculus / Level 5 rows as ``{id, problem, answer, subject, level, raw}``.

    ``id`` is ``math_`` plus the first 10 hex digits of the MD5 of the problem
    text, so ids are stable across runners.
    """
    from datasets import load_dataset

    rows: list[dict] = []
    for row in load_dataset(HF_DATASET)[HF_SPLIT]:
        if row.get("type") != SUBJECT or row.get("level") != LEVEL:
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
        rows.append(
            {
                "id": rid,
                "problem": problem,
                "answer": gold,
                "subject": row.get("type"),
                "level": row.get("level"),
                "raw": {"problem": problem, "solution": solution, "type": row.get("type"), "level": row.get("level")},
            }
        )
    rows = rows[offset:]
    return rows if limit is None else rows[:limit]


def format_prompt(problem: str) -> str:
    """User message for one problem: the problem statement itself."""
    return problem


# Answer extraction
def extract_boxed(text: str) -> str | None:
    """Inner content of the last ``\\boxed{...}`` (nested braces counted), or None."""
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
    """The model's boxed final answer, or None."""
    return extract_boxed(text)


# Scoring: Hendrycks' math_equivalence.py (NeurIPS 2021), the community-standard
# MATH scorer (https://github.com/hendrycks/math/blob/main/modeling/math_equivalence.py).
# Kept as published so scores stay comparable with prior work.
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
                except AssertionError:
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
        assert string == f"{a}/{b}"
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except (AssertionError, ValueError):
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
    """Hendrycks MATH equivalence: equal after LaTeX normalization."""
    if str1 is None and str2 is None:
        logger.warning("is_equiv: both answers are None")
        return True
    if str1 is None or str2 is None:
        return False
    try:
        ss1 = _strip_string(str1)
        ss2 = _strip_string(str2)
        if verbose:
            logger.info("is_equiv: %s vs %s", ss1, ss2)
        return ss1 == ss2
    except Exception:
        return str1 == str2


def exact_match_score(pred: str, gold: str) -> float:
    """``is_equiv`` as a float score."""
    return float(is_equiv(pred, gold))


# Aggregation over several agents' answers
def equivalence_buckets(items: list, answer: Callable = lambda item: item) -> list[list]:
    """Group ``items`` whose answers are equivalent to a bucket's first answer, in first-seen order."""
    buckets: list[list] = []
    for item in items:
        for bucket in buckets:
            if is_equiv(answer(item), answer(bucket[0])):
                bucket.append(item)
                break
        else:
            buckets.append([item])
    return buckets


def majority_vote(answers: list[dict]) -> str | None:
    """Answer of the largest equivalence bucket over ``{"answer": ...}`` records.

    Returns the raw text of the bucket's first answer; ties go to the bucket
    seen first (lowest agent index).
    """
    valid = [a for a in answers if a.get("answer") is not None]
    if not valid:
        return None
    best = max(equivalence_buckets(valid, lambda a: a["answer"]), key=len)
    return best[0]["answer"]


def best_of_n(answers: list[str | None]) -> str | None:
    """Majority over equivalence buckets of non-empty answers; ties go to the lowest index."""
    valid = [a for a in answers if a]
    if not valid:
        return None
    return max(equivalence_buckets(valid), key=len)[0]


# Batch records
def score(pred: str | None, gold: str) -> float:
    """Exact match of a prediction; a missing prediction scores 0."""
    return exact_match_score(pred, gold) if pred is not None else 0.0


def record(inst: dict, pred: str | None, **fields) -> dict:
    """Per-instance record: id, problem, gold, prediction and score, then runner ``fields`` in order."""
    gold = inst["answer"]
    return {
        "id": inst["id"],
        "problem": inst["problem"],
        "gold_answer": gold,
        "predicted_answer": pred,
        "em": score(pred, gold),
        **fields,
    }


def meta(inst: dict) -> dict:
    """The instance's subject and level record fields."""
    return {"subject": inst.get("subject"), "level": inst.get("level")}


def summarize(per_instance: list[dict]) -> dict:
    """Batch scores: n, extracted predictions, summed and mean EM (overall and on extracted rows)."""
    n = len(per_instance)
    n_extracted = sum(1 for rec in per_instance if rec["predicted_answer"] is not None)
    em_sum = sum((rec["em"] for rec in per_instance), 0.0)
    return {
        "n": n,
        "n_extracted": n_extracted,
        "em_sum": em_sum,
        "em": (em_sum / n) if n else 0.0,
        "extracted_em": (em_sum / n_extracted) if n_extracted else 0.0,
    }


def progress_line(index: int, total: int, rec: dict, done: list[dict]) -> str:
    """One verbose progress line after ``rec`` (``done`` holds the records so far)."""
    em, pred, gold = rec["em"], rec["predicted_answer"], rec["gold_answer"]
    running_em = sum(r["em"] for r in done) / len(done)
    mark = "✓" if em == 1.0 else ("?" if pred is None else "✗")
    return (
        f"[{index + 1:>3}/{total}] {rec['id'][:30]:<30} {mark}  "
        f"em={em:.0f}  pred={(pred or '-')[:30]!r} gold={(gold or '-')[:30]!r}  "
        f"EM={running_em:.3f} lat={rec['latency_s']:.1f}s"
    )


def banner(label: str, summary: dict) -> str:
    """End-of-batch report."""
    return (
        f"\n=== {label} batch complete ===\n"
        f"  n={summary['n']}  n_extracted={summary['n_extracted']}\n"
        f"  EM={summary['em']:.3f}  (on extracted only: {summary['extracted_em']:.3f})\n"
        f"  total_s={summary['total_s']}\n"
    )


def run_batch(
    instances: list[dict],
    row: Callable[[int, dict], dict],
    *,
    out_path: Path | None = None,
    verbose: bool = True,
    label: str = "MATH",
) -> dict:
    """:func:`core.batch.run_batch` with the MATH summary, progress lines and report."""
    return batch.run_batch(
        instances,
        row,
        summarize=summarize,
        out_path=out_path,
        verbose=verbose,
        progress=progress_line,
        banner=partial(banner, label),
    )


def print_demo_answer(answer: str | None) -> None:
    """Print the demo's extracted answer and its score against :data:`DEMO_ANSWER`."""
    print(f"\n=== Extracted boxed answer: {answer!r}  (expected: {DEMO_ANSWER!r}) ===")
    if answer is not None:
        print(f"=== EM (Hendrycks is_equiv): {exact_match_score(answer, DEMO_ANSWER):.2f} ===")
