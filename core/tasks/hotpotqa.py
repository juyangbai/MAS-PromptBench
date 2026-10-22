"""HotpotQA: data, Wikipedia tools, answer extraction, scoring and batch records.

The benchmark slice is the hotpot_qa validation split (7,405 multi-hop
questions; the distractor and fullwiki configs share questions and answers, and
distractor is the smaller download). Agents retrieve live from Wikipedia, so the
per-question context paragraphs are never read. Predictions are short-form
answers scored with the official HotpotQA EM and token F1.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from collections.abc import Callable
from functools import partial
from pathlib import Path

import wikipedia

from core import batch

DATASET = "hotpotqa"
HF_DATASET = "hotpot_qa"
HF_CONFIG = "distractor"
HF_SPLIT = "validation"
SOURCE = f"HotpotQA from {HF_DATASET} [{HF_CONFIG}/{HF_SPLIT}]"

# Appended to the single, independent and decentralized prompts: without it the
# model answers in prose, which scores EM=0 even when the reasoning is right.
OUTPUT_FORMAT_NUDGE = (
    "\n\nFINAL OUTPUT FORMAT:\n"
    "After your reasoning, end with a single line exactly of the form:\n"
    "  Answer: <short-form>\n"
    "The short-form MUST be the minimal string needed to answer — typically "
    "1-5 words. Examples of correct short-forms:\n"
    "  - For yes/no questions: 'yes' or 'no' (lowercase, no punctuation).\n"
    "  - For 'when/year' questions: just the year, e.g. '1997'.\n"
    "  - For 'who' questions: the person's full name, e.g. 'Paul McCartney'.\n"
    "  - For 'where/what city' questions: the place name, e.g. 'Paris'.\n"
    "Do NOT include explanations, lists, or sentences on the Answer line. "
    "Do NOT put the answer inside brackets, quotes, or markdown emphasis."
)

# Appended to the centralized manager prompt.
TERMINATE_NUDGE = (
    "\n\nFINAL OUTPUT FORMAT:\n"
    "Synthesize the workers' findings and end YOUR final message with a "
    "single line of the form:\n"
    "  Answer: <short-form>\n"
    "then immediately follow with the literal string TERMINATE on its own "
    "line so the group-chat knows to stop.\n"
    "The short-form MUST be the minimal string needed to answer — typically "
    "1-5 words:\n"
    "  - For yes/no questions: 'yes' or 'no' (lowercase, no punctuation).\n"
    "  - For 'when/year' questions: just the year, e.g. '1997'.\n"
    "  - For 'who' questions: the person's full name, e.g. 'Paul McCartney'.\n"
    "  - For 'where/what city' questions: the place name, e.g. 'Paris'.\n"
    "Do NOT include explanations, lists, or sentences on the Answer line. "
    "Do NOT put the answer inside brackets, quotes, or markdown emphasis."
)

DEMO_QUESTION = "Were Scott Derrickson and Ed Wood of the same nationality?"
DEMO_ANSWER = "yes"


# Data
def load_instances(limit: int | None = None, offset: int = 0, only: list[str] | None = None) -> list[dict]:
    """Load the validation rows as ``{id, question, answer, type, level, raw}``.

    ``id`` is the row's own HotpotQA id; ``type`` is "comparison" or "bridge"
    and ``level`` "easy", "medium" or "hard".
    """
    from datasets import load_dataset

    wanted = None if only is None else set(only)
    rows: list[dict] = []
    for row in load_dataset(HF_DATASET, HF_CONFIG, trust_remote_code=True)[HF_SPLIT]:
        rid = row.get("id")
        if wanted is not None and rid not in wanted:
            continue
        question = (row.get("question") or "").strip()
        answer = (row.get("answer") or "").strip()
        if not question or not answer:
            continue
        rows.append(
            {
                "id": rid,
                "question": question,
                "answer": answer,
                "type": row.get("type"),
                "level": row.get("level"),
                "raw": {k: row.get(k) for k in ("id", "question", "answer", "type", "level")},
            }
        )
    rows = rows[offset:]
    return rows if limit is None else rows[:limit]


def format_prompt(question: str) -> str:
    """User message for one question: the question itself."""
    return question


# Wikipedia tools. A tool's docstring is the description the model sees, so the
# three wordings used by the runners are kept verbatim, indentation included.
SEARCH_DOC = (
    "Search Wikipedia for an article matching the query.\n\n"
    "    Returns titles and short (~2-sentence) summaries of the top matching\n"
    "    articles. Use this first to locate the relevant article, then call\n"
    "    wikipedia_page on its exact title for full details.\n"
    "    "
)
PAGE_DOC = (
    "Return the full text of a Wikipedia article by its exact title.\n\n"
    "    Output is truncated to roughly 4000 characters. Use the exact title\n"
    "    returned by wikipedia_search.\n"
    "    "
)
SEARCH_DOC_CENTRALIZED = (
    "Search Wikipedia for articles matching `query`.\n\n"
    "    Returns titles + short (~2-sentence) summaries for the top matching\n"
    "    articles. Use this first to locate the relevant article, then call\n"
    "    wikipedia_page on its exact title for full details.\n"
    "    "
)
PAGE_DOC_CENTRALIZED = (
    "Return the full text of a Wikipedia article by its exact title.\n\n"
    "    Output is truncated to ~4000 characters. Use the exact title\n"
    "    returned by wikipedia_search.\n"
    "    "
)
SEARCH_DOC_SHORT = "Search Wikipedia; titles + ~2-sentence summaries of top matches."
PAGE_DOC_SHORT = "Full Wikipedia article text by exact title, truncated to ~4000 chars."

# The OpenAI Agents SDK tools: descriptions and parameter schemas.
SEARCH_DESCRIPTION = "Search Wikipedia and return matching article titles with short summaries."
SEARCH_PARAMETERS = {
    "type": "object",
    "properties": {"query": {"type": "string"}, "top_k": {"type": "integer"}},
    "required": ["query"],
}
PAGE_DESCRIPTION = "Read the native bounded text view of an exact Wikipedia article title."
PAGE_PARAMETERS = {"type": "object", "properties": {"title": {"type": "string"}}, "required": ["title"]}

PAGE_CHAR_BUDGET = 4000  # characters of an article a page lookup returns


def search(query: str, top_k: int = 3) -> str:
    """Titles and two-sentence summaries of the top ``top_k`` Wikipedia matches, or an error line."""
    try:
        titles = wikipedia.search(query, results=top_k)
    except Exception as e:
        return f"ERROR: {e}"
    if not titles:
        return f"[no Wikipedia results for '{query}']"
    chunks = []
    for title in titles:
        try:
            summary = wikipedia.summary(title, sentences=2, auto_suggest=False)
            chunks.append(f"- {title}: {summary}")
        except wikipedia.DisambiguationError as e:
            chunks.append(f"- {title}: disambiguation page; options include {e.options[:3]}")
        except wikipedia.PageError:
            chunks.append(f"- {title}: (no page)")
        except Exception as e:
            chunks.append(f"- {title}: error ({e})")
    return "\n".join(chunks)


def read_page(title: str, options_label: str = "options include") -> str:
    """The article titled ``title``, cut at :data:`PAGE_CHAR_BUDGET` characters, or an error line.

    ``options_label`` introduces the candidate titles of a disambiguation page.
    """
    try:
        # ``content`` is a second, lazy request: its failures are tool errors too.
        content = wikipedia.page(title, auto_suggest=False).content
    except wikipedia.DisambiguationError as e:
        return f"ERROR: '{title}' is a disambiguation page; {options_label} {e.options[:5]}"
    except wikipedia.PageError:
        return f"ERROR: no Wikipedia page titled '{title}'"
    except Exception as e:
        return f"ERROR: {e}"
    return content[:PAGE_CHAR_BUDGET] + ("..." if len(content) > PAGE_CHAR_BUDGET else "")


def make_wikipedia_search(doc: str) -> Callable[..., str]:
    """A ``wikipedia_search(query, top_k=3)`` function documented by ``doc``, ready for a framework's tool wrapper."""

    def wikipedia_search(query: str, top_k: int = 3) -> str:
        return search(query, top_k)

    wikipedia_search.__doc__ = doc
    return wikipedia_search


def make_wikipedia_page(doc: str, options_label: str = "options include") -> Callable[[str], str]:
    """A ``wikipedia_page(title)`` function documented by ``doc``, ready for a framework's tool wrapper."""

    def wikipedia_page(title: str) -> str:
        return read_page(title, options_label)

    wikipedia_page.__doc__ = doc
    return wikipedia_page


# Answer extraction
# "Answer: X", "**Answer:** X", "final answer: X", ... (any case) up to the end of the line.
_ANSWER_RE = re.compile(r"\banswer\b\s*(?:is\s+)?[:\s]+\**\s*(.+?)\s*\**\s*(?:\n|$)", re.IGNORECASE)


def extract_answer(text: str) -> str | None:
    """The short-form answer: the last ``Answer: X``, else the last non-empty line, else None."""
    matches = _ANSWER_RE.findall(text)
    if matches:
        return matches[-1].strip().rstrip(".,")
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    return lines[-1] if lines else None


def extract_manager_answer(text: str) -> str | None:
    """:func:`extract_answer` of a centralized manager message, ignoring its ``TERMINATE`` marker."""
    return extract_answer(re.sub(r"\bTERMINATE\b", "", text).strip())


# Scoring: the official HotpotQA normalization, EM and F1 (hotpot_evaluate_v1.py),
# kept as published so scores stay comparable with prior work.
def normalize_answer(s: str) -> str:
    """Lowercase, remove punctuation and articles (a/an/the), collapse whitespace."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = " ".join(s.split())
    return s


def exact_match_score(pred: str, gold: str) -> float:
    """1.0 iff the normalized strings match, else 0.0."""
    return float(normalize_answer(pred) == normalize_answer(gold))


def f1_score(pred: str, gold: str) -> tuple[float, float, float]:
    """Token-level ``(f1, precision, recall)``; a yes/no/noanswer mismatch scores zero."""
    normalized_pred = normalize_answer(pred)
    normalized_gold = normalize_answer(gold)

    zero = (0.0, 0.0, 0.0)
    if normalized_pred in {"yes", "no", "noanswer"} and normalized_pred != normalized_gold:
        return zero
    if normalized_gold in {"yes", "no", "noanswer"} and normalized_pred != normalized_gold:
        return zero

    pred_tokens = normalized_pred.split()
    gold_tokens = normalized_gold.split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return zero
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1, precision, recall


# Aggregation over several agents' answers
def best_of_n(answers: list[str | None]) -> str | None:
    """The first answer of the largest normalized-answer bucket of non-empty answers; ties go to the lowest index."""
    buckets: dict[str, list[str]] = {}
    for answer in answers:
        if answer:
            buckets.setdefault(normalize_answer(answer), []).append(answer)
    if not buckets:
        return None
    return max(buckets.values(), key=len)[0]


def majority_vote(answers: list[dict]) -> str | None:
    """:func:`best_of_n` over the ``answer`` of ``{"answer": ...}`` records."""
    return best_of_n([a.get("answer") for a in answers])


def vote_counts(answers: list[dict]) -> dict[str, int]:
    """Normalized answer -> number of records voting for it (records without an answer abstain)."""
    return dict(Counter(normalize_answer(a["answer"]) for a in answers if a.get("answer")))


# Batch records
def score(pred: str | None, gold: str) -> tuple[float, float, float, float]:
    """``(em, f1, precision, recall)`` of a prediction; a missing prediction scores 0."""
    if pred is None:
        return 0.0, 0.0, 0.0, 0.0
    return exact_match_score(pred, gold), *f1_score(pred, gold)


def record(inst: dict, pred: str | None, **fields) -> dict:
    """Per-instance record: id, question, gold, prediction and scores, then runner ``fields`` in order."""
    gold = inst["answer"]
    em, f1, precision, recall = score(pred, gold)
    return {
        "id": inst["id"],
        "question": inst["question"],
        "gold_answer": gold,
        "predicted_answer": pred,
        "em": em,
        "f1": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        **fields,
    }


def meta(inst: dict) -> dict:
    """The instance's question type and level record fields."""
    return {"type": inst.get("type"), "level": inst.get("level")}


def _f1(rec: dict) -> float:
    """The record's unrounded F1."""
    return score(rec["predicted_answer"], rec["gold_answer"])[1]


def summarize(per_instance: list[dict]) -> dict:
    """Batch scores: n, extracted predictions, summed and mean EM / F1 (overall and on extracted rows)."""
    n = len(per_instance)
    n_extracted = sum(1 for rec in per_instance if rec["predicted_answer"] is not None)
    em_sum = sum((rec["em"] for rec in per_instance), 0.0)
    f1_sum = sum((_f1(rec) for rec in per_instance), 0.0)
    return {
        "n": n,
        "n_extracted": n_extracted,
        "em_sum": em_sum,
        "f1_sum": round(f1_sum, 4),
        "em": (em_sum / n) if n else 0.0,
        "f1": (f1_sum / n) if n else 0.0,
        "extracted_em": (em_sum / n_extracted) if n_extracted else 0.0,
        "extracted_f1": (f1_sum / n_extracted) if n_extracted else 0.0,
    }


UNICODE_MARKS = ("✓", "✗")  # (exact match, wrong answer) marks of a progress line
ASCII_MARKS = ("OK", "X")


def messages_detail(rec: dict) -> str:
    """Progress-line detail of a group chat: its message count."""
    return f"msgs={rec['n_messages']}  "


def peers_detail(rec: dict) -> str:
    """Progress-line detail of a debate: each peer's answer."""
    return f"peers=[{','.join((p['answer'] or '-')[:10] for p in rec['per_peer'])}]  "


def progress_line(
    index: int,
    total: int,
    rec: dict,
    done: list[dict],
    *,
    marks: tuple[str, str] = UNICODE_MARKS,
    width: int = 40,
    detail: Callable[[dict], str] | None = None,
) -> str:
    """One verbose progress line after ``rec`` (``done`` holds the records so far)."""
    em, pred, gold, f1 = rec["em"], rec["predicted_answer"], rec["gold_answer"], _f1(rec)
    running_em = sum((r["em"] for r in done), 0.0) / len(done)
    running_f1 = sum((_f1(r) for r in done), 0.0) / len(done)
    mark = marks[0] if em == 1.0 else ("~" if f1 > 0 else ("?" if pred is None else marks[1]))
    return (
        f"[{index + 1:>3}/{total}] {rec['id']} {mark}  "
        f"em={em:.0f} f1={f1:.2f}  "
        f"pred={(pred or '-')[:width]!r} gold={gold[:width]!r}  "
        f"{detail(rec) if detail else ''}"
        f"EM={running_em:.3f} F1={running_f1:.3f} lat={rec['latency_s']:.1f}s"
    )


def banner(title: str, summary: dict) -> str:
    """End-of-batch report headed ``=== <title> ===``."""
    return (
        f"\n=== {title} ===\n"
        f"  n={summary['n']}  n_extracted={summary['n_extracted']}\n"
        f"  EM={summary['em']:.3f}  F1={summary['f1']:.3f}  "
        f"(on extracted only: EM={summary['extracted_em']:.3f}  "
        f"F1={summary['extracted_f1']:.3f})\n"
        f"  total_s={summary['total_s']}\n"
    )


def run_batch(
    instances: list[dict],
    row: Callable[[int, dict], dict],
    *,
    out_path: Path | None = None,
    verbose: bool = True,
    label: str = "HotpotQA",
    team: str = "",
    marks: tuple[str, str] = UNICODE_MARKS,
    width: int = 40,
    detail: Callable[[dict], str] | None = None,
) -> dict:
    """:func:`core.batch.run_batch` with the HotpotQA summary, progress lines and report.

    The report is headed ``<label> batch complete <team>``; ``marks``, ``width``
    and ``detail`` shape the progress lines (:func:`progress_line`).
    """
    title = f"{label} batch complete" + (f" {team}" if team else "")
    return batch.run_batch(
        instances,
        row,
        summarize=summarize,
        out_path=out_path,
        verbose=verbose,
        progress=partial(progress_line, marks=marks, width=width, detail=detail),
        banner=partial(banner, title),
    )


def print_demo_answer(answer: str | None, label: str = "Extracted answer") -> None:
    """Print the demo's answer and its scores against :data:`DEMO_ANSWER`."""
    print(f"\n=== {label}: {answer!r}  (expected: {DEMO_ANSWER!r}) ===")
    if answer is not None:
        em = exact_match_score(answer, DEMO_ANSWER)
        f1, precision, recall = f1_score(answer, DEMO_ANSWER)
        print(f"=== EM: {em:.2f}   F1: {f1:.2f}   P: {precision:.2f}   R: {recall:.2f} ===")
