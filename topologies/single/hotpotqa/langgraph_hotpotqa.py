"""Single-agent ReAct topology specialized for HotpotQA.

Open-domain multi-hop question answering. Uses real Wikipedia retrieval
(HotpotQA was built from Wikipedia, so Wikipedia is the canonical source).
Two tools let the agent search for candidate articles, then read the one
it needs:
    wikipedia_search(query)  -> titles + short summaries
    wikipedia_page(title)    -> full article text (truncated)

Requires: pip install wikipedia
The system prompt is loaded from configs/prompts/single/hotpotqa/solver.txt.
"""

# Config
from __future__ import annotations

import argparse
import json
import os
import re
import string
import sys
import time
from collections import Counter
from pathlib import Path

from topologies.output_contracts import append_output_contract_from_path

import wikipedia
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

# Shared telemetry (tokens + rounds).
_TOPO_ROOT = str(Path(__file__).resolve().parents[3])
if _TOPO_ROOT not in sys.path:
    sys.path.insert(0, _TOPO_ROOT)
from topologies.telemetry import langchain_telemetry, normalize  # noqa: E402


VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://lai:8001/v1")
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3.5-9B")


# System Prompt
_REPO_ROOT = Path(__file__).resolve().parents[3]
_PROMPT_PATH = _REPO_ROOT / "configs" / "prompts" / "single" / "hotpotqa" / "solver.txt"

# The generated solver.txt describes the multi-hop retrieval procedure but
# does NOT constrain the output format. HotpotQA's official scorer expects
# SHORT-form answers (e.g., "yes", "no", "1997", "Paris") and compares
# after normalize_answer. Without an explicit format constraint the 9B
# emits prose like "Both individuals were of the same nationality:
# American", which scores EM=0 even when the reasoning is correct. This
# nudge is appended at load time — the prompt file on disk is unchanged.
_OUTPUT_FORMAT_NUDGE = (
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

SYSTEM_PROMPT = append_output_contract_from_path(_PROMPT_PATH.read_text().strip() + _OUTPUT_FORMAT_NUDGE, __file__, _PROMPT_PATH.stem)


# Tools
_PAGE_CHAR_BUDGET = 4000   # cap per page to keep context small


@tool
def wikipedia_search(query: str, top_k: int = 3) -> str:
    """Search Wikipedia for an article matching the query.

    Returns titles and short (~2-sentence) summaries of the top matching
    articles. Use this first to locate the relevant article, then call
    wikipedia_page on its exact title for full details.
    """
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


@tool
def wikipedia_page(title: str) -> str:
    """Return the full text of a Wikipedia article by its exact title.

    Output is truncated to roughly 4000 characters. Use the exact title
    returned by wikipedia_search.
    """
    try:
        page = wikipedia.page(title, auto_suggest=False)
    except wikipedia.DisambiguationError as e:
        return f"ERROR: '{title}' is a disambiguation page; options include {e.options[:5]}"
    except wikipedia.PageError:
        return f"ERROR: no Wikipedia page titled '{title}'"
    except Exception as e:
        return f"ERROR: {e}"
    content = page.content
    return content[:_PAGE_CHAR_BUDGET] + ("..." if len(content) > _PAGE_CHAR_BUDGET else "")


# Agent
def format_prompt(question: str) -> str:
    """Build the user-facing prompt for one HotpotQA question."""
    return question


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
    return create_react_agent(model=llm, tools=[wikipedia_search, wikipedia_page], prompt=SYSTEM_PROMPT)


# Output Parsing
# Matches "Answer: X", "The answer is X", "**Answer:** X", etc.
_ANSWER_RE = re.compile(
    r"\banswer\b\s*(?:is\s+)?[:\s]+\**\s*(.+?)\s*\**\s*(?:\n|$)",
    re.IGNORECASE,
)


def strip_thinking(text: str) -> str:
    """Cut everything up through the last </think> tag."""
    index = text.lower().rfind("</think>")
    if index >= 0:
        text = text[index + len("</think>"):]
    return text.strip()


def extract_answer(text: str) -> str | None:
    """Return the model's short-form answer.

    Prefers the last 'Answer: X' pattern. Falls back to the last non-empty
    line of the cleaned text.
    """
    matches = _ANSWER_RE.findall(text)
    if matches:
        return matches[-1].strip().rstrip(".,")
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    return lines[-1] if lines else None


# Scoring
def normalize_answer(s: str) -> str:
    """HotpotQA normalization (verbatim from hotpot_evaluate_v1.py).

    Lowercase, remove punctuation, remove articles (a/an/the), fix whitespace.
    """
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = " ".join(s.split())
    return s


def exact_match_score(pred: str, gold: str) -> float:
    """HotpotQA EM (official): 1.0 iff normalized strings match, else 0.0."""
    return float(normalize_answer(pred) == normalize_answer(gold))


def f1_score(pred: str, gold: str) -> tuple[float, float, float]:
    """HotpotQA token-level F1 (official): returns (f1, precision, recall).

    For yes/no/noanswer questions, a non-matching prediction scores (0, 0, 0)
    — no partial credit from token overlap.
    """
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


# Orchestration
def solve(question: str, agent=None) -> dict:
    """Run the agent on one HotpotQA question.

    Strips Qwen3's <think>...</think> reasoning from every AI message so both
    the returned `raw` string and the `messages` list are clean.

    Optional `agent` param lets callers reuse a pre-built agent across a
    batch to avoid rebuild-per-instance cost.

    Returns {'answer': str | None, 'raw': str, 'messages': list}.
    """
    if agent is None:
        agent = build_agent()
    result = agent.invoke(
        {"messages": [("user", format_prompt(question))]},
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
_HF_DATASET = "hotpot_qa"
# `distractor` and `fullwiki` have the SAME 7,405 dev questions + same gold
# answers — only the per-question context paragraphs differ. Our agent
# retrieves live from Wikipedia via the `wikipedia` package, so the
# paragraphs in the HF record are never consumed. `distractor` is the
# smaller/faster download.
_HF_CONFIG = "distractor"
_HF_SPLIT = "validation"


def load_instances(
    limit: int | None = None,
    offset: int = 0,
    only: list[str] | None = None,
) -> list[dict]:
    """Load HotpotQA dev rows and emit topology-ready instances.

    HotpotQA rows already have stable string ids (`row["id"]`), so we use
    them directly — no hash-based row-id scheme needed like GPQA. The
    first 100 rows (offset=0, limit=100) are deterministic across
    topologies for direct comparison.

    Returns list of dicts:
        {
            "id":       str (HF row id),
            "question": str,
            "answer":   gold short-form answer (str),
            "type":     "comparison" | "bridge",
            "level":    "easy" | "medium" | "hard",
            "raw":      full HF row (for auditing).
        }
    """
    from datasets import load_dataset

    ds = load_dataset(_HF_DATASET, _HF_CONFIG, trust_remote_code=True)[_HF_SPLIT]
    rows: list[dict] = []
    for row in ds:
        rid = row.get("id")
        if only is not None and rid not in set(only):
            continue
        q = (row.get("question") or "").strip()
        a = (row.get("answer") or "").strip()
        if not q or not a:
            continue
        rows.append({
            "id": rid,
            "question": q,
            "answer": a,
            "type": row.get("type"),
            "level": row.get("level"),
            "raw": {k: row.get(k) for k in ("id", "question", "answer", "type", "level")},
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
    """Run `solve()` on every instance, compute EM + F1 vs gold, return
    aggregate summary + optionally write per-instance predictions to
    JSONL.

    Returns:
        {
            "n":         total attempted,
            "n_extracted": non-null predictions,
            "em_sum":    sum of EM over all instances,
            "f1_sum":    sum of F1 over all instances,
            "em":        em_sum / n (0 for extraction-fail),
            "f1":        f1_sum / n,
            "extracted_em": em_sum / n_extracted,
            "extracted_f1": f1_sum / n_extracted,
            "per_instance": list of per-row dicts,
        }
    """
    agent = build_agent()  # build once; reuse across the batch
    per_instance: list[dict] = []
    n = len(instances)
    em_sum = 0.0
    f1_sum = 0.0
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
                out = solve(inst["question"], agent=agent)
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
                f1, prec, rec = f1_score(pred, gold)
            else:
                em = 0.0
                f1 = prec = rec = 0.0
            em_sum += em
            f1_sum += f1

            telem = normalize(langchain_telemetry(out.get("messages") or []))
            rec_out = {
                "id": inst["id"],
                "question": inst["question"],
                "gold_answer": gold,
                "predicted_answer": pred,
                "em": em,
                "f1": round(f1, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "type": inst.get("type"),
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
                running_f1 = f1_sum / (i + 1)
                mark = "✓" if em == 1.0 else ("~" if f1 > 0 else ("?" if pred is None else "✗"))
                pred_disp = (pred or "-")[:40]
                gold_disp = gold[:40]
                print(
                    f"[{i + 1:>3}/{n}] {inst['id']} {mark}  "
                    f"em={em:.0f} f1={f1:.2f}  "
                    f"pred={pred_disp!r} gold={gold_disp!r}  "
                    f"EM={running_em:.3f} F1={running_f1:.3f} lat={latency_s:.1f}s",
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
        "f1_sum": round(f1_sum, 4),
        "em": (em_sum / n) if n else 0.0,
        "f1": (f1_sum / n) if n else 0.0,
        "extracted_em": (em_sum / n_extracted) if n_extracted else 0.0,
        "extracted_f1": (f1_sum / n_extracted) if n_extracted else 0.0,
        "total_s": round(elapsed, 1),
        "per_instance": per_instance,
    }
    if verbose:
        print(
            f"\n=== HotpotQA batch complete ===\n"
            f"  n={summary['n']}  n_extracted={summary['n_extracted']}\n"
            f"  EM={summary['em']:.3f}  F1={summary['f1']:.3f}  "
            f"(on extracted only: EM={summary['extracted_em']:.3f}  "
            f"F1={summary['extracted_f1']:.3f})\n"
            f"  total_s={summary['total_s']}\n"
        )
    return summary


# Demo
def _canned_demo() -> None:
    question = "Were Scott Derrickson and Ed Wood of the same nationality?"
    expected = "yes"
    out = solve(question)
    print(f"\n=== Extracted answer: {out['answer']!r}  (expected: {expected!r}) ===")
    if out["answer"] is not None:
        em = exact_match_score(out["answer"], expected)
        f1, precision, recall = f1_score(out["answer"], expected)
        print(f"=== EM: {em:.2f}   F1: {f1:.2f}   P: {precision:.2f}   R: {recall:.2f} ===\n")
    print("=== Full message trace ===")
    for msg in out["messages"]:
        msg.pretty_print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Single-topology HotpotQA runner (LangGraph)."
    )
    parser.add_argument(
        "--batch", action="store_true",
        help="Run the real HotpotQA eval (else: one canned demo).",
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="Max rows to run (batch mode).")
    parser.add_argument("--offset", type=int, default=0,
                        help="Row offset into the split (batch mode).")
    parser.add_argument("--out", type=str, default=None,
                        help="Per-instance JSONL output path (batch mode).")
    parser.add_argument("--only", nargs="*", default=None,
                        help="Restrict to specific HotpotQA row ids.")
    args = parser.parse_args()

    if not args.batch:
        _canned_demo()
        sys.exit(0)

    print(f"loading HotpotQA from {_HF_DATASET} [{_HF_CONFIG}/{_HF_SPLIT}] ...")
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
