"""Centralized topology specialized for HotpotQA, AutoGen."""

# Config
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import string
import sys
import time
from collections import Counter
from pathlib import Path

from topologies.output_contracts import append_output_contract_from_path
from typing import Sequence

import wikipedia
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage
from autogen_agentchat.teams import SelectorGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Shared telemetry.
_TOPO_ROOT = str(Path(__file__).resolve().parents[4])
if _TOPO_ROOT not in sys.path:
    sys.path.insert(0, _TOPO_ROOT)
from topologies.telemetry import autogen_telemetry, normalize  # noqa: E402


VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://lai:8001/v1")
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3.5-9B")

_REPO_ROOT = Path(__file__).resolve().parents[4]
_PROMPTS_DIR = _REPO_ROOT / "configs" / "prompts" / "centralized" / "hotpotqa"


def _load_prompt(role: str) -> str:
    return append_output_contract_from_path((_PROMPTS_DIR / f"{role}.txt").read_text().strip(), __file__, role)


# Tools
_PAGE_CHAR_BUDGET = 4000  # cap per page so retrieval results stay context-cheap


def wikipedia_search(query: str, top_k: int = 3) -> str:
    """Search Wikipedia for articles matching `query`.

    Returns titles + short (~2-sentence) summaries for the top matching
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


def wikipedia_page(title: str) -> str:
    """Return the full text of a Wikipedia article by its exact title.

    Output is truncated to ~4000 characters. Use the exact title
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


# LLM
def _build_client() -> OpenAIChatCompletionClient:
    """Build an OpenAI-compatible client pointed at our local vLLM.

    Qwen3-specific sampling params (repetition_penalty + enable_thinking=
    False) are threaded via `extra_body` — without enable_thinking=False
    the 9B burns its token budget inside a <think> block.
    """
    return OpenAIChatCompletionClient(
        model=MODEL_ID,
        base_url=VLLM_BASE_URL,
        api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        model_info={
            "vision": False,
            "function_calling": True,
            "json_output": True,
            "family": "qwen",
            "structured_output": False,
        },
        temperature=0.2,
        top_p=0.9,
        seed=0,
        max_tokens=2048,
        extra_body={
            "repetition_penalty": 1.05,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )


# Team
# Same short-form format guidance used in single/hotpotqa and
# independent/hotpotqa. Without it the 9B manager emits verbose
# confirmations like "Yes, both were American" which score EM=0.
_MANAGER_TERMINATE_NUDGE = (
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


def build_team() -> SelectorGroupChat:
    client = _build_client()

    manager = AssistantAgent(
        "manager",
        description="Coordinator that plans the multi-hop retrieval, delegates, and writes the final answer.",
        model_client=client,
        system_message=_load_prompt("manager") + _MANAGER_TERMINATE_NUDGE,
        tools=[wikipedia_search, wikipedia_page],
    )

    retriever_worker = AssistantAgent(
        "retriever_worker",
        description="Performs one Wikipedia lookup per manager instruction and returns structured facts.",
        model_client=client,
        system_message=_load_prompt("retriever_worker"),
        tools=[wikipedia_search, wikipedia_page],
    )

    reasoner_worker = AssistantAgent(
        "reasoner_worker",
        description="Chains retrieved facts into a logical derivation per manager instruction.",
        model_client=client,
        system_message=_load_prompt("reasoner_worker"),
        tools=[wikipedia_search, wikipedia_page],
    )

    writer_worker = AssistantAgent(
        "writer_worker",
        description="Formats the final short-form answer per manager instruction.",
        model_client=client,
        system_message=_load_prompt("writer_worker"),
        tools=[wikipedia_search, wikipedia_page],
    )

    # Force manager-routing: after any worker speaks, the manager MUST be
    # the next speaker (so workers never chain turns with each other).
    # When the last message is already the manager's, let the
    # SelectorGroupChat's default LLM-based selector pick the next worker
    # (or return None to end the conversation).
    def _selector_func(messages: Sequence[BaseAgentEvent | BaseChatMessage]) -> str | None:
        if not messages:
            return manager.name
        if messages[-1].source != manager.name:
            return manager.name
        return None

    selector_prompt = (
        "You are coordinating a 4-agent team on a multi-hop Wikipedia question.\n"
        "Select the next agent to act.\n\n{roles}\n\n"
        "Conversation so far:\n{history}\n\n"
        "Pick exactly one agent from {participants}."
    )

    termination = TextMentionTermination("TERMINATE") | MaxMessageTermination(_MAX_MESSAGES)

    return SelectorGroupChat(
        [manager, retriever_worker, reasoner_worker, writer_worker],
        model_client=client,
        termination_condition=termination,
        selector_prompt=selector_prompt,
        selector_func=_selector_func,
        allow_repeated_speaker=True,
    )


# Stall safeguards
# Per-row wall-clock cap + tighter MaxMessageTermination. Without these
# the manager/retriever/reasoner loop can spiral on hard bridge questions,
# with the retriever re-fetching pages via Wikipedia tools across many
# turns, burning >3 min per row.
PER_ROW_TIMEOUT_S = 120
_MAX_MESSAGES = 18  # was 24


# Output parsing
# Matches "Answer: X", "The answer is X", "**Answer:** X", etc.
_ANSWER_RE = re.compile(
    r"\banswer\b\s*(?:is\s+)?[:\s]+\**\s*(.+?)\s*\**\s*(?:\n|$)",
    re.IGNORECASE,
)


def extract_answer(text: str) -> str | None:
    """Return the model's short-form answer from the manager's final message.

    Prefers the last 'Answer: X' pattern. Falls back to the last non-empty
    line of the cleaned text (matches single/hotpotqa's behavior).
    """
    # Drop any trailing TERMINATE so it doesn't pollute the fallback path.
    text = re.sub(r"\bTERMINATE\b", "", text).strip()
    matches = _ANSWER_RE.findall(text)
    if matches:
        return matches[-1].strip().rstrip(".,")
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    return lines[-1] if lines else None


# Scoring (aligned to single/hotpotqa, HotpotQA official)
def normalize_answer(s: str) -> str:
    """HotpotQA normalization (verbatim from hotpot_evaluate_v1.py).

    Lowercase, remove punctuation, remove articles (a/an/the), fix
    whitespace.
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
async def solve_async(question: str) -> dict:
    """Run the centralized team on one HotpotQA question.

    Returns:
        {
            "answer":   short-form answer (str) or None,
            "raw":      manager's last message content,
            "messages": list of {source, content} from every turn,
        }
    """
    team = build_team()
    result = await asyncio.wait_for(team.run(task=question), timeout=PER_ROW_TIMEOUT_S)
    messages = [
        {
            "source": getattr(m, "source", None),
            "content": getattr(m, "content", None)
            if isinstance(getattr(m, "content", None), str)
            else str(getattr(m, "content", "")),
        }
        for m in result.messages
    ]
    manager_msgs = [m for m in messages if m["source"] == "manager"]
    final = manager_msgs[-1]["content"] if manager_msgs else ""
    return {
        "answer": extract_answer(final),
        "raw": final,
        "messages": messages,
        "telemetry": normalize(autogen_telemetry(result)),
    }


def solve(question: str) -> dict:
    return asyncio.run(solve_async(question))


# Dataset loader (same row ids as single/indep/seq hotpotqa)
_HF_DATASET = "hotpot_qa"
_HF_CONFIG = "distractor"
_HF_SPLIT = "validation"


def load_instances(
    limit: int | None = None,
    offset: int = 0,
    only: list[str] | None = None,
) -> list[dict]:
    """Load HotpotQA dev rows. HotpotQA has stable string ids per row;
    the first 100 rows at offset=0 are the same questions used by the
    other hotpotqa topologies for cross-topology parity."""
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
    """Run the centralized team on every instance, compute EM + F1 vs
    gold, return aggregate summary + optionally write per-instance
    predictions to JSONL.
    """
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
                out = solve(inst["question"])
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
                "n_messages": len(out.get("messages") or []),
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
                running_f1 = f1_sum / (i + 1)
                mark = "✓" if em == 1.0 else ("~" if f1 > 0 else ("?" if pred is None else "✗"))
                pred_disp = (pred or "-")[:40]
                gold_disp = gold[:40]
                print(
                    f"[{i + 1:>3}/{n}] {inst['id']} {mark}  "
                    f"em={em:.0f} f1={f1:.2f}  "
                    f"pred={pred_disp!r} gold={gold_disp!r}  "
                    f"msgs={rec_out['n_messages']}  "
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
            f"\n=== centralized/HotpotQA batch complete ===\n"
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
        print(f"=== EM: {em:.2f}   F1: {f1:.2f}   P: {precision:.2f}   R: {recall:.2f} ===")
    print(f"=== {len(out['messages'])} messages across the group chat ===")
    for m in out["messages"]:
        snippet = m["content"][:300].replace("\n", " ")
        print(f"  [{m['source']}] {snippet}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Centralized-topology HotpotQA runner (AutoGen)."
    )
    parser.add_argument(
        "--batch", action="store_true",
        help="Run the real HotpotQA eval (else: one canned demo).",
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
        f"loading HotpotQA from {_HF_DATASET} [{_HF_CONFIG}/{_HF_SPLIT}] "
        f"(base_url={VLLM_BASE_URL}) ..."
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
