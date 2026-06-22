"""Independent topology specialized for HotpotQA."""

# Config
from __future__ import annotations

import argparse
import asyncio
import json
import operator
import os
import re
import string
import sys
import time
from collections import Counter
from pathlib import Path

from topologies.output_contracts import append_output_contract_from_path
from typing import Annotated

import wikipedia
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
N_AGENTS = int(os.environ.get("INDEPENDENT_N_AGENTS", "4"))

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PROMPT_PATH = (
    _REPO_ROOT / "configs" / "prompts" / "independent" / "hotpotqa" / "solver.txt"
)

# Same answer-format nudge as single/hotpotqa — without it, Qwen3.5-9B
# emits verbose prose instead of the minimal short-form HotpotQA's
# official scorer needs.
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
_PAGE_CHAR_BUDGET = 4000


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
            chunks.append(
                f"- {title}: disambiguation page; options include {e.options[:3]}"
            )
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


def _build_one_agent(seed: int):
    """Build one replica's react agent, seeded differently from its siblings.

    Same model/tools/prompt across replicas; only the seed varies.
    """
    llm = ChatOpenAI(
        model=MODEL_ID,
        base_url=VLLM_BASE_URL,
        api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        max_tokens=4096,
        # House default sampling. Greedy (temp=0) would collapse all N
        # replicas to identical behavior, defeating the ensemble.
        temperature=0.2,
        top_p=0.9,
        seed=seed,
        extra_body={
            "repetition_penalty": 1.05,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    return create_react_agent(
        model=llm,
        tools=[wikipedia_search, wikipedia_page],
        prompt=SYSTEM_PROMPT,
    )


# Output Parsing
# Primary: "Answer: X", "The answer is X", "**Answer:** X" (until newline/EOS).
_ANSWER_RE = re.compile(
    r"\banswer\b\s*(?:is\s+)?[:\s]+\**\s*(.+?)\s*\**\s*(?:\n|$)",
    re.IGNORECASE,
)


def strip_thinking(text: str) -> str:
    """Cut everything up through the last </think> tag (Qwen3 convention)."""
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
# Verbatim HotpotQA normalization + EM + F1 from hotpot_evaluate_v1.py,
# matching topologies/single/hotpotqa/langgraph_hotpotqa.py byte-for-byte
# so ensemble-level numbers remain comparable to single-topology numbers.
def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = " ".join(s.split())
    return s


def exact_match_score(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def f1_score(pred: str, gold: str) -> tuple[float, float, float]:
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


# Aggregation
def majority_vote(answers: list[dict]) -> str | None:
    """Majority vote over per-replica answers after HotpotQA normalization.

    Each answer is bucketed by its normalized form; the most-common bucket
    wins. The returned string is the RAW answer text of the first replica
    in that bucket (preserves case/punctuation for downstream EM/F1, which
    re-normalizes anyway).
    """
    valid = [a for a in answers if a.get("answer")]
    if not valid:
        return None
    buckets: dict[str, list[dict]] = {}
    for a in valid:
        key = normalize_answer(a["answer"])
        buckets.setdefault(key, []).append(a)
    best_key = max(buckets, key=lambda k: len(buckets[k]))
    return buckets[best_key][0]["answer"]


# Graph
class State(TypedDict):
    question: str
    prompt: str
    answers: Annotated[list[dict], operator.add]


class AgentInput(TypedDict):
    agent_id: int
    seed: int
    prompt: str


# Stall safeguards
# Per-row wall-clock cap; without this, a hard bridge question can let
# 4 concurrent agents chew >15 min (seen in practice: 1068s max row).
# The recursion limit stays below the old 25-turn default but is configurable
# for msg sweeps, where too-low limits showed up as scored error rows.
PER_ROW_TIMEOUT_S = int(os.environ.get("HOTPOTQA_INDEPENDENT_ROW_TIMEOUT_S", "120"))
_RECURSION_LIMIT = int(os.environ.get("HOTPOTQA_INDEPENDENT_RECURSION_LIMIT", "20"))


async def _run_replica(inp: AgentInput) -> dict:
    """Run one replica's react agent and return its extracted answer."""
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
def solve(question: str) -> dict:
    """Run the ensemble on one HotpotQA question.

    Returns:
        {
            "answer":    majority-vote string answer (raw form), or None,
            "per_agent": list of {agent_id, seed, answer, raw, messages},
            "votes":     Counter of normalized-answer -> count,
        }
    """
    compiled = build_graph().compile()
    prompt = format_prompt(question)

    async def _run():
        return await asyncio.wait_for(
            compiled.ainvoke({"question": question, "prompt": prompt, "answers": []}),
            timeout=PER_ROW_TIMEOUT_S,
        )

    result = asyncio.run(_run())
    per_agent = sorted(result["answers"], key=lambda a: a["agent_id"])
    votes = Counter(
        normalize_answer(a["answer"]) for a in per_agent if a.get("answer")
    )
    return {
        "answer": majority_vote(per_agent),
        "per_agent": per_agent,
        "votes": dict(votes),
    }


# Dataset loader (same rows as single/hotpotqa by offset/limit)
_HF_DATASET = "hotpot_qa"
_HF_CONFIG = "distractor"
_HF_SPLIT = "validation"


def load_instances(
    limit: int | None = None,
    offset: int = 0,
    only: list[str] | None = None,
) -> list[dict]:
    """Load HotpotQA dev rows. HotpotQA has stable string ids per row;
    the first 100 rows at offset=0 are the same questions used by
    single/hotpotqa so per-row comparisons line up across topologies.
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
    _propagate_errors: bool = False,
) -> dict:
    """Run the N-replica ensemble on every instance, aggregate via
    normalized-majority, compute EM + F1 vs gold, return aggregate
    summary + optionally write per-instance predictions to JSONL.

    Per-instance record shape:
        {id, question, gold_answer, predicted_answer (majority-vote),
         em, f1, precision, recall, votes (normalized form -> count),
         per_agent: [{agent_id, seed, answer, raw}],  # omit messages list
         type, level, latency_s, error}
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
                if _propagate_errors:
                    raise
                out = {"answer": None, "per_agent": [], "votes": {}}
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

            compact_per_agent = [
                {"agent_id": a["agent_id"], "seed": a["seed"],
                 "answer": a["answer"], "raw": a["raw"]}
                for a in out.get("per_agent") or []
            ]
            telem = normalize(langchain_ensemble_telemetry(out.get("per_agent") or []))
            rec_out = {
                "id": inst["id"],
                "question": inst["question"],
                "gold_answer": gold,
                "predicted_answer": pred,
                "em": em,
                "f1": round(f1, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "votes": out.get("votes") or {},
                "per_agent": compact_per_agent,
                **telem,
                "type": inst.get("type"),
                "level": inst.get("level"),
                "latency_s": round(latency_s, 2),
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
            f"\n=== independent/HotpotQA batch complete (N={N_AGENTS}) ===\n"
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
    print(f"\n=== Ensemble ({N_AGENTS} replicas) normalized votes: {out['votes']}")
    print(f"=== Majority-vote answer: {out['answer']!r}  (expected: {expected!r}) ===")
    if out["answer"] is not None:
        em = exact_match_score(out["answer"], expected)
        f1, precision, recall = f1_score(out["answer"], expected)
        print(f"=== EM: {em:.2f}   F1: {f1:.2f}   P: {precision:.2f}   R: {recall:.2f} ===\n")
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
        description="Independent-topology HotpotQA runner (LangGraph ensemble)."
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
