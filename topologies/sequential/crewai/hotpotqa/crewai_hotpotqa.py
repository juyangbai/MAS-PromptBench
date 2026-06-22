"""Sequential topology specialized for HotpotQA, implemented in CrewAI."""

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
_PROMPTS_DIR = _REPO_ROOT / "configs" / "prompts" / "sequential" / "hotpotqa"


def _load_prompt(role: str) -> str:
    return append_output_contract_from_path((_PROMPTS_DIR / f"{role}.txt").read_text().strip(), __file__, role)


# Tools
_PAGE_CHAR_BUDGET = 4000   # cap per page to keep context small


@tool("wikipedia_search")
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


@tool("wikipedia_page")
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


# LLM
def _build_llm() -> LLM:
    """CrewAI routes completions through litellm; `openai/<model>` + api_base
    points it at our local vLLM OpenAI-compatible endpoint."""
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
    """Build the 3-stage retriever -> reasoner -> writer pipeline."""
    if llm is None:
        llm = _build_llm()

    planner = Agent(
        role="HotpotQA Planner",
        goal=(
            "Identify the entities in the question and plan the 2-3 "
            "hop Wikipedia search strategy BEFORE any retrieval happens. "
            "Do NOT retrieve."
        ),
        backstory=_load_prompt("planner"),
        tools=[],  # planning only
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    retriever = Agent(
        role="HotpotQA Retriever",
        goal=(
            "Execute the Planner's queries via wikipedia_search and "
            "wikipedia_page; return a structured fact dossier."
        ),
        backstory=_load_prompt("retriever"),
        tools=[wikipedia_search, wikipedia_page],
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    reasoner = Agent(
        role="HotpotQA Reasoner",
        goal=(
            "Chain facts from the retrieved articles into a logical path "
            "to the answer."
        ),
        backstory=_load_prompt("reasoner"),
        tools=[],  # no tools; reasons over retriever's output
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    writer = Agent(
        role="HotpotQA Writer",
        goal="Emit the concise final short-form answer.",
        backstory=_load_prompt("writer"),
        tools=[],  # no tools; produces the final answer string
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    plan_task = Task(
        description=(
            "Read the HotpotQA question and plan the search strategy. "
            "Identify the key entities, decide which Wikipedia pages "
            "should be consulted and in what order, and what facts from "
            "each page would settle the question. Do NOT retrieve — the "
            "Retriever handles that.\n\n"
            "QUESTION:\n{question}"
        ),
        expected_output=(
            "A short plan: (1) key entities, (2) ordered list of "
            "Wikipedia searches to perform, (3) what fact each search "
            "should return."
        ),
        agent=planner,
    )

    retrieve_task = Task(
        description=(
            "Execute the Planner's search plan via `wikipedia_search` "
            "and `wikipedia_page`. Extract verbatim the facts the "
            "Planner flagged as load-bearing. Do NOT commit to a final "
            "answer.\n\n"
            "QUESTION:\n{question}"
        ),
        expected_output=(
            "A structured dossier: a list of the Wikipedia article titles "
            "consulted and, under each title, 1-3 verbatim facts that "
            "bear on the question."
        ),
        agent=retriever,
        context=[plan_task],
    )

    reason_task = Task(
        description=(
            "Using the Retriever's dossier, reason step by step from the "
            "facts to the answer. Make the logical chain explicit: "
            "'Fact A says X; Fact B says Y; therefore ...'. Do NOT emit "
            "the final answer string yet.\n\n"
            "QUESTION:\n{question}"
        ),
        expected_output=(
            "A short chain-of-reasoning paragraph that derives the answer "
            "from the Retriever's facts."
        ),
        agent=reasoner,
        context=[plan_task, retrieve_task],
    )

    write_task = Task(
        description=(
            "Given the Reasoner's derivation, emit the final short-form "
            "answer. HotpotQA answers are typically 1-5 words "
            "(an entity name, a year, a yes/no, a number). Your output "
            "MUST end with a line matching 'Answer: <short form>'.\n\n"
            "QUESTION:\n{question}"
        ),
        expected_output=(
            "A one-line answer formatted as 'Answer: <short form>'."
        ),
        agent=writer,
        context=[plan_task, retrieve_task, reason_task],
    )

    return Crew(
        agents=[planner, retriever, reasoner, writer],
        tasks=[plan_task, retrieve_task, reason_task, write_task],
        process=Process.sequential,
        verbose=False,
    )


# Output Parsing
# Matches "Answer: X", "The answer is X", "**Answer:** X"; aligned
# to single/hotpotqa.
_ANSWER_RE = re.compile(
    r"\banswer\b\s*(?:is\s+)?[:\s]+\**\s*(.+?)\s*\**\s*(?:\n|$)",
    re.IGNORECASE,
)


def extract_answer(text: str) -> str | None:
    """Return the writer's short-form answer.

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
# aligned to topologies/single/hotpotqa/langgraph_hotpotqa.py so
# sequential-topology numbers are directly comparable.
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


# Orchestration
def solve(question: str) -> dict:
    """Run the 3-stage sequential crew on one HotpotQA question.

    Returns:
        {
            "answer":   final short-form answer string or None,
            "raw":      writer's full output text,
            "by_stage": {retriever, reasoner, writer} -> each stage's output,
        }
    """
    crew = build_crew()
    result = crew.kickoff(inputs={"question": question})

    final = result.raw
    stages = {}
    try:
        stages["planner"]   = result.tasks_output[0].raw
        stages["retriever"] = result.tasks_output[1].raw
        stages["reasoner"]  = result.tasks_output[2].raw
        stages["writer"]    = result.tasks_output[3].raw
    except (AttributeError, IndexError):
        stages = {"planner": "", "retriever": "", "reasoner": "", "writer": final}

    return {
        "answer": extract_answer(final),
        "raw": final,
        "by_stage": stages,
        "telemetry": normalize(crewai_telemetry(crew, n_stages=len(stages))),
    }


# Dataset loader (same row ids as single/independent hotpotqa)
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
    single/hotpotqa and independent/hotpotqa for cross-topology parity.
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
    """Run the 4-stage CrewAI pipeline on every instance, compute EM + F1
    vs gold, return aggregate summary + optionally write per-instance
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
                out = {"answer": None, "raw": "", "by_stage": {}}
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

            # Short per-stage excerpts keep the JSONL compact while still
            # surfacing how the pipeline reasoned (useful for debugging).
            by_stage = out.get("by_stage") or {}
            excerpts = {k: (v or "")[:800] for k, v in by_stage.items()}
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
                "by_stage": excerpts,
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
            f"\n=== sequential/HotpotQA batch complete ===\n"
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
    print(f"\n=== Planner (excerpt) ===\n{out['by_stage']['planner'][:400]}...")
    print(f"\n=== Retriever (excerpt) ===\n{out['by_stage']['retriever'][:400]}...")
    print(f"\n=== Reasoner (excerpt) ===\n{out['by_stage']['reasoner'][:400]}...")
    print(f"\n=== Writer (excerpt) ===\n{out['by_stage']['writer'][:400]}...")
    print(f"\n=== Extracted answer: {out['answer']!r}  (expected: {expected!r}) ===")
    if out["answer"] is not None:
        em = exact_match_score(out["answer"], expected)
        f1, precision, recall = f1_score(out["answer"], expected)
        print(f"=== EM: {em:.2f}   F1: {f1:.2f}   P: {precision:.2f}   R: {recall:.2f} ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sequential-topology HotpotQA runner (CrewAI)."
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
