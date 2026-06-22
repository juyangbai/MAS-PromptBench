"""Decentralized debate topology specialized for HotpotQA, OpenAI SDK."""

# Config
from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import signal
import string
import sys
import time
from collections import Counter
from pathlib import Path

from topologies.output_contracts import append_output_contract_from_path

import wikipedia
from openai import OpenAI

# Shared telemetry.
_TOPO_ROOT = str(Path(__file__).resolve().parents[4])
if _TOPO_ROOT not in sys.path:
    sys.path.insert(0, _TOPO_ROOT)
from topologies.telemetry import (  # noqa: E402
    openai_sdk_telemetry, openai_sdk_accumulate, normalize,
)


_TELEM_ACC: dict = {
    "prompt_tokens": 0, "completion_tokens": 0,
    "total_tokens": 0, "n_llm_calls": 0, "n_tool_calls": 0,
}


def _reset_telem_acc() -> None:
    for k in _TELEM_ACC:
        _TELEM_ACC[k] = 0


# Stall safeguards
# Per-row wall-clock cap using SIGALRM when running on the main thread. The
# concurrent runner may execute run_batch([row]) inside a ThreadPool worker,
# where Python disallows signal handlers, so the guard becomes a no-op there.
# Hard bridge questions can make peers chain Wikipedia tool calls for many
# minutes.
PER_ROW_TIMEOUT_S = 120
_MAX_TOOL_LOOPS = 4  # was 6


class _RowTimeout(Exception):
    """Raised when a single row exceeds PER_ROW_TIMEOUT_S."""


def _row_timeout_handler(signum, frame):
    raise _RowTimeout(f"row exceeded {PER_ROW_TIMEOUT_S}s")


@contextlib.contextmanager
def _row_timeout_guard(seconds: int):
    """Install SIGALRM for `seconds`; uninstall on exit regardless of
    outcome so timeouts in one row don't bleed into the next."""
    import threading
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    old = signal.signal(signal.SIGALRM, _row_timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://lai:8001/v1")
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3.5-9B")

N_AGENTS = int(os.environ.get("DECENTRALIZED_N_AGENTS", "4"))
N_ROUNDS = int(os.environ.get("DECENTRALIZED_N_ROUNDS", "2"))

_PAGE_CHAR_BUDGET = 4000

_REPO_ROOT = Path(__file__).resolve().parents[4]
_PROMPTS_DIR = _REPO_ROOT / "configs" / "prompts" / "decentralized" / "hotpotqa"


def _load_prompt(role: str) -> str:
    return append_output_contract_from_path((_PROMPTS_DIR / f"{role}.txt").read_text().strip(), __file__, role)


# Same short-form format nudge used in single/independent/centralized
# hotpotqa. Without it Qwen3.5-9B peers emit verbose prose ("Yes, both
# were American") that scores EM=0 on yes/no questions.
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

SYSTEM_PROMPT = _load_prompt("debater") + _OUTPUT_FORMAT_NUDGE


# Tools
def wikipedia_search(query: str, top_k: int = 3) -> str:
    """Search Wikipedia; titles + ~2-sentence summaries of top matches."""
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
    """Full Wikipedia article text by exact title, truncated to ~4000 chars."""
    try:
        page = wikipedia.page(title, auto_suggest=False)
    except wikipedia.DisambiguationError as e:
        return f"ERROR: '{title}' is a disambiguation page; options: {e.options[:5]}"
    except wikipedia.PageError:
        return f"ERROR: no Wikipedia page titled '{title}'"
    except Exception as e:
        return f"ERROR: {e}"
    content = page.content
    return content[:_PAGE_CHAR_BUDGET] + ("..." if len(content) > _PAGE_CHAR_BUDGET else "")


_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "wikipedia_search",
            "description": "Search Wikipedia; returns titles + 2-sentence summaries.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "default": 3},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wikipedia_page",
            "description": "Full Wikipedia article text by exact title.",
            "parameters": {
                "type": "object",
                "properties": {"title": {"type": "string"}},
                "required": ["title"],
            },
        },
    },
]


def _dispatch_tool(name: str, arguments: dict) -> str:
    if name == "wikipedia_search":
        return wikipedia_search(arguments.get("query", ""), arguments.get("top_k", 3))
    if name == "wikipedia_page":
        return wikipedia_page(arguments.get("title", ""))
    return f"ERROR: unknown tool {name!r}"


# Client
def _build_client() -> OpenAI:
    return OpenAI(base_url=VLLM_BASE_URL, api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"))


def _completion_kwargs() -> dict:
    return {
        "model": MODEL_ID,
        "temperature": 0.2,
        "top_p": 0.9,
        "seed": 0,
        "max_tokens": 2048,
        "extra_body": {
            "repetition_penalty": 1.05,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    }


def _chat_with_tools(client: OpenAI, messages: list[dict], max_tool_loops: int = _MAX_TOOL_LOOPS) -> dict:
    kwargs = _completion_kwargs()
    kwargs["tools"] = _TOOL_SCHEMAS
    kwargs["tool_choice"] = "auto"
    dump: dict = {}
    for _ in range(max_tool_loops):
        resp = client.chat.completions.create(messages=messages, **kwargs)
        openai_sdk_accumulate(_TELEM_ACC, resp)
        msg = resp.choices[0].message
        dump = msg.model_dump() if hasattr(msg, "model_dump") else dict(msg)
        tool_calls = dump.get("tool_calls") or []
        messages.append(dump)
        if not tool_calls:
            return dump
        for tc in tool_calls:
            fn = tc.get("function", {}) if isinstance(tc, dict) else {}
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            result = _dispatch_tool(fn.get("name", ""), args)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.get("id"),
                "content": result,
            })
    return dump


# Debate loop
def _peer_injection(others_final: list[dict], question: str) -> dict:
    body = ["These are the final responses from other peer agents in the previous round:"]
    for i, m in enumerate(others_final):
        body.append(f"\nPeer {i + 1}:\n```\n{m.get('content') or ''}\n```")
    body.append(
        "\nCompare their reasoning + Wikipedia evidence against your own. "
        "Revise your answer ONLY if a peer cites concretely stronger "
        "evidence. Re-emit a single `Answer: <short-form>` line at the "
        "end.\n\nOriginal question:\n" + question
    )
    return {"role": "user", "content": "\n".join(body)}


def run_debate(question: str) -> list[list[dict]]:
    client = _build_client()
    contexts: list[list[dict]] = [
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        for _ in range(N_AGENTS)
    ]
    round_finals: list[list[dict]] = []
    for r in range(N_ROUNDS):
        this_round: list[dict] = []
        for i, ctx in enumerate(contexts):
            if r > 0:
                others = [round_finals[r - 1][j] for j in range(N_AGENTS) if j != i]
                ctx.append(_peer_injection(others, question))
            final_msg = _chat_with_tools(client, ctx)
            this_round.append(final_msg)
        round_finals.append(this_round)
    return contexts


# Output parsing
_ANSWER_RE = re.compile(
    r"\banswer\b\s*(?:is\s+)?[:\s]+\**\s*(.+?)\s*\**\s*(?:\n|$)",
    re.IGNORECASE,
)


def extract_answer(text: str) -> str | None:
    matches = _ANSWER_RE.findall(text)
    if matches:
        return matches[-1].strip().rstrip(".,")
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    return lines[-1] if lines else None


# Scoring (aligned to single/hotpotqa)
def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = " ".join(s.split())
    return s


def exact_match_score(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def f1_score(pred: str, gold: str) -> tuple[float, float, float]:
    np_pred = normalize_answer(pred)
    np_gold = normalize_answer(gold)
    zero = (0.0, 0.0, 0.0)
    if np_pred in {"yes", "no", "noanswer"} and np_pred != np_gold:
        return zero
    if np_gold in {"yes", "no", "noanswer"} and np_pred != np_gold:
        return zero
    pred_tokens = np_pred.split()
    gold_tokens = np_gold.split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return zero
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1, precision, recall


# Aggregation (aligned to independent/hotpotqa)
def bucket_majority(answers: list[str]) -> str | None:
    """Majority over normalized buckets; return the RAW form of the first
    peer in the winning bucket (preserves capitalization for EM/F1)."""
    valid = [a for a in answers if a]
    if not valid:
        return None
    buckets: list[list[str]] = []
    for a in valid:
        na = normalize_answer(a)
        for b in buckets:
            if normalize_answer(b[0]) == na:
                b.append(a)
                break
        else:
            buckets.append([a])
    # max() returns first bucket of max len — first peer to land in that
    # bucket keeps insertion order, so tie-break by lowest peer index.
    best = max(buckets, key=len)
    return best[0]


# Orchestration
def solve(question: str) -> dict:
    """Run N-peer × R-round debate on one HotpotQA question.

    Returns:
        {
            "answer":       round-R bucket-majority answer (str) or None,
            "per_peer":     [{peer, answer, raw}],
            "all_contexts": raw OpenAI chat contexts (one per peer),
        }
    """
    _reset_telem_acc()
    with _row_timeout_guard(PER_ROW_TIMEOUT_S):
        contexts = run_debate(question)
    per_peer = []
    answers = []
    for i, ctx in enumerate(contexts):
        final = ctx[-1].get("content") or ""
        ans = extract_answer(final)
        per_peer.append({"peer": i, "answer": ans, "raw": final})
        if ans:
            answers.append(ans)
    telem = dict(_TELEM_ACC)
    if telem["n_llm_calls"] == 0:
        telem = openai_sdk_telemetry(contexts)
    return {
        "answer": bucket_majority(answers),
        "per_peer": per_peer,
        "all_contexts": contexts,
        "telemetry": normalize(telem),
    }


# Dataset loader (same row ids as other hotpotqa topologies)
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
    other 4 hotpotqa topologies for cross-topology parity."""
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
    """Run the N-peer × R-round debate on every instance, bucket-majority
    → short-form answer, compute EM + F1 vs gold, return aggregate
    summary + optionally write per-instance predictions to JSONL.
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
                out = {"answer": None, "per_peer": [], "all_contexts": []}
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

            compact_per_peer = [
                {
                    "peer": p["peer"], "answer": p["answer"],
                    "raw_tail": (p["raw"] or "")[-300:],
                }
                for p in out.get("per_peer") or []
            ]
            rec_out = {
                "id": inst["id"],
                "question": inst["question"],
                "gold_answer": gold,
                "predicted_answer": pred,
                "em": em,
                "f1": round(f1, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "per_peer": compact_per_peer,
                "type": inst.get("type"),
                "level": inst.get("level"),
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
                peer_ans = [(p["answer"] or "-")[:10] for p in compact_per_peer]
                pred_disp = (pred or "-")[:30]
                gold_disp = gold[:30]
                print(
                    f"[{i + 1:>3}/{n}] {inst['id']} {mark}  "
                    f"em={em:.0f} f1={f1:.2f}  "
                    f"pred={pred_disp!r} gold={gold_disp!r}  "
                    f"peers=[{','.join(peer_ans)}]  "
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
            f"\n=== decentralized/HotpotQA batch complete "
            f"(N={N_AGENTS} peers × R={N_ROUNDS} rounds) ===\n"
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
    print(f"\n=== Debate: N={N_AGENTS} peers × R={N_ROUNDS} rounds ===")
    for p in out["per_peer"]:
        ans = p["answer"] or "(none)"
        print(f"  peer {p['peer']}: answer={ans!r}")
    print(f"\n=== Majority-vote final answer: {out['answer']!r}  (expected: {expected!r}) ===")
    if out["answer"]:
        em = exact_match_score(out["answer"], expected)
        f1, precision, recall = f1_score(out["answer"], expected)
        print(f"=== EM: {em:.2f}   F1: {f1:.2f}   P: {precision:.2f}   R: {recall:.2f} ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Decentralized-topology HotpotQA runner (OpenAI SDK debate)."
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
        f"(base_url={VLLM_BASE_URL}, N={N_AGENTS} peers × R={N_ROUNDS} rounds) ..."
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
