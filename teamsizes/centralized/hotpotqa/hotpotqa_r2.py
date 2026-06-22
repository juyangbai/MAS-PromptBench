"""Centralized topology specialized for HotpotQA, LangGraph."""

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

from teamsizes.output_contracts import append_output_contract_from_path
from typing import Annotated, Optional

from typing_extensions import TypedDict

import wikipedia
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, create_react_agent

# Shared telemetry.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_TOPO_ROOT = str(_REPO_ROOT)
if _TOPO_ROOT not in sys.path:
    sys.path.insert(0, _TOPO_ROOT)
from topologies.telemetry import langchain_telemetry, normalize  # noqa: E402


VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://n12:8000/v1")
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3.5-9B")

_PROMPTS_DIR = _REPO_ROOT / "configs" / "prompts" / "centralized" / "hotpotqa"

# Same cap as AutoGen sibling's MaxMessageTermination(18). Centralized
# HotpotQA needs room for manager planning + multi-hop retrieval +
# reasoning + final answer.
MAX_TURNS = 9


def _load_prompt(role: str) -> str:
    return append_output_contract_from_path((_PROMPTS_DIR / f"{role}.txt").read_text().strip(), __file__, role)


# Tools
_PAGE_CHAR_BUDGET = 4000  # cap per page so retrieval results stay context-cheap


@tool
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


@tool
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


WIKI_TOOLS = [wikipedia_search, wikipedia_page]


# Delegation tools (routing markers)
# The manager "calls" these to hand the floor to a specific worker. The
# body echoes the instructions, producing a ToolMessage the worker can
# read as context. The router after `manager_tools` inspects the name
# to route to the right worker node.
@tool("delegate_to_retriever_worker")
def delegate_to_retriever_worker(instructions: str) -> str:
    """Hand the next turn to the retriever_worker. Use this when you need
    a single Wikipedia lookup (search or page fetch) performed and the
    results returned as structured facts.

    Args:
        instructions: what you want the retriever to look up this turn.
    """
    return instructions




DELEGATION_TOOLS = [
    delegate_to_retriever_worker,
]
DELEGATION_NAMES = {t.name for t in DELEGATION_TOOLS}

MANAGER_TOOLS = WIKI_TOOLS + DELEGATION_TOOLS


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


# Manager nudge (short-form answer + TERMINATE contract)
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
    "Do NOT put the answer inside brackets, quotes, or markdown emphasis.\n\n"
    "Delegation: when you want a specific worker to act, call the "
    "matching delegate_to_<worker> tool with clear instructions "
    "(instead of merely addressing them in free-form text). The three "
    "workers are: retriever_worker, reasoner_worker, writer_worker."
)


# State + nodes
class CentralizedState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    turn_count: int


def _tag_source(msg: BaseMessage, source: str) -> None:
    """Attach an AutoGen-style `source` field via additional_kwargs."""
    try:
        kw = dict(getattr(msg, "additional_kwargs", None) or {})
        kw["source"] = source
        msg.additional_kwargs = kw
    except Exception:
        pass


def _manager_system() -> str:
    return _load_prompt("manager") + _MANAGER_TERMINATE_NUDGE


def _manager_node(state: CentralizedState) -> dict:
    llm = _build_llm().bind_tools(MANAGER_TOOLS)
    sys_msg = SystemMessage(content=_manager_system())
    ai = llm.invoke([sys_msg] + state["messages"])
    # AutoGen messages carry a `.source` name; we mimic that on the
    # AIMessage via additional_kwargs for trace rendering parity.
    _tag_source(ai, "manager")
    return {"messages": [ai], "turn_count": int(state.get("turn_count", 0)) + 1}


_manager_tool_node = ToolNode(MANAGER_TOOLS)


def _route_from_manager(state: CentralizedState) -> str:
    msgs = state["messages"]
    if not msgs:
        return "manager"
    last = msgs[-1]
    if int(state.get("turn_count", 0)) >= MAX_TURNS:
        return END
    if isinstance(last, AIMessage):
        content = last.content or ""
        if isinstance(content, str) and "TERMINATE" in content:
            return END
        if getattr(last, "tool_calls", None):
            return "manager_tools"
    # No tool call, no TERMINATE — loop back and let the manager try again.
    return "manager"


def _route_from_manager_tools(state: CentralizedState) -> str:
    # Find the most recent AIMessage with tool_calls; its tool_calls tell
    # us whether any delegation was requested.
    for m in reversed(state["messages"]):
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            for tc in m.tool_calls:
                name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
                if name in DELEGATION_NAMES:
                    return name.removeprefix("delegate_to_")
            # Only real-tool calls were made; loop back to manager.
            return "manager"
    return "manager"


def _make_worker_node(name: str, tools: list, llm: ChatOpenAI):
    sys_prompt = _load_prompt(name)
    agent = create_react_agent(model=llm, tools=tools, prompt=sys_prompt)

    def node(state: CentralizedState) -> dict:
        # create_react_agent returns {"messages": [full history incl. input]},
        # so we splice out only the new messages it appended.
        prior = list(state["messages"])
        result = agent.invoke(
            {"messages": prior},
            config={"recursion_limit": 30},
        )
        full = result["messages"]
        new_msgs = full[len(prior):]
        # Tag worker outputs with source for trace rendering.
        for m in new_msgs:
            if isinstance(m, AIMessage):
                _tag_source(m, name)
        # Each new AIMessage counts as one turn; tool-executions don't.
        n_turns = sum(1 for m in new_msgs if isinstance(m, AIMessage))
        return {"messages": new_msgs, "turn_count": int(state.get("turn_count", 0)) + n_turns}

    return node


def _build_graph(llm: Optional[ChatOpenAI] = None):
    if llm is None:
        llm = _build_llm()

    graph = StateGraph(CentralizedState)
    graph.add_node("manager", _manager_node)
    graph.add_node("manager_tools", _manager_tool_node)

    worker_specs = [
        ("retriever_worker", WIKI_TOOLS),
    ]
    for name, tools in worker_specs:
        graph.add_node(name, _make_worker_node(name, tools, llm))

    graph.add_edge(START, "manager")
    graph.add_conditional_edges(
        "manager",
        _route_from_manager,
        {
            "manager_tools": "manager_tools",
            "manager": "manager",
            END: END,
        },
    )
    graph.add_conditional_edges(
        "manager_tools",
        _route_from_manager_tools,
        {
            "retriever_worker": "retriever_worker",
            "manager": "manager",
        },
    )
    for name, _ in worker_specs:
        graph.add_edge(name, "manager")

    return graph.compile(), ["manager"] + [n for n, _ in worker_specs]


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
def _communications_source(m: BaseMessage) -> str:
    kw = getattr(m, "additional_kwargs", None) or {}
    src = kw.get("source")
    if src:
        return src
    t = getattr(m, "type", None)
    return {"human": "user", "ai": "assistant", "tool": "tool"}.get(t, t or "?")


def _communications_to_record(m: BaseMessage) -> dict:
    content = getattr(m, "content", "") or ""
    if not isinstance(content, str):
        content = str(content)
    return {"source": _communications_source(m), "content": content}


def solve(question: str) -> dict:
    """Run the centralized team on one HotpotQA question.

    Returns:
        {
            "answer":   short-form answer (str) or None,
            "raw":      manager's last message content,
            "messages": list of {source, content} from every turn,
            "telemetry": normalized 5-key token/call counts,
        }
    """
    compiled, _ = _build_graph()
    result = compiled.invoke(
        {"messages": [HumanMessage(content=question)], "turn_count": 0},
        config={"recursion_limit": MAX_TURNS * 4},
    )
    msgs = result.get("messages") or []
    rendered = [_communications_to_record(m) for m in msgs]

    manager_msgs = [r for r in rendered if r["source"] == "manager"]
    final = manager_msgs[-1]["content"] if manager_msgs else ""
    answer = extract_answer(final)
    if answer is None:
        for r in reversed(rendered):
            a = extract_answer(r["content"] or "")
            if a is not None:
                answer = a
                break
    return {
        "answer": answer,
        "raw": final,
        "messages": rendered,
        "telemetry": normalize(langchain_telemetry(msgs)),
    }


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
                out = {"answer": None, "raw": "", "messages": [], "telemetry": {}}
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
                mark = "OK" if em == 1.0 else ("~" if f1 > 0 else ("?" if pred is None else "X"))
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
        description="Centralized-topology HotpotQA runner (LangGraph manager/worker)."
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
    _default_out = str(
        Path(__file__).resolve().parents[3] / "results" / "hotpotqa_centralized_r2" / "predictions.jsonl"
    )
    out_path = Path(args.out) if args.out else Path(_default_out)
    run_batch(instances, out_path=out_path)
    if out_path:
        print(f"  predictions written to {out_path}")
