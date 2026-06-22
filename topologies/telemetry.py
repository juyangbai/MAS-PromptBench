"""Per-framework telemetry extractors for topology runners."""
from __future__ import annotations

from typing import Any, Iterable


_ZERO = {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
    "n_llm_calls": 0,
    "n_tool_calls": 0,
}


# LangChain (single + independent topologies)
def _ai_token_usage(msg) -> dict:
    """Pull token usage off one LangChain AIMessage.

    LangChain's ChatOpenAI sets both `response_metadata["token_usage"]`
    and `usage_metadata` on AIMessage. The schemas differ slightly; we
    probe response_metadata first (what `invoke()` returns) then fall
    back to usage_metadata (what streaming exposes).
    """
    rm = getattr(msg, "response_metadata", None) or {}
    usage = rm.get("token_usage") or rm.get("usage") or {}
    if usage:
        return {
            "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
            "total_tokens": int(
                usage.get("total_tokens", 0)
                or (usage.get("prompt_tokens", 0) or 0)
                + (usage.get("completion_tokens", 0) or 0)
            ),
        }
    um = getattr(msg, "usage_metadata", None) or {}
    if um:
        return {
            "prompt_tokens": int(um.get("input_tokens", 0) or 0),
            "completion_tokens": int(um.get("output_tokens", 0) or 0),
            "total_tokens": int(
                um.get("total_tokens", 0)
                or (um.get("input_tokens", 0) or 0) + (um.get("output_tokens", 0) or 0)
            ),
        }
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def langchain_telemetry(messages: Iterable) -> dict:
    """Aggregate per-row telemetry for a LangChain message list.

    Sums token usage over all AIMessages and counts LLM calls + tool
    calls. Works for both `agent.invoke()["messages"]` and the flattened
    `result["messages"]` that LangGraph's react agent returns.
    """
    telem = dict(_ZERO)
    for msg in messages or []:
        mtype = getattr(msg, "type", None)
        if mtype == "ai":
            telem["n_llm_calls"] += 1
            tc = getattr(msg, "tool_calls", None) or []
            telem["n_tool_calls"] += len(tc)
            u = _ai_token_usage(msg)
            telem["prompt_tokens"] += u["prompt_tokens"]
            telem["completion_tokens"] += u["completion_tokens"]
            telem["total_tokens"] += u["total_tokens"]
    return telem


def langchain_ensemble_telemetry(per_agent: Iterable[dict]) -> dict:
    """Aggregate telemetry across the N replicas of an independent ensemble.

    Each replica dict must carry a `messages` list (LangGraph result's
    messages). Returns the ensemble TOTAL (sums, not averages) — so
    `n_llm_calls` on a row reflects work across all 4 agents combined.
    """
    total = dict(_ZERO)
    for rep in per_agent or []:
        t = langchain_telemetry(rep.get("messages") or [])
        for k in _ZERO:
            total[k] += t[k]
    return total


# CrewAI (sequential topology)
def crewai_telemetry(crew_or_metrics, n_stages: int | None = None) -> dict:
    """Pull usage off a CrewAI Crew (after kickoff) or its UsageMetrics.

    CrewAI aggregates token usage at the crew level, not per task, so
    `crew.usage_metrics` is authoritative. `successful_requests` gives
    n_llm_calls. Tool-call count isn't tracked by CrewAI; we leave it 0
    and let callers override with a hand-counted value if they care.
    """
    um = (
        getattr(crew_or_metrics, "usage_metrics", None)
        if hasattr(crew_or_metrics, "usage_metrics")
        else crew_or_metrics
    )
    if um is None:
        out = dict(_ZERO)
        if n_stages is not None:
            out["n_llm_calls"] = int(n_stages)
        return out

    def _get(obj, name, default=0):
        # UsageMetrics is a pydantic BaseModel in crewai; fall back to dict.
        return getattr(obj, name, None) or (
            obj.get(name, default) if isinstance(obj, dict) else default
        ) or default

    prompt = _get(um, "prompt_tokens", 0)
    completion = _get(um, "completion_tokens", 0)
    total = _get(um, "total_tokens", 0) or (prompt + completion)
    n_calls = _get(um, "successful_requests", 0) or (n_stages or 0)
    return {
        "prompt_tokens": int(prompt),
        "completion_tokens": int(completion),
        "total_tokens": int(total),
        "n_llm_calls": int(n_calls),
        "n_tool_calls": 0,
    }


# AutoGen (centralized topology)
def autogen_telemetry(task_result) -> dict:
    """Aggregate telemetry across all messages of an AutoGen TaskResult.

    AutoGen attaches `models_usage` (RequestUsage) to each agent message;
    user-sim messages and tool-result messages have None. One
    `models_usage` present ≈ one LLM call.
    """
    telem = dict(_ZERO)
    for m in getattr(task_result, "messages", None) or []:
        usage = getattr(m, "models_usage", None)
        if usage is None:
            continue
        telem["n_llm_calls"] += 1
        telem["prompt_tokens"] += int(getattr(usage, "prompt_tokens", 0) or 0)
        telem["completion_tokens"] += int(getattr(usage, "completion_tokens", 0) or 0)
    telem["total_tokens"] = telem["prompt_tokens"] + telem["completion_tokens"]

    # Count tool-call message types to approximate n_tool_calls. AutoGen
    # emits ToolCallRequestEvent / ToolCallSummaryMessage-like types; we
    # detect by class name to avoid a hard dep on the import location.
    for m in getattr(task_result, "messages", None) or []:
        cls = type(m).__name__
        if "ToolCall" in cls and "Request" in cls:
            telem["n_tool_calls"] += 1
    return telem


# OpenAI SDK (decentralized topology)
def openai_sdk_accumulate(acc: dict, response) -> None:
    """In-place accumulate usage from one openai response into `acc`.

    Decentralized runners make many `client.chat.completions.create`
    calls per row; wrap each call's response through this function to
    keep a running total. Call-site pattern:

        telem = dict(_ZERO)
        ...
        resp = client.chat.completions.create(...)
        openai_sdk_accumulate(telem, resp)

    If `acc` doesn't have the keys yet we seed them here.
    """
    for k, v in _ZERO.items():
        acc.setdefault(k, v)
    usage = getattr(response, "usage", None)
    if usage is None:
        return
    acc["n_llm_calls"] += 1
    acc["prompt_tokens"] += int(getattr(usage, "prompt_tokens", 0) or 0)
    acc["completion_tokens"] += int(getattr(usage, "completion_tokens", 0) or 0)
    acc["total_tokens"] += int(
        getattr(usage, "total_tokens", 0)
        or (getattr(usage, "prompt_tokens", 0) or 0)
        + (getattr(usage, "completion_tokens", 0) or 0)
    )


def openai_sdk_telemetry(contexts: Iterable[list[dict]]) -> dict:
    """Coarse fallback when the runner didn't accumulate per-call usage.

    Given the debate's per-peer message histories, count tool calls
    (every message with role='tool') and LLM calls (messages where role
    == 'assistant' — every AI response costs one call). Token counts
    are zero in this path; the recommended path is to wire
    `openai_sdk_accumulate` in _chat_with_tools for exact token counts.
    """
    telem = dict(_ZERO)
    for ctx in contexts or []:
        for msg in ctx or []:
            role = None
            if isinstance(msg, dict):
                role = msg.get("role")
            else:
                role = getattr(msg, "role", None)
            if role == "assistant":
                telem["n_llm_calls"] += 1
            elif role == "tool":
                telem["n_tool_calls"] += 1
    return telem



# Utility: coerce any dict-ish into the fixed 5-key shape
def normalize(telem: dict | None) -> dict:
    """Return a dict that has exactly the 5 telemetry keys; use as
    post-processing before writing a row so schema stays uniform even
    when a helper returns a sparser dict."""
    out = dict(_ZERO)
    if telem:
        for k in _ZERO:
            if k in telem:
                out[k] = int(telem[k] or 0)
    return out
