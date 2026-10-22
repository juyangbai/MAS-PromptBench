"""Runtime guards of the LangGraph runners: replayable message histories and a wall-clock cap."""

from __future__ import annotations

import contextlib
import json
import signal
import threading
from collections.abc import Iterator

from langchain_core.messages import AIMessage, BaseMessage


def openai_safe_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    """The history with every AI message rebuilt from its text, normalized tool calls and ``source`` tag.

    Replaying a malformed raw tool-call payload from ``additional_kwargs`` can make
    an OpenAI-compatible server reject the next request.
    """
    return [_safe_ai_message(msg) if isinstance(msg, AIMessage) else msg for msg in messages]


def _safe_ai_message(msg: AIMessage) -> AIMessage:
    source = (msg.additional_kwargs or {}).get("source")
    tool_calls = []
    for index, call in enumerate(msg.tool_calls or []):
        name = _field(call, "name")
        if name:
            call_id = _field(call, "id")
            args = _json_safe_args(_field(call, "args", {}))
            tool_calls.append({"name": str(name), "args": args, "id": str(call_id or f"call_{index}")})
    content = msg.content or ""
    return AIMessage(
        content=content if isinstance(content, str) else str(content),
        tool_calls=tool_calls,
        additional_kwargs={"source": source} if source else {},
    )


def _field(call, name: str, default=None):
    return call.get(name, default) if isinstance(call, dict) else getattr(call, name, default)


def _json_safe_args(args) -> dict:
    if not isinstance(args, dict):
        return {}
    try:
        safe = json.loads(json.dumps(args, default=str))
    except Exception:
        return {str(k): str(v) for k, v in args.items()}
    return safe if isinstance(safe, dict) else {}


class _RowTimeout(Exception):
    """Raised by :func:`row_timeout` when the wall-clock cap is reached."""


@contextlib.contextmanager
def row_timeout(seconds: int) -> Iterator[None]:
    """Raise after ``seconds`` of wall-clock time (SIGALRM; a no-op off the main thread)."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def expire(signum, frame):
        raise _RowTimeout(f"row exceeded {seconds}s")

    old = signal.signal(signal.SIGALRM, expire)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
