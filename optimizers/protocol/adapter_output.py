"""Turn a real-runner adapter's output into run-record fields.

* JSON-safe conversion, message and tool-event extraction from adapter output;
* five-key telemetry (prompt/completion/total tokens, LLM calls, tool calls);
* transport-failure classification of exceptions and returned error text;
* capture of completed task-model responses at the OpenAI SDK boundary, so a
  rollout with no observed model response can be classified as infrastructure.
"""

from __future__ import annotations

import inspect
import re
import threading
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Any

from .errors import TelemetryError
from .schema import Usage, canonical_json, content_hash

# Config
TELEMETRY_KEYS = ("prompt_tokens", "completion_tokens", "total_tokens", "n_llm_calls", "n_tool_calls")
TRANSPORT_NAMES = frozenset(
    {
        "APIConnectionError",
        "APITimeoutError",
        "ConnectError",
        "ConnectTimeout",
        "ConnectionError",
        "ConnectionResetError",
        "ReadError",
        "ReadTimeout",
        "RemoteProtocolError",
        "TimeoutError",
        "InternalServerError",
        "RateLimitError",
        "ServiceUnavailableError",
    }
)
# Methods whose native code joins message contents as strings: structured
# final outputs are serialized as canonical JSON text for them only.
TEXT_MESSAGE_METHODS = frozenset({"maspo", "maspob"})


# Classification
def transport_failure(exc: BaseException) -> bool:
    """Timeouts, connection errors, HTTP 408/429/5xx and named transport errors."""
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    if type(status) is int and (status in {408, 429} or 500 <= status <= 599):
        return True
    return any(cls.__name__ in TRANSPORT_NAMES for cls in type(exc).__mro__)


def transport_error_text(value: Any) -> str | None:
    """Transport error text that an adapter caught and returned as output."""
    if not isinstance(value, Mapping):
        return None
    candidates: list[str] = []
    nested = value.get("runner_output")
    for item in (value, nested):
        if not isinstance(item, Mapping):
            continue
        for key in ("communication_infra_error", "infrastructure_error", "error"):
            if item.get(key):
                candidates.append(str(item[key]))
        if item.get("error_type") in TRANSPORT_NAMES:
            candidates.append(f"{item['error_type']}: {item.get('error', '')}")
    for text in candidates:
        if "dynamic_pool_unavailable" in text:
            return text
        if re.search(r"\bError code:\s*(?:408|429|5[0-9]{2})\b", text):
            return text
        if any(name.lower() in text.lower() for name in TRANSPORT_NAMES):
            return text
    return None


# JSON-safe views
def json_safe(value: Any, *, depth: int = 0) -> Any:
    """JSON-compatible view of an adapter value (paths become file names, objects their fields)."""
    if depth > 12:
        return repr(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and value != value:
            return None
        return value
    if isinstance(value, Path):
        return value.name
    if isinstance(value, Mapping):
        return {str(key): json_safe(item, depth=depth + 1) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item, depth=depth + 1) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [json_safe(item, depth=depth + 1) for item in value]
        return sorted(items, key=canonical_json)
    if hasattr(value, "toDict"):
        try:
            return json_safe(value.toDict(), depth=depth + 1)
        except Exception:
            pass
    if hasattr(value, "model_dump"):
        try:
            return json_safe(value.model_dump(mode="json"), depth=depth + 1)
        except Exception:
            pass
    if hasattr(value, "to_dict"):
        try:
            return json_safe(value.to_dict(), depth=depth + 1)
        except Exception:
            pass
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    attrs: dict[str, Any] = {}
    for name in (
        "type",
        "role",
        "source",
        "name",
        "content",
        "tool_calls",
        "additional_kwargs",
        "response_metadata",
        "usage_metadata",
    ):
        if hasattr(value, name):
            attrs[name] = json_safe(getattr(value, name), depth=depth + 1)
    return attrs or {"type": type(value).__name__, "repr": repr(value)}


def mapping_messages(value: Any) -> tuple[Mapping[str, Any], ...]:
    """Collect native messages from common runner-output layouts, deduplicated."""
    records: list[Mapping[str, Any]] = []
    seen: set[str] = set()

    def append(item: Any) -> None:
        safe = json_safe(item)
        record = dict(safe) if isinstance(safe, Mapping) else {"content": str(safe)}
        digest = content_hash(record)
        if digest not in seen:
            seen.add(digest)
            records.append(record)

    def visit(item: Any, depth: int = 0) -> None:
        if depth > 8 or not isinstance(item, Mapping):
            return
        for key in ("messages", "all_messages"):
            messages = item.get(key)
            if isinstance(messages, Sequence) and not isinstance(messages, (str, bytes)):
                for message in messages:
                    append(message)
        contexts = item.get("all_contexts")
        if isinstance(contexts, Sequence) and not isinstance(contexts, (str, bytes)):
            for context in contexts:
                if isinstance(context, Sequence) and not isinstance(context, (str, bytes)):
                    for message in context:
                        append(message)
        for key in ("by_stage", "by_role", "per_role"):
            entries = item.get(key)
            if isinstance(entries, Mapping):
                for role, message in entries.items():
                    if isinstance(message, Mapping) and "messages" in message:
                        visit(message, depth + 1)
                    else:
                        append({"source": str(role), "content": json_safe(message)})
        for key in ("runner_output", "per_agent", "per_peer", "stage_outputs", "workers"):
            nested = item.get(key)
            if isinstance(nested, Mapping):
                visit(nested, depth + 1)
            elif isinstance(nested, Sequence) and not isinstance(nested, (str, bytes)):
                for child in nested:
                    visit(child, depth + 1)

    visit(value)
    return tuple(records)


def synthetic_final_message(safe: Mapping[str, Any], method: str | None) -> dict[str, Any]:
    """Single ``final`` message standing in for an output that carries no transcript."""
    value = json_safe(
        safe.get("answer")
        or safe.get("predicted_answer")
        or safe.get("answer_text")
        or safe.get("model_output")
        or safe.get("output")
        or ""
    )
    if method in TEXT_MESSAGE_METHODS and not isinstance(value, str):
        value = canonical_json(value)
    return {"source": "final", "content": value, "synthetic_from_final_output": True}


def tool_events(value: Any, messages: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any], ...]:
    """Explicit tool events plus every tool call and tool result found in the messages."""
    events: list[Mapping[str, Any]] = []
    seen_calls: set[str] = set()
    seen_results: set[str] = set()
    if isinstance(value, Mapping):
        explicit = value.get("tool_events")
        if isinstance(explicit, Sequence) and not isinstance(explicit, (str, bytes)):
            events.extend(dict(json_safe(item)) for item in explicit if isinstance(json_safe(item), Mapping))
    for index, message in enumerate(messages):
        calls = message.get("tool_calls")
        if isinstance(calls, Sequence) and not isinstance(calls, (str, bytes)):
            for call in calls:
                safe = json_safe(call)
                identity = (
                    str(safe.get("id") or content_hash(safe)) if isinstance(safe, Mapping) else content_hash(safe)
                )
                if identity in seen_calls:
                    continue
                seen_calls.add(identity)
                events.append({"type": "tool_call", "message_index": index, "call": safe})
        role = str(message.get("role") or message.get("type") or "").lower()
        if "tool" in role and message.get("content") is not None:
            identity = str(message.get("tool_call_id") or content_hash(message.get("content")))
            if identity in seen_results:
                continue
            seen_results.add(identity)
            events.append({"type": "tool_result", "message_index": index, "content": json_safe(message.get("content"))})
    return tuple(events)


# Telemetry
def find_telemetry(value: Any) -> Mapping[str, Any] | None:
    """First five-key telemetry mapping in the output (breadth first through nested runner output)."""
    queue = [value]
    seen: set[int] = set()
    while queue:
        item = queue.pop(0)
        if not isinstance(item, Mapping) or id(item) in seen:
            continue
        seen.add(id(item))
        telemetry = item.get("telemetry")
        if isinstance(telemetry, Mapping) and all(key in telemetry for key in TELEMETRY_KEYS):
            return telemetry
        if all(key in item for key in TELEMETRY_KEYS):
            return item
        for key in ("runner_output", "runtime_output", "metadata"):
            nested = item.get(key)
            if isinstance(nested, Mapping):
                queue.append(nested)
    return None


def usage_from_telemetry(
    telemetry: Mapping[str, Any] | None, *, derived_tool_calls: int = 0, allow_zero_calls: bool = False
) -> Usage:
    """Usage counters from telemetry; zero model calls is a ``TelemetryError`` unless allowed."""
    if telemetry is None:
        raise TelemetryError("execution reported no model calls and no telemetry")
    try:
        input_tokens = int(telemetry["prompt_tokens"])
        output_tokens = int(telemetry["completion_tokens"])
        model_calls = int(telemetry["n_llm_calls"])
        tool_calls = max(int(telemetry["n_tool_calls"]), int(derived_tool_calls))
    except (KeyError, TypeError, ValueError) as exc:
        raise TelemetryError("telemetry is incomplete or non-integral") from exc
    if min(input_tokens, output_tokens, model_calls, tool_calls) < 0:
        raise TelemetryError("telemetry contains a negative counter")
    if model_calls <= 0 and not allow_zero_calls:
        raise TelemetryError("execution reported zero model calls")
    return Usage(
        model_calls=model_calls,
        tool_calls=tool_calls,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
    )


# Model-response capture
_ACTIVE_CAPTURE = None


def _observe_active(response: Any, kwargs: Mapping[str, Any]) -> None:
    capture = _ACTIVE_CAPTURE
    if capture is not None and kwargs.get("model") == capture.model:
        capture.observe(response, kwargs)


def _safe_response(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return {str(k): _safe_response(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe_response(v) for v in value]
    return str(value)


class ModelResponseCapture:
    """Completed task-model responses observed during one rollout."""

    def __init__(self, model: str) -> None:
        self.model = model
        self.events: list[dict[str, Any]] = []
        self._observed: set[str] = set()
        self._lock = threading.Lock()

    def observe(self, response: Any, kwargs: Mapping[str, Any]) -> None:
        """Record one completed response (each response ID once)."""
        value = _safe_response(response)
        if not isinstance(value, Mapping):
            value = {"unparsed_response_type": type(response).__name__}
        messages = _safe_response(kwargs.get("messages", kwargs.get("input", [])))
        event = {
            "response_id": value.get("id"),
            "seed": kwargs.get("seed"),
            "usage": value.get("usage"),
            "tool_results": sum(1 for m in messages if isinstance(m, Mapping) and m.get("role") == "tool")
            if isinstance(messages, list)
            else 0,
        }
        identity = str(value.get("id") or id(response))
        with self._lock:
            if identity not in self._observed:
                self._observed.add(identity)
                self.events.append(event)

    def telemetry(self) -> dict[str, int] | None:
        """Five-key telemetry of the observed responses, or None when nothing was observed."""
        if not self.events:
            return None
        result = dict.fromkeys(TELEMETRY_KEYS, 0)
        result["n_llm_calls"] = len(self.events)
        for event in self.events:
            usage = event["usage"] if isinstance(event["usage"], Mapping) else {}
            prompt = int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
            completion = int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0)
            result["prompt_tokens"] += prompt
            result["completion_tokens"] += completion
            result["total_tokens"] += prompt + completion
        result["n_tool_calls"] = max((event["tool_results"] for event in self.events), default=0)
        return result


@contextmanager
def capture_model_responses(model: str, enabled: bool = True) -> Iterator[ModelResponseCapture]:
    """Hook the OpenAI SDK (sync/async, chat/responses) for responses of ``model``.

    LangChain, LiteLLM (CrewAI), AutoGen and the Agents SDK all cross this
    boundary. The caller serializes rollouts, so the hook is process-global.
    When ``enabled`` is false or the SDK cannot be imported, nothing is hooked
    and the capture stays empty.
    """
    global _ACTIVE_CAPTURE
    capture = ModelResponseCapture(model)
    previous = _ACTIVE_CAPTURE
    targets = _sdk_hook_targets() if enabled else None
    if targets is None:
        yield capture
        return
    clients, resources = targets
    _ACTIVE_CAPTURE = capture
    restore: list[tuple[Any, str, Any]] = []
    for cls, asynchronous in clients:
        original = cls._process_response
        restore.append((cls, "_process_response", original))
        cls._process_response = _hooked_process_response(original, model, asynchronous)
    for cls, asynchronous in resources:
        for name in ("create", "parse"):
            original = getattr(cls, name, None)
            if not callable(original):
                continue
            restore.append((cls, name, original))
            setattr(cls, name, _hooked_resource_call(original, model, asynchronous))
    try:
        yield capture
    finally:
        for cls, name, original in reversed(restore):
            setattr(cls, name, original)
        _ACTIVE_CAPTURE = previous


def _sdk_hook_targets() -> tuple[tuple[tuple[type, bool], ...], tuple[tuple[type, bool], ...]] | None:
    """The OpenAI SDK's ``(class, asynchronous)`` base clients and resources, or None without the SDK."""
    try:
        from openai._base_client import AsyncAPIClient, SyncAPIClient
        from openai.resources.chat.completions import AsyncCompletions, Completions
        from openai.resources.responses import AsyncResponses, Responses
    except Exception:
        return None
    clients = ((SyncAPIClient, False), (AsyncAPIClient, True))
    resources = ((Completions, False), (AsyncCompletions, True), (Responses, False), (AsyncResponses, True))
    return clients, resources


def _parsed(response: Any) -> Any:
    """The parsed body of a raw SDK response (the response itself when it has no ``parse``)."""
    return response.parse() if callable(getattr(response, "parse", None)) else response


def _model_request_body(kwargs: Mapping[str, Any], model: str) -> Mapping[str, Any] | None:
    """The JSON body of a non-streamed ``_process_response`` call for ``model``, else None."""
    body = getattr(kwargs.get("options"), "json_data", None)
    if not kwargs.get("stream") and isinstance(body, Mapping) and body.get("model") == model:
        return body
    return None


def _hooked_process_response(original: Any, model: str, asynchronous: bool) -> Any:
    """A base client's ``_process_response`` that also observes each non-streamed response of ``model``."""
    if asynchronous:

        @wraps(original)
        async def process_async(self, *args, **kwargs):
            result = await original(self, *args, **kwargs)
            body = _model_request_body(kwargs, model)
            if body is not None:
                parsed = _parsed(result)
                if inspect.isawaitable(parsed):
                    parsed = await parsed
                _observe_active(parsed, body)
            return result

        return process_async

    @wraps(original)
    def process_sync(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        body = _model_request_body(kwargs, model)
        if body is not None:
            _observe_active(_parsed(result), body)
        return result

    return process_sync


def _hooked_resource_call(original: Any, model: str, asynchronous: bool) -> Any:
    """A resource's ``create`` / ``parse`` that also observes the response of each request for ``model``."""
    if asynchronous:

        @wraps(original)
        async def async_call(self, *args, **kwargs):
            response = await original(self, *args, **kwargs)
            if kwargs.get("model") == model:
                parsed = _parsed(response)
                if inspect.isawaitable(parsed):
                    parsed = await parsed
                _observe_active(parsed, kwargs)
            return response

        return async_call

    @wraps(original)
    def sync_call(self, *args, **kwargs):
        response = original(self, *args, **kwargs)
        if kwargs.get("model") == model:
            _observe_active(_parsed(response), kwargs)
        return response

    return sync_call


__all__ = [
    "ModelResponseCapture",
    "TELEMETRY_KEYS",
    "TEXT_MESSAGE_METHODS",
    "TRANSPORT_NAMES",
    "capture_model_responses",
    "find_telemetry",
    "json_safe",
    "mapping_messages",
    "synthetic_final_message",
    "tool_events",
    "transport_error_text",
    "transport_failure",
    "usage_from_telemetry",
]
