"""Reflection requests that outgrow the served context window.

The reflection model is served at its native window and the protocol asks for
up to 48,000 output tokens. A GEPA or MIPRO reflection prompt built from long
traces (BFCL tool schemas, APPS/LiveCodeBench tests) can exceed what is left,
and the server refuses it with HTTP 400.

``ContextFitLM`` sizes every request *before* it is sent, with the served
model's own tokenizer (when available locally) and the window the server
reports: the middle of the longest prompt texts is removed, with a marker
saying so, until the prompt plus the full answer ceiling fits. A refusal that
still happens is retried a few times with a request that fits (first the
answer ceiling is lowered to the room the prompt leaves, never below
``MIN_REFLECTION_OUTPUT_TOKENS``; then the prompt is cut). Each refit is
recorded next to the call's response observations as a
``mas-promptbench-prompt-fit`` entry. Requests that fit pass through
untouched.

``ContextFitLM`` is a transparent proxy, so it can wrap the common reflection
LM (``dspy_bridge.build_reflection_lm``) or any DSPy LM.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import urllib.request
from collections.abc import Callable
from typing import Any

from dspy.clients.base_lm import BaseLM

from ..reflection import REFLECTION_MAX_OUTPUT_TOKENS, active_response_observations
from ..schema import schema_name
from ..settings import ProtocolSettings

logger = logging.getLogger(__name__)

# vLLM states the refusal in one of three ways:
#   A: "...you requested R output tokens and your prompt contains at least P input tokens"
#   B: "...you requested T tokens (P in the messages, R in the completion)"
#   C: "...you requested R output tokens and your prompt contains C characters (more than ...)"
#      -- a pre-tokenization refusal: only a character count is known.
_CONTEXT_TOKENS_A = re.compile(
    r"maximum context length is (\d+) tokens\. However, you requested (\d+) output tokens "
    r"and your prompt contains at least (\d+) input tokens"
)
_CONTEXT_TOKENS_B = re.compile(
    r"maximum context length is (\d+) tokens\. However, you requested \d+ tokens "
    r"\((\d+) in the messages, (\d+) in the completion\)"
)
_CONTEXT_CHARS_C = re.compile(
    r"maximum context length is (\d+) tokens\. However, you requested (\d+) output tokens "
    r"and your prompt contains (\d+) characters"
)
_CHARS_PER_TOKEN_GUESS = 2.5  # tool-schema JSON measures ~2.3 chars/token; over-estimating cuts more
_NO_TOKENIZER_CHARS_PER_TOKEN = 1.5  # pre-fit fallback without a tokenizer (digit-heavy payloads)
MIN_REFLECTION_OUTPUT_TOKENS = 16384
_CONTEXT_MARGIN_TOKENS = 256
_PROMPT_FIT_ROUNDS = 5
PROMPT_FIT_SCHEMA = schema_name("prompt-fit")
ASSUMED_CONTEXT_LIMIT = 65536  # smallest window a reflection server is expected to offer


def _marker(cut: int, limit: int | None = None) -> str:
    window = f"{limit}-token " if limit is not None else ""
    return (
        f"\n\n[context fit: {cut} characters removed from the middle of this message so the "
        f"request fits the reflection model's {window}context window]\n\n"
    )


def _exception_chain(exc: BaseException):
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        yield exc
        exc = exc.__cause__ or exc.__context__


def _context_limit(exc: BaseException) -> tuple[int, int, int] | None:
    """(context limit, requested output tokens, prompt tokens) from a vLLM refusal, or None."""
    for item in _exception_chain(exc):
        text = str(item)
        match = _CONTEXT_TOKENS_A.search(text)
        if match:
            limit, requested, prompt = (int(group) for group in match.groups())
            return limit, requested, prompt
        match = _CONTEXT_TOKENS_B.search(text)
        if match:
            limit, prompt, requested = (int(group) for group in match.groups())
            return limit, requested, prompt
        match = _CONTEXT_CHARS_C.search(text)
        if match:
            limit, requested, chars = (int(group) for group in match.groups())
            return limit, requested, int(chars / _CHARS_PER_TOKEN_GUESS) + 1
    return None


def _is_context_window_error(exc: BaseException) -> bool:
    return any(
        "ContextWindowExceeded" in type(item).__name__ or "maximum context length" in str(item)
        for item in _exception_chain(exc)
    )


def _assumed_limit(args: tuple, kwargs: dict) -> tuple[int, int, int]:
    """Numbers to fit against when the refusal carries none: the smallest expected
    window, with the prompt estimated from its characters."""
    chars = sum(len(text) for _, _, text in _prompt_texts(args, kwargs))
    requested = int(kwargs.get("max_tokens") or 0)
    return ASSUMED_CONTEXT_LIMIT, requested, int(chars / _CHARS_PER_TOKEN_GUESS) + 1


def _prompt_texts(args: tuple, kwargs: dict) -> list[tuple[Any, Any, str]]:
    """(container, key, text) for every string the request sends as prompt."""
    found: list[tuple[Any, Any, str]] = []
    messages = kwargs.get("messages")
    if isinstance(messages, list):
        for index, message in enumerate(messages):
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                found.append((messages, index, message["content"]))
            elif isinstance(message, dict) and isinstance(message.get("content"), list):
                for part_index, part in enumerate(message["content"]):
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        found.append((message["content"], part_index, part["text"]))
    if isinstance(kwargs.get("prompt"), str):
        found.append((kwargs, "prompt", kwargs["prompt"]))
    elif args and isinstance(args[0], str):
        found.append((None, 0, args[0]))
    return found


def fit_reflection_request(
    args: tuple, kwargs: dict, limit: int, requested: int, prompt_tokens: int
) -> tuple[tuple, dict, dict] | None:
    """Return (args, kwargs, note) that fit the model's window, or None when nothing sensible fits."""
    output = int(kwargs.get("max_tokens") or requested)
    adjustable = "max_tokens" in kwargs  # a backend without the knob can only shrink its prompt
    room = limit - prompt_tokens - _CONTEXT_MARGIN_TOKENS
    note = {"context_limit": limit, "prompt_tokens_reported": prompt_tokens, "removed_chars": 0}
    if room >= MIN_REFLECTION_OUTPUT_TOKENS and adjustable:
        if room >= output:
            return None  # the refusal was not about this request's size
        return args, {**kwargs, "max_tokens": room}, {**note, "max_tokens": room}
    if room >= limit - _CONTEXT_MARGIN_TOKENS - (requested or 0) and not adjustable and room >= output:
        return None
    texts = _prompt_texts(args, kwargs)
    if not texts or prompt_tokens <= 0:
        return None
    total_chars = sum(len(text) for _, _, text in texts)
    chars_per_token = total_chars / prompt_tokens
    target_tokens = limit - MIN_REFLECTION_OUTPUT_TOKENS - _CONTEXT_MARGIN_TOKENS
    remove = int((prompt_tokens - target_tokens) * chars_per_token * 1.05) + 64
    # Cut from the largest texts first (a reflection prompt can carry long tool
    # schemas in more than one message), each down to a 2,000-char floor.
    plan: list[tuple[Any, Any, str, int]] = []
    left = remove
    for container, key, text in sorted(texts, key=lambda item: -len(item[2])):
        if left <= 0:
            break
        cut = min(left, len(text) - 2000)
        if cut <= 0:
            continue
        plan.append((container, key, text, cut))
        left -= cut
    if left > 0:
        return None  # nothing meaningful would be left of the prompt
    new_args, new_kwargs = args, dict(kwargs)
    messages = None
    if isinstance(kwargs.get("messages"), list):
        messages = [dict(message) if isinstance(message, dict) else message for message in kwargs["messages"]]
        new_kwargs["messages"] = messages
    for container, key, text, cut in plan:
        keep = len(text) - cut
        head = int(keep * 0.6)
        trimmed = text[:head] + _marker(cut, limit) + text[len(text) - (keep - head) :]
        if container is None:
            new_args = (trimmed, *args[1:])
        elif container is kwargs.get("messages"):
            messages[key] = {**messages[key], "content": trimmed}
        elif isinstance(container, list):
            container[key] = {**container[key], "text": trimmed}
        else:
            new_kwargs["prompt"] = trimmed
    max_tokens = min(output, MIN_REFLECTION_OUTPUT_TOKENS)
    if adjustable:
        new_kwargs["max_tokens"] = max_tokens
    return new_args, new_kwargs, {**note, "removed_chars": remove, "max_tokens": max_tokens if adjustable else None}


def call_within_context(lm: Any, args: tuple, kwargs: dict) -> tuple[Any, dict[str, Any] | None]:
    """Call the LM; on a context-window refusal, refit the request and retry a few times."""
    note: dict[str, Any] | None = None
    for _ in range(_PROMPT_FIT_ROUNDS):
        try:
            return lm(*args, **kwargs), note
        except Exception as exc:
            if not _is_context_window_error(exc):
                raise
            logger.warning(
                "[context_fit] refused after pre-fit: %s",
                " | ".join(f"{type(item).__name__}: {str(item)[:300]}" for item in _exception_chain(exc)),
            )
            numbers = _context_limit(exc) or _assumed_limit(args, kwargs)
            if note:
                # vLLM reports the prompt as "at least N tokens" -- a lower bound.
                # A second refusal after lowering the answer ceiling means the
                # prompt itself is bigger than reported: size it from its
                # characters so this round trims it instead of trying again;
                # every further refusal assumes denser text still.
                limit, requested, reported = numbers
                chars = sum(len(text) for _, _, text in _prompt_texts(args, kwargs))
                density = _CHARS_PER_TOKEN_GUESS / (1 + 0.5 * (note.get("rounds", 0) - 1))
                numbers = (limit, requested, max(reported, int(chars / max(density, 1.0)) + 1))
            fitted = fit_reflection_request(args, kwargs, *numbers)
            if fitted is None:
                raise
            args, kwargs, step = fitted
            note = {
                **(note or {}),
                **step,
                "rounds": (note or {}).get("rounds", 0) + 1,
                "removed_chars": (note or {}).get("removed_chars", 0) + step["removed_chars"],
            }
    return lm(*args, **kwargs), note


def note_prompt_fit(note: dict[str, Any]) -> None:
    """Record a refit next to the response observations kept for this call."""
    observations = active_response_observations()
    if isinstance(observations, list):
        observations.append({"schema": PROMPT_FIT_SCHEMA, "length_limited": None, "empty_final": None, **note})


_MESSAGE_OVERHEAD_TOKENS = 4  # role and separator tokens the chat template adds per message
_limit_cache: dict[str, int] = {}
_counter_cache: dict[str, Callable[[str], int] | None] = {}
_cache_lock = threading.Lock()


def _served_context_limit(api_base: str | None, model: str | None = None) -> int:
    """The served window: ``REFLECTION_CONTEXT_LIMIT`` when set, else what
    ``<api_base>/models`` reports (vLLM's ``max_model_len``), else the smallest
    expected window."""
    configured = ProtocolSettings.from_env().reflection_context_limit
    if configured is not None:
        return configured
    name = str(model or "")
    if name.startswith("openai/"):
        name = name[len("openai/") :]
    key = f"{api_base or ''}|{name}"
    with _cache_lock:
        if key in _limit_cache:
            return _limit_cache[key]
    limit = ASSUMED_CONTEXT_LIMIT
    if api_base:
        try:
            with urllib.request.urlopen(api_base.rstrip("/") + "/models", timeout=5) as response:
                data = json.loads(response.read().decode("utf-8"))
            items = [
                item
                for item in data.get("data", [])
                if isinstance(item, dict) and isinstance(item.get("max_model_len"), int)
            ]
            matching = [item for item in items if item.get("id") == name] or items
            if matching:
                limit = min(int(item["max_model_len"]) for item in matching)
        except Exception:
            limit = ASSUMED_CONTEXT_LIMIT
    with _cache_lock:
        if limit != ASSUMED_CONTEXT_LIMIT:
            _limit_cache[key] = limit  # a failed probe is retried on the next call
    return limit


def _token_counter(model: str | None) -> Callable[[str], int] | None:
    """A counter backed by the served model's tokenizer (local files only), or None."""
    name = str(model or "")
    if name.startswith("openai/"):
        name = name[len("openai/") :]
    with _cache_lock:
        if name in _counter_cache:
            return _counter_cache[name]
    counter = None
    if name:
        try:
            from transformers import AutoTokenizer
        except ImportError:
            AutoTokenizer = None
        # Extra tokenizer cache roots (colon-separated) for model files kept
        # outside the default Hugging Face cache.
        roots = [None] + list(ProtocolSettings.from_env().tokenizer_cache_dirs)
        for root in roots if AutoTokenizer is not None else ():
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    name, local_files_only=True, trust_remote_code=False, **({"cache_dir": root} if root else {})
                )

                def counter(text: str, _tokenizer=tokenizer) -> int:
                    return len(_tokenizer(text, add_special_tokens=False)["input_ids"])

                break
            except Exception:
                counter = None
    with _cache_lock:
        if counter is not None:
            _counter_cache[name] = counter  # a miss is retried on the next call (cheap), never cached
    return counter


_COUNT_SAFETY = 1.02  # server count = local count + a few chat-template tokens;
_COUNT_SAFETY_TOKENS = 512  # this also covers a tokenizer that drifts by 2%


def count_prompt_tokens(request: dict, counter: Callable[[str], int] | None) -> int:
    """Conservative prompt-token count of a request (tokenizer when available, else characters)."""
    texts = _prompt_texts((), request)
    if counter is None:
        # Without the tokenizer, assume the densest text seen (digit-heavy test
        # payloads at ~1.5 chars/token).
        tokens = sum(int(len(text) / _NO_TOKENIZER_CHARS_PER_TOKEN) + 1 for _, _, text in texts)
    else:
        tokens = sum(counter(text) for _, _, text in texts)
    # A request whose prompt lives in a shape _prompt_texts does not know is
    # still sized, from its serialized length, so it is never sent blind.
    serialized = len(json.dumps({k: v for k, v in request.items() if k in ("messages", "prompt")}, default=str))
    known = sum(len(text) for _, _, text in texts)
    if serialized - known > 4 * _CHARS_PER_TOKEN_GUESS * 256:
        tokens += int((serialized - known) / _CHARS_PER_TOKEN_GUESS) + 1
    messages = request.get("messages")
    tokens += _MESSAGE_OVERHEAD_TOKENS * (len(messages) if isinstance(messages, list) else 1)
    return int(tokens * _COUNT_SAFETY) + _COUNT_SAFETY_TOKENS


_PROTOCOL_ANSWER_TOKENS = int(REFLECTION_MAX_OUTPUT_TOKENS)
_HARD_TRIM_FLOOR_CHARS = 500


def _hard_trim(request: dict, prompt_tokens: int, target_tokens: int) -> tuple[dict, int]:
    """Cut every prompt text in proportion so about ``target_tokens`` remain (head 60% / tail 40%,
    with a marker). Used when the gentle fit gives up: sending a prompt many times the window is
    a certain refusal, so a drastic cut is the better of two bad outcomes."""
    texts = _prompt_texts((), request)
    total = sum(len(t) for _, _, t in texts)
    if not texts or total <= 0 or prompt_tokens <= 0:
        return request, 0
    keep_total = int(total * target_tokens / prompt_tokens * 0.9)
    new_kwargs = dict(request)
    messages = None
    if isinstance(request.get("messages"), list):
        messages = [dict(m) if isinstance(m, dict) else m for m in request["messages"]]
        new_kwargs["messages"] = messages
    removed = 0
    for container, key, text in texts:
        keep = max(_HARD_TRIM_FLOOR_CHARS, int(keep_total * len(text) / total))
        if keep >= len(text):
            continue
        cut = len(text) - keep
        head = int(keep * 0.6)
        trimmed = text[:head] + _marker(cut) + text[len(text) - (keep - head) :]
        removed += cut
        if container is None:
            new_kwargs["prompt"] = trimmed
        elif container is request.get("messages"):
            messages[key] = {**messages[key], "content": trimmed}
        elif isinstance(container, list):
            container[key] = {**container[key], "text": trimmed}
        else:
            new_kwargs["prompt"] = trimmed
    return new_kwargs, removed


def fit_before_call(lm: Any, request: dict) -> tuple[dict, dict[str, Any] | None]:
    """Size the request to the served window before it is sent.

    The reflection policy sends the protocol's answer ceiling whatever
    ``max_tokens`` a call carries, so only the prompt can shrink here and it
    must leave room for the full ceiling. Counted with the served model's
    tokenizer when it is available locally, otherwise estimated from
    characters. Requests that fit are passed through untouched.
    """
    settings = dict(getattr(lm, "kwargs", {}) or {})
    limit = _served_context_limit(request.get("api_base") or settings.get("api_base"), getattr(lm, "model", None))
    requested = max(int(request.get("max_tokens") or settings.get("max_tokens") or 0), _PROTOCOL_ANSWER_TOKENS)
    target = limit - requested - _CONTEXT_MARGIN_TOKENS
    counter = _token_counter(getattr(lm, "model", None))
    note: dict[str, Any] | None = None
    first_count = None
    # A prompt many times the window is cut by characters first, so the
    # tokenizer never has to count tens of millions of tokens.
    texts = _prompt_texts((), request)
    total_chars = sum(len(t) for _, _, t in texts)
    if total_chars > 8 * target * 4:
        est = int(total_chars / (_CHARS_PER_TOKEN_GUESS if counter is not None else _NO_TOKENIZER_CHARS_PER_TOKEN))
        request, removed = _hard_trim(request, est, target)
        note = {
            "context_limit": limit,
            "prompt_tokens_reported": est,
            "removed_chars": removed,
            "max_tokens": None,
            "pre_fit": True,
            "tokenizer": counter is not None,
            "rounds": 1,
            "hard_trim": True,
        }
        first_count = est
    for _ in range(_PROMPT_FIT_ROUNDS):
        prompt_tokens = count_prompt_tokens(request, counter)
        first_count = prompt_tokens if first_count is None else first_count
        if prompt_tokens + requested + _CONTEXT_MARGIN_TOKENS <= limit:
            break
        # With the answer ceiling fixed the prompt must reach ``target``: cut
        # in proportion straight to it (re-counted next round).
        request, removed = _hard_trim(request, prompt_tokens, target)
        step = {
            "context_limit": limit,
            "prompt_tokens_reported": prompt_tokens,
            "removed_chars": removed,
            "max_tokens": None,
            "hard_trim": True,
        }
        note = {
            **(note or {}),
            **step,
            "pre_fit": True,
            "tokenizer": counter is not None,
            "rounds": (note or {}).get("rounds", 0) + 1,
            "removed_chars": (note or {}).get("removed_chars", 0) + step["removed_chars"],
        }
    level = logging.DEBUG if note is None else logging.INFO
    if logger.isEnabledFor(level):
        logger.log(
            level,
            "[context_fit] pre-fit: limit=%s requested=%s prompt_tokens=%s->%s texts=%s tokenizer=%s action=%s",
            limit,
            requested,
            first_count,
            count_prompt_tokens(request, counter),
            len(_prompt_texts((), request)),
            counter is not None,
            "none" if note is None else note,
        )
    return request, note


class ContextFitLM(BaseLM):
    """DSPy LM proxy whose requests are refitted to the served context window.

    Attribute access (``kwargs``, ``model``, ``usage``, ...) is delegated to
    the wrapped LM, so the logical-seed wrapper and native call sites see the
    wrapped LM's state; ``copy`` returns a refitting copy.
    """

    def __init__(self, lm: Any) -> None:
        self.lm = lm

    def __getattr__(self, name: str) -> Any:
        if name == "lm" or (name.startswith("__") and name.endswith("__")):
            raise AttributeError(name)
        return getattr(self.lm, name)

    def __deepcopy__(self, memo: dict[int, Any]) -> ContextFitLM:
        import copy

        clone = ContextFitLM(copy.deepcopy(self.lm, memo))
        memo[id(self)] = clone
        return clone

    def copy(self, **kwargs: Any) -> ContextFitLM:
        return ContextFitLM(self.lm.copy(**kwargs))

    def __call__(self, prompt: Any = None, messages: Any = None, **kwargs: Any) -> Any:
        request = dict(kwargs)
        if prompt is not None:
            request["prompt"] = prompt
        if messages is not None:
            request["messages"] = messages
        request, pre_note = fit_before_call(self.lm, request)
        result, note = call_within_context(lambda **fitted: self.lm(**fitted), (), request)
        if pre_note or note:
            merged = {**(pre_note or {}), **(note or {})}
            if pre_note and note:
                merged["removed_chars"] = pre_note.get("removed_chars", 0) + note.get("removed_chars", 0)
                merged["rounds"] = pre_note.get("rounds", 0) + note.get("rounds", 0)
            note_prompt_fit(merged)
        return result


def fit_to_context(lm: Any) -> Any:
    """Wrap ``lm`` in :class:`ContextFitLM` unless it already refits its requests."""
    return lm if isinstance(lm, ContextFitLM) else ContextFitLM(lm)


__all__ = [
    "ASSUMED_CONTEXT_LIMIT",
    "ContextFitLM",
    "MIN_REFLECTION_OUTPUT_TOKENS",
    "PROMPT_FIT_SCHEMA",
    "call_within_context",
    "count_prompt_tokens",
    "fit_before_call",
    "fit_reflection_request",
    "fit_to_context",
    "note_prompt_fit",
]
