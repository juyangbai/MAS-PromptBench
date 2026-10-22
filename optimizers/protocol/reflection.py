"""The reflection model: its common policy, the one client every text method uses, and request telemetry.

Policy (``config.REFLECTION_SAMPLING``): thinking on, 48,000 output tokens,
temperature/top-p at each method's native call-site values (1.0 when a call
site names none), a logical request seed per call.

* :class:`ReflectionClient` sends one chat request per ``complete`` call to an
  OpenAI-compatible endpoint and keeps exact request telemetry. HiveMind,
  MAMUT-GEPA, MAPRO (also for its task-model judge), MASPO, MASPOB and TAVO
  use it; GEPA and MIPRO apply the same policy through their DSPy LMs
  (``methods.dspy_bridge``).
* :class:`LogicalTextReflectionClient` serves call sites written against the
  upstream ``complete(prompt, temperature, max_tokens)`` shape: it derives the
  logical seed from the prompt and enforces thinking and the output ceiling.
* :func:`capture_reflection_responses` / :func:`observe_reflection_response`
  record finish reasons and token counts of completed responses.

Importing this module contacts no endpoint.
"""

from __future__ import annotations

import hashlib
import itertools
import threading
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict
from typing import Any, Protocol

from .config import REFLECTION_MODEL_ID, REFLECTION_SAMPLING
from .errors import OptimizerContractError
from .schema import CellSpec, Usage, content_hash
from .seeding import logical_request_seed
from .settings import ProtocolSettings

REFLECTION_SAMPLING_POLICY = REFLECTION_SAMPLING["sampling_policy"]
REFLECTION_MAX_OUTPUT_TOKENS = int(REFLECTION_SAMPLING["max_output_tokens"])
REFLECTION_DEFAULT_TEMPERATURE = float(REFLECTION_SAMPLING["default_temperature"])
REFLECTION_DEFAULT_TOP_P = float(REFLECTION_SAMPLING["default_top_p"])
# Transport time only; independent of generation and optimizer budgets.
REFLECTION_REQUEST_TIMEOUT_SECONDS = 1800.0
TASK_REQUEST_TIMEOUT_SECONDS = 1200.0


def model_request_timeout_seconds(model: str) -> float:
    """Request timeout: the reflection model gets the longer one."""
    return REFLECTION_REQUEST_TIMEOUT_SECONDS if model == REFLECTION_MODEL_ID else TASK_REQUEST_TIMEOUT_SECONDS


def reflection_model_id() -> str:
    """``REFLECTION_MODEL_ID``, else the protocol reflection model."""
    return ProtocolSettings.from_env().reflection_model


def reflection_base_url() -> str:
    """``REFLECTION_MODEL_BASE_URL``, else ``http://localhost:8200/v1``."""
    return ProtocolSettings.from_env().reflection_endpoint()


# Response observation
_reflection_responses: ContextVar[list[dict[str, Any]] | None] = ContextVar(
    "reflection_response_observations", default=None
)


@contextmanager
def capture_reflection_responses() -> Iterator[list[dict[str, Any]]]:
    """Collect the response observations made inside the block."""
    observations: list[dict[str, Any]] = []
    token = _reflection_responses.set(observations)
    try:
        yield observations
    finally:
        _reflection_responses.reset(token)


def active_response_observations() -> list[dict[str, Any]] | None:
    """The list collecting observations for the current call, if any."""
    return _reflection_responses.get(None)


def _response_field(value: Any, name: str, default: Any = None) -> Any:
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


def _observed_tokens(value: Any) -> int | None:
    # Missing usage is unknown, never a claim of zero consumption.
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0 else None


def _any_observed(values: Sequence[bool | None]) -> bool | None:
    if any(value is True for value in values):
        return True
    return False if values and all(value is False for value in values) else None


def observe_reflection_response(response: Any) -> dict[str, Any]:
    """Observe completed chat responses; never reject, trim, retry or continue."""
    choices = []
    missing = object()
    for choice in _response_field(response, "choices", ()) or ():
        reason = _response_field(choice, "finish_reason")
        reason = reason if isinstance(reason, str) else None
        content = _response_field(_response_field(choice, "message"), "content", missing)
        empty = (content is None or not content.strip()) if content is None or isinstance(content, str) else None
        choices.append(
            {
                "index": _response_field(choice, "index"),
                "finish_reason": reason,
                "length_limited": reason == "length" if reason is not None else None,
                "empty_final": empty,
                "final_characters": len(content) if isinstance(content, str) else (0 if content is None else None),
            }
        )
    usage = _response_field(response, "usage")
    details = _response_field(usage, "completion_tokens_details")
    metadata = {
        "schema": "reflection-response-observation",
        "finish_reason": choices[0]["finish_reason"] if len(choices) == 1 else None,
        "length_limited": _any_observed([choice["length_limited"] for choice in choices]),
        "empty_final": _any_observed([choice["empty_final"] for choice in choices]),
        "completion_tokens": _observed_tokens(_response_field(usage, "completion_tokens")),
        "reasoning_tokens": _observed_tokens(_response_field(details, "reasoning_tokens")),
        "choices": choices,
    }
    collector = _reflection_responses.get()
    if collector is not None:
        collector.append(metadata)
    return metadata


def reflection_response_summary(observations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Telemetry fields summarizing the observations of one call."""
    responses = [dict(item) for item in observations]
    tokens = [item.get("completion_tokens") for item in responses]
    return {
        "response_metadata": responses,
        "finish_reason": responses[0].get("finish_reason") if len(responses) == 1 else None,
        "length_limited": _any_observed([item.get("length_limited") for item in responses]),
        "empty_final": _any_observed([item.get("empty_final") for item in responses]),
        "completion_tokens": sum(tokens) if tokens and all(value is not None for value in tokens) else None,
    }


# The client
class ReflectionBackend(Protocol):
    """What the text methods call: one completion with an explicit seed and sampling."""

    def complete(
        self,
        prompt: str,
        *,
        request_seed: int,
        temperature: float,
        top_p: float,
        max_output_tokens: int,
        thinking: bool,
        system: str | None = None,
        phase: str | None = None,
        role: str | None = None,
    ) -> str: ...

    def snapshot(self) -> Mapping[str, Any]:
        """Usage and per-request telemetry so far."""
        ...


class ReflectionClient:
    """OpenAI-compatible chat client with exact request telemetry.

    ``model`` defaults to ``REFLECTION_MODEL_ID`` (the protocol reflection
    model) and the endpoint to ``REFLECTION_MODEL_BASE_URL`` (else
    ``http://localhost:8200/v1``); ``base_urls`` serves requests round robin
    (MAPRO's task-model judge). ``max_retries`` and ``timeout`` are the SDK's
    transport settings (``timeout`` defaults to the model's request timeout).
    The SDK client is created on first use; ``client`` injects a ready one.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        base_urls: Sequence[str] | None = None,
        max_retries: int = 0,
        timeout: float | None = None,
        http_client: Any = None,
        client: Any = None,
    ) -> None:
        self.model = model or reflection_model_id()
        self.base_urls = tuple(base_urls) if base_urls is not None else None
        self.max_retries = int(max_retries)
        self.timeout = timeout
        self._http_client = http_client
        self._injected = client
        self._clients: dict[str, Any] = {}
        self._cycle = itertools.cycle(self.base_urls) if self.base_urls else None
        self._client_lock = threading.Lock()
        self._lock = threading.Lock()
        self._requests: list[dict[str, Any]] = []
        self._usage = Usage()

    @property
    def endpoint(self) -> str:
        """The (first) endpoint requests go to."""
        return self.base_urls[0] if self.base_urls else reflection_base_url()

    def _client(self) -> Any:
        with self._client_lock:
            if self._injected is not None:
                return self._injected
            endpoint = next(self._cycle) if self._cycle is not None else self.endpoint
            if endpoint not in self._clients:
                from openai import OpenAI

                extra = {"http_client": self._http_client} if self._http_client is not None else {}
                timeout = self.timeout if self.timeout is not None else model_request_timeout_seconds(self.model)
                self._clients[endpoint] = OpenAI(
                    base_url=endpoint,
                    api_key=ProtocolSettings.from_env().api_key,
                    timeout=timeout,
                    max_retries=self.max_retries,
                    **extra,
                )
            return self._clients[endpoint]

    def complete(
        self,
        prompt: str,
        *,
        request_seed: int,
        temperature: float,
        top_p: float,
        max_output_tokens: int,
        thinking: bool,
        system: str | None = None,
        phase: str | None = None,
        role: str | None = None,
    ) -> str:
        """Send one request and return the final message text ("" when empty)."""
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        response = self._client().chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=float(temperature),
            top_p=float(top_p),
            max_tokens=int(max_output_tokens),
            seed=int(request_seed),
            extra_body={"chat_template_kwargs": {"enable_thinking": bool(thinking)}},
        )
        metadata = reflection_response_summary([observe_reflection_response(response)])
        text = response.choices[0].message.content or ""
        raw_usage = getattr(response, "usage", None)
        prompt_tokens = int(getattr(raw_usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(raw_usage, "completion_tokens", 0) or 0)
        request = {
            "phase": phase,
            "role": role,
            "model": self.model,
            "temperature": float(temperature),
            "top_p": float(top_p),
            "max_output_tokens": int(max_output_tokens),
            "thinking": bool(thinking),
            "request_seed": int(request_seed),
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "output_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            **metadata,
        }
        with self._lock:
            self._requests.append(request)
            self._usage = self._usage + Usage(
                model_calls=1,
                input_tokens=prompt_tokens,
                output_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            )
        return text

    @property
    def usage(self) -> dict[str, int]:
        """Token and call counts (``prompt_tokens``, ``completion_tokens``, ``n_calls``)."""
        with self._lock:
            return {
                "prompt_tokens": self._usage.input_tokens,
                "completion_tokens": self._usage.output_tokens,
                "n_calls": self._usage.model_calls,
            }

    def snapshot(self) -> Mapping[str, Any]:
        """Usage counters and every request's telemetry."""
        with self._lock:
            return {"usage": asdict(self._usage), "requests": list(self._requests)}


class LogicalTextReflectionClient:
    """Inject logical seeds and enforce the common reflection mode and ceiling.

    Native call sites keep their sampling arguments; thinking and the total
    output-token ceiling are fixed for every reflection request. The seed of a
    call is keyed by its phase, the prompt and how often that prompt was sent.
    """

    def __init__(
        self, *, cell: CellSpec, client: Any, method: str, default_temperature: float, default_top_p: float
    ) -> None:
        self.cell = cell
        self.client = client
        self.method = method
        self.default_temperature = float(default_temperature)
        self.default_top_p = float(default_top_p)
        self.requests: list[dict[str, Any]] = []
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "n_calls": 0}
        self._counts: dict[str, int] = {}
        self._lock = threading.Lock()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.client, name)

    def complete(
        self,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        *,
        top_p: float | None = None,
        phase: str | None = None,
    ) -> str:
        """Upstream call-site shape; ``max_tokens`` is superseded by the common ceiling."""
        del max_tokens
        temperature = self.default_temperature if temperature is None else float(temperature)
        top_p = self.default_top_p if top_p is None else float(top_p)
        max_tokens = REFLECTION_MAX_OUTPUT_TOKENS
        thinking = True
        prompt_hash = content_hash({"prompt": prompt})
        with self._lock:
            occurrence = self._counts.get(prompt_hash, 0)
            self._counts[prompt_hash] = occurrence + 1
        effective_phase = phase or f"{self.method}_reflection"
        seed = logical_request_seed(
            self.cell.optimizer_seed,
            self.cell.cell_id,
            effective_phase,
            occurrence,
            prompt_hash,
            "reflection",
            occurrence,
        )
        base = {
            "phase": effective_phase,
            "prompt_sha256": prompt_hash,
            "temperature": temperature,
            "top_p": top_p,
            "max_output_tokens": max_tokens,
            "thinking": thinking,
            "sampling_policy": REFLECTION_SAMPLING_POLICY,
            "request_seed": seed,
            "occurrence": occurrence,
        }
        observations: list[dict[str, Any]] = []
        try:
            with capture_reflection_responses() as captured:
                try:
                    text = self.client.complete(
                        prompt,
                        request_seed=seed,
                        temperature=temperature,
                        top_p=top_p,
                        max_output_tokens=max_tokens,
                        thinking=thinking,
                        phase=effective_phase,
                    )
                finally:
                    observations.extend(captured)
        except Exception as exc:
            with self._lock:
                if self._counts.get(prompt_hash) == occurrence + 1:
                    self._counts[prompt_hash] = occurrence
                self.requests.append(
                    {
                        **base,
                        "status": "infrastructure_failure",
                        **reflection_response_summary(observations),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            raise
        if not isinstance(text, str):
            raise OptimizerContractError("reflection client returned a non-string response")
        source_usage = getattr(self.client, "usage", {}) or {}
        with self._lock:
            self.requests.append(
                {
                    **base,
                    "output_sha256": content_hash({"output": text}),
                    "status": "success",
                    **reflection_response_summary(observations),
                }
            )
            self.usage = {
                "prompt_tokens": int(source_usage.get("prompt_tokens", 0)),
                "completion_tokens": int(source_usage.get("completion_tokens", 0)),
                "n_calls": len(self.requests),
            }
        return text


__all__ = [
    "LogicalTextReflectionClient",
    "REFLECTION_DEFAULT_TEMPERATURE",
    "REFLECTION_DEFAULT_TOP_P",
    "REFLECTION_MAX_OUTPUT_TOKENS",
    "REFLECTION_REQUEST_TIMEOUT_SECONDS",
    "REFLECTION_SAMPLING_POLICY",
    "ReflectionBackend",
    "ReflectionClient",
    "TASK_REQUEST_TIMEOUT_SECONDS",
    "active_response_observations",
    "capture_reflection_responses",
    "model_request_timeout_seconds",
    "observe_reflection_response",
    "reflection_base_url",
    "reflection_model_id",
    "reflection_response_summary",
]
