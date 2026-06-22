"""LM endpoint helpers for isolated real-runner MIPRO experiments."""
from __future__ import annotations

import itertools
import os
import threading
from typing import Iterable

import dspy


TASK_MODEL = os.environ.get("MIPRO_TASK_MODEL") or os.environ.get("TASK_MODEL", "Qwen/Qwen3.5-9B")
REFL_MODEL = os.environ.get("MIPRO_REFL_MODEL") or os.environ.get("REFL_MODEL", "Qwen/Qwen3.5-122B-A10B-FP8")

DEFAULT_TASK_ENDPOINTS = tuple(f"http://localhost:{port}/v1" for port in range(15000, 15008))
DEFAULT_REFL_ENDPOINT = "http://localhost:15000/v1"


def _env_list(names: tuple[str, ...], default: Iterable[str]) -> tuple[str, ...]:
    for name in names:
        raw = os.environ.get(name)
        if raw:
            values = tuple(s.strip() for s in raw.split(",") if s.strip())
            if values:
                return values
    return tuple(default)


def task_endpoints() -> tuple[str, ...]:
    return _env_list(("MIPRO_TASK_ENDPOINTS", "GEPA_TASK_ENDPOINTS"), DEFAULT_TASK_ENDPOINTS)


def reflection_endpoint() -> str:
    return os.environ.get("MIPRO_REFL_ENDPOINT") or os.environ.get("GEPA_REFL_ENDPOINT") or DEFAULT_REFL_ENDPOINT


def task_endpoint_for_seed(seed: int) -> str:
    endpoints = task_endpoints()
    if not endpoints:
        return os.environ.get("VLLM_BASE_URL", DEFAULT_TASK_ENDPOINTS[0])
    return endpoints[seed % len(endpoints)]


_adapter_endpoint_cycle = itertools.cycle(task_endpoints())
_adapter_endpoint_lock = threading.Lock()


def next_task_endpoint() -> str:
    """Return the next task endpoint for non-DSPy real-runner calls."""
    with _adapter_endpoint_lock:
        return next(_adapter_endpoint_cycle)


def task_sampling() -> dict:
    return {
        "temperature": 0.2,
        "top_p": 0.9,
        "seed": 0,
        "max_tokens": 1024,
        "extra_body": {
            "repetition_penalty": 1.05,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    }


def reflection_sampling() -> dict:
    return {
        "temperature": 1.0,
        "max_tokens": 48000,
    }


class RoundRobinLM(dspy.LM):
    """DSPy LM wrapper that rotates across the task endpoint pool per call."""

    def __init__(self, endpoints: tuple[str, ...]):
        if not endpoints:
            raise ValueError("RoundRobinLM needs at least one endpoint")
        super().__init__(
            f"openai/{TASK_MODEL}",
            api_base=endpoints[0],
            api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
            **task_sampling(),
        )
        self._endpoints = tuple(endpoints)
        self._cycle = itertools.cycle(self._endpoints)
        self._lock = threading.Lock()

    def _next_base(self) -> str:
        with self._lock:
            return next(self._cycle)

    def __call__(self, *args, **kwargs):
        self.kwargs["api_base"] = self._next_base()
        return super().__call__(*args, **kwargs)


def build_task_pool() -> RoundRobinLM:
    return RoundRobinLM(task_endpoints())


def build_reflection_lm() -> dspy.LM:
    return dspy.LM(
        f"openai/{REFL_MODEL}",
        api_base=reflection_endpoint(),
        api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        **reflection_sampling(),
    )
