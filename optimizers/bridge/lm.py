"""LM endpoints and decoding for real-runner rollouts (variable names: ``optimizers.bridge.env``)."""

from __future__ import annotations

import contextlib
import itertools
import os
import threading
from collections.abc import Iterable

from optimizers.bridge import env

TASK_MODEL = env.get("TASK_MODEL", "Qwen/Qwen3.5-9B")
REFL_MODEL = env.get("REFL_MODEL", "Qwen/Qwen3.5-122B-A10B-FP8")

# Port convention (see models/): task model 8000+, Llama 8100+, reflection 8200.
DEFAULT_TASK_ENDPOINTS = ("http://localhost:8000/v1",)


def _env_list(name: str, default: Iterable[str]) -> tuple[str, ...]:
    raw = env.get(name)
    if not raw:
        return tuple(default)
    return tuple(s.strip() for s in raw.split(",") if s.strip())


def task_endpoints() -> tuple[str, ...]:
    return _env_list("TASK_ENDPOINTS", DEFAULT_TASK_ENDPOINTS)


_adapter_endpoint_cycle = itertools.cycle(task_endpoints())
_adapter_endpoint_lock = threading.Lock()


def next_task_endpoint() -> str:
    """Return the next task endpoint for real-runner calls (round robin)."""
    with _adapter_endpoint_lock:
        return next(_adapter_endpoint_cycle)


# Decoding protocol: optimization rollouts sample at TASK_TEMP_OPTIMIZE (0.2);
# every measurement pass (seed-prompt baseline, final validation, test) runs
# greedy at TASK_TEMP_EVAL (0.0) so the same prompts score the same twice. Runner
# modules read TASK_MODEL_TEMPERATURE at call time, so the active mode is
# mirrored into that variable as well. Process-global: evaluation never
# overlaps optimization inside one process.
_EVAL_MODE = False


def task_temperature() -> float:
    if _EVAL_MODE:
        return float(os.environ.get("TASK_TEMP_EVAL", "0"))
    return float(os.environ.get("TASK_TEMP_OPTIMIZE", "0.2"))


def task_request_seed() -> int:
    return int(os.environ.get("REQUEST_SEED", "0"))


def task_max_tokens() -> int:
    return int(os.environ.get("TASK_MODEL_MAX_TOKENS", "32768"))


def _sync_runner_temperature() -> None:
    os.environ["TASK_MODEL_TEMPERATURE"] = str(task_temperature())


@contextlib.contextmanager
def eval_mode(enabled: bool = True):
    """Force greedy decoding for everything executed inside the block."""
    global _EVAL_MODE
    prev = _EVAL_MODE
    _EVAL_MODE = enabled
    _sync_runner_temperature()
    try:
        yield
    finally:
        _EVAL_MODE = prev
        _sync_runner_temperature()


_sync_runner_temperature()


def task_sampling() -> dict:
    return {
        "temperature": task_temperature(),
        "top_p": 0.9,
        "seed": task_request_seed(),
        "max_tokens": task_max_tokens(),
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
