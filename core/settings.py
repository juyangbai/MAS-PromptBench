"""Runner settings, read from the environment at call time.

Every runner talks to one OpenAI-compatible endpoint with one decoding
protocol; this module is the single place that documents and reads those
variables. Values are read on every call, so a change to ``os.environ``
(e.g. an optimizer exporting its rollout temperature) applies to the next
client that is built.

======================== ======================== ==============================================
Variable                 Default                  Meaning
======================== ======================== ==============================================
VLLM_BASE_URL            http://localhost:8000/v1 OpenAI-compatible endpoint
MODEL_ID                 Qwen/Qwen3.5-9B          served model name
OPENAI_API_KEY           EMPTY                    API key (a local vLLM endpoint accepts any)
TASK_MODEL_TEMPERATURE   0.0                      sampling temperature (optimizer rollouts: 0.2)
TASK_MODEL_TOP_P         0.9                      nucleus-sampling mass
TASK_MODEL_MAX_TOKENS    32768                    output-token cap per model call
REQUEST_SEED             0                        per-request sampling seed
INDEPENDENT_N_AGENTS     4 (team-size runners: r) replicas in the independent topology
DECENTRALIZED_N_AGENTS   4 (team-size runners: r) peers in the decentralized debate
DECENTRALIZED_N_ROUNDS   2                        debate rounds
======================== ======================== ==============================================

ToolHop and API-Bank runners read ``TOOLHOP_<VAR>`` / ``APIBANK_<VAR>`` first
for the three team-shape variables (``dataset=`` below).
"""

from __future__ import annotations

import os
from typing import Any

DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_MODEL_ID = "Qwen/Qwen3.5-9B"
DEFAULT_API_KEY = "EMPTY"

# vLLM sampling extras sent with every task-model request.
REPETITION_PENALTY = 1.05
ENABLE_THINKING = False


def base_url() -> str:
    return os.environ.get("VLLM_BASE_URL", DEFAULT_BASE_URL)


def model_id() -> str:
    return os.environ.get("MODEL_ID", DEFAULT_MODEL_ID)


def api_key(*, blank_as_unset: bool = False) -> str:
    """``OPENAI_API_KEY`` or "EMPTY". With ``blank_as_unset`` an empty value
    also falls back to "EMPTY" (the raw OpenAI and Agents SDK clients)."""
    if blank_as_unset:
        return os.environ.get("OPENAI_API_KEY") or DEFAULT_API_KEY
    return os.environ.get("OPENAI_API_KEY", DEFAULT_API_KEY)


def temperature() -> float:
    return float(os.environ.get("TASK_MODEL_TEMPERATURE", "0.0"))


def top_p() -> float:
    return float(os.environ.get("TASK_MODEL_TOP_P", "0.9"))


def max_tokens() -> int:
    return int(os.environ.get("TASK_MODEL_MAX_TOKENS", "32768"))


def request_seed() -> int:
    return int(os.environ.get("REQUEST_SEED", "0"))


def extra_body() -> dict[str, Any]:
    """vLLM ``extra_body`` (a fresh dict per call)."""
    return {
        "repetition_penalty": REPETITION_PENALTY,
        "chat_template_kwargs": {"enable_thinking": ENABLE_THINKING},
    }


def decoding(*, seed: int | None = None, include_max_tokens: bool = True) -> dict[str, Any]:
    """Decoding kwargs the runners pass to their chat clients.

    ``seed`` defaults to ``REQUEST_SEED`` (independent replicas pass their
    own); ``include_max_tokens=False`` leaves the output cap to the server.
    """
    kwargs: dict[str, Any] = {
        "temperature": temperature(),
        "top_p": top_p(),
        "seed": request_seed() if seed is None else seed,
    }
    if include_max_tokens:
        kwargs["max_tokens"] = max_tokens()
    kwargs["extra_body"] = extra_body()
    return kwargs


def _team_int(name: str, default: int, dataset: str | None) -> int:
    value = os.environ.get(name, str(default))
    if dataset:
        value = os.environ.get(f"{dataset.upper()}_{name}", value)
    return int(value)


def independent_n_agents(default: int = 4, *, dataset: str | None = None) -> int:
    return _team_int("INDEPENDENT_N_AGENTS", default, dataset)


def decentralized_n_agents(default: int = 4, *, dataset: str | None = None) -> int:
    return _team_int("DECENTRALIZED_N_AGENTS", default, dataset)


def decentralized_n_rounds(default: int = 2, *, dataset: str | None = None) -> int:
    return _team_int("DECENTRALIZED_N_ROUNDS", default, dataset)
