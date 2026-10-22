"""Shared BFCL real-runner adapter utilities.

Question flattening, schema tools, call parsing and the vote are the runners'
own (:mod:`core.tasks.bfcl`, :mod:`core.bfcl_calls`).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from core.bfcl_calls import extract_canonical  # noqa: F401  (adapter API)
from core.paths import PROMPTS_DIR
from core.tasks.bfcl import (  # noqa: F401  (adapter API)
    extract_first_tool_calls,
    majority_vote,
    render_schemas,
    schema_to_tool,
    select_call,
    stage_inputs,
    to_canonical,
    vote_counts,
)
from core.tasks.bfcl import flatten_question as flatten_user_request  # noqa: F401  (adapter API)
from optimizers.bridge.lm import TASK_MODEL, next_task_endpoint, task_max_tokens, task_request_seed, task_temperature
from optimizers.bridge.output_contracts import append_output_contract

BFCL_DATASET = "bfcl"
DEFAULT_RECURSION_LIMIT = 100


def prompt_path(topology: str, role: str) -> Path:
    return PROMPTS_DIR / topology / "bfcl" / f"{role}.txt"


def load_prompt(topology: str, role: str) -> str:
    return prompt_path(topology, role).read_text().strip()


def load_prompts(topology: str, roles: list[str], overrides: dict[str, str] | None = None) -> dict[str, str]:
    overrides = overrides or {}
    return {role: overrides.get(role, load_prompt(topology, role)) for role in roles}


def execution_prompt(prompt: str, topology: str, role: str) -> str:
    return append_output_contract(prompt, BFCL_DATASET, topology, role)


def coerce_instance(example: Any) -> dict:
    if isinstance(example, dict):
        return example
    if hasattr(example, "toDict"):
        return example.toDict()
    data = dict(getattr(example, "__dict__", {}))
    if not data:
        raise TypeError(f"Cannot coerce {type(example).__name__} to instance dict")
    return data


def schemas_text(instance: dict) -> str:
    return render_schemas(instance.get("function") or [])


def default_chat_model(seed: int = 0):
    """Task-model client; `seed` is an offset (replica / peer / role index) added to REQUEST_SEED."""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=os.environ.get("MODEL_ID", TASK_MODEL),
        base_url=next_task_endpoint(),
        api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        temperature=task_temperature(),
        top_p=0.9,
        seed=task_request_seed() + seed,
        max_tokens=task_max_tokens(),
        timeout=600.0,
        max_retries=5,
        extra_body={
            "repetition_penalty": 1.05,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )


def recursion_limit(env_name: str | None = None, default: int = DEFAULT_RECURSION_LIMIT) -> int:
    """Read a positive LangGraph recursion limit from the environment."""
    raw = os.environ.get(env_name or "", "") if env_name else ""
    raw = raw or os.environ.get("REAL_RUNNER_RECURSION_LIMIT", "")
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def compact_role_trace(
    *,
    role: str,
    model_output: list[dict],
    winner: Any = None,
    buckets: Any = None,
    details: list[str] | None = None,
) -> str:
    lines = [
        f"role={role}",
        f"winner={winner}",
        f"majority_buckets={buckets or []}",
        f"selected_model_output={model_output or []}",
    ]
    lines.extend(details or [])
    return "\n".join(lines)
