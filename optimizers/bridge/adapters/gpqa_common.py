"""Shared GPQA real-runner adapter utilities."""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import Any

from core.paths import PROMPTS_DIR
from optimizers.bridge.datasets.gpqa import extract_letter, format_prompt, strip_thinking
from optimizers.bridge.lm import TASK_MODEL, next_task_endpoint, task_max_tokens, task_request_seed, task_temperature
from optimizers.bridge.output_contracts import append_output_contract

GPQA_DATASET = "gpqa"


def prompt_path(topology: str, role: str) -> Path:
    return PROMPTS_DIR / topology / "gpqa" / f"{role}.txt"


def load_prompt(topology: str, role: str) -> str:
    return prompt_path(topology, role).read_text().strip()


def load_prompts(topology: str, roles: list[str], overrides: dict[str, str] | None = None) -> dict[str, str]:
    overrides = overrides or {}
    return {role: overrides.get(role, load_prompt(topology, role)) for role in roles}


def execution_prompt(prompt: str, topology: str, role: str) -> str:
    return append_output_contract(prompt, GPQA_DATASET, topology, role)


def coerce_instance(example: Any) -> dict:
    if isinstance(example, dict):
        return example
    if hasattr(example, "toDict"):
        return example.toDict()
    data = dict(getattr(example, "__dict__", {}))
    if not data:
        raise TypeError(f"Cannot coerce {type(example).__name__} to instance dict")
    return data


def default_chat_model(seed: int = 0, max_tokens: int | None = None):
    """Task-model client; `seed` is an offset (replica / peer / role index) added to REQUEST_SEED."""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=os.environ.get("MODEL_ID", TASK_MODEL),
        base_url=next_task_endpoint(),
        api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        temperature=task_temperature(),
        top_p=0.9,
        seed=task_request_seed() + seed,
        max_tokens=max_tokens if max_tokens is not None else task_max_tokens(),
        timeout=600.0,
        max_retries=5,
        extra_body={
            "repetition_penalty": 1.05,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )


def calculator_tool():
    from langchain_core.tools import tool

    @tool
    def calculator(expression: str) -> str:
        """Evaluate a numeric Python expression with arithmetic and math functions."""
        import math

        allowed = {
            k: getattr(math, k)
            for k in (
                "sqrt",
                "log",
                "log10",
                "log2",
                "exp",
                "sin",
                "cos",
                "tan",
                "asin",
                "acos",
                "atan",
                "floor",
                "ceil",
                "pow",
                "pi",
                "e",
            )
        }
        allowed["__builtins__"] = {}
        try:
            return str(eval(expression, allowed))
        except Exception as exc:
            return f"ERROR: {exc}"

    return calculator


def majority_vote(answers: list[dict]) -> str | None:
    letters = [a["answer"] for a in answers if a.get("answer") is not None]
    if not letters:
        return None
    counts = Counter(letters)
    top_count = max(counts.values())
    for letter in letters:
        if counts[letter] == top_count:
            return letter
    return None


def answer_text(letter: str | None, raw: str | None = None) -> str:
    if raw:
        return raw
    return f"Answer: {letter}" if letter else ""


def compact_role_trace(
    *,
    role: str,
    answer: str | None,
    winner: Any = None,
    votes: Any = None,
    details: list[str] | None = None,
) -> str:
    lines = [
        f"role={role}",
        f"winner={winner}",
        f"votes={votes or {}}",
        f"selected_answer={answer}",
    ]
    lines.extend(details or [])
    return "\n".join(lines)


def clean_messages(messages: list) -> list:
    for msg in messages:
        if getattr(msg, "type", None) == "ai" and isinstance(getattr(msg, "content", None), str):
            msg.content = strip_thinking(msg.content)
    return messages


def final_answer_from_messages(messages: list) -> tuple[str | None, str]:
    final = messages[-1].content if messages else ""
    raw = strip_thinking(str(final))
    return extract_letter(raw), raw


def prompt_from_instance(instance: dict) -> str:
    return instance.get("prompt") or format_prompt(instance["question"], instance["choices"])
