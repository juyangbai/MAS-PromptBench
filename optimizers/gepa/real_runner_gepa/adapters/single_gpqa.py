"""Single/GPQA real-runner adapter."""
from __future__ import annotations

import os
import time
from collections.abc import Callable
from typing import Any

from real_runner_gepa.adapters.gpqa_common import (
    answer_text,
    calculator_tool,
    clean_messages,
    coerce_instance,
    compact_role_trace,
    default_chat_model,
    execution_prompt,
    final_answer_from_messages,
    load_prompts,
    prompt_from_instance,
)


ROLE = "solver"
TOPOLOGY = "single"
DATASET = "gpqa"


class SingleGPQAAdapter:
    topology = TOPOLOGY
    dataset = DATASET

    def __init__(
        self,
        prompts: dict[str, str] | None = None,
        model_factory: Callable[[int], Any] | None = None,
        keep_messages: bool | None = None,
    ):
        self._prompts = load_prompts(TOPOLOGY, [ROLE], prompts)
        self.model_factory = model_factory or default_chat_model
        self.keep_messages = (
            keep_messages
            if keep_messages is not None
            else os.environ.get("REAL_RUNNER_KEEP_MESSAGES", "0") == "1"
        )

    def roles(self) -> list[str]:
        return [ROLE]

    def get_prompt(self, role: str) -> str:
        self._check_role(role)
        return self._prompts[role]

    def set_prompt(self, role: str, text: str) -> None:
        self._check_role(role)
        self._prompts[role] = text

    def reset(self) -> None:
        return None

    def __getstate__(self):
        return self.__dict__.copy()

    def run_example(self, example: Any) -> dict:
        from langgraph.prebuilt import create_react_agent

        instance = coerce_instance(example)
        agent = create_react_agent(
            model=self.model_factory(0),
            tools=[calculator_tool()],
            prompt=execution_prompt(self._prompts[ROLE], TOPOLOGY, ROLE),
        )
        start = time.time()
        try:
            result = agent.invoke(
                {"messages": [("user", prompt_from_instance(instance))]},
                config={"recursion_limit": int(os.environ.get("GPQA_SINGLE_RECURSION_LIMIT", "25"))},
            )
            messages = clean_messages(result["messages"])
            letter, raw = final_answer_from_messages(messages)
            error = None
        except Exception as exc:
            messages = []
            letter = None
            raw = ""
            error = f"{type(exc).__name__}: {exc}"

        role_data = {
            "answer": letter,
            "raw": raw,
            "solve_s": round(time.time() - start, 2),
            "error": error,
        }
        if self.keep_messages:
            role_data["messages"] = messages
        return {
            "model_output": [],
            "answer": letter,
            "answer_text": answer_text(letter, raw),
            "winner": 0,
            "buckets": {},
            "per_role": {ROLE: role_data},
        }

    def format_role_trace(self, role: str, output: Any) -> str:
        self._check_role(role)
        if not isinstance(output, dict):
            return str(output)
        data = (output.get("per_role") or {}).get(role) or {}
        details = [
            f"role_answer={data.get('answer')} solve_s={data.get('solve_s')} error={data.get('error') or 'None'}",
            f"raw_tail={(data.get('raw') or '')[-600:]}",
        ]
        return compact_role_trace(
            role=role,
            answer=output.get("answer"),
            winner=output.get("winner"),
            votes=output.get("buckets") or {},
            details=details,
        )

    def describe_runtime(self, example: Any | None = None) -> dict:
        return {
            "topology": self.topology,
            "dataset": self.dataset,
            "roles": self.roles(),
            "prompt_prefix": self.get_prompt(ROLE)[:80],
            "example_id": coerce_instance(example).get("id") if example is not None else None,
        }

    @staticmethod
    def _check_role(role: str) -> None:
        if role != ROLE:
            raise KeyError(f"Unknown role {role!r}; expected {ROLE!r}")
