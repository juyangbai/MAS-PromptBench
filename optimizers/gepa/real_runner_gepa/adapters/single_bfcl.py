"""Single/BFCL real-runner adapter."""
from __future__ import annotations

import os
import time
from collections.abc import Callable
from typing import Any

from real_runner_gepa.adapters.bfcl_common import (
    coerce_instance,
    compact_role_trace,
    default_chat_model,
    execution_prompt,
    extract_first_tool_calls,
    load_prompts,
    recursion_limit,
    schema_to_tool,
    to_canonical,
)


ROLE = "solver"
TOPOLOGY = "single"
DATASET = "bfcl"


class SingleBFCLAdapter:
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
        tools = [schema_to_tool(schema) for schema in instance["function"]]
        agent = create_react_agent(
            model=self.model_factory(0),
            tools=tools,
            prompt=execution_prompt(self._prompts[ROLE], TOPOLOGY, ROLE),
        )
        start = time.time()
        result = agent.invoke(
            {"messages": instance["question"][0]},
            config={"recursion_limit": recursion_limit("BFCL_SINGLE_RECURSION_LIMIT")},
        )
        tool_calls = extract_first_tool_calls(result["messages"])
        model_output = to_canonical(tool_calls)
        out = {
            "model_output": model_output,
            "winner": 0,
            "buckets": [],
            "per_role": {
                ROLE: {
                    "model_output": model_output,
                    "solve_s": round(time.time() - start, 2),
                    "tool_calls": tool_calls,
                }
            },
        }
        if self.keep_messages:
            out["per_role"][ROLE]["messages"] = result["messages"]
        return out

    def format_role_trace(self, role: str, output: Any) -> str:
        self._check_role(role)
        details = []
        if isinstance(output, dict):
            data = (output.get("per_role") or {}).get(role) or {}
            details.append(
                f"role_output={data.get('model_output') or []} "
                f"solve_s={data.get('solve_s')} error={data.get('error') or 'None'}"
            )
            return compact_role_trace(
                role=role,
                model_output=output.get("model_output") or [],
                winner=output.get("winner"),
                buckets=output.get("buckets") or [],
                details=details,
            )
        return str(output)

    def describe_runtime(self, example: Any | None = None) -> dict:
        instance = coerce_instance(example) if example is not None else {}
        return {
            "topology": self.topology,
            "dataset": self.dataset,
            "roles": self.roles(),
            "tool_count": len(instance.get("function") or []),
            "prompt_prefix": self.get_prompt(ROLE)[:80],
        }

    @staticmethod
    def _check_role(role: str) -> None:
        if role != ROLE:
            raise KeyError(f"Unknown role {role!r}; expected {ROLE!r}")
