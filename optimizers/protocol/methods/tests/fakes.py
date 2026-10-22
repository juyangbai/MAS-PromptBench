"""Fake DSPy LMs (reflection / prompt and task models) for the GEPA, MIPRO and context-fit tests."""

from __future__ import annotations

import re
import threading
from types import SimpleNamespace
from typing import Any

from dspy.clients.base_lm import BaseLM

from ...schema import CellSpec
from ...tests.fakes import fake_cell

# Config
GOOD = "GOOD"  # the fake adapter answers every item once the writer prompt contains it
_CURRENT_INSTRUCTION = re.compile(r"```\n(.*?)\n```", re.DOTALL)
_OUTPUT_FIELDS = re.compile(r"Your output fields are:\n(.*?)\nAll interactions will be structured", re.DOTALL)
_FIELD_NAME = re.compile(r"`(\w+)`")
_INPUT_FIELD = re.compile(r"\[\[ ## (\w+) ## \]\]\n(.*?)(?=\n\n\[\[ ## |\n\nRespond with|\Z)", re.DOTALL)


def improved(instruction: str) -> str:
    text = instruction.strip()
    return text if GOOD in text else f"{text} {GOOD}"


class FakeLM(BaseLM):
    """Offline DSPy LM that records every request.

    * GEPA instruction prompts: the current instruction (first fenced block)
      comes back with ``GOOD`` appended, inside a fenced block.
    * DSPy chat-adapter signatures (MIPRO proposer): every output field is
      filled; ``proposed_instruction`` is ``basic_instruction`` + ``GOOD``.

    ``forbid_calls`` makes any call fail (an LM that must stay unused).
    """

    def __init__(self, model: str = "fake/reflection", *, forbid_calls: bool = False) -> None:
        super().__init__(model=model, temperature=1.0, max_tokens=1000)
        self.forbid_calls = forbid_calls
        self.calls: list[dict[str, Any]] = []
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "n_calls": 0}
        self._lock = threading.Lock()

    def __deepcopy__(self, memo: dict[int, Any]) -> FakeLM:
        del memo
        return self

    def forward(self, prompt: Any = None, messages: Any = None, **kwargs: Any) -> Any:
        if self.forbid_calls:
            raise AssertionError("this LM must never be called")
        text = self._answer(prompt, messages)
        with self._lock:
            self.calls.append({"prompt": prompt, "messages": messages, "kwargs": dict(kwargs), "output": text})
            self.usage["n_calls"] += 1
            self.usage["prompt_tokens"] += 10
            self.usage["completion_tokens"] += 5
        choices = [
            SimpleNamespace(index=index, finish_reason="stop", message=SimpleNamespace(content=text, tool_calls=None))
            for index in range(int(kwargs.get("n") or 1))
        ]
        return SimpleNamespace(
            choices=choices, model=self.model, usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        )

    @staticmethod
    def _answer(prompt: Any, messages: Any) -> str:
        if messages:
            system = str(messages[0].get("content", ""))
            fields = _OUTPUT_FIELDS.search(system)
            names = _FIELD_NAME.findall(fields.group(1)) if fields else []
            inputs = dict(_INPUT_FIELD.findall(str(messages[-1].get("content", ""))))
            values = {
                name: improved(inputs.get("basic_instruction", ""))
                if name == "proposed_instruction"
                else f"fake {name}"
                for name in names
            }
            return (
                "".join(f"[[ ## {name} ## ]]\n{value}\n\n" for name, value in values.items()) + "[[ ## completed ## ]]"
            )
        match = _CURRENT_INSTRUCTION.search(str(prompt or ""))
        return f"```\n{improved(match.group(1) if match else '')}\n```"


def method_cell(method: str, *, seed: int = 0, budget: int = 600) -> CellSpec:
    """The protocol test fake cell (off-grid ``fake`` task) for one method."""
    return fake_cell(method=method, seed=seed, budget=budget)


__all__ = ["FakeLM", "GOOD", "improved", "method_cell"]
