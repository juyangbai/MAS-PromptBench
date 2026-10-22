"""Scripted chat models, tiny splits, a scorer and a reflection backend for offline HiveMind tests.

The real ``optimizers.bridge`` adapters run unchanged; only the chat model is
replaced by :class:`ScriptedChatModel`, so LangGraph graphs, tool nodes,
routers and the module patches execute exactly as in a job. Scripted answers
are correct when the first worker of the team actually responded and either
the item is even-numbered or some executed prompt carries learned lessons.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Mapping
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from optimizers.protocol.runner import AdapterRuntime, ProtocolRunner, ScoreResult, TaskData
from optimizers.protocol.schema import CellSpec
from optimizers.protocol.tests import fakes as protocol_fakes
from optimizers.protocol.tests.fakes import FakeReflectionBackend

from ..runtime import CoalitionExecutionHook

# Config
SPLITS = {
    "train": [f"tr{i}" for i in range(10)],
    "validation": [f"va{i}" for i in range(6)],
    "test": ["te0", "te1"],
}
ANSWER_SCHEMA = {
    "name": "answer",
    "description": "Submit the final answer.",
    "parameters": {
        "type": "dict",
        "properties": {"value": {"type": "string", "description": "answer"}},
        "required": ["value"],
    },
}
HOTPOTQA_WORKERS = ("retriever_worker", "reasoner_worker", "writer_worker")
BFCL_WORKERS = ("inspector_worker", "caller_worker", "validator_worker")
LESSONS_MARKER = "Lessons learned"
IMPROVED = "[improved]"
USAGE = {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}


# Data
def fake_examples(dataset: str) -> list[dict[str, Any]]:
    rows = []
    for ids in SPLITS.values():
        for item in ids:
            gold = f"value-{item}"
            question = f"question {item} | gold: {gold}"
            if dataset == "bfcl":
                instance = {
                    "id": item,
                    "question": [[{"role": "user", "content": question}]],
                    "function": [ANSWER_SCHEMA],
                    "ground_truth": [{"answer": {"value": [gold]}}],
                }
            else:
                instance = {"id": item, "question": question}
            rows.append({"id": item, "question": question, "answer": gold, "task_instance": instance})
    return rows


def fake_task_data(dataset: str) -> TaskData:
    return TaskData(dataset, SPLITS, fake_examples(dataset))


def hivemind_cell(task: str, topology: str, *, budget: int = 600, seed: int = 0) -> CellSpec:
    return CellSpec(
        method="hivemind",
        task=task,
        topology=topology,
        framework="langgraph",
        optimizer_seed=seed,
        split_hash=fake_task_data(task).split_hash,
        budget=budget,
    )


class AnswerScorer:
    """Exact match of the MAS answer (text answer or the BFCL ``answer`` call)."""

    scorer_id = "fake-answer-match"

    def __call__(self, native: Any, result) -> ScoreResult:
        output = result.final_output or {}
        answer = output.get("answer")
        calls = output.get("model_output") or []
        if not answer and calls and isinstance(calls[0], Mapping):
            answer = (calls[0].get("answer") or {}).get("value")
        score = 1.0 if answer == native["answer"] else 0.0
        return ScoreResult(
            score=score, status="success" if score else "semantic_failure", metadata={"feedback": "offline exact match"}
        )


def protocol_runner(cell: CellSpec, runtime: Any) -> tuple[ProtocolRunner, Any, TaskData]:
    """The protocol runner over ``runtime``, the HiveMind splits and the answer scorer."""
    return protocol_fakes.protocol_runner(cell, runtime, data=fake_task_data(cell.task), scorer=AnswerScorer())


# Scripted model
def _text(message: Any) -> str:
    content = getattr(message, "content", "")
    return content if isinstance(content, str) else str(content)


def _source(message: Any) -> str | None:
    return (getattr(message, "additional_kwargs", None) or {}).get("source")


def _final(answer: str) -> str:
    """Both a BFCL canonical call block and a HotpotQA ``Answer:`` line."""
    return f'```json\n[{{"answer": {{"value": "{answer}"}}}}]\n```\nAnswer: {answer}'


def _delegations(messages: list) -> list[str]:
    return [
        call["name"]
        for message in messages
        if isinstance(message, AIMessage)
        for call in (message.tool_calls or [])
        if call["name"].startswith("delegate_to_")
    ]


class Script:
    """Deterministic policies for every node of the tested runtimes; logs each model call."""

    def __init__(self, workers: tuple[str, ...] = ()) -> None:
        self.workers = tuple(workers)
        self.calls: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def model(self, seed: int | None = 0, **_: Any) -> ScriptedChatModel:
        return ScriptedChatModel(script=self, seed=int(seed or 0))

    def respond(self, messages: list, tool_names: tuple[str, ...], seed: int) -> AIMessage:
        system = _text(messages[0]) if messages and isinstance(messages[0], SystemMessage) else ""
        conversation = "\n".join(_text(message) for message in messages[1:])
        found = re.search(r"question (\w+) \| gold: (\S+)", conversation)
        item, gold = (found.group(1), found.group(2)) if found else ("x1", "unknown")
        even = int(item[-1]) % 2 == 0
        if "delegate_to_" in system:
            role = "manager"
            message = self._manager(messages, system, gold, even)
        elif _delegations(messages):
            worker = _delegations(messages)[-1].removeprefix("delegate_to_")
            role = f"worker:{worker}"
            message = AIMessage(content=f"{worker} report" + (f" {IMPROVED}" if LESSONS_MARKER in system else ""))
        elif "answer" in tool_names:
            role = "replica"
            if isinstance(messages[-1], ToolMessage):
                message = AIMessage(content="submitted")
            else:
                message = AIMessage(
                    content="",
                    tool_calls=[{"name": "answer", "args": {"value": gold}, "id": f"call-{seed}-{len(self.calls)}"}],
                )
        else:
            role = "stage"
            answer = gold if even or LESSONS_MARKER in system else "wrong"
            message = AIMessage(content=_final(answer))
        message.usage_metadata = dict(USAGE)
        with self._lock:
            self.calls.append({"role": role, "seed": seed, "tools": tuple(tool_names)})
        return message

    def _manager(self, messages: list, system: str, gold: str, even: bool) -> AIMessage:
        turn = sum(1 for message in messages if isinstance(message, AIMessage) and _source(message) == "manager")
        if turn < len(self.workers):
            # Delegate to every worker in turn, masked or not: a masked worker's
            # tool does not exist, so the call must bounce back to the manager.
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": f"delegate_to_{self.workers[turn]}",
                        "args": {"instructions": "help"},
                        "id": f"call-m{turn}",
                    }
                ],
            )
        reports = [
            message
            for message in messages
            if isinstance(message, AIMessage) and _source(message) in self.workers and _text(message)
        ]
        responded = {_source(message) for message in reports}
        improved = LESSONS_MARKER in system or any(IMPROVED in _text(message) for message in reports)
        answer = gold if self.workers and self.workers[0] in responded and (even or improved) else "wrong"
        return AIMessage(content=_final(answer) + "\nTERMINATE")

    def roles_called(self) -> list[str]:
        return [call["role"] for call in self.calls]


class ScriptedChatModel(BaseChatModel):
    """LangChain chat model answering from a :class:`Script`; ``bind_tools`` records the bound tool names."""

    script: Any = None
    seed: int = 0
    tool_names: tuple = ()

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> ScriptedChatModel:
        names = tuple(getattr(tool, "name", None) or dict(tool).get("name") for tool in tools)
        return self.model_copy(update={"tool_names": names})

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        return ChatResult(
            generations=[ChatGeneration(message=self.script.respond(list(messages), self.tool_names, self.seed))]
        )


# Hook for adapters without native telemetry
class TelemetryAdapter:
    """Reports the scripted calls of one rollout as native telemetry (BFCL adapters report none)."""

    def __init__(self, inner: Any, script: Script) -> None:
        self._inner = inner
        self._script = script

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def run_example(self, example: Any) -> Any:
        before = len(self._script.calls)
        output = self._inner.run_example(example)
        calls = len(self._script.calls) - before
        if calls:
            output["telemetry"] = {
                "prompt_tokens": 3 * calls,
                "completion_tokens": 2 * calls,
                "total_tokens": 5 * calls,
                "n_llm_calls": calls,
                "n_tool_calls": 0,
            }
        return output


class TelemetryHook(CoalitionExecutionHook):
    def __init__(self, script: Script) -> None:
        super().__init__()
        self.script = script

    def build_adapter(self, runtime: Any, request: Any, control: Any) -> Any:
        return TelemetryAdapter(super().build_adapter(runtime, request, control), self.script)


def bfcl_runtime(cell: CellSpec, adapter_class: type, script: Script, **kwargs: Any) -> AdapterRuntime:
    return AdapterRuntime(
        cell, adapter_class=adapter_class, capture_usage=False, adapter_kwargs={"model_factory": script.model, **kwargs}
    )


def hotpotqa_runtime(cell: CellSpec, adapter_class: type) -> AdapterRuntime:
    return AdapterRuntime(cell, adapter_class=adapter_class, capture_usage=False)


# Reflection
class FakeReflection(FakeReflectionBackend):
    """Deterministic reflection backend that always proposes one lesson block."""

    LESSON = "===BEGIN LESSONS===\n- check every retrieved fact twice\n===END LESSONS==="

    def __init__(self, text: str = LESSON) -> None:
        super().__init__()
        self.text = text

    def respond(self, prompt: str, request: Mapping[str, Any]) -> str:
        return self.text
