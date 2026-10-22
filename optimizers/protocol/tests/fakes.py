"""Fakes shared by every protocol and method test: adapter, dataset, scorer, runners, toy optimizers."""

from __future__ import annotations

import hashlib
import os
import threading
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

from ..artifacts import ArtifactStore
from ..budget import BudgetLedger
from ..config import DEFAULT_TASK_MODEL
from ..rollouts import BudgetedRunner
from ..run import JobHooks
from ..runner import AdapterRuntime, CellExecutor, ProtocolRunner, ScoreResult, TaskData
from ..schema import SEARCH_LAYOUT, CellSpec, OptimizerResult, PromptBundle
from ..seeding import request_seeds
from ..session import RunnerSession

# Config
DATASET = "fake"
ROLES = ("planner", "writer")
SEED_PROMPTS = {"planner": "Plan the answer.", "writer": "Write the final answer."}
SPLITS = {
    "train": [f"tr{i}" for i in range(6)],
    "validation": [f"va{i}" for i in range(4)],
    "test": [f"te{i}" for i in range(6)],
}


class BadRequestError(Exception):
    """Same class name as the OpenAI SDK's HTTP 400 error."""


class PreObservationFailure(RuntimeError):
    """Same class name as the Agents SDK pre-observation failure."""


class FakeAdapter:
    """Two-role prompt-mutable adapter whose answers depend on the prompts.

    Seed prompts answer even-numbered items correctly; a writer prompt with
    ``GOOD`` answers everything; ``BAD`` answers nothing. ``SCRIPT`` maps an
    example ID to a list of per-attempt behaviors consumed in order.
    """

    topology = "sequential"
    dataset = DATASET
    SCRIPT: dict[str, list[str]] = {}
    CALLS: list[dict[str, Any]] = []

    def __init__(self, prompts: dict[str, str] | None = None) -> None:
        self._prompts = dict(SEED_PROMPTS, **(prompts or {}))

    def roles(self) -> list[str]:
        return list(ROLES)

    def get_prompt(self, role: str) -> str:
        return self._prompts[role]

    def set_prompt(self, role: str, text: str) -> None:
        self._prompts[role] = text

    def reset(self) -> None:
        return None

    def run_example(self, example: dict) -> dict:
        item = str(example["id"])
        observed = {
            "id": item,
            "seed": int(os.environ["REQUEST_SEED"]),
            "temperature": float(os.environ["TASK_MODEL_TEMPERATURE"]),
            "model": os.environ["MODEL_ID"],
            "writer": self._prompts["writer"],
            "keys": sorted(example),
        }
        FakeAdapter.CALLS.append(observed)
        script = FakeAdapter.SCRIPT.get(item)
        behavior = script.pop(0) if script else "ok"
        if behavior == "connection":
            raise ConnectionError("endpoint refused the connection")
        if behavior == "bad_request":
            raise BadRequestError("Error code: 400 - context length exceeded")
        if behavior == "sdk_pre_observation":
            raise PreObservationFailure("Agents SDK failed before an observation")
        if behavior == "runtime_bug":
            raise KeyError("unexpected native failure")
        writer = self._prompts["writer"]
        correct = "BAD" not in writer and ("GOOD" in writer or int(item[2:]) % 2 == 0)
        output = {
            "answer": example["question"].upper() if correct else "wrong",
            "runner_output": {
                "messages": [{"source": "planner", "content": "plan"}, {"source": "writer", "content": "final"}]
            },
            "telemetry": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "n_llm_calls": 0 if behavior == "zero_calls" else 2,
                "n_tool_calls": 0,
            },
        }
        if behavior == "transport_text":
            output["error"] = "APITimeoutError: Request timed out."
        if behavior == "scorer_crash":
            output["answer"] = None
        return output


class FakeScorer:
    scorer_id = "fake-exact-match"

    def __call__(self, native: Any, result) -> ScoreResult:
        answer = result.final_output["answer"]
        if answer is None:
            raise RuntimeError("scorer could not parse the output")
        score = 1.0 if answer == native["answer"] else 0.0
        return ScoreResult(
            score=score, status="success" if score else "semantic_failure", metadata={"feedback": "fake"}
        )


def fake_examples() -> list[dict[str, Any]]:
    rows = []
    for split, ids in SPLITS.items():
        for item in ids:
            question = f"question {item}"
            rows.append(
                {
                    "id": item,
                    "question": question,
                    "answer": question.upper(),
                    "split_hint": split,
                    "task_instance": {"id": item, "question": question, "ground_truth": "hidden"},
                }
            )
    return rows


def fake_task_data(dataset: str = DATASET) -> TaskData:
    return TaskData(dataset, SPLITS, fake_examples())


def fake_runtime(cell: CellSpec, adapter_class: type = FakeAdapter, **kwargs: Any) -> AdapterRuntime:
    """The cell's runtime over a fake adapter class (no OpenAI response capture)."""
    return AdapterRuntime(cell, adapter_class=adapter_class, capture_usage=False, **kwargs)


def fake_hooks() -> JobHooks:
    return JobHooks(
        load_task_data=fake_task_data,
        build_runtime=fake_runtime,
        build_scorer=lambda cell: FakeScorer(),
        configure_environment=False,
    )


def fake_cell(
    *,
    method: str = "toy",
    seed: int = 0,
    budget: int = 600,
    task: str = DATASET,
    topology: str = "sequential",
    framework: str = "langgraph",
    team_size: int = 4,
    task_model: str = DEFAULT_TASK_MODEL,
    communication: str = "freeform",
    source_tables: tuple[int, ...] = (),
) -> CellSpec:
    """A cell over the fake dataset rows (``task`` selects the task name, e.g. a Table-6 grid task)."""
    return CellSpec(
        method=method,
        task=task,
        topology=topology,
        framework=framework,
        communication=communication,
        team_size=team_size,
        task_model=task_model,
        optimizer_seed=seed,
        split_hash=fake_task_data(task).split_hash,
        budget=budget,
        source_tables=source_tables,
    )


def protocol_runner(
    cell: CellSpec,
    runtime: Any = None,
    *,
    data: TaskData | None = None,
    scorer: Any = None,
    store: ArtifactStore | None = None,
    directory: Any = None,
    retries: int = 2,
) -> tuple[ProtocolRunner, BudgetLedger, TaskData]:
    """A budget-owning protocol runner over ``runtime`` (default: the fake adapter) and the fake dataset."""
    data = data if data is not None else fake_task_data(cell.task)
    runtime = runtime if runtime is not None else fake_runtime(cell)
    executor = CellExecutor(cell=cell, runtime=runtime, scorer=scorer or FakeScorer(), data=data)
    budget = BudgetLedger(cell.budget)
    runner = ProtocolRunner(
        cell=cell,
        executor=executor,
        budget=budget,
        seed_bundle=runtime.seed_bundle(),
        max_infrastructure_retries=retries,
        artifact_store=store,
        artifact_directory=directory,
    )
    return runner, budget, data


def fake_runner(cell: CellSpec | None = None, *, retries: int = 2) -> tuple[ProtocolRunner, BudgetLedger, TaskData]:
    """The protocol runner of the fake cell."""
    return protocol_runner(cell or fake_cell(), retries=retries)


def reset_fake() -> None:
    FakeAdapter.SCRIPT = {}
    FakeAdapter.CALLS = []


# Reflection
class FakeReflectionBackend:
    """Offline reflection backend (the ``ReflectionBackend`` protocol) that records every request.

    Subclasses answer through :meth:`respond`; it runs under the backend's
    lock, so stateful fakes need no lock of their own. A request whose
    ``respond`` raises stays recorded without an ``output``.
    """

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "n_calls": 0}
        self._lock = threading.RLock()

    def respond(self, prompt: str, request: Mapping[str, Any]) -> str:
        raise NotImplementedError

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
        request = {
            "phase": phase,
            "role": role,
            "request_seed": int(request_seed),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "max_output_tokens": int(max_output_tokens),
            "thinking": bool(thinking),
            "system": system,
            "system_sha256": hashlib.sha256((system or "").encode()).hexdigest(),
            "prompt": prompt,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        }
        with self._lock:
            self.requests.append(request)
            self.usage["n_calls"] += 1
            self.usage["prompt_tokens"] += 10
            self.usage["completion_tokens"] += 5
            request["output"] = self.respond(prompt, request)
        return request["output"]

    def snapshot(self) -> Mapping[str, Any]:
        with self._lock:
            requests = [{key: value for key, value in request.items() if key != "prompt"} for request in self.requests]
        return {
            "usage": {
                "model_calls": len(requests),
                "tool_calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            },
            "requests": requests,
        }


class FakeOpenAI:
    """OpenAI SDK stand-in for :class:`~optimizers.protocol.reflection.ReflectionClient`: records request bodies."""

    def __init__(self, content: str = "improved prompt") -> None:
        self.bodies: list[dict[str, Any]] = []
        self.content = content
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **body: Any) -> Any:
        self.bodies.append(body)
        message = SimpleNamespace(role="assistant", content=self.content)
        return SimpleNamespace(
            choices=[SimpleNamespace(index=0, finish_reason="stop", message=message)],
            usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7, completion_tokens_details=None),
        )


# Toy optimizers (registered by tests)
class ToyOptimizer:
    """Spend train rollouts, then propose a writer prompt with ``WORD`` and check it on validation."""

    WORD = "GOOD"

    def __init__(self, initial_bundle: PromptBundle) -> None:
        self.initial_bundle = initial_bundle

    def optimize(self, cell, runner, budget, training, validation) -> OptimizerResult:
        seed = self.initial_bundle
        rollout = BudgetedRunner(cell=cell, runner=runner, budget=budget)
        rollout.run_batch(training, seed, request_seeds(cell, training, phase="toy/train", iteration=0, bundle=seed))
        candidate = PromptBundle(
            roles={**seed.roles, "writer": seed.roles["writer"] + f" {self.WORD}"},
            metadata={"proposal": self.WORD.lower()},
        )
        session = RunnerSession(cell=cell, runner=runner, budget=budget, seed_bundle=candidate)
        session.set_context(iteration=1, event="candidate_validation")
        scores = [session.run_record(row).score for row in validation]
        return OptimizerResult(
            layout=SEARCH_LAYOUT,
            schema="toy",
            method=cell.method,
            implementation_kind="toy",
            cell_id=cell.cell_id,
            seed_bundle=seed,
            incumbent_bundle=candidate,
            budget_snapshot=budget.snapshot(),
            stop_reason="toy_done",
            metadata={"native_validation": sum(scores) / len(scores)},
        )


class BadToyOptimizer(ToyOptimizer):
    WORD = "BAD"


class InfraToyOptimizer(ToyOptimizer):
    """Hits an item whose every attempt fails before observation."""

    def optimize(self, cell, runner, budget, training, validation):
        FakeAdapter.SCRIPT[str(training[0]["id"])] = ["connection"] * 3
        return super().optimize(cell, runner, budget, training, validation)
