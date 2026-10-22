"""TRUCE (TAVO) on the shared run protocol: fixed batches, validation gate, retry and patience.

Default ``truce-release`` lifecycle (release-faithful, ``max_outer_rounds`` 5):

* fixed batches sampled once with ``random.Random(233 + optimizer_seed)``: train
  batch 6, validation batch ``max(3, (B - 5 * train) // 6)`` capped at the split;
* the seed bundle is scored on the validation batch first (charged), then each
  outer round runs one attempt (``driver.run_optimization_attempt``: train batch,
  credit assignment, meta-knowledge, shared overlay) and scores the candidate on
  the validation batch (charged);
* a candidate is adopted only if it beats the seed reference by at least 0.01 and
  is not below the best so far; a rejected round gets one retry; two consecutive
  rejected rounds stop the search; a round or retry starts only if its train and
  validation rollouts both fit in the remaining budget.

The train chain always advances to the newest candidate, adopted or not (upstream
``meta_knowledge_path`` semantics). Every rollout goes through the protocol runner.
"""

from __future__ import annotations

import json
import random
from collections.abc import Mapping, Sequence
from typing import Any

from optimizers.protocol.cells import validate_cell
from optimizers.protocol.errors import OptimizerContractError
from optimizers.protocol.journal import native_checkpoint, persist_optimizer_artifact
from optimizers.protocol.learning_curve import LearningCurveRecorder
from optimizers.protocol.reflection import LogicalTextReflectionClient, ReflectionClient
from optimizers.protocol.rollouts import BudgetedRunner, load_seed_bundle, ordered_prompt_roles
from optimizers.protocol.schema import (
    SEARCH_LAYOUT,
    CellSpec,
    LearningCurvePoint,
    OptimizerResult,
    PromptBundle,
    RunRecord,
    StopReason,
    example_id,
)
from optimizers.protocol.seeding import request_seeds

from .driver import (
    BENCHMARK_ADAPTATIONS,
    DEFAULT_OPTIMIZER_MODE,
    MODE_CLASSIFICATION,
    OPTIMIZER_MODES,
    TRUCE_RELEASE,
    batch_mean,
    run_optimization_attempt,
)
from .settings import DEFAULTS, TAVOSettings, trajectory_credit_enabled


def _ordered_roles(cell: CellSpec, bundle: PromptBundle) -> list[str]:
    """Native role order of the runtime; a centralized manager always comes first."""
    roles = list(ordered_prompt_roles(cell, bundle))
    if cell.topology == "centralized":
        manager = next((role for role in roles if role == "manager" or role.startswith("manager_r")), None)
        if manager is not None:
            roles = [manager] + [role for role in roles if role != manager]
    return roles


def _messages(record: RunRecord) -> list[dict[str, Any]]:
    """Source-tag every message; structured contents become JSON text."""
    normalized = []
    for message in record.messages:
        item = dict(message)
        if not item.get("source"):
            kind = item.get("role") or item.get("type")
            item["source"] = (
                item.get("name")
                or item.get("agent")
                or {"human": "user", "ai": "assistant"}.get(kind, kind)
                or "runtime"
            )
        content = item.get("content")
        if content is not None and not isinstance(content, str):
            item["content"] = json.dumps(content, ensure_ascii=False, sort_keys=True, default=str)
        calls = item.get("tool_calls")
        if calls is not None:
            item["tool_calls"] = (
                [dict(call) for call in calls if isinstance(call, Mapping)] if isinstance(calls, (list, tuple)) else []
            )
        normalized.append(item)
    return normalized


class _TavoRunnerAdapter:
    """Prompt-mutable view of the cell whose batches run through the budgeted runner."""

    def __init__(
        self, *, cell: CellSpec, runner: BudgetedRunner, seed_bundle: PromptBundle, roles: Sequence[str]
    ) -> None:
        self.cell = cell
        self.dataset = cell.task
        self.topology = cell.topology
        self.framework = cell.framework
        self._runner = runner
        self._roles = list(roles)
        self._prompts = dict(seed_bundle.roles)
        self._demos = tuple(seed_bundle.demos)
        self._metadata = dict(seed_bundle.metadata)

    def roles(self) -> list[str]:
        return list(self._roles)

    def get_prompt(self, role: str) -> str:
        return self._prompts[role]

    def set_prompt(self, role: str, text: str) -> None:
        if role not in self._prompts:
            raise KeyError(role)
        if not isinstance(text, str) or not text.strip():
            raise ValueError("role prompts must be non-empty strings")
        self._prompts[role] = text

    def reset(self) -> None:
        return None

    def bundle(self, prompts: Mapping[str, str]) -> PromptBundle:
        if set(prompts) != set(self._prompts):
            raise OptimizerContractError("TAVO candidate changed the runtime role interface")
        return PromptBundle(
            roles={role: str(prompts[role]) for role in self._roles}, demos=self._demos, metadata=self._metadata
        )

    def execute_batch(
        self, prompts: Mapping[str, str], rows: Sequence[Any], *, phase: str, iteration: int
    ) -> list[dict[str, Any]]:
        bundle = self.bundle(prompts)
        seeds = request_seeds(self.cell, rows, phase=phase, iteration=iteration, bundle=bundle)
        records = self._runner.run_batch(rows, bundle, seeds)
        trajectories = []
        for index, (row, record) in enumerate(zip(rows, records)):
            if not record.usable or record.score is None:
                raise OptimizerContractError(f"TAVO received unusable rollout {record.example_id!r} after retries")
            final = record.final_output
            answer = final.get("answer") if isinstance(final, Mapping) else final
            trajectories.append(
                {
                    "id": record.example_id or example_id(row, index),
                    "question": str(
                        row.get("problem", row.get("question", ""))
                        if isinstance(row, Mapping)
                        else getattr(row, "problem", getattr(row, "question", ""))
                    ),
                    "messages": _messages(record),
                    "topology": self.topology,
                    "team_size": self.cell.team_size,
                    "task_model": self.cell.task_model,
                    "prompt_roles": list(self._roles),
                    "answer": answer,
                    "score": float(record.score),
                    "error": record.error,
                    "telemetry": {**dict(record.metadata), "total_tokens": int(record.usage.total_tokens)},
                    "executed": True,
                    "request_seed": record.request_seed,
                }
            )
        return trajectories


class _BudgetView:
    """Read-only native pacing view over the runner-owned ledger."""

    def __init__(self, budget: Any) -> None:
        self.ledger = budget
        self.cap = int(budget.maximum)
        self.truncations = 0
        self.phase = "tavo"
        self.iteration = 0

    @property
    def used(self) -> int:
        return int(self.ledger.snapshot()["charged"])

    def remaining(self) -> int:
        return int(self.ledger.snapshot()["remaining"])

    def can_afford(self, count: int) -> bool:
        return self.remaining() >= int(count)

    def note_truncation(self) -> None:
        self.truncations += 1


class TAVOOptimizer:
    """Release-faithful TRUCE overlay optimization on protocol-runner trajectories.

    ``mode`` selects ``truce-release`` (default), ``tavo-hybrid`` or
    ``tavo-paper-reproduction``. ``trajectory_credit`` toggles Eq. 3 credit
    (default from ``TAVO_CREDIT``, on unless set to 0); off is the no-credit
    ablation. ``reflection_client`` defaults to the protocol :class:`ReflectionClient`.
    """

    method = "tavo"
    native_iteration_event = "completed outer optimization round"

    def __init__(
        self,
        *,
        initial_bundle: PromptBundle | None = None,
        reflection_client: Any | None = None,
        mode: str = DEFAULT_OPTIMIZER_MODE,
        trajectory_credit: bool | None = None,
        reflection_inflight: int = DEFAULTS.reflection_inflight,
        validation_one_cycle: bool = False,
    ) -> None:
        if mode not in OPTIMIZER_MODES:
            raise ValueError(f"unsupported TAVO mode {mode!r}; choices={OPTIMIZER_MODES}")
        if reflection_inflight <= 0:
            raise ValueError("concurrency limits must be positive")
        self.settings = TAVOSettings(reflection_inflight=int(reflection_inflight))
        self.initial_bundle = initial_bundle
        self.reflection_client = (
            reflection_client
            if reflection_client is not None
            else ReflectionClient(max_retries=self.settings.reflection_max_retries)
        )
        self.mode = mode
        self.trajectory_credit = trajectory_credit_enabled() if trajectory_credit is None else bool(trajectory_credit)
        self.validation_one_cycle = bool(validation_one_cycle)

    @property
    def implementation_kind(self) -> str:
        """Result label of the selected mode."""
        return f"native_{self.mode.replace('-', '_')}_common_runner"

    def optimize(
        self, cell: CellSpec, runner: Any, budget: Any, training: list[Any], validation: list[Any]
    ) -> OptimizerResult:
        """Score the seed on the validation batch, then run outer rounds while their rollouts fit the budget."""
        settings = self.settings
        validate_cell(self.method, cell)
        if not training or not validation:
            raise ValueError("native TAVO requires non-empty training and validation splits")

        seed_bundle = load_seed_bundle(cell, runner, self.initial_bundle)
        roles = _ordered_roles(cell, seed_bundle)
        protocol_runner = BudgetedRunner(cell=cell, runner=runner, budget=budget)
        native_runner = _TavoRunnerAdapter(cell=cell, runner=protocol_runner, seed_bundle=seed_bundle, roles=roles)
        reflection = LogicalTextReflectionClient(
            cell=cell,
            client=self.reflection_client,
            method="tavo",
            default_temperature=settings.default_temperature,
            default_top_p=settings.default_top_p,
        )
        view = _BudgetView(budget)
        rng_seed = settings.rng_seed_base + int(cell.optimizer_seed)
        rng = random.Random(rng_seed)
        train_size = 1 if self.validation_one_cycle else min(len(training), settings.train_batch_size)
        train_batch = rng.sample(list(training), train_size)
        validation_size = (
            1 if self.validation_one_cycle else settings.validation_batch_size(cell.budget, train_size, len(validation))
        )
        validation_batch = rng.sample(list(validation), validation_size)

        def native_batch_runner(
            _runner: Any,
            _metric: Any,
            prompts: Mapping[str, str],
            rows: Sequence[Any],
            _num_threads: int,
            native_budget: _BudgetView,
        ) -> list[dict[str, Any]]:
            del _runner, _metric, _num_threads
            return native_runner.execute_batch(
                prompts, rows, phase=native_budget.phase, iteration=native_budget.iteration
            )

        initial_snapshot = dict(budget.snapshot())
        learning_curve = [LearningCurvePoint("initial", 0, int(initial_snapshot["charged"]), None, seed_bundle.digest)]
        curve = LearningCurveRecorder(maximum=int(budget.maximum))
        curve.observe(int(initial_snapshot["charged"]), seed_bundle.digest, 0, None, "initial_state")
        checkpoints: list[Mapping[str, Any]] = []
        seed_prompts = {role: seed_bundle.roles[role] for role in roles}
        view.phase = "tavo_seed_validation"
        base_trajectories = native_runner.execute_batch(seed_prompts, validation_batch, phase=view.phase, iteration=0)
        base_score = batch_mean(base_trajectories)
        learning_curve.append(
            LearningCurvePoint(
                "seed_validation",
                0,
                view.used,
                base_score,
                seed_bundle.digest,
                {"validation_ids": [item["id"] for item in base_trajectories]},
            )
        )
        curve.observe(view.used, seed_bundle.digest, 0, base_score, "seed_validation")
        best_score = base_score
        best_prompts = dict(seed_prompts)
        current_prompts = dict(seed_prompts)
        best_iteration = 0
        patience_counter = 0
        iteration_summaries = []
        stopped_reason = None
        effective_max_iterations = 1 if self.validation_one_cycle else settings.max_outer_rounds

        for iteration in range(1, effective_max_iterations + 1):
            iteration_cost = len(train_batch) + len(validation_batch)
            if not view.can_afford(iteration_cost):
                view.note_truncation()
                stopped_reason = StopReason.BUDGET_BEFORE_OUTER_ROUND
                break
            adopted = False
            attempts = []
            for attempt in range(1, settings.attempts_per_round + 1):
                if attempt > 1 and not view.can_afford(iteration_cost):
                    view.note_truncation()
                    stopped_reason = StopReason.BUDGET_BEFORE_RETRY
                    break
                view.phase = f"tavo_train/iteration_{iteration}/attempt_{attempt}"
                view.iteration = iteration
                # Scores come from the protocol runner, so no native metric is passed.
                candidate, detail = run_optimization_attempt(
                    native_runner,
                    None,
                    roles,
                    seed_prompts,
                    current_prompts,
                    train_batch,
                    reflection,
                    1,
                    view,
                    cell.task,
                    settings.reflection_inflight,
                    iteration=iteration,
                    optimizer_mode=self.mode,
                    batch_runner=native_batch_runner,
                    trajectory_credit=self.trajectory_credit,
                )
                view.phase = f"tavo_validation/iteration_{iteration}/attempt_{attempt}"
                validation_trajectories = native_runner.execute_batch(
                    candidate, validation_batch, phase=view.phase, iteration=iteration
                )
                validation_score = batch_mean(validation_trajectories)
                improvement_vs_base = validation_score - base_score
                improvement_over_best = validation_score - best_score
                adopt = (
                    improvement_vs_base >= settings.adoption_threshold
                    and improvement_over_best >= settings.validation_delta_tolerance
                )
                detail.update(
                    {
                        "iteration": iteration,
                        "attempt": attempt,
                        "validation_score": validation_score,
                        "improvement_vs_base": improvement_vs_base,
                        "improvement_over_best": improvement_over_best,
                        "adopted": adopt,
                        "rollouts_used_after": view.used,
                    }
                )
                attempts.append(detail)
                current_prompts = dict(candidate)
                if adopt:
                    best_score = validation_score
                    best_prompts = dict(candidate)
                    best_iteration = iteration
                    patience_counter = 0
                    adopted = True
                    break
            if not adopted:
                patience_counter += 1
            iteration_summaries.append(
                {
                    "iteration": iteration,
                    "attempts": attempts,
                    "adopted": adopted,
                    "best_score": best_score,
                    "patience_counter": patience_counter,
                    "rollouts_used_after": view.used,
                }
            )
            best_bundle = PromptBundle(roles=best_prompts, demos=seed_bundle.demos, metadata=seed_bundle.metadata)
            learning_curve.append(
                LearningCurvePoint(
                    "completed_outer_round",
                    iteration,
                    view.used,
                    best_score,
                    best_bundle.digest,
                    {"adopted": adopted, "attempts": len(attempts)},
                )
            )
            curve.observe(
                view.used,
                best_bundle.digest,
                iteration,
                best_score,
                "native_iteration_end",
                metadata={"adopted": adopted, "attempts": len(attempts)},
            )
            checkpoints.append(
                native_checkpoint(
                    method=self.method,
                    iteration=iteration,
                    bundle=best_bundle,
                    budget=budget,
                    state={
                        "current_prompts": current_prompts,
                        "best_prompts": best_prompts,
                        "best_score": best_score,
                        "best_iteration": best_iteration,
                        "patience_counter": patience_counter,
                        "fixed_train_ids": [example_id(row, i) for i, row in enumerate(train_batch)],
                        "fixed_validation_ids": [example_id(row, i) for i, row in enumerate(validation_batch)],
                    },
                )
            )
            if patience_counter >= settings.patience:
                stopped_reason = StopReason.NATIVE_PATIENCE
                break

        curve.carry_forward_after_stop()
        selected = PromptBundle(
            roles=best_prompts,
            demos=seed_bundle.demos,
            metadata={**dict(seed_bundle.metadata), "optimizer": "tavo", "native_mode": self.mode},
        )
        artifact = OptimizerResult(
            layout=SEARCH_LAYOUT,
            method=self.method,
            implementation_kind=self.implementation_kind,
            production_eligible=not self.validation_one_cycle,
            cell_id=cell.cell_id,
            seed_bundle=seed_bundle,
            incumbent_bundle=selected,
            budget_snapshot=dict(budget.snapshot()),
            stop_reason=stopped_reason or StopReason.MAX_OUTER_ROUNDS,
            learning_curve=tuple(learning_curve),
            checkpoints=tuple(checkpoints),
            reflection_requests=tuple(reflection.requests),
            metadata={
                "native_backend": "TRUCE release overlay pipeline",
                "native_optimizer_executed": True,
                "validation_only": self.validation_one_cycle,
                "effective_lifecycle": {
                    "max_iterations": effective_max_iterations,
                    "train_batch_size": train_size,
                    "validation_batch_size": validation_size,
                },
                "settings": {
                    "adoption_threshold": settings.adoption_threshold,
                    "validation_delta_tolerance": settings.validation_delta_tolerance,
                    "attempts_per_round": settings.attempts_per_round,
                    "patience": settings.patience,
                    "max_outer_rounds": settings.max_outer_rounds,
                    "reflection_inflight": settings.reflection_inflight,
                },
                "mode": self.mode,
                "mode_classification": MODE_CLASSIFICATION[self.mode],
                "method_provenance": {
                    "official_release": dict(TRUCE_RELEASE),
                    "benchmark_adaptations": list(BENCHMARK_ADAPTATIONS),
                },
                "equation_3_credit": self.trajectory_credit,
                "trajectory_credit_ablation": None if self.trajectory_credit else "eq3-disabled",
                "rng_seed": rng_seed,
                "role_order": roles,
                "topology": cell.topology,
                "train_batch_size": train_size,
                "validation_batch_size": validation_size,
                "train_batch_ids": [example_id(row, i) for i, row in enumerate(train_batch)],
                "validation_batch_ids": [example_id(row, i) for i, row in enumerate(validation_batch)],
                "base_score": base_score,
                "best_score": best_score,
                "best_iteration": best_iteration,
                "iterations": iteration_summaries,
                "budget_truncations": view.truncations,
                "stopped_reason": stopped_reason,
                "reflection_usage": dict(reflection.usage),
            },
        )
        persist_optimizer_artifact(runner, artifact)
        store, directory = getattr(runner, "artifact_store", None), getattr(runner, "artifact_directory", None)
        if store is not None and directory is not None:
            for row in curve.rows:
                store.append_jsonl(directory, "learning_curve.jsonl", row)
        return artifact


__all__ = ["TAVOOptimizer"]
