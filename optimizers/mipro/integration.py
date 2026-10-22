"""MIPRO (``dspy.teleprompt.MIPROv2``) on the shared run protocol.

Instruction proposal, demo bootstrapping, Optuna trial selection and the
non-minibatch policy remain DSPy MIPROv2 operations; only real-runner
execution is replaced by the protocol runner (budget ledger, logical request
seeds, optimization decoding). Selected demos are rendered into the role
prompts, so the incumbent bundle's prompts are exactly what ran.

Rows MIPRO asks for after the ledger is spent are answered without running
(``optimizers.protocol.rollouts.BudgetStopRunner``); MIPRO then finishes its
schedule and returns its best program.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

from optimizers.protocol.config import PROTOCOL_ID
from optimizers.protocol.errors import OptimizerContractError
from optimizers.protocol.journal import native_checkpoint, persist_optimizer_artifact
from optimizers.protocol.methods.context_fit import fit_to_context
from optimizers.protocol.methods.dspy_bridge import (
    LogicalSeedLM,
    ProtocolRunnerAdapter,
    build_reflection_lm,
    build_task_lm,
    dspy_examples,
    pin_dspy_parallel_executor,
    require_grid_cell,
    runner_score,
)
from optimizers.protocol.rollouts import BudgetedRunner, BudgetStopRunner, load_seed_bundle
from optimizers.protocol.schema import (
    SEARCH_LAYOUT,
    CellSpec,
    LearningCurvePoint,
    OptimizerResult,
    PromptBundle,
    StopReason,
    example_id,
)
from optimizers.protocol.seeding import logical_request_seed

METHOD = "mipro"
OPTIMIZATION_PHASE = "optimization/mipro"
REFLECTION_PHASE = "mipro_reflection"


# Settings
@dataclass(frozen=True)
class MIPROPolicy:
    """MIPRO's settings: the frozen DSPy MIPROv2 configuration (no environment knobs).

    The budget split is MIPROv2's own: ``num_candidates`` instruction/demo
    sets, ``num_trials`` full-validation trials, rows past B answered without
    running.
    """

    backend: str = "dspy.teleprompt.MIPROv2"
    dspy_version: str = "3.2.0"
    auto: None = None
    num_candidates: int = 3
    num_trials: int = 3
    max_bootstrapped_demos: int = 4
    max_labeled_demos: int = 0
    metric_threshold: None = None
    minibatch: bool = False
    minibatch_size: int = 35
    minibatch_full_eval_steps: int = 5
    seed: int = 9
    init_temperature: float = 1.0
    view_data_batch_size: int = 10
    program_aware_proposer: bool = True
    data_aware_proposer: bool = True
    tip_aware_proposer: bool = True
    fewshot_aware_proposer: bool = True
    provide_traceback: bool = False
    max_errors: None = None
    requires_permission_to_run: None = None
    candidate_selection: str = "Optuna TPESampler through DSPy MIPROv2"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


FROZEN_MIPRO_POLICY = MIPROPolicy()


def validate_frozen_policy() -> None:
    """Fail closed if the installed DSPy surface cannot express the policy."""
    import dspy
    from dspy.teleprompt import MIPROv2

    policy = FROZEN_MIPRO_POLICY
    if dspy.__version__ != policy.dspy_version:
        raise RuntimeError(f"MIPRO requires dspy=={policy.dspy_version}; found {dspy.__version__}")
    constructor = inspect.signature(MIPROv2).parameters
    compile_parameters = inspect.signature(MIPROv2.compile).parameters
    required_constructor = {
        "max_bootstrapped_demos",
        "max_labeled_demos",
        "auto",
        "num_candidates",
        "num_threads",
        "max_errors",
        "seed",
        "init_temperature",
        "metric_threshold",
    }
    required_compile = {
        "num_trials",
        "seed",
        "minibatch",
        "minibatch_size",
        "minibatch_full_eval_steps",
        "program_aware_proposer",
        "data_aware_proposer",
        "view_data_batch_size",
        "tip_aware_proposer",
        "fewshot_aware_proposer",
        "requires_permission_to_run",
        "provide_traceback",
    }
    missing_constructor = sorted(required_constructor - set(constructor))
    missing_compile = sorted(required_compile - set(compile_parameters))
    if missing_constructor or missing_compile:
        raise RuntimeError(
            "installed MIPROv2 API cannot express the frozen policy: "
            f"constructor={missing_constructor}, compile={missing_compile}"
        )


# Adapter
class MIPROCommonRunnerAdapter(ProtocolRunnerAdapter):
    """MIPRO's adapter over the protocol runner.

    Request seed of one rollout: ``logical_request_seed(optimizer_seed,
    cell_id, "optimization/mipro", k, example_id, bundle_sha256, k)`` where
    ``k`` counts earlier rollouts of the same bundle on the same example.
    """

    def __init__(self, *, cell: CellSpec, runner: Any, bundle: PromptBundle) -> None:
        super().__init__(cell=cell, runner=runner, bundle=bundle, phase=OPTIMIZATION_PHASE)

    def request_seed(self, item_id: str, bundle: PromptBundle, occurrence: int) -> int:
        return logical_request_seed(
            self.cell.optimizer_seed,
            self.cell.cell_id,
            OPTIMIZATION_PHASE,
            occurrence,
            item_id,
            bundle.digest,
            occurrence,
        )


# Optimizer
class MIPROOptimizer:
    """Run the frozen DSPy MIPROv2 lifecycle through the protocol runner.

    ``reflection_lm`` (the prompt model) is wrapped in
    :class:`~optimizers.protocol.methods.context_fit.ContextFitLM` and
    :class:`LogicalSeedLM` unless it already is a ``LogicalSeedLM``.
    ``task_lm`` is MIPROv2's task model; the adapter-backed program never
    calls it (every rollout is a runner rollout). ``optimizer_factory``
    (default ``dspy.teleprompt.MIPROv2``) is a test hook.
    """

    method = METHOD
    implementation_kind = "native_dspy_mipro_v2_protocol_runner"
    production_eligible = True

    def __init__(
        self,
        *,
        initial_bundle: PromptBundle | None = None,
        reflection_lm: Any | None = None,
        task_lm: Any | None = None,
        optimizer_factory: Callable[..., Any] | None = None,
        num_threads: int = 1,
    ) -> None:
        if num_threads <= 0:
            raise ValueError("num_threads must be positive")
        self.initial_bundle = initial_bundle
        self.reflection_lm = reflection_lm
        self.task_lm = task_lm
        self.optimizer_factory = optimizer_factory
        self.num_threads = int(num_threads)

    @classmethod
    def _validate_cell(cls, cell: CellSpec) -> None:
        if cell.protocol_id != PROTOCOL_ID:
            raise ValueError(f"MIPRO requires protocol_id={PROTOCOL_ID!r}, got {cell.protocol_id!r}")
        if cell.method != cls.method:
            raise ValueError(f"expected method={cls.method!r}, got {cell.method!r}")
        require_grid_cell(cell, ValueError)

    @staticmethod
    def _metric(_example: Any, prediction: Any, *args: Any, **kwargs: Any) -> float:
        del _example, args, kwargs
        return runner_score(prediction, "MIPROv2")

    def optimize(
        self, cell: CellSpec, runner: Any, budget: Any, training: list[Any], validation: list[Any]
    ) -> OptimizerResult:
        """Run native MIPROv2 on the training/validation rows; demos are rendered into the incumbent's prompts."""
        self._validate_cell(cell)
        if not training or not validation:
            raise ValueError("native MIPRO requires non-empty training and validation splits")

        import dspy
        from dspy.teleprompt import MIPROv2

        from optimizers.bridge.mipro_programs import MIPROAdapterBackedProgram

        validate_frozen_policy()
        pin_dspy_parallel_executor()
        seed_bundle = load_seed_bundle(cell, runner, self.initial_bundle)
        protocol_runner = BudgetStopRunner(BudgetedRunner(cell=cell, runner=runner, budget=budget))
        common_adapter = MIPROCommonRunnerAdapter(cell=cell, runner=protocol_runner, bundle=seed_bundle)
        program = MIPROAdapterBackedProgram(common_adapter)
        trainset = dspy_examples(training)
        valset = dspy_examples(validation)
        policy = FROZEN_MIPRO_POLICY
        raw_reflection_lm = self.reflection_lm or build_reflection_lm()
        reflection_lm = (
            raw_reflection_lm
            if isinstance(raw_reflection_lm, LogicalSeedLM)
            else LogicalSeedLM(cell=cell, lm=fit_to_context(raw_reflection_lm), phase=REFLECTION_PHASE)
        )
        task_lm = self.task_lm or build_task_lm(cell.task_model)
        dspy.configure(lm=task_lm, track_usage=True)
        effective_seed = int(policy.seed) + int(cell.optimizer_seed)
        factory = self.optimizer_factory or MIPROv2
        optimizer = factory(
            metric=self._metric,
            prompt_model=reflection_lm,
            task_model=task_lm,
            max_bootstrapped_demos=policy.max_bootstrapped_demos,
            max_labeled_demos=policy.max_labeled_demos,
            auto=policy.auto,
            num_candidates=policy.num_candidates,
            num_threads=self.num_threads,
            max_errors=policy.max_errors,
            seed=effective_seed,
            init_temperature=policy.init_temperature,
            verbose=False,
            track_stats=True,
            # Candidate programs stay in memory: the job's artifacts are JSON only.
            log_dir=None,
            metric_threshold=policy.metric_threshold,
        )
        initial_snapshot = dict(budget.snapshot())
        compiled = optimizer.compile(
            program,
            trainset=trainset,
            valset=valset,
            num_trials=policy.num_trials,
            seed=effective_seed,
            minibatch=policy.minibatch,
            minibatch_size=policy.minibatch_size,
            minibatch_full_eval_steps=policy.minibatch_full_eval_steps,
            program_aware_proposer=policy.program_aware_proposer,
            data_aware_proposer=policy.data_aware_proposer,
            view_data_batch_size=policy.view_data_batch_size,
            tip_aware_proposer=policy.tip_aware_proposer,
            fewshot_aware_proposer=policy.fewshot_aware_proposer,
            provide_traceback=policy.provide_traceback,
            requires_permission_to_run=policy.requires_permission_to_run,
        )
        if hasattr(compiled, "sync_prompts_to_adapter"):
            compiled.sync_prompts_to_adapter()
        role_artifacts = compiled.role_artifacts()
        roles = {str(item["role"]): str(item["prompt"]) for item in role_artifacts}
        if set(roles) != set(seed_bundle.roles):
            raise OptimizerContractError(
                f"compiled MIPRO roles changed: expected={sorted(seed_bundle.roles)}, actual={sorted(roles)}"
            )
        selected_demos = {str(item["role"]): tuple(item.get("demos") or ()) for item in role_artifacts}
        selected = PromptBundle(
            roles=roles,
            demos=tuple(seed_bundle.demos),
            metadata={
                **dict(seed_bundle.metadata),
                "optimizer": METHOD,
                "native_compiled": True,
                "selected_demos_by_role": selected_demos,
            },
        )
        final_snapshot = dict(budget.snapshot())
        checkpoint = native_checkpoint(
            method=METHOD,
            iteration=policy.num_trials,
            bundle=selected,
            budget=budget,
            state={
                "native_compile_complete": True,
                "effective_seed": effective_seed,
                "selected_demos_by_role": selected_demos,
            },
        )
        artifact = OptimizerResult(
            layout=SEARCH_LAYOUT,
            method=METHOD,
            implementation_kind=self.implementation_kind,
            production_eligible=True,
            cell_id=cell.cell_id,
            seed_bundle=seed_bundle,
            incumbent_bundle=selected,
            budget_snapshot=final_snapshot,
            stop_reason=StopReason.ROLLOUT_BUDGET_SPENT if protocol_runner.answered else StopReason.NUM_TRIALS_COMPLETE,
            learning_curve=(
                LearningCurvePoint("initial", 0, int(initial_snapshot["charged"]), None, seed_bundle.digest),
                LearningCurvePoint(
                    "completed_native_mipro_trials",
                    policy.num_trials,
                    int(final_snapshot["charged"]),
                    None,
                    selected.digest,
                ),
            ),
            checkpoints=(checkpoint,),
            reflection_requests=tuple(getattr(reflection_lm, "requests", ())),
            metadata={
                "native_backend": policy.backend,
                "native_optimizer_executed": True,
                "native_policy": policy.to_dict(),
                "effective_lifecycle": {
                    "num_candidates": policy.num_candidates,
                    "num_trials": policy.num_trials,
                    "max_bootstrapped_demos": policy.max_bootstrapped_demos,
                    "view_data_batch_size": policy.view_data_batch_size,
                },
                "effective_seed": effective_seed,
                "rows_answered_past_budget": protocol_runner.answered,
                "selected_demos_by_role": selected_demos,
                "training_ids": [example_id(row, index) for index, row in enumerate(training)],
                "validation_ids": [example_id(row, index) for index, row in enumerate(validation)],
            },
        )
        persist_optimizer_artifact(runner, artifact)
        return artifact


__all__ = [
    "FROZEN_MIPRO_POLICY",
    "MIPROCommonRunnerAdapter",
    "MIPROOptimizer",
    "MIPROPolicy",
    "validate_frozen_policy",
]
