"""GEPA (``dspy.teleprompt.GEPA``) on the shared run protocol.

The DSPy program is the real-runner ``AdapterBackedProgram``; only its
execution adapter is replaced (:class:`CommonRunnerAdapter`), so every
rollout is a protocol-runner rollout: budget ledger, logical request seeds and
optimization decoding. Candidate mutation, Pareto selection, merge,
full-evaluation pacing and early stopping are native GEPA behavior.

Rows GEPA asks for after the ledger is spent are answered without running
(``optimizers.protocol.rollouts.BudgetStopRunner``) and the full-evaluation
stopper halts the search once B rollouts have been requested, so the incumbent
is the best fully evaluated candidate.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from importlib.metadata import version
from typing import Any

from optimizers.protocol.config import PROTOCOL_ID
from optimizers.protocol.errors import OptimizerContractError, UnsupportedGEPACell
from optimizers.protocol.journal import native_checkpoint, persist_optimizer_artifact
from optimizers.protocol.methods.context_fit import fit_to_context
from optimizers.protocol.methods.dspy_bridge import (
    LogicalSeedLM,
    ProtocolRunnerAdapter,
    build_reflection_lm,
    dspy_examples,
    pin_dspy_parallel_executor,
    require_grid_cell,
    runner_score,
)
from optimizers.protocol.rollouts import BudgetedRunner, BudgetStopRunner, Runner, load_seed_bundle
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

logger = logging.getLogger(__name__)

METHOD = "gepa"
OPTIMIZATION_PHASE = "optimization/gepa"
REFLECTION_PHASE = "gepa_reflection"
SUPPORTED_TASKS = frozenset({"gpqa", "hotpotqa", "math", "lcb", "apps", "swe", "bfcl", "toolhop", "apibank"})
SUPPORTED_FRAMEWORKS_BY_TOPOLOGY: Mapping[str, frozenset[str]] = {
    "single": frozenset({"langgraph"}),
    "independent": frozenset({"langgraph"}),
    "sequential": frozenset({"langgraph", "crewai"}),
    "centralized": frozenset({"langgraph", "autogen"}),
    "decentralized": frozenset({"langgraph", "openai_agents"}),
}


# Settings
@dataclass(frozen=True)
class GEPAPolicy:
    """GEPA's settings: the frozen DSPy GEPA configuration (no environment knobs).

    The budget split is GEPA's own: ``max_full_evals`` full validation passes
    and reflection minibatches of 3, stopped at B (:class:`FullEvalPlateauStopper`).
    """

    backend: str = "dspy.teleprompt.GEPA"
    dspy_version: str = "3.2.0"
    gepa_version: str = "0.0.27"
    auto: None = None
    max_full_evals: int = 5
    reflection_minibatch_size: int = 3
    candidate_selection_strategy: str = "pareto"
    component_selector: str = "round_robin"
    use_merge: bool = True
    max_merge_invocations: int = 5
    early_stop_patience: int = 3
    skip_perfect_score: bool = True
    failure_score: float = 0.0
    perfect_score: float = 1.0
    seed: int = 0
    track_stats: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


FROZEN_GEPA_POLICY = GEPAPolicy()


def validate_frozen_policy() -> None:
    """Fail closed when installed DSPy/GEPA cannot express the frozen policy."""
    import dspy
    from dspy.teleprompt import GEPA

    policy = FROZEN_GEPA_POLICY
    if dspy.__version__ != policy.dspy_version:
        raise RuntimeError(f"GEPA requires dspy=={policy.dspy_version}; found {dspy.__version__}")
    if version("gepa") != policy.gepa_version:
        raise RuntimeError(f"GEPA requires gepa=={policy.gepa_version}; found {version('gepa')}")
    parameters = inspect.signature(GEPA).parameters
    required = {
        "auto",
        "max_full_evals",
        "reflection_minibatch_size",
        "candidate_selection_strategy",
        "component_selector",
        "use_merge",
        "max_merge_invocations",
        "skip_perfect_score",
        "failure_score",
        "perfect_score",
        "seed",
        "track_stats",
        "gepa_kwargs",
    }
    missing = sorted(required - set(parameters))
    if missing:
        raise RuntimeError(f"installed DSPy GEPA cannot express frozen policy: {missing}")


# Early stopping
class FullEvalPlateauStopper:
    """Stop after N full validation evaluations without improvement, or at B.

    Each GEPA metric call is one rollout, so once GEPA has requested B the
    search is over; further rows are answered without running
    (``BudgetStopRunner``).
    """

    def __init__(self, patience: int, rollout_budget: int):
        self._patience = patience
        self._rollout_budget = int(rollout_budget)
        self._budget_logged = False
        self._best_score = float("-inf")
        self._no_improve_full_evals = 0
        self._last_full_eval_count = 0
        self._last_logged_full_evals = -1

    @property
    def reason(self) -> str | None:
        """Why the search stopped early, if it did."""
        if self._budget_logged:
            return StopReason.ROLLOUT_BUDGET_SPENT
        if self._no_improve_full_evals >= self._patience:
            return StopReason.FULL_EVAL_PLATEAU
        return None

    def __call__(self, gepa_state) -> bool:
        evals = int(getattr(gepa_state, "total_num_evals", 0) or 0)
        if evals >= self._rollout_budget:
            if not self._budget_logged:
                self._budget_logged = True
                logger.info("[early-stop] halting: %d rollouts requested, B=%d spent", evals, self._rollout_budget)
            return True
        scores = getattr(gepa_state, "program_full_scores_val_set", None) or []
        n_full_evals = len(scores)

        if n_full_evals > self._last_full_eval_count:
            current_best = max(scores) if scores else 0.0
            if current_best > self._best_score:
                self._best_score = current_best
                self._no_improve_full_evals = 0
            else:
                self._no_improve_full_evals += 1
            self._last_full_eval_count = n_full_evals

            if n_full_evals != self._last_logged_full_evals:
                self._last_logged_full_evals = n_full_evals
                logger.info(
                    "[early-stop] full-evals=%d best_val=%.3f no-improve=%d/%d",
                    n_full_evals,
                    self._best_score,
                    self._no_improve_full_evals,
                    self._patience,
                )

        fired = self._no_improve_full_evals >= self._patience
        if fired and self._last_logged_full_evals != -2:
            self._last_logged_full_evals = -2
            logger.info("[early-stop] halting after %d full-evals without improvement", self._no_improve_full_evals)
        return fired


def build_early_stopper(patience: int, rollout_budget: int) -> FullEvalPlateauStopper | None:
    """The full-evaluation stopper, or None when patience is 0."""
    return FullEvalPlateauStopper(patience, rollout_budget) if patience > 0 else None


# Cells
def validate_supported_cell(cell: CellSpec) -> None:
    """Fail before any rollout unless GEPA can run ``cell`` (any off-grid cell ``run.py`` admitted)."""
    if cell.method != METHOD:
        raise UnsupportedGEPACell(f"expected method={METHOD!r}, got {cell.method!r}")
    if cell.protocol_id != PROTOCOL_ID:
        raise UnsupportedGEPACell(f"GEPA requires protocol_id={PROTOCOL_ID!r}, got {cell.protocol_id!r}")
    if not cell.source_tables:
        return  # off-grid job admitted by run.py --allow-any-cell
    if cell.task not in SUPPORTED_TASKS:
        raise UnsupportedGEPACell(f"unsupported task {cell.task!r}; supported={sorted(SUPPORTED_TASKS)}")
    frameworks = SUPPORTED_FRAMEWORKS_BY_TOPOLOGY.get(cell.topology)
    if frameworks is None:
        raise UnsupportedGEPACell(f"unsupported topology {cell.topology!r}")
    if cell.framework not in frameworks:
        raise UnsupportedGEPACell(
            f"unsupported framework/topology pair {cell.framework!r}/{cell.topology!r}; "
            f"supported frameworks={sorted(frameworks)}"
        )
    if cell.topology == "single" and cell.communication != "freeform":
        raise UnsupportedGEPACell("single-agent cells support only freeform communication")
    require_grid_cell(cell, UnsupportedGEPACell)


# Adapter
class CommonRunnerAdapter(ProtocolRunnerAdapter):
    """GEPA's adapter over the protocol runner.

    Request seed of one rollout: ``logical_request_seed(optimizer_seed,
    cell_id, "optimization/gepa", iteration, example_id, bundle_sha256, k)``
    where ``k`` counts earlier rollouts of the same bundle on the same example.
    """

    def __init__(
        self,
        *,
        cell: CellSpec,
        runner: Runner,
        bundle: PromptBundle,
        phase: str = OPTIMIZATION_PHASE,
        iteration: int = 0,
    ) -> None:
        validate_supported_cell(cell)
        super().__init__(cell=cell, runner=runner, bundle=bundle, phase=phase)
        self._iteration = iteration

    def request_seed(self, item_id: str, bundle: PromptBundle, occurrence: int) -> int:
        return logical_request_seed(
            self.cell.optimizer_seed,
            self.cell.cell_id,
            self._phase,
            self._iteration,
            item_id,
            bundle.digest,
            occurrence,
        )


# Optimizer
class GEPAOptimizer:
    """Execute the frozen DSPy GEPA policy through the protocol runner.

    ``reflection_lm`` is wrapped in
    :class:`~optimizers.protocol.methods.context_fit.ContextFitLM` and
    :class:`LogicalSeedLM` (logical seeds, thinking, 48,000 tokens,
    temperature/top-p 1.0) unless it already is a ``LogicalSeedLM``.
    ``optimizer_factory`` (default ``dspy.teleprompt.GEPA``) is a test hook.
    """

    method = METHOD
    implementation_kind = "native_dspy_gepa_protocol_runner"
    production_eligible = True

    def __init__(
        self,
        *,
        initial_bundle: PromptBundle | None = None,
        reflection_lm: Any | None = None,
        optimizer_factory: Callable[..., Any] | None = None,
        num_threads: int = 1,
    ) -> None:
        if num_threads <= 0:
            raise ValueError("num_threads must be positive")
        self.initial_bundle = initial_bundle
        self.reflection_lm = reflection_lm
        self.optimizer_factory = optimizer_factory
        self.num_threads = int(num_threads)

    @staticmethod
    def _metric(_example: Any, prediction: Any, *args: Any, **kwargs: Any) -> float:
        del _example, args, kwargs
        return runner_score(prediction, "GEPA")

    @staticmethod
    def _compiled_bundle(compiled: Any, seed: PromptBundle) -> PromptBundle:
        if hasattr(compiled, "sync_prompts_to_adapter"):
            compiled.sync_prompts_to_adapter()
        adapter = getattr(compiled, "_adapter", None)
        if adapter is not None and all(hasattr(adapter, name) for name in ("roles", "get_prompt")):
            roles = {role: adapter.get_prompt(role) for role in adapter.roles()}
        else:
            roles = {}
            for name, predictor in compiled.named_predictors():
                role = getattr(predictor, "role", name)
                roles[str(role)] = (predictor.signature.instructions or "").strip() + "\n"
        if set(roles) != set(seed.roles):
            raise OptimizerContractError(
                f"compiled GEPA roles changed: expected={sorted(seed.roles)}, actual={sorted(roles)}"
            )
        return PromptBundle(
            roles=roles,
            demos=tuple(seed.demos),
            metadata={**dict(seed.metadata), "optimizer": METHOD, "native_compiled": True},
        )

    def optimize(
        self, cell: CellSpec, runner: Runner, budget: Any, training: list[Any], validation: list[Any]
    ) -> OptimizerResult:
        """Run native GEPA on the training/validation rows; the incumbent is GEPA's compiled program."""
        validate_supported_cell(cell)
        if not training or not validation:
            raise ValueError("native GEPA requires non-empty training and validation splits")

        from dspy.teleprompt import GEPA

        from optimizers.bridge.programs import AdapterBackedProgram

        validate_frozen_policy()
        pin_dspy_parallel_executor()
        seed_bundle = load_seed_bundle(cell, runner, self.initial_bundle)
        protocol_runner = BudgetStopRunner(BudgetedRunner(cell=cell, runner=runner, budget=budget))
        common_adapter = CommonRunnerAdapter(
            cell=cell, runner=protocol_runner, bundle=seed_bundle, phase=OPTIMIZATION_PHASE
        )
        program = AdapterBackedProgram(common_adapter)
        trainset = dspy_examples(training)
        valset = dspy_examples(validation)
        policy = FROZEN_GEPA_POLICY
        base_lm = self.reflection_lm or build_reflection_lm()
        reflection_lm = (
            base_lm
            if isinstance(base_lm, LogicalSeedLM)
            else LogicalSeedLM(cell=cell, lm=fit_to_context(base_lm), phase=REFLECTION_PHASE)
        )
        factory = self.optimizer_factory or GEPA
        stopper = build_early_stopper(policy.early_stop_patience, rollout_budget=int(budget.maximum))
        gepa_kwargs = {"stop_callbacks": [stopper]} if stopper is not None else {}
        effective_seed = int(policy.seed) + int(cell.optimizer_seed)
        optimizer = factory(
            metric=self._metric,
            auto=policy.auto,
            max_full_evals=policy.max_full_evals,
            reflection_minibatch_size=policy.reflection_minibatch_size,
            candidate_selection_strategy=policy.candidate_selection_strategy,
            reflection_lm=reflection_lm,
            skip_perfect_score=policy.skip_perfect_score,
            component_selector=policy.component_selector,
            use_merge=policy.use_merge,
            max_merge_invocations=policy.max_merge_invocations,
            num_threads=self.num_threads,
            failure_score=policy.failure_score,
            perfect_score=policy.perfect_score,
            # Native state stays in memory: the job's artifacts are JSON only.
            log_dir=None,
            track_stats=policy.track_stats,
            seed=effective_seed,
            gepa_kwargs=gepa_kwargs,
        )
        initial_snapshot = dict(budget.snapshot())
        compiled = optimizer.compile(program, trainset=trainset, valset=valset)
        common_adapter.raise_for_runtime_failures("GEPA")
        selected = self._compiled_bundle(compiled, seed_bundle)
        final_snapshot = dict(budget.snapshot())
        stop_reason = (stopper.reason if stopper is not None else None) or StopReason.MAX_FULL_EVALS
        checkpoints = (
            native_checkpoint(
                method=METHOD,
                iteration=policy.max_full_evals,
                bundle=selected,
                budget=budget,
                state={"native_compile_complete": True, "effective_seed": effective_seed},
            ),
        )
        curve = (
            LearningCurvePoint("initial", 0, int(initial_snapshot["charged"]), None, seed_bundle.digest),
            LearningCurvePoint(
                "completed_native_gepa_compile",
                policy.max_full_evals,
                int(final_snapshot["charged"]),
                None,
                selected.digest,
            ),
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
            stop_reason=stop_reason,
            learning_curve=curve,
            checkpoints=checkpoints,
            reflection_requests=tuple(getattr(reflection_lm, "requests", ())),
            metadata={
                "native_backend": policy.backend,
                "native_optimizer_executed": True,
                "native_policy": policy.to_dict(),
                "effective_lifecycle": {
                    "max_full_evals": policy.max_full_evals,
                    "reflection_minibatch_size": policy.reflection_minibatch_size,
                },
                "effective_seed": effective_seed,
                "rows_answered_past_budget": protocol_runner.answered,
                "training_ids": [example_id(row, index) for index, row in enumerate(training)],
                "validation_ids": [example_id(row, index) for index, row in enumerate(validation)],
            },
        )
        persist_optimizer_artifact(runner, artifact)
        return artifact


__all__ = [
    "CommonRunnerAdapter",
    "FROZEN_GEPA_POLICY",
    "FullEvalPlateauStopper",
    "GEPAOptimizer",
    "GEPAPolicy",
    "SUPPORTED_FRAMEWORKS_BY_TOPOLOGY",
    "SUPPORTED_TASKS",
    "UnsupportedGEPACell",
    "build_early_stopper",
    "validate_frozen_policy",
    "validate_supported_cell",
]
