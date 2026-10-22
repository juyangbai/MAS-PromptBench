"""Run one optimizer job: optimize -> final validation + locked selection -> test.

    python -m optimizers.protocol.run --method gepa --dataset hotpotqa \\
        --topology sequential --model qwen --seed 0 --out runs/gepa-hotpotqa-seq-qwen-0

Artifacts (all JSON, relative paths only) under ``--out``:

    job.json                 cell identity, seed bundle, evaluation condition
    optimization/            runner attempts/records, optimizer artifacts
    optimization.json        optimizer envelope (status, budget, incumbent)
    evaluations/             content-addressed uncharged evaluations
    selection.json           sealed deployment decision (before any test row)
    test.json, result.json   paired test scores and the job summary

``--phase`` resumes from saved artifacts: ``optimize``, ``validate`` (needs
optimization.json) or ``test`` (needs selection.json); ``all`` runs whatever is
missing. An interrupted optimization is never restarted silently.

Progress and errors are logged to stderr (``--log-level``, default
``$LOG_LEVEL`` or INFO; ``--quiet`` drops the progress lines); stdout carries
only the one-line JSON summary.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core import logs
from topologies.decentralized.openai_agents.agents_sdk_base import FRAMEWORK as SDK_FRAMEWORK
from topologies.decentralized.openai_agents.agents_sdk_base import reexec_with_sdk_first

from .adapter_output import json_safe
from .artifacts import ArtifactStore, atomic_write_json, read_json, scrub_text
from .budget import BudgetLedger
from .cells import find_cell, parse_registry_key, registry_key, runner_condition_in_grid
from .config import BUDGET, PROTOCOL_ID, TASK_MODELS, TASKS, normalize_task, protocol_hash, task_model_id
from .errors import (
    EvaluationError,
    JobError,
    MethodUnavailable,
    NativeInfrastructureExhausted,
    OptimizerInfrastructureFailure,
    PreObservationInfrastructureFailure,
    RunnerContractError,
)
from .evaluation import Evaluator, condition_identity, load_selection, lock_selection, optimization_is_reportable
from .runner import (
    AdapterRuntime,
    CellExecutor,
    DatasetMetricScorer,
    ProtocolRunner,
    TaskData,
    configure_task_endpoints,
    load_task_data,
    usage_dict,
)
from .schema import (
    CellSpec,
    PromptBundle,
    StopReason,
    bundle_from_dict,
    bundle_to_dict,
    content_hash,
    incumbent_of,
    schema_name,
    sealed,
    verify_sealed,
)
from .settings import ProtocolSettings

logger = logging.getLogger(__name__)

PHASES = ("all", "optimize", "validate", "test")


@dataclass
class JobOptions:
    """Command-line options of one job."""

    method: str
    dataset: str
    topology: str
    model: str
    seed: int
    out: Path
    framework: str | None = None
    team_size: int | None = None
    communication: str | None = None
    budget: int = BUDGET
    phase: str = "all"
    allow_any_cell: bool = False
    task_endpoints: Sequence[str] = ()
    evaluation_cache: Path | None = None
    quiet: bool = False


@dataclass
class JobHooks:
    """Injection points for offline use; real jobs leave every field unset."""

    load_task_data: Callable[[str], TaskData] | None = None
    build_runtime: Callable[[CellSpec], Any] | None = None
    build_scorer: Callable[[CellSpec], Any] | None = None
    optimizer_kwargs: Mapping[str, Any] = field(default_factory=dict)
    configure_environment: bool = True


# Cell resolution
def resolve_cell_args(options: JobOptions) -> dict[str, Any]:
    """Merge ``--topology`` (bare topology or registry key) with the explicit flags."""
    task = normalize_task(options.dataset)
    if task not in TASKS and not options.allow_any_cell:
        raise JobError(f"unknown dataset {options.dataset!r}; choices={sorted(TASKS)}")
    parsed = parse_registry_key(options.topology)
    topology = str(parsed["topology"])
    explicit = {"framework": options.framework, "communication": options.communication, "team_size": options.team_size}
    for name, value in explicit.items():
        if value is not None and name in parsed and parsed[name] != value:
            raise JobError(f"--{name.replace('_', '-')} {value} conflicts with --topology {options.topology}")
    framework = str(parsed.get("framework") or options.framework or "langgraph")
    communication = str(parsed.get("communication") or options.communication or "freeform")
    team_size = 1 if topology == "single" else int(parsed.get("team_size") or options.team_size or 4)
    return {
        "task": task,
        "topology": topology,
        "framework": framework,
        "communication": communication,
        "team_size": team_size,
        "task_model": task_model_id(options.model),
    }


def grid_membership(method: str, args: Mapping[str, Any], allow_any_cell: bool) -> tuple[tuple[int, ...], bool]:
    """(source tables, in grid). ``identity`` may run on any grid runtime condition."""
    if method == "identity":
        return (), runner_condition_in_grid(**args)
    cell = find_cell(method, **args)
    if cell is None:
        if not allow_any_cell:
            raise JobError(
                f"{method} {args} is not a cell of the experiment grid (use --allow-any-cell to run it anyway)"
            )
        return (), False
    return cell.source_tables, True


# Job
class Job:
    """One (method, cell, optimizer seed) job and its three phases over one ``--out`` directory."""

    def __init__(self, options: JobOptions, hooks: JobHooks | None = None) -> None:
        self.options = options
        self.hooks = hooks or JobHooks()
        if options.phase not in PHASES:
            raise JobError(f"--phase must be one of {PHASES}")
        self.out = Path(options.out)
        self.store = ArtifactStore(self.out)
        args = resolve_cell_args(options)
        self.source_tables, self.in_grid = grid_membership(options.method, args, options.allow_any_cell)
        if not self.in_grid and not options.allow_any_cell:
            raise JobError(f"runtime condition {args} is not in the experiment grid")
        self.registry_key = registry_key(args["topology"], args["framework"], args["communication"], args["team_size"])
        if self.hooks.configure_environment:
            self._configure_environment(args["task_model"])
        self.data = (self.hooks.load_task_data or load_task_data)(args["task"])
        self.cell = CellSpec(
            method=options.method,
            optimizer_seed=options.seed,
            split_hash=self.data.split_hash,
            protocol_hash=protocol_hash(),
            budget=options.budget,
            source_tables=self.source_tables,
            reflection_model=ProtocolSettings.from_env().reflection_model,
            **args,
        )
        self.runtime = (self.hooks.build_runtime or (lambda cell: AdapterRuntime(cell)))(self.cell)
        self.scorer = (self.hooks.build_scorer or (lambda cell: DatasetMetricScorer(cell.task)))(self.cell)
        self.executor = CellExecutor(cell=self.cell, runtime=self.runtime, scorer=self.scorer, data=self.data)
        self.seed_bundle: PromptBundle = self.runtime.seed_bundle()
        self.executor.validate_bundle(self.seed_bundle)
        self.condition = condition_identity(
            self.cell,
            self.seed_bundle,
            runtime_id=self.executor.implementation_id,
            scorer_id=self.executor.scorer_id,
            split_hash=self.data.split_hash,
        )
        cache = Path(options.evaluation_cache) if options.evaluation_cache else self.out / "evaluations"
        self.evaluator = Evaluator(executor=self.executor, condition=self.condition, directory=cache)
        self._write_job_identity()

    def _configure_environment(self, task_model: str) -> None:
        endpoints = list(self.options.task_endpoints) or list(ProtocolSettings.from_env().task_endpoints)
        if not endpoints:
            raise JobError("no task endpoint: pass --task-endpoints or set TASK_ENDPOINTS / VLLM_BASE_URL")
        configure_task_endpoints(endpoints, task_model=task_model)
        os.environ.setdefault("DSPY_CACHEDIR", str((self.out / "optimization" / "dspy_cache").resolve()))

    @property
    def identity(self) -> dict[str, Any]:
        """The sealed ``job.json`` content."""
        return {
            "schema": schema_name("job"),
            "protocol_id": PROTOCOL_ID,
            "cell": dict(self.cell.identity),
            "cell_id": self.cell.cell_id,
            "source_tables": list(self.cell.source_tables),
            "in_grid": self.in_grid,
            "protocol_conformant": self.cell.protocol_conformant and self.in_grid,
            "registry_key": self.registry_key,
            "runtime_implementation_id": self.executor.implementation_id,
            "scorer_id": self.executor.scorer_id,
            "seed_bundle": bundle_to_dict(self.seed_bundle),
            "condition": self.condition,
            "condition_id": self.evaluator.condition_id,
            "split_sizes": {split: len(ids) for split, ids in self.data.split_ids.items()},
        }

    def _write_job_identity(self) -> None:
        path = self.out / "job.json"
        identity = self.identity
        if path.exists():
            if verify_sealed(read_json(path)) != identity:
                raise JobError("this --out directory belongs to a different job or code/data revision")
            return
        atomic_write_json(path, sealed(identity), indent=1)

    def log(self, message: str) -> None:
        """Log a progress line (silent with ``--quiet``)."""
        if not self.options.quiet:
            logger.info("[protocol] %s", message)

    # Phase 1: optimization
    def optimize(self) -> dict[str, Any]:
        """Run the method on the charged ledger and seal ``optimization.json`` (or reuse it)."""
        path = self.out / "optimization.json"
        if path.exists():
            envelope = verify_sealed(read_json(path))
            if not optimization_is_reportable(envelope):
                raise JobError(f"the saved optimization failed ({envelope.get('failure_kind')}); use a fresh --out")
            self.log(f"optimization already {envelope['status']}; reusing it")
            return envelope
        directory = self.out / "optimization"
        if (directory / "attempts.jsonl").exists() or (directory / "records.jsonl").exists():
            raise JobError(
                "an interrupted optimization attempt exists; it is never restarted silently, use a fresh --out"
            )
        from .methods import build_optimizer, load_method

        load_method(self.cell.method)
        if "dspy" in sys.modules:
            from .methods.dspy_bridge import pin_dspy_parallel_executor

            pin_dspy_parallel_executor()
        directory = self.store.named("optimization")
        budget = BudgetLedger(self.cell.budget)
        progress = _Progress(self.log, budget)
        runner = ProtocolRunner(
            cell=self.cell,
            executor=self.executor,
            budget=budget,
            seed_bundle=self.seed_bundle,
            artifact_store=self.store,
            artifact_directory=directory,
            event_sink=progress,
        )
        envelope: dict[str, Any] = {
            "schema": schema_name("optimization"),
            "protocol_id": PROTOCOL_ID,
            "method": self.cell.method,
            "cell": dict(self.cell.identity),
            "cell_id": self.cell.cell_id,
            "status": "failed",
            "seed_bundle": bundle_to_dict(self.seed_bundle),
            "training_ids": list(self.data.split_ids["train"]),
            "validation_ids": list(self.data.split_ids["validation"]),
        }
        started = time.monotonic()
        optimizer = None
        self.log(
            f"optimizing {self.cell.method} on {self.cell.task}/{self.registry_key} "
            f"model={self.cell.task_model} seed={self.cell.optimizer_seed} B={self.cell.budget}"
        )
        try:
            optimizer = build_optimizer(
                self.cell.method,
                self.cell,
                self.seed_bundle,
                run_dir=directory / "native",
                extra=self.hooks.optimizer_kwargs,
            )
            artifact = optimizer.optimize(
                self.cell, runner, budget, list(self.data.rows("train")), list(self.data.rows("validation"))
            )
            payload = json_safe(artifact.to_dict())
            incumbent = incumbent_of(artifact)
            if incumbent is None:
                raise JobError("optimizer did not return an incumbent PromptBundle")
            if payload.get("method") != self.cell.method or payload.get("cell_id") != self.cell.cell_id:
                raise JobError("optimizer result identity differs from the job")
            if (
                payload.get("production_eligible") is False
                or (payload.get("metadata") or {}).get("validation_only") is True
            ):
                raise JobError("validation-only optimizer result rejected")
            if dict(payload.get("budget_snapshot") or {}) != budget.snapshot() or budget.reserved:
                raise JobError("optimizer result differs from the committed budget")
            runner.validate_bundle(incumbent)
            self.store.write_json(
                directory, "optimizer_result.json", {**payload, "artifact_sha256": content_hash(payload)}
            )
            metadata_stop = (payload.get("metadata") or {}).get("stop_reason", StopReason.NATIVE_RETURN)
            envelope.update(
                status="completed",
                incumbent_bundle=bundle_to_dict(incumbent),
                stop_reason=payload.get("stop_reason", metadata_stop),
            )
        except Exception as exc:
            envelope.update(
                error_type=type(exc).__name__,
                error=scrub_text(str(exc))[:2000],
                failure_kind=_failure_kind(exc, budget),
            )
        envelope.update(
            budget=budget.snapshot(),
            elapsed_seconds=round(time.monotonic() - started, 3),
            usage={
                **{k: usage_dict(v) for k, v in runner.usage_totals.items()},
                "reflection": _reflection_usage(optimizer),
            },
            optimizer_artifact="optimization/optimizer_result.json" if envelope["status"] == "completed" else None,
        )
        atomic_write_json(path, sealed(envelope), indent=1)
        self.log(
            f"optimization {envelope['status']}"
            + (
                f" ({envelope.get('failure_kind')}: {envelope.get('error')})"
                if envelope["status"] != "completed"
                else ""
            )
            + f"; charged {budget.charged}/{budget.maximum}"
        )
        if not optimization_is_reportable(envelope):
            raise JobError(f"optimization failed ({envelope.get('failure_kind')}): {envelope.get('error')}")
        return envelope

    # Phase 2: final validation and locked selection
    def validate(self) -> dict[str, Any]:
        """Evaluate seed and incumbent on the full validation split and lock ``selection.json``."""
        path = self.out / "selection.json"
        if path.exists():
            self.log("selection already locked; reusing it")
            return load_selection(path)
        envelope = self._envelope()
        forced = None
        try:
            incumbent = bundle_from_dict(envelope["incumbent_bundle"])
            self.evaluator.validate_bundle(incumbent)
        except (KeyError, ValueError, TypeError, RunnerContractError):
            incumbent, forced = self.seed_bundle, "invalid_incumbent_bundle"
        if envelope.get("status") != "completed":
            incumbent, forced = self.seed_bundle, "infrastructure_invalid_optimization"
        rows = self.data.rows("validation")
        self.log(f"final validation: seed and incumbent on {len(rows)} validation rows (uncharged, greedy)")
        baseline = self.evaluator.evaluate(
            self.seed_bundle, phase="baseline", split="validation", evaluation_seed=self.cell.optimizer_seed, rows=rows
        )
        candidate = self.evaluator.evaluate(
            incumbent, phase="final_validation", split="validation", evaluation_seed=self.cell.optimizer_seed, rows=rows
        )
        selection = lock_selection(
            path,
            envelope=envelope,
            seed_bundle=self.seed_bundle,
            incumbent_bundle=incumbent,
            baseline=baseline,
            candidate=candidate,
            forced_fallback_reason=forced,
        )
        self.log(
            f"selection locked: selected_candidate={selection['selected_candidate']} "
            f"fallback={selection['fallback_reason']} seed={selection['baseline_validation_score']} "
            f"incumbent={selection['incumbent_validation_score']}"
        )
        return selection

    # Phase 3: held-out test
    def test(self) -> dict[str, Any]:
        """Evaluate seed and deployed bundles on the test split; write ``test.json`` and ``result.json``."""
        selection_path = self.out / "selection.json"
        if not selection_path.exists():
            raise JobError("test requires a locked selection (run --phase validate first)")
        selection = load_selection(selection_path)
        envelope = self._envelope()
        if content_hash(envelope) != selection["optimization_envelope_sha256"]:
            raise JobError("optimization envelope changed after selection")
        if selection["condition_id"] != self.evaluator.condition_id:
            raise JobError("selection was locked for a different evaluation condition")
        self.data.unlock_test(selection)
        rows = self.data.rows("test")
        deployed_bundle = bundle_from_dict(selection["deployed_bundle"])
        self.log(f"test: seed and deployed bundles on {len(rows)} test rows (uncharged, greedy)")
        baseline = self.evaluator.evaluate(
            self.seed_bundle,
            phase="baseline",
            split="test",
            evaluation_seed=self.cell.optimizer_seed,
            rows=rows,
            selection=selection,
        )
        deployed = self.evaluator.evaluate(
            deployed_bundle,
            phase="test",
            split="test",
            evaluation_seed=self.cell.optimizer_seed,
            rows=rows,
            selection=selection,
        )
        valid = baseline.valid and deployed.valid
        base_mean, deployed_mean = baseline.mean_score(), deployed.mean_score()
        test = {
            "schema": schema_name("test"),
            "selection_id": selection["selection_id"],
            "condition_id": self.evaluator.condition_id,
            "evaluation_seed": self.cell.optimizer_seed,
            "baseline_evaluation_id": baseline.evaluation_id,
            "deployed_evaluation_id": deployed.evaluation_id,
            "example_ids": [record.example_id for record in baseline.records],
            "baseline_scores": [record.score for record in baseline.records],
            "deployed_scores": [record.score for record in deployed.records],
            "baseline_status": [record.status for record in baseline.records],
            "deployed_status": [record.status for record in deployed.records],
            "valid_for_aggregation": valid,
            "baseline_mean": base_mean,
            "deployed_mean": deployed_mean,
            "delta_pp": None if not valid else 100.0 * (deployed_mean - base_mean),
            "fallback_reason": selection["fallback_reason"],
            "selected_candidate": selection["selected_candidate"],
            "usage": {"baseline": dict(baseline.usage), "deployed": dict(deployed.usage)},
            "search_budget_charged": 0,
        }
        atomic_write_json(self.out / "test.json", sealed(test), indent=1)
        result = self._result(envelope, selection, test)
        atomic_write_json(self.out / "result.json", sealed(result), indent=1)
        self.log(f"test: seed={base_mean} deployed={deployed_mean} valid={valid}")
        return result

    def _envelope(self) -> dict[str, Any]:
        path = self.out / "optimization.json"
        if not path.exists():
            raise JobError("no optimization.json (run --phase optimize first)")
        envelope = verify_sealed(read_json(path))
        if envelope.get("cell") != dict(self.cell.identity):
            raise JobError("optimization.json belongs to a different cell")
        if not optimization_is_reportable(envelope):
            raise JobError(f"optimization failed ({envelope.get('failure_kind')}); nothing to report")
        return envelope

    def _result(
        self, envelope: Mapping[str, Any], selection: Mapping[str, Any], test: Mapping[str, Any]
    ) -> dict[str, Any]:
        cell = self.cell
        return {
            "schema": schema_name("job-result"),
            "protocol_id": PROTOCOL_ID,
            "status": "completed",
            "protocol_conformant": cell.protocol_conformant and self.in_grid,
            "cell": dict(cell.identity),
            "cell_id": cell.cell_id,
            "optimizer_seed": cell.optimizer_seed,
            "grid_cell": {
                "method": cell.method,
                "task": cell.task,
                "topology": cell.topology,
                "framework": cell.framework,
                "communication": cell.communication,
                "team_size": cell.team_size,
                "task_model": cell.task_model,
            },
            "registry_key": self.registry_key,
            "source_tables": list(cell.source_tables),
            "condition_id": self.evaluator.condition_id,
            "optimization": {
                key: envelope.get(key)
                for key in ("status", "failure_kind", "stop_reason", "budget", "usage", "elapsed_seconds")
            },
            "selection": {
                key: selection.get(key)
                for key in (
                    "selection_id",
                    "selected_candidate",
                    "fallback_reason",
                    "baseline_validation_score",
                    "incumbent_validation_score",
                    "validation_delta",
                )
            },
            "test": {
                key: test[key]
                for key in (
                    "example_ids",
                    "baseline_scores",
                    "deployed_scores",
                    "valid_for_aggregation",
                    "baseline_mean",
                    "deployed_mean",
                    "delta_pp",
                )
            },
        }

    def run(self) -> dict[str, Any]:
        """Run the requested phase(s); ``all`` runs whatever is missing."""
        phase = self.options.phase
        if phase in {"all", "optimize"}:
            self.optimize()
        if phase in {"all", "validate"}:
            self.validate()
        if phase == "optimize":
            return {"phase": phase, "out": "optimization.json"}
        if phase == "validate":
            return {"phase": phase, "out": "selection.json"}
        if (self.out / "result.json").exists() and phase == "all":
            return verify_sealed(read_json(self.out / "result.json"))
        return self.test()


class _Progress:
    """Compact progress line every 10 charged rollouts."""

    def __init__(self, log: Callable[[str], None], budget: BudgetLedger) -> None:
        self.log, self.budget, self.last = log, budget, -1

    def __call__(self, event: Mapping[str, Any]) -> None:
        charged = int(event["budget"]["charged"])
        if charged // 10 != self.last // 10 or not event.get("usable"):
            self.last = charged
            self.log(
                f"rollouts charged={charged}/{self.budget.maximum} attempted={event['budget']['attempted']} "
                f"infra={event['budget']['infrastructure_failures']} last={event['status']}"
            )


def _reflection_usage(optimizer: Any) -> dict[str, int]:
    """Best-effort reflection/auxiliary LM usage reported by the optimizer's clients."""
    total = {"model_calls": 0, "input_tokens": 0, "output_tokens": 0, "cache_hits": 0}
    if optimizer is None:
        return total
    for name in ("reflection_lm", "reflection", "reflection_client", "task_judge", "task_lm"):
        backend = getattr(optimizer, name, None)
        if backend is None:
            continue
        usage = getattr(backend, "usage", None)
        if usage is None and callable(getattr(backend, "snapshot", None)):
            try:
                usage = dict(backend.snapshot()).get("usage")
            except Exception:
                usage = None
        if not isinstance(usage, Mapping):
            continue
        total["model_calls"] += int(usage.get("model_calls", usage.get("n_calls", 0)) or 0)
        total["input_tokens"] += int(usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0)
        total["output_tokens"] += int(usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0)
        total["cache_hits"] += int(usage.get("cache_hits", 0) or 0)
    return total


def _failure_kind(exc: BaseException, budget: BudgetLedger) -> str:
    if budget.reserved:
        return "ambiguous_external"
    infrastructure = (
        PreObservationInfrastructureFailure,
        NativeInfrastructureExhausted,
        OptimizerInfrastructureFailure,
    )
    if isinstance(exc, infrastructure):
        return "infrastructure_invalid"
    if isinstance(exc, (JobError, RunnerContractError, EvaluationError, ValueError)):
        return "contract_failure"
    return "program_failure"


# CLI
def build_parser() -> argparse.ArgumentParser:
    """The ``python -m optimizers.protocol.run`` argument parser."""
    from .methods import method_names

    parser = argparse.ArgumentParser(description="Run one protocol job (optimize -> validate/select -> test).")
    parser.add_argument("--method", required=True, choices=method_names())
    parser.add_argument("--dataset", required=True, help=f"one of {sorted(TASKS)}")
    parser.add_argument(
        "--topology",
        required=True,
        help="base topology (single, independent, sequential, centralized, decentralized) "
        "or a registry key such as sequential_crewai, independent_r8, "
        "centralized_communications_structured_soft",
    )
    parser.add_argument("--framework", choices=("langgraph", "crewai", "autogen", "openai_agents"))
    parser.add_argument("--team-size", type=int, choices=(2, 4, 8, 10))
    parser.add_argument("--communication", choices=("freeform", "semi_structured", "structured_soft"))
    parser.add_argument("--model", required=True, help=f"{sorted(TASK_MODELS)} or a full task-model ID")
    parser.add_argument("--seed", type=int, required=True, choices=(0, 1, 2))
    parser.add_argument(
        "--budget",
        type=int,
        default=BUDGET,
        help=f"usable rollouts (protocol: {BUDGET}; smaller values are reported as non-conformant)",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--phase", choices=PHASES, default="all")
    parser.add_argument("--allow-any-cell", action="store_true", help="run a cell that is not in the experiment grid")
    parser.add_argument(
        "--task-endpoints",
        default="",
        help="comma-separated OpenAI-compatible endpoints serving the task model "
        "(default: $TASK_ENDPOINTS, else $VLLM_BASE_URL)",
    )
    parser.add_argument(
        "--evaluation-cache",
        type=Path,
        default=None,
        help="shared directory for content-addressed evaluations (reuses identical baselines)",
    )
    parser.add_argument("--quiet", action="store_true", help="no progress lines (errors are still logged)")
    logs.add_argument(parser)
    return parser


def options_from_args(args: argparse.Namespace) -> JobOptions:
    """Job options from parsed arguments."""
    return JobOptions(
        method=args.method,
        dataset=args.dataset,
        topology=args.topology,
        model=args.model,
        seed=args.seed,
        out=args.out,
        framework=args.framework,
        team_size=args.team_size,
        communication=args.communication,
        budget=args.budget,
        phase=args.phase,
        allow_any_cell=args.allow_any_cell,
        task_endpoints=[item for item in args.task_endpoints.split(",") if item.strip()],
        evaluation_cache=args.evaluation_cache,
        quiet=args.quiet,
    )


def main(argv: Sequence[str] | None = None, hooks: JobHooks | None = None) -> int:
    """Run one job; exit code 2 for job, evaluation and installation errors."""
    args = build_parser().parse_args(argv)
    logs.configure(args.log_level)
    try:
        result = Job(options_from_args(args), hooks).run()
    except (JobError, EvaluationError, MethodUnavailable) as exc:
        logger.error("[protocol] error: %s", scrub_text(str(exc)))
        return 2
    summary = {key: result.get(key) for key in ("status", "cell_id", "phase", "out") if key in result}
    if "test" in result:
        summary.update(
            {
                key: result["test"][key]
                for key in ("baseline_mean", "deployed_mean", "delta_pp", "valid_for_aggregation")
            }
        )
        summary["fallback_reason"] = result["selection"]["fallback_reason"]
    print(json.dumps(summary, sort_keys=True))
    return 0


def reexec_for_sdk(argv: Sequence[str] | None = None) -> None:
    """Restart an OpenAI Agents SDK job with the SDK directory first on PYTHONPATH.

    For the ``__main__`` block, before the job imports anything (see
    :func:`~topologies.decentralized.openai_agents.agents_sdk_base.reexec_with_sdk_first`).
    Returns at once for other frameworks and for a cell that does not resolve,
    which :func:`main` then reports.
    """
    try:
        framework = resolve_cell_args(options_from_args(build_parser().parse_args(argv)))["framework"]
    except JobError:
        return
    if framework == SDK_FRAMEWORK:
        reexec_with_sdk_first()


if __name__ == "__main__":
    reexec_for_sdk()
    raise SystemExit(main())
