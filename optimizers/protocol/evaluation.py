"""Uncharged final validation, locked deployment selection and held-out test.

Seed and candidate bundles are evaluated on the complete ordered split at
temperature 0 with paired per-item request seeds that do not depend on the
bundle or the method, so the seed-bundle baseline of one runtime condition is
identical (and cacheable) across methods. Test rows are released only after a
selection has been sealed to disk.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .artifacts import atomic_write_json, exclusive, read_json
from .budget import closed_ledger
from .config import MAX_INFRASTRUCTURE_RETRIES, PROTOCOL_ID, TASK_DECODING
from .errors import EvaluationError
from .runner import CellExecutor
from .schema import (
    CellSpec,
    PromptBundle,
    RunRecord,
    Usage,
    bundle_to_dict,
    content_hash,
    record_from_dict,
    schema_name,
    sealed,
    verify_sealed,
)
from .seeding import paired_evaluation_seed
from .selection import assert_test_unexposed, select_for_deployment


def condition_identity(
    cell: CellSpec, seed_bundle: PromptBundle, *, runtime_id: str, scorer_id: str, split_hash: str
) -> dict[str, Any]:
    """Method-independent runtime condition; split and evaluation seed are added per run."""
    return {
        "schema": schema_name("baseline-condition"),
        "protocol_id": PROTOCOL_ID,
        "task": cell.task,
        "topology": cell.topology,
        "framework": cell.framework,
        "communication": cell.communication,
        "team_size": cell.team_size,
        "task_model": cell.task_model,
        "runtime_implementation_id": runtime_id,
        "initial_bundle_sha256": seed_bundle.digest,
        "evaluation_decoding": dict(TASK_DECODING["evaluation"]),
        "scorer_id": scorer_id,
        "split_sha256": split_hash,
        "paired_seed_schema": schema_name("paired-evaluation-seed"),
    }


@dataclass(frozen=True)
class EvaluationResult:
    """One complete uncharged evaluation: its identity, per-item records and usage."""

    identity: Mapping[str, Any]
    records: tuple[RunRecord, ...]
    usage: Mapping[str, Any]

    @property
    def evaluation_id(self) -> str:
        """Content hash of the evaluation identity."""
        return content_hash(self.identity)

    @property
    def complete(self) -> bool:
        """Every ordered item has a record."""
        return [r.example_id for r in self.records] == list(self.identity["ordered_example_ids"])

    @property
    def valid(self) -> bool:
        """Complete and every record usable."""
        return self.complete and all(record.usable for record in self.records)

    def mean_score(self) -> float | None:
        """Mean score of a valid evaluation, else None."""
        if not self.valid or not self.records:
            return None
        return sum(float(record.score) for record in self.records) / len(self.records)

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluation_id": self.evaluation_id,
            "identity": dict(self.identity),
            "records": [record.to_dict() for record in self.records],
            "usage": dict(self.usage),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> EvaluationResult:
        result = cls(
            dict(value["identity"]), tuple(record_from_dict(r) for r in value["records"]), dict(value["usage"])
        )
        if value.get("evaluation_id") != result.evaluation_id:
            raise EvaluationError("evaluation digest mismatch")
        return result


class Evaluator:
    """Execute every item of one split once (after infrastructure retries), uncharged.

    Each evaluation is content-addressed under ``directory`` and journaled per
    item, so an interrupted evaluation resumes from its committed prefix and an
    identical evaluation (same condition, split, seed and bundle) is reused.
    """

    def __init__(
        self,
        *,
        executor: CellExecutor,
        condition: Mapping[str, Any],
        directory: Path,
        max_infrastructure_retries: int = MAX_INFRASTRUCTURE_RETRIES,
    ) -> None:
        self.executor = executor
        self.cell = executor.cell
        self.condition = dict(condition)
        self.condition_id = content_hash(self.condition)
        self.directory = Path(directory)
        self.max_infrastructure_retries = max_infrastructure_retries

    def validate_bundle(self, bundle: PromptBundle) -> None:
        """Reject a bundle the runtime cannot execute or whose metadata exposes test material."""
        self.executor.validate_bundle(bundle)
        assert_test_unexposed(dict(bundle.metadata))

    def evaluate(
        self,
        bundle: PromptBundle,
        *,
        phase: str,
        split: str,
        evaluation_seed: int,
        rows: Sequence[Mapping[str, Any]],
        selection: Mapping[str, Any] | None = None,
    ) -> EvaluationResult:
        """Evaluate ``bundle`` on every row of ``split`` (resumed or reused when already journaled).

        ``rows`` must be the whole split in its fixed order. A test evaluation
        needs the locked ``selection`` of this condition and one of the two
        bundles it locked. Each item gets the paired request seed of
        ``evaluation_seed``; the sealed result is stored under its content hash.
        """
        self._check_access(bundle, phase=phase, split=split, selection=selection)
        self.validate_bundle(bundle)
        expected = list(self.executor.data.split_ids[split])
        ids = [str(row.get("id", "")) for row in rows]
        if not expected or ids != expected:
            raise EvaluationError("evaluation rows must exactly match the full ordered split")
        seed_context = content_hash({"condition": self.condition_id, "split": split, "seed": evaluation_seed})
        seeds = [paired_evaluation_seed(self.condition_id, evaluation_seed, split, item) for item in ids]
        identity = {
            "schema": schema_name("evaluation"),
            "condition": self.condition,
            "condition_id": self.condition_id,
            "split": split,
            "evaluation_seed": evaluation_seed,
            "bundle_sha256": bundle.digest,
            "ordered_example_ids": ids,
            "rows_sha256": content_hash(list(rows)),
            "evaluation_seed_identity": seed_context,
        }
        evaluation_id = content_hash(identity)
        final_path = self.directory / f"{evaluation_id}.json"
        run = _EvaluationPass(
            bundle=bundle,
            phase=phase,
            split=split,
            evaluation_seed=evaluation_seed,
            seed_identity=seed_context,
            evaluation_id=evaluation_id,
            journal_path=self.directory / f"{evaluation_id}.partial.jsonl",
        )
        with exclusive(self.directory / f"{evaluation_id}.lock"):
            if final_path.exists():
                result = EvaluationResult.from_dict(verify_sealed(read_json(final_path)))
                if result.identity != identity:
                    raise EvaluationError("stored evaluation identity differs")
                return result
            records, attempts = self._replay(run.journal_path, ids)
            done = len(records)
            for item_id, seed in zip(ids[done:], seeds[done:]):
                record = self._evaluate_item(run, item_id, seed, attempts)
                records.append(record)
                self._journal(run.journal_path, {"kind": "record", "record": record.to_dict()})
            result = EvaluationResult(identity, tuple(records), _usage_summary(attempts))
            validate_evaluation(
                result,
                expected_ids=expected,
                bundle_sha256=bundle.digest,
                condition_id=self.condition_id,
                evaluation_seed=evaluation_seed,
                split=split,
            )
            atomic_write_json(final_path, sealed(result.to_dict()))
            run.journal_path.unlink(missing_ok=True)
            return result

    def _check_access(
        self, bundle: PromptBundle, *, phase: str, split: str, selection: Mapping[str, Any] | None
    ) -> None:
        """Reject an unknown phase or split, and a test evaluation that the locked selection does not allow."""
        if phase not in {"baseline", "final_validation", "test"} or split not in {"validation", "test"}:
            raise EvaluationError("invalid evaluation phase or split")
        if split == "test":
            if selection is None or selection.get("condition_id") != self.condition_id:
                raise EvaluationError("test evaluation requires the locked selection of this condition")
            if bundle.digest not in {
                selection["seed_bundle"]["bundle_sha256"],
                selection["deployed_bundle"]["bundle_sha256"],
            }:
                raise EvaluationError("test bundle was not locked by the selection")
        elif phase == "test":
            raise EvaluationError("test phase cannot execute another split")

    def _evaluate_item(self, run: _EvaluationPass, item_id: str, seed: int, attempts: list[dict]) -> RunRecord:
        """The final record of one item: retried until usable or out of infrastructure retries.

        Attempts journaled before a resume count towards the retry limit. Every
        new attempt is appended to ``attempts`` and journaled; the record carries
        the usage and latency of all of the item's attempts.
        """
        native = self.executor.data.native(item_id)
        prior = [a for a in attempts if a["example_id"] == item_id]
        task_usage = _component_usage(prior, "task")
        judge_usage = _component_usage(prior, "judge")
        record = None
        for attempt_index in range(len(prior), self.max_infrastructure_retries + 1):
            record, _outcome, attempt = self.executor.attempt(
                example_id=item_id,
                native=native,
                bundle=run.bundle,
                request_seed=seed,
                attempt_index=attempt_index,
                phase=run.phase,
                evaluation_seed_identity=run.seed_identity,
                strict=True,
            )
            attempts.append(attempt)
            self._journal(run.journal_path, {"kind": "attempt", "attempt": attempt})
            task_usage += Usage(**attempt["usage_by_component"]["task"])
            judge_usage += Usage(**attempt["usage_by_component"]["judge"])
            if record.usable or attempt_index == self.max_infrastructure_retries:
                break
        if record is None:
            # Every attempt for this item was already spent before a resume.
            record = RunRecord(
                cell_id=self.cell.cell_id,
                example_id=item_id,
                status="infrastructure_failure",
                request_seed=seed,
                model_id=self.cell.task_model,
                framework=self.cell.framework,
                error="infrastructure retries exhausted before resume",
            )
        item_attempts = [a for a in attempts if a["example_id"] == item_id]
        record.usage = task_usage
        record.latency_seconds = sum(a["latency_seconds"] for a in item_attempts)
        record.metadata.update(
            {
                "condition_id": self.condition_id,
                "evaluation_id": run.evaluation_id,
                "bundle_sha256": run.bundle.digest,
                "split": run.split,
                "evaluation_seed": run.evaluation_seed,
                "evaluation_seed_identity": run.seed_identity,
                "phase": run.phase,
                "execution_attempts": len(item_attempts),
                "infrastructure_failures": sum(a["outcome"] == "infrastructure_failure" for a in item_attempts),
                "usage_by_component": {"task": asdict(task_usage), "judge": asdict(judge_usage)},
                "search_budget_charged": 0,
            }
        )
        return record

    @staticmethod
    def _journal(path: Path, value: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(value, sort_keys=True, ensure_ascii=False) + "\n")
            handle.flush()

    @staticmethod
    def _replay(path: Path, ids: Sequence[str]) -> tuple[list[RunRecord], list[dict[str, Any]]]:
        records: list[RunRecord] = []
        attempts: list[dict[str, Any]] = []
        if not path.is_file():
            return records, attempts
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                break  # torn final line from an interrupted write
            if entry.get("kind") == "attempt":
                attempts.append(entry["attempt"])
            elif entry.get("kind") == "record":
                records.append(record_from_dict(entry["record"]))
        if [r.example_id for r in records] != list(ids[: len(records)]):
            raise EvaluationError("evaluation journal is not a committed prefix of the split")
        return records, attempts


@dataclass(frozen=True)
class _EvaluationPass:
    """What every item of one evaluation in progress shares."""

    bundle: PromptBundle
    phase: str
    split: str
    evaluation_seed: int
    seed_identity: str
    evaluation_id: str
    journal_path: Path


def _component_usage(attempts: Sequence[Mapping[str, Any]], component: str) -> Usage:
    """Total ``task`` or ``judge`` usage of ``attempts``."""
    return sum((Usage(**a["usage_by_component"][component]) for a in attempts), Usage())


def _usage_summary(attempts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Usage of every attempt of one evaluation; evaluations are never charged to the search budget."""
    return {
        "task": asdict(_component_usage(attempts, "task")),
        "judge": asdict(_component_usage(attempts, "judge")),
        "attempts": len(attempts),
        "infrastructure_failures": sum(a["outcome"] == "infrastructure_failure" for a in attempts),
        "service_seconds": sum(a["latency_seconds"] for a in attempts),
        "search_budget_charged": 0,
    }


def validate_evaluation(
    result: EvaluationResult,
    *,
    expected_ids: Sequence[str],
    bundle_sha256: str,
    condition_id: str,
    evaluation_seed: int,
    split: str,
) -> None:
    """Reject an evaluation whose identity, pairing or records differ from what was requested."""
    identity = result.identity
    expected = list(expected_ids)
    if (
        identity.get("condition_id") != condition_id
        or identity.get("bundle_sha256") != bundle_sha256
        or identity.get("evaluation_seed") != evaluation_seed
        or identity.get("split") != split
        or identity.get("ordered_example_ids") != expected
        or [r.example_id for r in result.records] != expected
        or len(set(expected)) != len(expected)
    ):
        raise EvaluationError("evaluation identity or full item pairing differs")
    if content_hash(identity.get("condition")) != condition_id:
        raise EvaluationError("evaluation condition digest mismatch")
    for record in result.records:
        if record.metadata.get("evaluation_id") != result.evaluation_id:
            raise EvaluationError("record belongs to a different evaluation")
        if record.request_seed != paired_evaluation_seed(condition_id, evaluation_seed, split, record.example_id):
            raise EvaluationError("record uses a different paired request seed")
        if record.status not in {"success", "semantic_failure", "infrastructure_failure"}:
            raise EvaluationError("unclassified evaluation record")
        if record.usable and (record.score is None or not 0 <= record.score <= 1):
            raise EvaluationError("invalid usable evaluation score")


# Selection
def optimization_is_reportable(envelope: Mapping[str, Any]) -> bool:
    """Completed, or failed only for exhausted infrastructure (documented fallback)."""
    status = envelope.get("status")
    if status != "completed" and not (status == "failed" and envelope.get("failure_kind") == "infrastructure_invalid"):
        return False
    budget = envelope.get("budget")
    return isinstance(budget, dict) and closed_ledger(budget)


def lock_selection(
    path: Path,
    *,
    envelope: Mapping[str, Any],
    seed_bundle: PromptBundle,
    incumbent_bundle: PromptBundle,
    baseline: EvaluationResult,
    candidate: EvaluationResult,
    forced_fallback_reason: str | None = None,
) -> dict[str, Any]:
    """Seal the deployment decision before any test row is loaded."""
    if not optimization_is_reportable(envelope):
        raise EvaluationError("ambiguous or unclassified optimization failure cannot enter selection")
    assert_test_unexposed(dict(incumbent_bundle.metadata))
    decision = select_for_deployment(seed_bundle, incumbent_bundle, baseline.records, candidate.records)
    reason = forced_fallback_reason or decision.fallback_reason
    if envelope.get("status") != "completed":
        reason = "infrastructure_invalid_optimization"
    deployed = seed_bundle if reason else decision.deployed
    payload = {
        "schema": schema_name("locked-selection"),
        "protocol_id": PROTOCOL_ID,
        "cell": dict(envelope["cell"]),
        "condition_id": baseline.identity["condition_id"],
        "optimization_envelope_sha256": content_hash(dict(envelope)),
        "seed_bundle": bundle_to_dict(seed_bundle),
        "incumbent_bundle": bundle_to_dict(incumbent_bundle),
        "deployed_bundle": bundle_to_dict(deployed),
        "selected_candidate": reason is None and decision.selected_candidate,
        "fallback_reason": reason,
        "baseline_validation_score": decision.seed_score,
        "incumbent_validation_score": None if forced_fallback_reason else decision.candidate_score,
        "validation_delta": None if forced_fallback_reason else decision.delta,
        "baseline_validation_id": baseline.evaluation_id,
        "candidate_validation_id": candidate.evaluation_id,
        "ordered_validation_ids": list(baseline.identity["ordered_example_ids"]),
        "test_exposed": False,
    }
    payload["selection_id"] = content_hash(payload)
    with exclusive(path.with_suffix(".lock")):
        if path.exists() and verify_sealed(read_json(path)) != payload:
            raise EvaluationError("locked selection changed")
        atomic_write_json(path, sealed(payload))
    return payload


def load_selection(path: Path) -> dict[str, Any]:
    """Read and verify a sealed ``selection.json``."""
    try:
        value = verify_sealed(read_json(path))
    except ValueError as exc:
        raise EvaluationError("locked selection is corrupt") from exc
    body = {key: item for key, item in value.items() if key != "selection_id"}
    if value.get("selection_id") != content_hash(body):
        raise EvaluationError("selection identity mismatch")
    if value.get("schema") != schema_name("locked-selection") or value.get("test_exposed") is not False:
        raise EvaluationError("invalid locked selection contract")
    return value


__all__ = [
    "EvaluationResult",
    "Evaluator",
    "condition_identity",
    "load_selection",
    "lock_selection",
    "optimization_is_reportable",
    "validate_evaluation",
]
