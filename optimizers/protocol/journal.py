"""Zero-call records of an optimization: events, learning curve, checkpoints and the result file.

:class:`LifecycleJournal` records the native loops of the ``RunnerSession``
methods (one event, learning-curve rows and a canonical checkpoint per
native step); :func:`native_checkpoint` and :func:`persist_optimizer_artifact`
serve the search-layout methods. Nothing here calls a model.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .artifacts import ArtifactStore
from .errors import NativeIntegrationError
from .learning_curve import CommittedState, LearningCurveRecorder
from .schema import (
    JOURNAL_LAYOUT,
    CellSpec,
    OptimizerResult,
    PromptBundle,
    canonical_json,
    content_hash,
    schema_name,
    to_jsonable,
)
from .session import RunnerSession


def _bundle_payload(bundle: PromptBundle) -> dict[str, Any]:
    return {
        "roles": dict(bundle.roles),
        "demos": list(bundle.demos),
        "metadata": dict(bundle.metadata),
        "sha256": bundle.digest,
    }


class LifecycleJournal:
    """Zero-call event, learning-curve and canonical checkpoint recorder."""

    def __init__(self, *, method: str, cell: CellSpec, session: RunnerSession) -> None:
        self.method = method
        self.cell = cell
        self.session = session
        self.started = time.monotonic()
        self.events: list[dict[str, Any]] = []
        self.checkpoints: list[dict[str, Any]] = []
        self.curve = LearningCurveRecorder(maximum=int(session.budget.maximum))
        self.store: ArtifactStore | None = getattr(session.runner, "artifact_store", None)
        self.directory: Path | None = getattr(session.runner, "artifact_directory", None)

    def restore_curve(self, checkpoint: Mapping[str, Any]) -> None:
        """Restore zero-call learning-curve state from a canonical checkpoint."""
        if self.events or self.checkpoints or self.curve.rows:
            raise NativeIntegrationError("checkpoint must be restored into a fresh journal")
        raw = checkpoint.get("learning_curve")
        if not isinstance(raw, list) or not raw:
            raise NativeIntegrationError("checkpoint lacks resumable learning-curve state")
        rows = [dict(row) for row in raw]
        last = rows[-1]
        expected = int(dict(checkpoint.get("ledger") or {}).get("charged", -1))
        if int(last.get("charged_rollouts", -2)) != expected:
            raise NativeIntegrationError("checkpoint learning curve/ledger mismatch")
        grids = [int(row["rollout_grid"]) for row in rows if row.get("rollout_grid") is not None]
        self.curve.rows = rows
        self.curve._next_grid = (max(grids) + self.curve.step) if grids else 0
        self.curve._last = CommittedState(
            charged_rollouts=expected,
            bundle_hash=str(last["bundle_hash"]),
            native_iteration=int(last["native_iteration"]),
            native_score=last.get("native_observed_score"),
            metadata=dict(last.get("metadata") or {}),
        )

    def observe(
        self,
        *,
        iteration: int,
        native_event: str,
        current: PromptBundle,
        incumbent: PromptBundle,
        native_score: float | None,
        accepted: bool,
        reasons: Sequence[str],
        state: Mapping[str, Any],
        coordinates: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        """Record one native step: event, learning-curve rows, checkpoint and bundles."""
        state_snapshot = json.loads(canonical_json(dict(state)))
        snap = dict(self.session.budget.snapshot())
        event_base = {
            "method": self.method,
            "cell_id": self.cell.cell_id,
            "optimizer_seed": self.cell.optimizer_seed,
            "native_iteration": iteration,
            "native_event": native_event,
            "reasons": list(dict.fromkeys(reasons)),
            "elapsed_seconds": time.monotonic() - self.started,
            "charged_rollouts": int(snap["charged"]),
            "attempted_rollouts": int(snap["attempted"]),
            "successful_rollouts": int(snap.get("successful", 0)),
            "semantic_failures": int(snap.get("semantic_failures", 0)),
            "infrastructure_failures": int(snap.get("infrastructure_failures", 0)),
            "retries": int(snap.get("retries", 0)),
            "current_bundle_sha256": current.digest,
            "incumbent_bundle_sha256": incumbent.digest,
            "accepted": bool(accepted),
            "native_observed_score": native_score,
            "coordinates": dict(coordinates or {}),
            "optimizer_state_sha256": content_hash(state_snapshot),
        }
        event = {"event_id": content_hash(event_base), **event_base}
        self.events.append(event)
        curve_rows = self.curve.observe(
            int(snap["charged"]),
            incumbent.digest,
            iteration,
            native_score,
            reasons,
            metadata={
                "method": self.method,
                "native_event": native_event,
                "event_id": event["event_id"],
                "current_bundle_sha256": current.digest,
                "accepted": bool(accepted),
                "coordinates": dict(coordinates or {}),
            },
        )
        checkpoint_base = {
            "schema": schema_name("native-checkpoint"),
            "method": self.method,
            "protocol_id": self.cell.protocol_id,
            "cell_id": self.cell.cell_id,
            "cell_identity_sha256": content_hash(dict(self.cell.identity)),
            "optimizer_seed": self.cell.optimizer_seed,
            "native_iteration": iteration,
            "native_event": native_event,
            "ledger": snap,
            "current_bundle": _bundle_payload(current),
            "incumbent_bundle": _bundle_payload(incumbent),
            "state": state_snapshot,
            "optimizer_state_sha256": event_base["optimizer_state_sha256"],
            "learning_curve": list(self.curve.rows),
        }
        checkpoint = {"checkpoint_id": content_hash(checkpoint_base), **checkpoint_base}
        self.checkpoints.append(checkpoint)
        if self.store is not None and self.directory is not None:
            self.store.append_jsonl(self.directory, "optimizer_events.jsonl", event)
            for row in curve_rows:
                self.store.append_jsonl(self.directory, "learning_curve.jsonl", row)
            self.store.append_jsonl(self.directory, "optimizer_checkpoints.jsonl", checkpoint)
            for bundle in (current, incumbent):
                self.store.write_json(
                    self.directory,
                    f"optimizer-bundle-{bundle.digest}.json",
                    {
                        "schema": schema_name("prompt-bundle"),
                        "bundle_sha256": bundle.digest,
                        "roles": dict(bundle.roles),
                        "demos": list(bundle.demos),
                        "metadata": dict(bundle.metadata),
                    },
                )
        return checkpoint

    def finalize_curve(self) -> None:
        """Carry the final incumbent forward to B on the learning curve."""
        rows = self.curve.carry_forward_after_stop()
        if self.store is not None and self.directory is not None:
            for row in rows:
                self.store.append_jsonl(self.directory, "learning_curve.jsonl", row)

    def artifact(
        self,
        *,
        seed_bundle: PromptBundle,
        incumbent_bundle: PromptBundle,
        native_iterations: int,
        stop_reason: str,
        metadata: Mapping[str, Any],
    ) -> OptimizerResult:
        """The journal-layout optimizer result (also written to ``optimizer_result.json``)."""
        artifact = OptimizerResult(
            method=self.method,
            protocol_id=self.cell.protocol_id,
            cell_id=self.cell.cell_id,
            seed_bundle=seed_bundle,
            incumbent_bundle=incumbent_bundle,
            native_iterations=native_iterations,
            stop_reason=stop_reason,
            records=tuple(self.session.records),
            request_ledger=tuple(asdict(value) for value in self.session.request_events),
            events=tuple(self.events),
            checkpoints=tuple(self.checkpoints),
            learning_curve=tuple(self.curve.rows),
            budget_snapshot=dict(self.session.budget.snapshot()),
            metadata=dict(metadata),
            layout=JOURNAL_LAYOUT,
        )
        if self.store is not None and self.directory is not None:
            self.store.write_json(self.directory, "optimizer_result.json", artifact.to_dict())
        return artifact


def require_resume_checkpoint(
    checkpoint: Mapping[str, Any] | None, *, method: str, cell: CellSpec, budget: Any
) -> Mapping[str, Any] | None:
    """Verify a checkpoint against the method, the cell and the current ledger (None passes through)."""
    if checkpoint is None:
        return None
    if checkpoint.get("schema") != schema_name("native-checkpoint"):
        raise NativeIntegrationError("unsupported native checkpoint schema")
    if checkpoint.get("method") != method or checkpoint.get("cell_id") != cell.cell_id:
        raise NativeIntegrationError("checkpoint method/cell identity mismatch")
    if checkpoint.get("cell_identity_sha256") != content_hash(dict(cell.identity)):
        raise NativeIntegrationError("checkpoint CellSpec hash mismatch")
    base = {key: value for key, value in checkpoint.items() if key != "checkpoint_id"}
    if checkpoint.get("checkpoint_id") != content_hash(base):
        raise NativeIntegrationError("checkpoint content hash mismatch")
    expected_ledger = dict(checkpoint.get("ledger") or {})
    actual_ledger = dict(budget.snapshot())
    for key in (
        "maximum",
        "charged",
        "attempted",
        "successful",
        "semantic_failures",
        "infrastructure_failures",
        "retries",
    ):
        if int(expected_ledger.get(key, -1)) != int(actual_ledger.get(key, -2)):
            raise NativeIntegrationError(f"checkpoint ledger mismatch: {key}")
    return checkpoint


def bundle_from_checkpoint(value: Mapping[str, Any], key: str = "incumbent_bundle") -> PromptBundle:
    """A bundle stored in a checkpoint, verified against its hash."""
    raw = value.get(key)
    if not isinstance(raw, Mapping):
        raise NativeIntegrationError(f"checkpoint lacks {key}")
    bundle = PromptBundle(
        roles=dict(raw.get("roles") or {}),
        demos=tuple(raw.get("demos") or ()),
        metadata=dict(raw.get("metadata") or {}),
    )
    if raw.get("sha256") != bundle.digest:
        raise NativeIntegrationError(f"checkpoint {key} hash mismatch")
    return bundle


def persist_optimizer_artifact(runner: Any, artifact: OptimizerResult) -> None:
    """Write the result and its checkpoints beside the runner artifacts (when the runner has a store)."""
    store = getattr(runner, "artifact_store", None)
    directory = getattr(runner, "artifact_directory", None)
    if store is None or directory is None:
        return
    payload = artifact.to_dict()
    payload["artifact_sha256"] = content_hash(payload)
    store.write_json(directory, "optimizer_result.json", payload)
    for index, checkpoint in enumerate(artifact.checkpoints):
        store.write_json(directory, f"optimizer_checkpoint_{index:04d}.json", to_jsonable(checkpoint))


def native_checkpoint(
    *, method: str, iteration: int, bundle: PromptBundle, budget: Any, state: Mapping[str, Any]
) -> dict[str, Any]:
    """Content-hashed checkpoint of a search-layout method (bundle, ledger, native state)."""
    payload = {
        "schema": schema_name("native-optimizer-checkpoint"),
        "method": method,
        "iteration": int(iteration),
        "bundle": to_jsonable(bundle),
        "budget_snapshot": dict(budget.snapshot()),
        "state": to_jsonable(state),
    }
    payload["checkpoint_sha256"] = content_hash(payload)
    return payload


__all__ = [
    "LifecycleJournal",
    "bundle_from_checkpoint",
    "native_checkpoint",
    "persist_optimizer_artifact",
    "require_resume_checkpoint",
]
