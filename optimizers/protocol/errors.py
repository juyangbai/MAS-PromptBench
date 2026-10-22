"""Every exception the run protocol and the optimizer methods raise.

Class names are part of the job artifacts (``optimization.json`` records
``error_type``) and of the failure classification in ``run.py``:

* infrastructure (the documented seed fallback): :class:`PreObservationInfrastructureFailure`,
  :class:`OptimizerInfrastructureFailure`, :class:`NativeInfrastructureExhausted`;
* contract failures: :class:`JobError`, :class:`RunnerContractError`,
  :class:`EvaluationError` and every ``ValueError``;
* anything else is a program failure.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


# Runner and artifacts
class RunnerContractError(RuntimeError):
    """A binding returned data that cannot be reported under the protocol."""


class UnsupportedRuntimeBinding(RunnerContractError):
    """No real-runner adapter exists for the resolved cell."""


class PreObservationInfrastructureFailure(RuntimeError):
    """A retryable failure that produced no usable task observation.

    Malformed answers, tool failures and wrong answers are semantic results
    and must be returned so they are charged. Retries reuse the request seed.
    """

    def __init__(
        self,
        message: str,
        *,
        stage: str = "task_runtime",
        usage: Any = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        from .schema import Usage

        super().__init__(message)
        self.stage = stage
        self.usage = usage or Usage()
        self.metadata = dict(metadata or {})


class TelemetryError(RuntimeError):
    """Native output carries no usable telemetry."""


class ArtifactError(RuntimeError):
    """A job artifact cannot be read or would leave the job directory."""


class EvaluationError(RuntimeError):
    """An uncharged evaluation or the locked selection violates the protocol."""


class JobError(RuntimeError):
    """A job cannot run as requested (cell, phase or saved artifacts)."""


class MethodUnavailable(ImportError):
    """A registered method's module is not installed."""


# Optimizer methods
class OptimizerContractError(RuntimeError):
    """A native optimizer cannot satisfy the runner contract."""


class OptimizerInfrastructureFailure(OptimizerContractError):
    """Infrastructure retries were exhausted without a usable observation."""


class NativeIntegrationError(RuntimeError):
    """A native optimizer loop cannot continue without corrupting its trajectory."""


class NativeInfrastructureExhausted(NativeIntegrationError):
    """The runner exhausted transport retries before a usable observation."""


class UnsupportedBaselineCell(ValueError):
    """A cell is outside the method's grid (raised before any model call)."""


class UnsupportedGEPACell(ValueError):
    """The GEPA protocol bridge cannot represent the requested cell."""


class MetricCallCapReached(NativeIntegrationError):
    """MAMUT-GEPA requested a batch that cannot fit the exact metric-call cap."""


class MASPOBDependencyError(ImportError):
    """A heavy MASPOB dependency (numpy, torch, torch_geometric, sentence-transformers) is not installed."""


__all__ = [
    "ArtifactError",
    "EvaluationError",
    "JobError",
    "MASPOBDependencyError",
    "MethodUnavailable",
    "MetricCallCapReached",
    "NativeInfrastructureExhausted",
    "NativeIntegrationError",
    "OptimizerContractError",
    "OptimizerInfrastructureFailure",
    "PreObservationInfrastructureFailure",
    "RunnerContractError",
    "TelemetryError",
    "UnsupportedBaselineCell",
    "UnsupportedGEPACell",
    "UnsupportedRuntimeBinding",
]
