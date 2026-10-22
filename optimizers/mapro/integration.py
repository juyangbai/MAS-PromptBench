"""MAPRO on the shared run protocol.

MAPRO (Zhang et al., "MAPRO: Recasting Multi-Agent Prompt Optimization as
Maximum a Posteriori Inference", arXiv:2510.07475) keeps a K-candidate prompt
pool per role, scores candidates with LLM-judged node/edge potentials on the
cell's prompt graph, selects the joint assignment by exact max-product MAP and
refines the pools from topology-aware blame feedback (``native.py``).

Every full-MAS rollout goes through the protocol runner via ``RunnerSession``.
Pool initialization, blame and mutation use the reflection model under the
common reflection policy (thinking on, 48,000 output tokens, native
temperature/top-p). Candidate probes and node/edge judges are single-agent
calls on the cell's task model at the task endpoints configured by the
protocol (thinking off); they are not MAS rollouts and are never charged.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from optimizers.protocol.cells import validate_cell
from optimizers.protocol.errors import NativeIntegrationError
from optimizers.protocol.journal import LifecycleJournal
from optimizers.protocol.reflection import REFLECTION_MAX_OUTPUT_TOKENS, ReflectionBackend, ReflectionClient
from optimizers.protocol.rollouts import native_role_order
from optimizers.protocol.schema import CellSpec, OptimizerResult, PromptBundle
from optimizers.protocol.seeding import reflection_seed
from optimizers.protocol.session import RunnerSession
from optimizers.protocol.settings import ProtocolSettings

from .regime import ADAPTATION_ID, MAPROSettings, regime_metadata

DEFAULTS = MAPROSettings()


def task_model_client(
    task_model: str, *, endpoints: tuple[str, ...] | None = None, http_client: Any = None
) -> ReflectionClient:
    """Request-logged client for MAPRO's task-model judge and probes.

    The model is the cell's task model, so a Llama cell is judged by Llama;
    requests rotate over the protocol's task endpoints (``TASK_ENDPOINTS``,
    else ``VLLM_BASE_URL``).
    """
    endpoints = tuple(endpoints) if endpoints is not None else ProtocolSettings.from_env().task_endpoints
    if not endpoints:
        raise NativeIntegrationError("MAPRO judge needs a task endpoint: set TASK_ENDPOINTS or VLLM_BASE_URL")
    return ReflectionClient(model=task_model, base_urls=endpoints, http_client=http_client)


class _SeededAsyncChat:
    """Native ``LLMClient.chat_text`` shape over a request-logged backend."""

    def __init__(self, cell: CellSpec, backend: ReflectionBackend, phase: str, *, thinking: bool = False) -> None:
        self.cell, self.backend, self.phase = cell, backend, phase
        # Thinking belongs to the model tier, not the native generation config:
        # the same shim serves the task-model judge, which keeps thinking off.
        self.thinking = bool(thinking)
        self._prompt_turns: dict[tuple[str, str | None], int] = {}
        self._lock = threading.Lock()
        self.n_errors = 0

    @property
    def usage(self) -> Mapping[str, int]:
        usage = dict(self.backend.snapshot().get("usage") or {})
        return {
            "prompt_tokens": int(usage.get("input_tokens", 0)),
            "completion_tokens": int(usage.get("output_tokens", 0)),
            "n_calls": int(usage.get("model_calls", 0)),
            "n_errors": self.n_errors,
        }

    async def chat_text(self, prompt: str, system: str | None = None, cfg: Any = None) -> str:
        with self._lock:
            key = (prompt, system)
            turn = self._prompt_turns.get(key, 0)
            self._prompt_turns[key] = turn + 1
        temperature = float(getattr(cfg, "temperature", 0.7))
        top_p = float(getattr(cfg, "top_p", 1.0))
        max_tokens = REFLECTION_MAX_OUTPUT_TOKENS if self.thinking else int(getattr(cfg, "max_tokens", 1024))
        return await asyncio.to_thread(
            self.backend.complete,
            prompt,
            request_seed=reflection_seed(self.cell, phase=self.phase, iteration=turn, role=self.phase, prompt=prompt),
            phase=self.phase,
            role=self.phase,
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_tokens,
            thinking=self.thinking,
            system=system,
        )


class _SeededProbe:
    """Native candidate-probe shape (system = candidate prompt, user = role context)."""

    def __init__(
        self, cell: CellSpec, backend: ReflectionBackend, num_threads: int, settings: MAPROSettings = DEFAULTS
    ) -> None:
        self.cell, self.backend = cell, backend
        self.num_threads = max(1, int(num_threads))
        self.settings = settings
        self._prompt_turns: dict[tuple[str, str], int] = {}
        self._lock = threading.Lock()

    @property
    def usage(self) -> Mapping[str, int]:
        usage = dict(self.backend.snapshot().get("usage") or {})
        return {
            "prompt_tokens": int(usage.get("input_tokens", 0)),
            "completion_tokens": int(usage.get("output_tokens", 0)),
            "n_calls": int(usage.get("model_calls", 0)),
            "n_errors": 0,
        }

    def complete(self, system: str, prompt: str) -> str:
        with self._lock:
            key = (prompt, system)
            turn = self._prompt_turns.get(key, 0)
            self._prompt_turns[key] = turn + 1
        return self.backend.complete(
            prompt,
            request_seed=reflection_seed(
                self.cell,
                phase="mapro_candidate_probe",
                iteration=turn,
                role="candidate_probe",
                prompt=system + "\n" + prompt,
            ),
            phase="candidate_probe",
            role="candidate_probe",
            temperature=self.settings.probe_temperature,
            top_p=self.settings.probe_top_p,
            max_output_tokens=self.settings.probe_max_tokens,
            thinking=False,
            system=system,
        )

    def batch(self, jobs: list[tuple[str, str]]) -> list[str]:
        if not jobs:
            return []
        with ThreadPoolExecutor(max_workers=min(self.num_threads, len(jobs))) as pool:
            return list(pool.map(lambda item: self.complete(*item), jobs))


class MAPROOptimizer:
    """MAPRO on the protocol runner; the keyword knobs are :class:`~.regime.MAPROSettings`.

    ``reflection`` defaults to the protocol :class:`ReflectionClient`;
    ``task_judge`` (node/edge judges and candidate probes) to
    :func:`task_model_client` on the cell's task model.
    """

    method = "mapro"
    native_iteration_event = "completed outer mutation/selection round"

    def __init__(
        self,
        *,
        seed_bundle: PromptBundle | None = None,
        reflection: ReflectionBackend | None = None,
        task_judge: ReflectionBackend | None = None,
        candidate_count: int = DEFAULTS.candidate_count,
        max_iterations: int = DEFAULTS.max_iterations,
        patience: int = DEFAULTS.patience,
        epsilon: float = DEFAULTS.epsilon,
        scoring_batch: int = DEFAULTS.scoring_batch,
        evaluation_batch: int = DEFAULTS.evaluation_batch,
        feedback_count: int = DEFAULTS.feedback_count,
        num_threads: int = DEFAULTS.num_threads,
        use_demos: bool = DEFAULTS.use_demos,
    ) -> None:
        self.seed_bundle = seed_bundle
        self.reflection = reflection or ReflectionClient()
        # Default: the cell's task model on the protocol task endpoints (bound in optimize).
        self.task_judge = task_judge
        self.settings = MAPROSettings(
            candidate_count=int(candidate_count),
            max_iterations=int(max_iterations),
            patience=int(patience),
            epsilon=float(epsilon),
            scoring_batch=int(scoring_batch),
            evaluation_batch=int(evaluation_batch),
            feedback_count=int(feedback_count),
            num_threads=int(num_threads),
            use_demos=bool(use_demos),
        )
        if self.settings.candidate_count < 1 or self.settings.max_iterations < 1 or self.settings.patience < 1:
            raise ValueError("invalid MAPRO native configuration")

    def optimize(
        self, cell: CellSpec, runner: Any, budget: Any, training: list[Any], validation: list[Any]
    ) -> OptimizerResult:
        """Run the MAPRO loop on the training rows (final validation is the protocol's, uncharged)."""
        del validation
        validate_cell(self.method, cell)
        if self.seed_bundle is None:
            raise NativeIntegrationError("MAPRO requires the canonical seed PromptBundle")
        if not training:
            raise NativeIntegrationError("MAPRO requires a non-empty training split")

        from . import native

        if native.MAPRO_LISTWISE_FLAG or native.MAPRO_PAPER_ANCHOR_FLAG:
            raise NativeIntegrationError("MAPRO requires the frozen pointwise/best-so-far native regime")

        seed = self.seed_bundle
        roles = native_role_order(cell, list(seed.roles))
        if cell.topology == "centralized":
            managers = [role for role in roles if role == "manager" or role.startswith("manager_r")]
            if len(managers) != 1:
                raise NativeIntegrationError("MAPRO centralized binding requires one exact manager")
            roles = managers + [role for role in roles if role not in managers]
        if self.task_judge is None:
            self.task_judge = task_model_client(cell.task_model)
        session = RunnerSession(cell=cell, runner=runner, budget=budget, seed_bundle=seed)
        journal = LifecycleJournal(method=self.method, cell=cell, session=session)
        journal.observe(
            iteration=0,
            native_event="initial_state",
            current=seed,
            incumbent=seed,
            native_score=None,
            accepted=False,
            reasons=("initial_state",),
            state={"iteration": 0, "candidate": dict(seed.roles)},
        )
        reflection_chat = _SeededAsyncChat(cell, self.reflection, "mapro_reflection", thinking=True)
        judge_chat = _SeededAsyncChat(cell, self.task_judge, "mapro_node_edge_judge")
        probe = _SeededProbe(cell, self.task_judge, self.settings.num_threads, self.settings)
        iteration_count = 0

        def on_iteration(payload: Mapping[str, Any]) -> None:
            nonlocal iteration_count
            iteration_count = int(payload["iteration"])
            selected = PromptBundle(
                roles=dict(payload["selected_assignment"]), demos=seed.demos, metadata=seed.metadata
            )
            incumbent = PromptBundle(roles=dict(payload["best_assignment"]), demos=seed.demos, metadata=seed.metadata)
            changed = incumbent.digest != journal.events[-1]["incumbent_bundle_sha256"]
            journal.observe(
                iteration=iteration_count,
                native_event=self.native_iteration_event,
                current=selected,
                incumbent=incumbent,
                native_score=float(payload["selected_score"]),
                accepted=changed,
                reasons=("native_iteration_end", "incumbent_change") if changed else ("native_iteration_end",),
                state=dict(payload),
                coordinates={"round": iteration_count, "candidate_pool": self.settings.candidate_count},
            )

        def unreachable_metric(_row: Any, _prediction: Any) -> float:
            raise NativeIntegrationError("MAPRO runner output unexpectedly bypassed its protocol score")

        result = asyncio.run(
            native.optimize_mapro(
                self.settings.native_args(),
                session,
                roles,
                dict(seed.roles),
                training,
                unreachable_metric,
                session.pacing,
                log=lambda _line: None,
                shim=reflection_chat,
                judge=judge_chat,
                probe=probe,
                on_iteration=on_iteration,
                topology=cell.topology,
                team_size=cell.team_size,
            )
        )
        if session.pacing.pending:
            raise NativeIntegrationError("MAPRO ended with unsettled native rollout reservations")
        incumbent = PromptBundle(roles=dict(result["best_assignment"]), demos=seed.demos, metadata=seed.metadata)
        journal.observe(
            iteration=iteration_count,
            native_event="final_state",
            current=incumbent,
            incumbent=incumbent,
            native_score=float(result["best_train"]),
            accepted=False,
            reasons=("final_state", "early_stop"),
            state={
                "iteration": iteration_count,
                "best_assignment": dict(result["best_assignment"]),
                "best_train": float(result["best_train"]),
                "seed_train": float(result["seed_train"]),
                "trajectory": list(result["trajectory"]),
                "history": list(result["history"]),
                "stop_reason": str(result["stop_reason"]),
            },
        )
        journal.finalize_curve()
        return journal.artifact(
            seed_bundle=seed,
            incumbent_bundle=incumbent,
            native_iterations=iteration_count,
            stop_reason=str(result["stop_reason"]),
            metadata={
                "adaptation_id": ADAPTATION_ID,
                **regime_metadata(native.MAPRO_LISTWISE_FLAG, native.MAPRO_PAPER_ANCHOR_FLAG),
                "native_result": result,
                "reflection": dict(self.reflection.snapshot()),
                "task_judge_and_probe": dict(self.task_judge.snapshot()),
            },
        )


__all__ = ["MAPROOptimizer", "task_model_client"]
