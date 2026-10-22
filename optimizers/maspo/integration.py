"""MASPO on the shared run protocol: the retained full beam search over one cell.

Every full-MAS rollout goes through the protocol runner (``RunnerSession`` over
the budget-owning ledger), every proposal/judge call through one reflection
backend with logical request seeds. Topology wiring:

* centralized: workers in bundle order, then the exact manager (terminal);
  each worker's successor is the manager, the manager's predecessors are the
  workers;
* sequential: the native stage order, final stage terminal, successor = next
  stage, predecessor = previous stage;
* single/independent/decentralized: the one shared replica prompt, scored by
  full-system global credit.

Search settings follow the retained code: beam 2, offspring 2, minibatch 10,
T=3 depth steps per role turn and a maximum search depth of 9 per role (the
method table lists D=3; the code value is kept). Search stops when the ledger
is exhausted or every role reaches the maximum depth. Final validation is the
protocol's (uncharged); the incumbent is the coordinate-ascent best bundle.
"""

from __future__ import annotations

import random
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from optimizers.protocol.cells import validate_cell
from optimizers.protocol.config import normalize_task
from optimizers.protocol.errors import NativeIntegrationError
from optimizers.protocol.journal import LifecycleJournal
from optimizers.protocol.reflection import REFLECTION_MAX_OUTPUT_TOKENS, ReflectionBackend, ReflectionClient
from optimizers.protocol.rollouts import native_role_order
from optimizers.protocol.schema import CellSpec, OptimizerResult, PromptBundle, StopReason
from optimizers.protocol.seeding import reflection_seed
from optimizers.protocol.session import RunnerSession

from . import search
from .search import PROPOSAL_TEMPERATURE, TaskType


@dataclass(frozen=True)
class MASPOSettings:
    """MASPO's knobs (no environment variables).

    Beam ``beam_width`` with ``offspring`` proposals per parent, ``minibatch``
    rows per node, ``rounds_per_turn`` depth steps per role turn up to
    ``maximum_search_depth``. Budget split: every parent and candidate
    minibatch is charged (reserved before dispatch); the search stops when the
    ledger is spent. ``num_threads`` / ``reflection_inflight`` bound concurrent
    rollouts / judge calls.
    """

    beam_width: int = 2
    offspring: int = 2
    minibatch: int = 10
    maximum_search_depth: int = 9
    rounds_per_turn: int = 3
    num_threads: int = 12
    reflection_inflight: int = 4
    reflection_top_p: float = 1.0


DEFAULTS = MASPOSettings()


class _MASPOReflectionClient:
    """Upstream ``complete(prompt, temperature, max_tokens)`` shape with logical seeds.

    Temperature > 0 is a proposal, 0.0 a pairwise judgment. Thinking and the
    output-token ceiling follow the common reflection policy.
    """

    def __init__(self, cell: CellSpec, backend: ReflectionBackend, top_p: float = DEFAULTS.reflection_top_p) -> None:
        self.cell, self.backend, self.top_p = cell, backend, top_p
        self._prompt_turns: dict[tuple[str, float, int], int] = {}
        self._lock = threading.Lock()

    @property
    def usage(self) -> Mapping[str, int]:
        usage = dict(self.backend.snapshot().get("usage") or {})
        return {
            "prompt_tokens": int(usage.get("input_tokens", 0)),
            "completion_tokens": int(usage.get("output_tokens", 0)),
            "n_calls": int(usage.get("model_calls", 0)),
        }

    def complete(self, prompt: str, temperature: float = PROPOSAL_TEMPERATURE, max_tokens: int = 4096) -> str:
        with self._lock:
            key = (prompt, float(temperature), int(max_tokens))
            turn = self._prompt_turns.get(key, 0)
            self._prompt_turns[key] = turn + 1
        phase = "proposal" if float(temperature) > 0 else "pairwise_judge"
        return self.backend.complete(
            prompt,
            request_seed=reflection_seed(self.cell, phase=f"maspo_{phase}", iteration=turn, role=phase, prompt=prompt),
            phase=phase,
            role=phase,
            temperature=float(temperature),
            top_p=self.top_p,
            max_output_tokens=REFLECTION_MAX_OUTPUT_TOKENS,
            thinking=True,
            system=None,
        )


class MASPOOptimizer:
    """MASPO's beam search on the protocol runner; the keyword knobs are :class:`MASPOSettings`.

    ``reflection`` (proposals and pairwise judgments) defaults to the protocol
    :class:`ReflectionClient`.
    """

    method = "maspo"
    native_iteration_event = "completed role-depth beam update"

    def __init__(
        self,
        *,
        seed_bundle: PromptBundle | None = None,
        reflection: ReflectionBackend | None = None,
        beam_width: int = DEFAULTS.beam_width,
        offspring: int = DEFAULTS.offspring,
        minibatch: int = DEFAULTS.minibatch,
        maximum_search_depth: int = DEFAULTS.maximum_search_depth,
        rounds_per_turn: int = DEFAULTS.rounds_per_turn,
        num_threads: int = DEFAULTS.num_threads,
        reflection_inflight: int = DEFAULTS.reflection_inflight,
    ) -> None:
        self.seed_bundle = seed_bundle
        self.reflection = reflection or ReflectionClient()
        self.settings = MASPOSettings(
            beam_width=int(beam_width),
            offspring=int(offspring),
            minibatch=int(minibatch),
            maximum_search_depth=int(maximum_search_depth),
            rounds_per_turn=int(rounds_per_turn),
            num_threads=int(num_threads),
            reflection_inflight=int(reflection_inflight),
        )
        settings = self.settings
        if (
            min(
                settings.beam_width,
                settings.offspring,
                settings.minibatch,
                settings.maximum_search_depth,
                settings.rounds_per_turn,
                settings.num_threads,
                settings.reflection_inflight,
            )
            < 1
        ):
            raise ValueError("invalid MASPO native configuration")

    @staticmethod
    def _fork_session(parent: RunnerSession) -> RunnerSession:
        """A second prompt holder sharing the parent's pacing, records and dispatch order."""
        child = RunnerSession(
            cell=parent.cell,
            runner=parent.runner,
            budget=parent.budget,
            seed_bundle=parent.seed_bundle,
            phase=parent.phase,
        )
        child.pacing = parent.pacing
        child.records = parent.records
        child.request_events = parent.request_events
        child._dispatch_lock = parent._dispatch_lock
        child._dispatch_state = parent._dispatch_state
        return child

    @staticmethod
    def wiring(cell: CellSpec, roles: list[str]) -> tuple[list[str], str, dict[str, str | None], dict[str, list[str]]]:
        """(role order, terminal role, successors, predecessors) of the native topology."""
        roles = native_role_order(cell, roles)
        if cell.topology == "centralized":
            managers = [role for role in roles if role == "manager" or role.startswith("manager_r")]
            if len(managers) != 1:
                raise NativeIntegrationError("MASPO requires the exact centralized manager")
            terminal = managers[0]
            role_order = [role for role in roles if role != terminal] + [terminal]
        else:
            terminal = roles[-1]
            role_order = roles
        workers = [role for role in role_order if role != terminal]
        if cell.topology == "centralized":
            successor: dict[str, str | None] = {role: terminal if role != terminal else None for role in role_order}
            predecessors = {terminal: list(workers), **{role: [terminal] for role in workers}}
        elif cell.topology == "sequential":
            successor = {
                role: role_order[i + 1] if i + 1 < len(role_order) else None for i, role in enumerate(role_order)
            }
            predecessors = {role: role_order[i - 1 : i] if i else [] for i, role in enumerate(role_order)}
        elif cell.topology in {"single", "independent", "decentralized"} and len(roles) == 1:
            successor, predecessors = {terminal: None}, {terminal: []}
        else:
            raise NativeIntegrationError("MASPO unsupported native prompt-variable topology")
        return role_order, terminal, successor, predecessors

    def optimize(
        self, cell: CellSpec, runner: Any, budget: Any, training: list[Any], validation: list[Any]
    ) -> OptimizerResult:
        """Run the beam search on the training rows (final validation is the protocol's, uncharged)."""
        del validation
        settings = self.settings
        validate_cell(self.method, cell)
        if self.seed_bundle is None:
            raise NativeIntegrationError("MASPO requires the canonical seed PromptBundle")
        if not training:
            raise NativeIntegrationError("MASPO requires a non-empty training split")
        regime = search.regime_mode(settings.rounds_per_turn)

        seed = self.seed_bundle
        session = RunnerSession(cell=cell, runner=runner, budget=budget, seed_bundle=seed)
        candidate_sessions = [self._fork_session(session) for _ in range(max(1, settings.offspring))]
        journal = LifecycleJournal(method=self.method, cell=cell, session=session)
        journal.observe(
            iteration=0,
            native_event="initial_state",
            current=seed,
            incumbent=seed,
            native_score=None,
            accepted=False,
            reasons=("initial_state",),
            state={"step": 0, "best_prompt": dict(seed.roles)},
        )

        role_order, terminal, successor, predecessors = self.wiring(cell, list(seed.roles))
        best_prompt = dict(seed.roles)
        states = {role: search.RoleState(role, seed.roles[role]) for role in role_order}
        mis_buffers: dict[str, list] = {role: [] for role in role_order}
        reflection_client = _MASPOReflectionClient(cell, self.reflection, settings.reflection_top_p)
        task = normalize_task(cell.task)
        task_type = TaskType.MATH if task == "math" else TaskType.CODE
        judge = search.Judge(reflection_client, task_type, search.dataset_contract(task), task)
        judge.topology, judge.terminal_role = cell.topology, terminal
        rng = random.Random(cell.optimizer_seed)
        steps: list[dict[str, Any]] = []
        step_number = 0

        def observe(step: Mapping[str, Any], event: str, coordinates: Mapping[str, Any]) -> None:
            incumbent = PromptBundle(roles=dict(best_prompt), demos=seed.demos, metadata=seed.metadata)
            accepted = bool(step.get("anchor_updated"))
            journal.observe(
                iteration=step_number,
                native_event=event,
                current=incumbent,
                incumbent=incumbent,
                native_score=float(step.get("best_cumulative", 0.0)),
                accepted=accepted,
                reasons=("native_iteration_end", "incumbent_change") if accepted else ("native_iteration_end",),
                state=self._state(role_order, states, best_prompt, mis_buffers, steps),
                coordinates=coordinates,
            )

        while not session.pacing.exhausted and any(
            state.depth < settings.maximum_search_depth for state in states.values()
        ):
            for role in role_order:
                if session.pacing.exhausted:
                    break
                state = states[role]
                if 0 < state.depth < settings.maximum_search_depth and not session.pacing.exhausted:
                    # Eq. 8: refresh at every role revisit, before this turn's depth steps.
                    session.set_context(iteration=step_number + 1, event=f"maspo_refresh:{role}:{state.depth}")
                    refresh = search.beam_refresh(
                        state,
                        session,
                        judge,
                        best_prompt,
                        training,
                        session.pacing,
                        rng,
                        settings.num_threads,
                        settings.reflection_inflight,
                        settings.minibatch,
                        is_terminal=role == terminal,
                        beam_width=settings.beam_width,
                    )
                    steps.append(refresh)
                    step_number += 1
                    observe(
                        refresh,
                        "completed beam refresh",
                        {
                            "outer_sweep": min(s.depth for s in states.values()),
                            "role": role,
                            "depth": state.depth,
                            "refresh": True,
                        },
                    )
                for _ in range(settings.rounds_per_turn):
                    if session.pacing.exhausted or state.depth >= settings.maximum_search_depth:
                        break
                    event = f"maspo_step:{role}:{state.depth}"
                    session.set_context(iteration=step_number + 1, event=event)
                    for candidate_session in candidate_sessions:
                        candidate_session.set_context(iteration=step_number + 1, event=event)
                    step = search.step_role(
                        state,
                        session,
                        candidate_sessions,
                        judge,
                        best_prompt,
                        training,
                        session.pacing,
                        rng,
                        settings.num_threads,
                        settings.reflection_inflight,
                        settings.beam_width,
                        settings.offspring,
                        settings.minibatch,
                        is_terminal=role == terminal,
                        successor_role=successor[role],
                        mis_buffers=mis_buffers,
                        predecessors=predecessors[role],
                    )
                    steps.append(step)
                    step_number += 1
                    observe(
                        step,
                        self.native_iteration_event,
                        {"outer_sweep": min(s.depth for s in states.values()), "role": role, "depth": state.depth},
                    )

        if session.pacing.pending:
            raise NativeIntegrationError("MASPO ended with unsettled native rollout reservations")
        stop_reason = StopReason.BUDGET if session.pacing.exhausted else StopReason.MAXIMUM_SEARCH_DEPTH
        incumbent = PromptBundle(roles=dict(best_prompt), demos=seed.demos, metadata=seed.metadata)
        final_state = self._state(role_order, states, best_prompt, mis_buffers, steps)
        final_state["stop_reason"] = stop_reason
        journal.observe(
            iteration=step_number,
            native_event="final_state",
            current=incumbent,
            incumbent=incumbent,
            native_score=None,
            accepted=False,
            reasons=("final_state", "early_stop"),
            state=final_state,
        )
        journal.finalize_curve()
        return journal.artifact(
            seed_bundle=seed,
            incumbent_bundle=incumbent,
            native_iterations=step_number,
            stop_reason=stop_reason,
            metadata={
                "mode": regime,
                "beam_width": settings.beam_width,
                "offspring": settings.offspring,
                "minibatch": settings.minibatch,
                "maximum_search_depth": settings.maximum_search_depth,
                "rounds_per_turn": settings.rounds_per_turn,
                "optimizer_rng_seed": cell.optimizer_seed,
                "reward_weights": {
                    "lookahead": list(search.LOOKAHEAD_WEIGHTS),
                    "fallback_local_global": list(search.FALLBACK_WEIGHTS),
                    "offset": search.SCORE_OFFSET,
                },
                "misalignment": {"buffer_capacity": search.MIS_CAP, "injection": search.MIS_INJECT},
                "task_type": task_type.value,
                "template_source": "upstream MASPO MATH templates" if task == "math" else "role-aware adaptation",
                "role_order": role_order,
                "terminal_role": terminal,
                "steps": steps,
                "topology": cell.topology,
                "successors": successor,
                "predecessors": predecessors,
                "proposal_calls": judge.n_proposal_calls,
                "judge_calls": judge.n_judge_calls,
                "proposal_extract_failures": judge.n_proposal_extract_failures,
                "reflection": dict(self.reflection.snapshot()),
            },
        )

    @staticmethod
    def _state(
        role_order: list[str],
        states: Mapping[str, Any],
        best_prompt: Mapping[str, str],
        mis_buffers: Mapping[str, list[Any]],
        steps: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "best_prompt": dict(best_prompt),
            "states": {
                role: {
                    "depth": int(states[role].depth),
                    "beam": [dict(node) for node in states[role].beam],
                    "best_overall": dict(states[role].best_overall),
                }
                for role in role_order
            },
            "misalignment_buffer_ids": {role: [native_id(row) for row in rows] for role, rows in mis_buffers.items()},
            "steps": list(steps),
        }


def native_id(row: Any) -> str:
    """Identifier of a training row as recorded in MASPO checkpoints."""
    if isinstance(row, Mapping):
        return str(row.get("id", row.get("example_id", id(row))))
    return str(getattr(row, "id", getattr(row, "example_id", id(row))))


__all__ = ["MASPOOptimizer", "native_id"]
