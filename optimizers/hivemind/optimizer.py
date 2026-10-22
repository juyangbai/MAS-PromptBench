"""Behavior-preserving protocol-runner integration of HiveMind (CG-OPO).

Retains the audited CG-OPO cycle: deterministic coalition planning, full-MAS
coalition values, Shapley credit, bottleneck/manager rotation, native lessons
metamorphosis, and strict validation-minibatch acceptance.  Coalition masking
is an explicit runtime metadata contract (``optimizer_control``) executed by
the hook in :mod:`.runtime` and must be acknowledged fail-closed.
"""

from __future__ import annotations

import math
import random
import re
from collections.abc import Mapping, Sequence
from itertools import combinations
from pathlib import Path
from statistics import fmean
from typing import Any

from optimizers.protocol.cells import validate_cell
from optimizers.protocol.errors import NativeIntegrationError
from optimizers.protocol.journal import LifecycleJournal, bundle_from_checkpoint, require_resume_checkpoint
from optimizers.protocol.reflection import REFLECTION_MAX_OUTPUT_TOKENS, ReflectionBackend, ReflectionClient
from optimizers.protocol.schema import CellSpec, OptimizerResult, PromptBundle, StopReason, content_hash, example_id
from optimizers.protocol.seeding import reflection_seed
from optimizers.protocol.session import RunnerSession

from .regime import (
    ADAPTATION_ID,
    DEFAULTS,
    METHOD_SETTINGS,
    HiveMindSettings,
    build_coalition_plan,
    coalition_key,
    plan_provenance,
    plan_seed,
    row_order_seed,
)
from .runtime import coalition_execution
from .topology import coalition_game, expected_control_evidence

_BEGIN, _END = "===BEGIN LESSONS===", "===END LESSONS==="
_LESSONS_HEADER = "\n\n=== Lessons learned (experience-based guidance) ==="
_REFLECTION_FILE = Path(__file__).resolve().parent / "prompts" / "reflection.txt"
_DOMAIN_SENTENCE = (
    "that solves math word problems. The system runs three style-diverse reasoners, "
    "then three verifiers, then a finalizer."
)
_REFLECT_SYSTEM = (
    "You improve one agent's role prompt inside a multi-agent reasoning system. "
    "You output only the requested LESSONS block, nothing else."
)
# Task names as they appear in the reflection prompt.
_TASK_DISPLAY_NAMES = {"lcb": "livecodebench"}


def exact_shapley_values(agents: Sequence[str], values: Mapping[frozenset[str], float]) -> dict[str, float]:
    """Exact Shapley value (HiveMind Eq. 2) over the full coalition table."""
    count, result = len(agents), {}
    for agent in agents:
        others, total = [value for value in agents if value != agent], 0.0
        for size in range(len(others) + 1):
            weight = math.factorial(size) * math.factorial(count - size - 1) / math.factorial(count)
            for combo in combinations(others, size):
                coalition = frozenset(combo)
                total += weight * (values[coalition | {agent}] - values[coalition])
        result[agent] = total
    return result


def permutation_shapley_values(
    agents: Sequence[str], permutations: Sequence[Sequence[str]], values: Mapping[frozenset[str], float]
) -> dict[str, float]:
    """Monte-Carlo Shapley estimate: mean marginal contribution along sampled permutations."""
    result = {agent: 0.0 for agent in agents}
    for permutation in permutations:
        for index, agent in enumerate(permutation):
            result[agent] += values[frozenset(permutation[: index + 1])] - values[frozenset(permutation[:index])]
    return {agent: value / max(1, len(permutations)) for agent, value in result.items()}


def extract_lessons(text: str) -> str:
    """HiveMind lesson parsing: the last non-empty LESSONS block, one bullet per line."""
    pattern = re.compile(re.escape(_BEGIN) + r"(.*?)" + re.escape(_END), re.DOTALL)
    blocks = [block for block in pattern.findall(text) if block.strip()]
    body = blocks[-1] if blocks else text
    raw = [line.strip() for line in body.splitlines() if line.strip().lstrip("-* ").strip()]
    raw = [
        re.sub(r"^([-*])\s+", r"\1 ", cleaned)
        for cleaned in (re.sub(r"<lesson\s*\d*>", "", line, flags=re.I).strip() for line in raw)
        if cleaned.lstrip("-* ").strip()
    ]
    if not blocks:
        raw = [line for line in raw if re.match(r"^(\-|\*|\d+[.)])\s+", line)]
    lines = [re.sub(r"^\d+[.)]\s+", "", line) for line in raw]
    return "\n".join(line if line.startswith(("-", "*")) else f"- {line}" for line in lines).strip()


def metamorphose(base_prompt: str, lessons_log: Sequence[str], max_lessons: int | None = 6) -> str:
    """Append-only prompt update (HiveMind Eq. 5): seed prompt + the latest lessons."""
    unbounded = max_lessons is None or not max_lessons or max_lessons == float("inf")
    nonempty = [lesson for lesson in lessons_log if lesson.strip()]
    kept = nonempty if unbounded else nonempty[-int(max_lessons) :]
    return base_prompt if not kept else base_prompt.rstrip() + _LESSONS_HEADER + "\n" + "\n".join(kept)


def _question(example: Any) -> str:
    if isinstance(example, Mapping):
        for key in ("problem", "question", "prompt", "input"):
            if example.get(key):
                return str(example[key])
    for key in ("problem", "question", "prompt", "input"):
        value = getattr(example, key, None)
        if value:
            return str(value)
    return str(example)


def _role_output(messages: Sequence[Mapping[str, Any]], role: str) -> str:
    values = [
        str(message.get("content") or "")
        for message in messages
        if (message.get("source") or message.get("role") or message.get("name")) == role
    ]
    return "\n---\n".join(value for value in values[-3:] if value) or "(this agent was never activated in this rollout)"


def _parameter_output(record: Any, role: str, game: Any) -> str:
    if game.topology not in {"independent", "decentralized"}:
        return _role_output(record.messages, role)
    # Shared prompt reflection sees every replica trajectory, retaining each
    # player's identity instead of pretending the parameter is a deployed agent.
    value = getattr(record, "final_output", {}) or {}
    nested = value.get("runner_output", value)
    players = nested.get("per_agent") or nested.get("per_peer") or []
    outputs = []
    for player in players:
        text = player.get("raw") or "\n".join(
            str(m.get("content", "") if isinstance(m, Mapping) else getattr(m, "content", ""))
            for m in player.get("messages", [])
        )
        if not text:
            text = str(player.get("model_output") or player.get("call") or "")
        outputs.append(f"{role} replica {player.get('agent_id', player.get('peer'))}: {text}")
    return "\n---\n".join(outputs) or _role_output(record.messages, role)


def _format_cases(cases: Sequence[tuple[str, str, float]], limit: int = 6) -> str:
    if not cases:
        return "(none)"
    lines = []
    for index, (question, output, _reward) in enumerate(cases[:limit], 1):
        lines.append(
            f"[case {index}] problem: {' '.join(question.strip().split())[:300]}\n"
            f"         agent output: {' '.join(output.strip().split())[:400]}"
        )
    return "\n".join(lines)


def _reflection_prompt(
    *,
    task: str,
    workers: Sequence[str],
    role: str,
    phi: float,
    failures: Sequence[tuple[str, str, float]],
    successes: Sequence[tuple[str, str, float]],
    nonce: str,
) -> str:
    template = _REFLECTION_FILE.read_text(encoding="utf-8")
    domain = (
        f"that solves {_TASK_DISPLAY_NAMES.get(task, task)} tasks. The system preserves the frozen benchmark "
        f"topology; the coalition-capable runtime exposes workers ({', '.join(workers)}) and returns "
        "the topology's canonical final answer."
    )
    if _DOMAIN_SENTENCE not in template:
        raise NativeIntegrationError("retained HiveMind reflection template drifted")
    return (
        _REFLECT_SYSTEM
        + "\n\n"
        + template.replace(_DOMAIN_SENTENCE, domain).format(
            role=role,
            aid=role,
            phi=phi,
            failures=_format_cases(failures),
            successes=_format_cases(successes),
            nonce=nonce,
        )
    )


def _bundle(seed: PromptBundle, roles: Mapping[str, str], **metadata: Any) -> PromptBundle:
    return PromptBundle(roles=dict(roles), demos=tuple(seed.demos), metadata={**dict(seed.metadata), **metadata})


def _require_control_ack(record: Any, expected: Mapping[str, Any]) -> None:
    metadata = dict(getattr(record, "metadata", {}) or {})
    nested = metadata.get("runtime_metadata")
    raw = None
    if isinstance(nested, Mapping):
        raw = nested.get("execution_control", nested.get("optimizer_control"))
    if raw is None:
        raw = metadata.get("execution_control", metadata.get("optimizer_control"))
    ack = dict(raw or {})
    canonical = {
        "schema": expected["schema"],
        "request_sha256": content_hash(dict(expected)),
        **expected_control_evidence(expected),
    }
    observed = {key: ack.get(key) for key in canonical}
    for key in ("active_workers", "masked_workers", "routable_workers", "bound_delegation_tools"):
        observed[key] = sorted(observed[key] or [])
    if observed != canonical or not ack.get("runtime_implementation_id"):
        raise NativeIntegrationError("runtime did not acknowledge exact HiveMind coalition masking")


class HiveMindOptimizer:
    """CG-OPO with coalition Shapley credit on the shared protocol runner.

    Constructed with the frozen ``seed_bundle``; the keyword knobs are
    :class:`~.regime.HiveMindSettings`. ``reflection`` defaults to the protocol
    :class:`ReflectionClient` (native call-site sampling: temperature 0.7,
    top-p 1.0, thinking, 48,000 tokens). ``execution_hook`` replaces the
    coalition hook registered while ``optimize`` runs.
    """

    method = "hivemind"
    native_iteration_event = "completed CG-OPO/Shapley cycle"

    def __init__(
        self,
        *,
        seed_bundle: PromptBundle | None = None,
        reflection: ReflectionBackend | None = None,
        max_cycles: int = DEFAULTS.max_cycles,
        coalition_batch: int = DEFAULTS.coalition_batch,
        acceptance_batch: int = DEFAULTS.acceptance_batch,
        max_coalitions: int = DEFAULTS.max_coalitions,
        manager_every_k: int = DEFAULTS.manager_every_k,
        max_lessons: int = DEFAULTS.max_lessons,
        fail_below: float = DEFAULTS.fail_below,
        hm_seed: int = DEFAULTS.hm_seed,
        resume_checkpoint: Mapping[str, Any] | None = None,
        execution_hook: Any = None,
    ) -> None:
        self.seed_bundle = seed_bundle
        self.reflection = reflection or ReflectionClient()
        self.settings = HiveMindSettings(
            max_cycles=int(max_cycles),
            coalition_batch=int(coalition_batch),
            acceptance_batch=int(acceptance_batch),
            max_coalitions=int(max_coalitions),
            manager_every_k=int(manager_every_k),
            max_lessons=int(max_lessons),
            fail_below=float(fail_below),
            hm_seed=int(hm_seed),
        )
        self.resume_checkpoint = resume_checkpoint
        self.execution_hook = execution_hook
        if self.settings.max_cycles < 0 or min(self.settings.coalition_batch, self.settings.acceptance_batch) <= 0:
            raise ValueError("invalid HiveMind native cycle configuration")

    def optimize(
        self, cell: CellSpec, runner: Any, budget: Any, training: list[Any], validation: list[Any]
    ) -> OptimizerResult:
        """Run CG-OPO cycles until the budget (or ``max_cycles``) is spent."""
        validate_cell(self.method, cell)
        if self.seed_bundle is None:
            raise NativeIntegrationError("HiveMind requires the canonical seed PromptBundle")
        if not training or not validation:
            raise NativeIntegrationError("HiveMind requires non-empty training and validation splits")
        with coalition_execution(self.execution_hook):
            return self._optimize(cell, runner, budget, training, validation)

    def _optimize(
        self, cell: CellSpec, runner: Any, budget: Any, training: list[Any], validation: list[Any]
    ) -> OptimizerResult:
        settings = self.settings
        checkpoint = require_resume_checkpoint(self.resume_checkpoint, method=self.method, cell=cell, budget=budget)
        seed = self.seed_bundle
        session = RunnerSession(cell=cell, runner=runner, budget=budget, seed_bundle=seed)
        journal = LifecycleJournal(method=self.method, cell=cell, session=session)
        roles = list(seed.roles)
        try:
            game = coalition_game(cell.topology, roles, cell.team_size)
        except ValueError as exc:
            raise NativeIntegrationError(str(exc)) from exc
        manager, workers = game.manager, list(game.players)

        def key_for(coalition):
            return coalition_key(coalition) if coalition or manager is not None else "empty"

        game_state = {
            **game.state(),
            "continuation_contract": {
                "seed_bundle_sha256": seed.digest,
                "coalition_batch": settings.coalition_batch,
                "acceptance_batch": settings.acceptance_batch,
                "max_coalitions": settings.max_coalitions,
                "manager_every_k": settings.manager_every_k,
                "max_lessons": settings.max_lessons,
                "fail_below": settings.fail_below,
                "hm_seed": settings.hm_seed,
                "training_order_identity": [[example_id(row), _question(row)] for row in training],
                "validation_order_identity": [[example_id(row), _question(row)] for row in validation],
            },
        }
        coalitions, permutations = build_coalition_plan(
            workers, random.Random(plan_seed(cell.optimizer_seed, settings.hm_seed)), settings.max_coalitions
        )
        full = frozenset(workers)
        if full not in coalitions or frozenset() not in coalitions:
            raise NativeIntegrationError("HiveMind coalition plan lacks empty/full anchors")
        train_order, val_order = list(range(len(training))), list(range(len(validation)))
        order_rng = random.Random(row_order_seed(cell.optimizer_seed, settings.hm_seed))
        order_rng.shuffle(train_order)
        order_rng.shuffle(val_order)
        train_cursor = val_cursor = cycle = 0
        current, lessons, history = dict(seed.roles), {role: [] for role in roles}, []
        if checkpoint is not None:
            raw_state = checkpoint.get("state")
            if not isinstance(raw_state, Mapping):
                raise NativeIntegrationError("HiveMind checkpoint state is malformed")
            if raw_state.get("topology_game") != game_state:
                raise NativeIntegrationError("HiveMind checkpoint topology game differs from the resumed cell")
            current = dict(bundle_from_checkpoint(checkpoint).roles)
            lessons = {role: list(values) for role, values in dict(raw_state.get("lessons") or {}).items()}
            cycle, train_cursor, val_cursor = (
                int(raw_state.get("cycle", 0)),
                int(raw_state.get("train_cursor", 0)),
                int(raw_state.get("val_cursor", 0)),
            )
            history = list(raw_state.get("history") or [])
            journal.restore_curve(checkpoint)
        initial = _bundle(seed, current)
        journal.observe(
            iteration=cycle,
            native_event="initial_state" if checkpoint is None else "checkpoint_resumed",
            current=initial,
            incumbent=initial,
            native_score=None,
            accepted=False,
            reasons=("initial_state",) if checkpoint is None else ("initial_state", "checkpoint_resumed"),
            state={
                "cycle": cycle,
                "lessons": lessons,
                "train_cursor": train_cursor,
                "val_cursor": val_cursor,
                "history": history,
                "topology_game": game_state,
            },
        )
        batch, accept_n = settings.cycle_batches(
            len(coalitions), len(training), len(validation), session.pacing.remaining()
        )
        planned, stop_reason = len(coalitions) * batch + 2 * accept_n, StopReason.BUDGET
        while planned <= session.pacing.remaining():
            if settings.max_cycles and cycle >= settings.max_cycles:
                stop_reason = StopReason.MAX_CYCLES
                break
            cycle += 1
            picked = [train_order[(train_cursor + index) % len(train_order)] for index in range(batch)]
            train_cursor = (train_cursor + batch) % len(train_order)
            train_rows = [training[index] for index in picked]
            coalition_records, values = {}, {}
            for coalition in coalitions:
                control = game.control(coalition)
                candidate_bundle = _bundle(seed, current, optimizer_control=control)
                for role, prompt in candidate_bundle.roles.items():
                    session.set_prompt(role, prompt)
                session.set_bundle_metadata(**candidate_bundle.metadata)
                session.set_context(iteration=cycle, event=f"coalition:{key_for(coalition)}")
                if not session.pacing.try_charge(len(train_rows)):
                    raise NativeIntegrationError("atomic HiveMind coalition batch was truncated")
                records = session.run_records(train_rows)
                for record in records:
                    _require_control_ack(record, control)
                coalition_records[coalition] = records
                values[coalition] = fmean(float(record.score or 0.0) for record in records)
            phi = (
                exact_shapley_values(workers, values)
                if permutations is None
                else permutation_shapley_values(workers, permutations, values)
            )
            manager_cycle = (
                manager is not None and settings.manager_every_k > 0 and cycle % settings.manager_every_k == 0
            )
            # Tied replicas expose one trainable parameter: total contribution of
            # that parameter is the sum of its players' exact Shapley credits.
            parameter_phi = {
                role: sum(phi[player] for player in workers if game.player_roles[player] == role)
                for role in sorted(set(game.player_roles.values()))
            }
            target = manager if manager_cycle else min(parameter_phi, key=parameter_phi.get)
            cases = [
                (_question(row), _parameter_output(record, target, game), float(record.score or 0.0))
                for row, record in zip(train_rows, coalition_records[full])
            ]
            failures = [c for c in cases if c[2] < settings.fail_below]
            successes = [c for c in cases if c[2] >= settings.fail_below]
            prompt = _reflection_prompt(
                task=cell.task,
                workers=workers,
                role=target,
                phi=0.0 if manager_cycle else parameter_phi[target],
                failures=failures,
                successes=successes,
                nonce=str(cycle),
            )
            raw = self.reflection.complete(
                prompt,
                request_seed=reflection_seed(
                    cell, phase="hivemind_reflection", iteration=cycle, role=target, prompt=prompt
                ),
                phase="coalition_credit_and_reflection",
                role=target,
                temperature=settings.reflection_temperature,
                top_p=settings.reflection_top_p,
                max_output_tokens=REFLECTION_MAX_OUTPUT_TOKENS,
                thinking=True,
                system=None,
            )
            lesson, accepted, current_score, candidate_score = extract_lessons(raw), False, None, None
            candidate = dict(current)
            if lesson:
                proposed = metamorphose(seed.roles[target], [*lessons[target], lesson], settings.max_lessons)
                if proposed.strip() != current[target].strip():
                    candidate[target] = proposed
                    picked_val = [val_order[(val_cursor + index) % len(val_order)] for index in range(accept_n)]
                    val_cursor = (val_cursor + accept_n) % len(val_order)
                    val_rows, scores = [validation[index] for index in picked_val], []
                    for label, prompts in (("current", current), ("candidate", candidate)):
                        control = game.control(workers)
                        bnd = _bundle(seed, prompts, optimizer_control=control)
                        for role, text in bnd.roles.items():
                            session.set_prompt(role, text)
                        session.set_bundle_metadata(**bnd.metadata)
                        session.set_context(iteration=cycle, event=f"accept_{label}")
                        if not session.pacing.try_charge(len(val_rows)):
                            raise NativeIntegrationError("atomic HiveMind acceptance batch was truncated")
                        records = session.run_records(val_rows)
                        for record in records:
                            _require_control_ack(record, control)
                        scores.append(fmean(float(record.score or 0.0) for record in records))
                    current_score, candidate_score = scores
                    if candidate_score > current_score + 1e-9:
                        current, accepted = candidate, True
                        lessons[target].append(lesson)
            incumbent = _bundle(seed, current)
            history = [
                *history,
                {
                    "cycle": cycle,
                    "coalition_values": {
                        key_for(k): v
                        for k, v in sorted(values.items(), key=lambda item: (len(item[0]), key_for(item[0])))
                    },
                    "phi": phi,
                    "parameter_phi": parameter_phi,
                    "target": target,
                    "manager_cycle": manager_cycle,
                    "accepted": accepted,
                    "current_score": current_score,
                    "candidate_score": candidate_score,
                },
            ]
            state = {
                "cycle": cycle,
                "lessons": lessons,
                "train_cursor": train_cursor,
                "val_cursor": val_cursor,
                "history": history,
                "topology_game": game_state,
            }
            journal.observe(
                iteration=cycle,
                native_event=self.native_iteration_event,
                current=_bundle(seed, candidate),
                incumbent=incumbent,
                native_score=candidate_score if candidate_score is not None else values[full],
                accepted=accepted,
                reasons=("native_iteration_end", "incumbent_change") if accepted else ("native_iteration_end",),
                state=state,
                coordinates={"cycle": cycle, "selected_agent": target},
            )
        if settings.max_cycles and cycle >= settings.max_cycles:
            stop_reason = StopReason.MAX_CYCLES
        final_state = {
            "cycle": cycle,
            "lessons": lessons,
            "train_cursor": train_cursor,
            "val_cursor": val_cursor,
            "history": history,
            "stop_reason": stop_reason,
            "topology_game": game_state,
        }
        journal.observe(
            iteration=cycle,
            native_event="final_state",
            current=_bundle(seed, current),
            incumbent=_bundle(seed, current),
            native_score=None,
            accepted=False,
            reasons=("final_state", "early_stop"),
            state=final_state,
        )
        journal.finalize_curve()
        return journal.artifact(
            seed_bundle=seed,
            incumbent_bundle=_bundle(seed, current),
            native_iterations=cycle,
            stop_reason=stop_reason,
            metadata={
                "adaptation_id": ADAPTATION_ID,
                "settings": dict(METHOD_SETTINGS),
                "topology_game": game_state,
                "coalition_plan": plan_provenance(
                    workers,
                    coalitions,
                    permutations,
                    max_coalitions=settings.max_coalitions,
                    seed_offset=cell.optimizer_seed,
                    hm_seed=settings.hm_seed,
                    empty_key="manager_only" if manager is not None else "empty",
                ),
                "reflection": dict(self.reflection.snapshot()),
                "history": history,
            },
        )


__all__ = ["HiveMindOptimizer", "exact_shapley_values", "extract_lessons", "metamorphose", "permutation_shapley_values"]
