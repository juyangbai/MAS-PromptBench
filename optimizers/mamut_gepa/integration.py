"""MAMUT-GEPA on the shared run protocol: one joint GEPA search over all role prompts.

Adaptation of the MAMUT recipe ("Build, Judge, Optimize: A Blueprint for
Continuous Improvement of Multi-Agent Consumer Assistants", arXiv:2603.03565)
on the official ``gepa`` engine ("GEPA: Reflective Prompt Evolution Can
Outperform Reinforcement Learning", arXiv:2507.19457): a single
``gepa.optimize`` whose seed candidate carries every role prompt, round-robin
component selection, Pareto candidate selection, merge enabled and reflection
minibatches of 3. No public MAMUT reference implementation exists, so this is
an adaptation, not official MAMUT code.

Every GEPA metric call is one charged full-MAS rollout through the protocol
runner (``RunnerSession``). ``max_metric_calls`` equals the rollout budget and
the adapter rejects any evaluation batch that would cross it before dispatch,
so the ledger can never overshoot. Reflection prompts carry the source-tagged
trajectory of every role plus the global task score. The incumbent is GEPA's
best candidate when its validation aggregate is at least the seed's; the
protocol's final validation then makes the deployment decision.

Requires ``gepa==0.0.27`` (the version ``dspy==3.2`` pins): ``gepa.optimize``
must accept ``callbacks``, ``cache_evaluation``, ``module_selector`` and
``use_merge``, and ``InstructionProposalSignature.prompt_renderer`` must exist.
"""

from __future__ import annotations

import inspect
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

import gepa
from gepa.core.adapter import EvaluationBatch
from gepa.strategies.instruction_proposal import InstructionProposalSignature

from optimizers.protocol.cells import validate_cell
from optimizers.protocol.config import BUDGET
from optimizers.protocol.errors import MetricCallCapReached, NativeIntegrationError
from optimizers.protocol.journal import LifecycleJournal
from optimizers.protocol.reflection import REFLECTION_MAX_OUTPUT_TOKENS, ReflectionBackend, ReflectionClient
from optimizers.protocol.schema import CellSpec, OptimizerResult, PromptBundle, StopReason
from optimizers.protocol.seeding import reflection_seed
from optimizers.protocol.session import RunnerSession

GEPA_VERSION = "0.0.27"
REQUIRED_GEPA_KWARGS = (
    "callbacks",
    "cache_evaluation",
    "module_selector",
    "use_merge",
    "candidate_selection_strategy",
    "reflection_minibatch_size",
    "max_metric_calls",
    "seed",
)


@dataclass(frozen=True)
class MAMUTGEPASettings:
    """MAMUT-GEPA's knobs (no environment variables).

    The budget split: ``gepa.optimize`` gets ``max_metric_calls`` =
    min(``metric_call_limit``, B) and every metric call is one charged rollout.
    """

    metric_call_limit: int = BUDGET
    candidate_selection: str = "pareto"
    module_selector: str = "round_robin"
    use_merge: bool = True
    reflection_minibatch_size: int = 3
    reflection_temperature: float = 0.7
    reflection_top_p: float = 1.0


DEFAULTS = MAMUTGEPASettings()

_BEGIN, _END = "===BEGIN NEW INSTRUCTION===", "===END NEW INSTRUCTION==="
REFLECTION_TEMPLATE = f"""I gave an assistant (one role in a multi-agent system) the following instruction:
```
<curr_instructions>
```

Below are task inputs, the assistant's source-tagged outputs, and global task feedback:
```
<inputs_outputs_feedback>
```

Write an improved instruction for this role. Keep it self-contained and specific,
preserve its topology-visible responsibilities and exact output contract, and fold in
recurring generalizable strategies. Output ONLY the final instruction between:
{_BEGIN}
<the new instruction text>
{_END}"""


def extract_instruction(text: str, fallback: str) -> str:
    """Delimited block, else the last fenced block, else the text after the opening delimiter."""
    match = re.search(re.escape(_BEGIN) + r"(.*?)" + re.escape(_END), text, re.DOTALL)
    if match and match.group(1).strip():
        return match.group(1).strip()
    blocks = re.findall(r"```[^\n]*\n(.*?)```", text, re.DOTALL)
    if blocks and blocks[-1].strip():
        return blocks[-1].strip()
    if _BEGIN in text:
        tail = text.split(_BEGIN, 1)[1].replace(_END, "").strip()
        if tail:
            return tail
    return fallback


def require_gepa_api() -> str | None:
    """Fail before any rollout when the installed ``gepa`` lacks the retained API; return its version."""
    parameters = inspect.signature(gepa.optimize).parameters
    missing = [name for name in REQUIRED_GEPA_KWARGS if name not in parameters]
    if missing or not callable(getattr(InstructionProposalSignature, "prompt_renderer", None)):
        raise NativeIntegrationError(
            f"MAMUT-GEPA requires gepa=={GEPA_VERSION}; the installed gepa.optimize "
            f"lacks {missing or ['InstructionProposalSignature.prompt_renderer']}"
        )
    try:
        return metadata.version("gepa")
    except metadata.PackageNotFoundError:
        return getattr(gepa, "__version__", None)


def _question(item: Any) -> str:
    if isinstance(item, Mapping):
        for key in ("question", "problem", "prompt", "input"):
            if item.get(key):
                return str(item[key])
    return str(item)


class CommonMAMUTGEPAAdapter:
    """GEPA adapter with a joint candidate; scores are the runner's canonical scores."""

    def __init__(
        self,
        session: RunnerSession,
        reflection: ReflectionBackend,
        cell: CellSpec,
        metric_call_limit: int,
        settings: MAMUTGEPASettings = DEFAULTS,
    ) -> None:
        self.session, self.reflection, self.cell = session, reflection, cell
        self.settings = settings
        self.base_prompts = dict(session.seed_bundle.roles)
        self.native_iteration = 0
        self.proposal_calls = 0
        self.metric_call_limit = int(metric_call_limit)
        self.start_charged = int(session.budget.snapshot()["charged"])

    def evaluate(self, batch: list[Any], candidate: dict[str, str], capture_traces: bool = False) -> EvaluationBatch:
        """GEPA adapter hook: charged full-MAS rollouts of one candidate on a batch."""
        full = {**self.base_prompts, **candidate}
        if tuple(full) != tuple(self.base_prompts):
            raise NativeIntegrationError("MAMUT candidate changed the ordered role interface")
        charged = int(self.session.budget.snapshot()["charged"])
        if charged + len(batch) > self.start_charged + self.metric_call_limit:
            raise MetricCallCapReached("GEPA atomic evaluation would exceed the exact metric-call cap")
        if not self.session.pacing.try_charge(len(batch)):
            raise MetricCallCapReached("GEPA atomic evaluation cannot fit the remaining rollout budget")
        for role, prompt in full.items():
            self.session.set_prompt(role, prompt)
        self.session.set_context(iteration=self.native_iteration, event="joint_gepa_evaluate")
        trajectories, scores, outputs = [], [], []
        records = self.session.run_records(batch)
        for item, record in zip(batch, records, strict=True):
            score = float(record.score or 0.0)
            calls = []
            for message in record.messages:
                source = str(message.get("source") or message.get("role") or message.get("name") or "unknown")
                shared = len(full) == 1 and source not in {"user", "human", "system"}
                calls.append(
                    {
                        "component": next(iter(full)) if shared else source,
                        "native_source": source,
                        "input": "",
                        "output": str(message.get("content") or ""),
                    }
                )
            overall = f"Global CommonRunner score={score:.6f}; status={record.status}"  # retained feedback text
            trajectories.append(
                {
                    "id": str(record.example_id),
                    "question": _question(item),
                    "component_calls": calls,
                    "final_answer": record.final_output,
                    "score": score,
                    "common_status": record.status,
                    "_rubric": {
                        "checks": dict(record.metadata.get("scorer_metadata") or {}),
                        "score": score,
                        "node_score": score,
                        "feedback": {role: overall for role in full},
                        "overall": overall,
                    },
                }
            )
            scores.append(score)
            outputs.append({"final_answer": record.final_output, "score": score})
        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories if capture_traces else None)

    def make_reflective_dataset(
        self, candidate: dict[str, str], eval_batch: EvaluationBatch, components_to_update: list[str]
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        """GEPA adapter hook: per-role reflection records of an evaluated batch."""
        del candidate
        result: dict[str, list[dict[str, Any]]] = {}
        for component in components_to_update:
            rows = []
            for trajectory in eval_batch.trajectories or []:
                all_calls = list(trajectory.get("component_calls") or [])
                target = [call for call in all_calls if call.get("component") == component]
                if not target:
                    continue
                rubric = dict(trajectory.get("_rubric") or {})
                feedback = rubric.get("feedback", {}).get(component, rubric.get("overall", ""))
                rows.append(
                    {
                        "Inputs": json.dumps(
                            {
                                "task": trajectory.get("question", ""),
                                "target_component": component,
                                "source_tagged_trajectory": all_calls,
                            },
                            ensure_ascii=False,
                        ),
                        "Generated Outputs": json.dumps(
                            {"component": component, "invocations": target}, ensure_ascii=False
                        ),
                        "Feedback": feedback,
                    }
                )
            if rows:
                result[component] = rows
        return result

    def propose_new_texts(
        self,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        """GEPA adapter hook: MAMUT's joint reflection proposing new role prompts."""
        self.native_iteration += 1
        proposed: dict[str, str] = {}
        for component in components_to_update:
            dataset = reflective_dataset.get(component, ())
            if not dataset:
                proposed[component] = candidate[component]
                continue
            prompt = InstructionProposalSignature.prompt_renderer(
                {
                    "current_instruction_doc": candidate[component],
                    "dataset_with_feedback": dataset,
                    "prompt_template": REFLECTION_TEMPLATE,
                }
            )
            raw = self.reflection.complete(
                prompt,
                request_seed=reflection_seed(
                    self.cell,
                    phase="mamut_gepa_reflection",
                    iteration=self.native_iteration,
                    role=component,
                    prompt=prompt,
                ),
                phase="joint_gepa_reflection",
                role=component,
                temperature=self.settings.reflection_temperature,
                top_p=self.settings.reflection_top_p,
                max_output_tokens=REFLECTION_MAX_OUTPUT_TOKENS,
                thinking=True,
                system=None,
            )
            self.proposal_calls += 1
            proposed[component] = extract_instruction(raw, candidate[component])
        if components_to_update and all(
            proposed[component].strip() == candidate[component].strip() for component in components_to_update
        ):
            # GEPA skips the iteration (its proposer catches proposal errors).
            raise NativeIntegrationError("MAMUT-GEPA produced a no-op joint proposal")
        return proposed


class _JournalCallback:
    """GEPA ``on_iteration_end`` hook: one journal event per completed GEPA iteration."""

    def __init__(self, journal: LifecycleJournal, adapter: CommonMAMUTGEPAAdapter, seed: PromptBundle) -> None:
        self.journal, self.adapter, self.seed = journal, adapter, seed
        self.iterations = 0
        self.last_state: Any | None = None

    @staticmethod
    def _best(state: Any) -> tuple[dict[str, str], float | None]:
        scores = list(getattr(state, "program_full_scores_val_set", ()) or ())
        candidates = list(getattr(state, "program_candidates", ()) or ())
        if not candidates:
            return {}, None
        index = max(range(len(scores)), key=scores.__getitem__) if scores else len(candidates) - 1
        return dict(candidates[index]), float(scores[index]) if scores else None

    def on_iteration_end(self, event: Mapping[str, Any]) -> None:
        self.iterations = int(event["iteration"])
        self.last_state = event["state"]
        candidate, score = self._best(event["state"])
        bundle = PromptBundle(
            roles=candidate or dict(self.seed.roles), demos=self.seed.demos, metadata=self.seed.metadata
        )
        state = event["state"]
        accepted = bool(event.get("proposal_accepted"))
        self.journal.observe(
            iteration=self.iterations,
            native_event="completed joint GEPA proposal/update step",
            current=bundle,
            incumbent=bundle,
            native_score=score,
            accepted=accepted,
            reasons=("native_iteration_end", "incumbent_change") if accepted else ("native_iteration_end",),
            state={
                "iteration": self.iterations,
                "total_metric_calls": int(getattr(state, "total_num_evals", 0)),
                "program_candidates": list(getattr(state, "program_candidates", ()) or ()),
                "program_full_scores_val_set": list(getattr(state, "program_full_scores_val_set", ()) or ()),
                "full_program_trace": list(getattr(state, "full_program_trace", ()) or ()),
            },
            coordinates={"generation": self.iterations, "accepted": accepted},
        )


def _relative_state_dir(journal: LifecycleJournal, run_dir: Path | None) -> str | None:
    """GEPA state directory relative to the job (artifacts carry no absolute paths)."""
    if run_dir is None:
        return None
    if journal.store is not None:
        try:
            return journal.store.relative(run_dir)
        except Exception:
            pass
    return Path(run_dir).name


class MAMUTGEPAOptimizer:
    """One joint ``gepa.optimize`` over all role prompts; every metric call is a charged rollout.

    ``reflection`` defaults to the protocol :class:`ReflectionClient`;
    ``metric_call_limit`` is :attr:`MAMUTGEPASettings.metric_call_limit`;
    ``run_dir`` holds GEPA's own state (default: inside the job's optimization directory).
    """

    method = "mamut_gepa"
    native_iteration_event = "completed joint GEPA proposal/update step"

    def __init__(
        self,
        *,
        seed_bundle: PromptBundle | None = None,
        reflection: ReflectionBackend | None = None,
        metric_call_limit: int = DEFAULTS.metric_call_limit,
        run_dir: str | Path | None = None,
    ) -> None:
        self.seed_bundle = seed_bundle
        self.reflection = reflection or ReflectionClient()
        if not 1 <= int(metric_call_limit) <= BUDGET:
            raise ValueError("MAMUT-GEPA metric_call_limit must be in [1, 600]")
        self.settings = MAMUTGEPASettings(metric_call_limit=int(metric_call_limit))
        self.run_dir = Path(run_dir) if run_dir is not None else None

    def optimize(
        self, cell: CellSpec, runner: Any, budget: Any, training: list[Any], validation: list[Any]
    ) -> OptimizerResult:
        """Run the joint GEPA search; the incumbent is GEPA's best candidate unless it trails the seed."""
        settings = self.settings
        validate_cell(self.method, cell)
        if self.seed_bundle is None:
            raise NativeIntegrationError("MAMUT-GEPA requires the canonical seed PromptBundle")
        if not training or not validation:
            raise NativeIntegrationError("MAMUT-GEPA requires non-empty training and validation splits")
        gepa_version = require_gepa_api()
        # The metric-call cap is the rollout budget (600 under the protocol; smaller for smoke ledgers).
        metric_call_limit = min(settings.metric_call_limit, int(budget.maximum))
        seed = self.seed_bundle
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
        native_adapter = CommonMAMUTGEPAAdapter(session, self.reflection, cell, metric_call_limit, settings)
        callback = _JournalCallback(journal, native_adapter, seed)
        run_dir = self.run_dir
        if run_dir is None and journal.directory is not None:
            run_dir = Path(journal.directory) / "mamut-gepa-native-state"
        stop_reason = StopReason.NATIVE_GEPA_STOP
        try:
            result = gepa.optimize(
                seed_candidate=dict(seed.roles),
                trainset=training,
                valset=validation,
                adapter=native_adapter,
                max_metric_calls=metric_call_limit,
                candidate_selection_strategy=settings.candidate_selection,
                module_selector=settings.module_selector,
                reflection_minibatch_size=settings.reflection_minibatch_size,
                use_merge=settings.use_merge,
                display_progress_bar=False,
                seed=cell.optimizer_seed,
                raise_on_exception=True,
                run_dir=str(run_dir) if run_dir is not None else None,
                callbacks=[callback],
                cache_evaluation=False,
            )
        except MetricCallCapReached:
            stop_reason = StopReason.METRIC_CALL_CAP
            state = callback.last_state
            if state is None:
                best = dict(seed.roles)
                scores: list[float] = []
                candidates = [dict(seed.roles)]
                total_metric_calls = int(session.budget.snapshot()["charged"])
            else:
                best, _ = callback._best(state)
                best = best or dict(seed.roles)
                scores = [float(value) for value in (getattr(state, "program_full_scores_val_set", ()) or ())]
                candidates = list(getattr(state, "program_candidates", ()) or ())
                total_metric_calls = int(getattr(state, "total_num_evals", 0))
        else:
            best = dict(result.best_candidate)
            scores = [float(value) for value in (result.val_aggregate_scores or ())]
            candidates = list(result.candidates or ())
            total_metric_calls = int(result.total_metric_calls or 0)
        baseline_score = float(scores[0]) if scores else None
        best_score = max(float(value) for value in scores) if scores else None
        identical = all(best.get(role, "").strip() == prompt.strip() for role, prompt in seed.roles.items())
        accepted = (
            not identical
            and best_score is not None
            and baseline_score is not None
            and best_score >= baseline_score - 1e-9
        )
        incumbent = PromptBundle(roles=best if accepted else dict(seed.roles), demos=seed.demos, metadata=seed.metadata)
        journal.observe(
            iteration=callback.iterations,
            native_event="final_state",
            current=PromptBundle(roles=best, demos=seed.demos, metadata=seed.metadata),
            incumbent=incumbent,
            native_score=best_score,
            accepted=accepted,
            reasons=("final_state", "incumbent_change") if accepted else ("final_state",),
            state={
                "iteration": callback.iterations,
                "candidates": candidates,
                "val_aggregate_scores": scores,
                "best_candidate": best,
                "total_metric_calls": total_metric_calls,
            },
        )
        journal.finalize_curve()
        return journal.artifact(
            seed_bundle=seed,
            incumbent_bundle=incumbent,
            native_iterations=callback.iterations,
            stop_reason=stop_reason,
            metadata={
                "adaptation_id": "mamut-gepa-joint-adaptation",
                "optimizer_engine": "gepa",
                "gepa_version": gepa_version,
                "candidate_selection": settings.candidate_selection,
                "component_selection": settings.module_selector,
                "use_merge": settings.use_merge,
                "reflection_minibatch_size": settings.reflection_minibatch_size,
                "metric_call_limit": metric_call_limit,
                "gepa_seed": cell.optimizer_seed,
                "native_total_metric_calls": total_metric_calls,
                "baseline_native_validation": baseline_score,
                "best_native_validation": best_score,
                "accepted": accepted,
                "proposal_calls": native_adapter.proposal_calls,
                "reflection": dict(self.reflection.snapshot()),
                "native_state_dir": _relative_state_dir(journal, run_dir),
            },
        )


__all__ = [
    "CommonMAMUTGEPAAdapter",
    "MAMUTGEPAOptimizer",
    "MAMUTGEPASettings",
    "REFLECTION_TEMPLATE",
    "extract_instruction",
    "require_gepa_api",
]
