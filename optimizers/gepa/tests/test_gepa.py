"""GEPA port: native DSPy GEPA through the protocol runner with a fake MAS and a fake reflection LM."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from optimizers.protocol import REPO_ROOT, methods, run
from optimizers.protocol.artifacts import ArtifactStore, read_json
from optimizers.protocol.errors import OptimizerInfrastructureFailure, UnsupportedGEPACell
from optimizers.protocol.methods.context_fit import ContextFitLM
from optimizers.protocol.methods.dspy_bridge import LogicalSeedLM
from optimizers.protocol.methods.tests.fakes import GOOD, FakeLM, method_cell
from optimizers.protocol.schema import incumbent_of, verify_sealed
from optimizers.protocol.seeding import logical_request_seed
from optimizers.protocol.tests.fakes import FakeAdapter, fake_hooks, fake_runner

from .. import integration

# GEPA settings of the reference method configuration.
GEPA_SETTINGS = {
    "auto": None,
    "max_full_evals": 5,
    "reflection_minibatch_size": 3,
    "candidate_selection_strategy": "pareto",
    "component_selector": "round_robin",
    "use_merge": True,
    "max_merge_invocations": 5,
    "early_stop_patience": 3,
    "skip_perfect_score": True,
    "failure_score": 0.0,
    "perfect_score": 1.0,
    "seed": 0,
    "track_stats": True,
    "dspy_version": "3.2.0",
    "gepa_version": "0.0.27",
}


def _spy(runner):
    """Record (example IDs, bundle digest, request seeds) of every dispatched batch."""
    calls = []
    native = runner.run_batch

    def run_batch(examples, bundle, seeds):
        calls.extend((str(example["id"]), bundle.digest, int(seed)) for example, seed in zip(examples, seeds))
        return native(examples, bundle, seeds)

    runner.run_batch = run_batch
    return calls


def _recording_factory(captured):
    from dspy.teleprompt import GEPA

    def factory(**kwargs):
        captured.update(kwargs)
        return GEPA(**kwargs)

    return factory


def _optimize(tmp_path, *, seed=0, budget=600, script=None):
    cell = method_cell("gepa", seed=seed, budget=budget)
    runner, ledger, data = fake_runner(cell)
    runner.artifact_store = ArtifactStore(tmp_path)
    runner.artifact_directory = runner.artifact_store.named("optimization")
    calls = _spy(runner)
    FakeAdapter.SCRIPT.update(script or {})
    lm, captured = FakeLM(), {}
    optimizer = methods.build_optimizer(
        "gepa", cell, runner.seed_bundle, extra={"reflection_lm": lm, "optimizer_factory": _recording_factory(captured)}
    )
    artifact = optimizer.optimize(cell, runner, ledger, list(data.rows("train")), list(data.rows("validation")))
    return cell, runner, ledger, artifact, calls, lm, captured


def test_policy_equals_the_reference_settings():
    policy = integration.FROZEN_GEPA_POLICY.to_dict()
    assert {key: policy[key] for key in GEPA_SETTINGS} == GEPA_SETTINGS
    assert methods.BUNDLE_PARAMETERS["gepa"] == "initial_bundle"
    assert methods.load_method("gepa") is integration.GEPAOptimizer
    integration.validate_frozen_policy()


@pytest.mark.parametrize("seed", [0, 2])
def test_native_gepa_runs_through_the_protocol_runner(tmp_path, seed):
    cell, runner, ledger, artifact, calls, lm, captured = _optimize(tmp_path, seed=seed)

    # Settings reach dspy GEPA unchanged; seed = policy seed 0 + optimizer seed.
    for key in (
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
        "track_stats",
    ):
        assert captured[key] == GEPA_SETTINGS[key], key
    assert captured["seed"] == seed and captured["log_dir"] is None and captured["num_threads"] == 1
    stopper = captured["gepa_kwargs"]["stop_callbacks"][0]
    assert stopper._patience == 3 and stopper._rollout_budget == 600
    reflection = captured["reflection_lm"]
    assert isinstance(reflection, LogicalSeedLM) and reflection.phase == "gepa_reflection"
    assert isinstance(reflection.lm, ContextFitLM) and reflection.lm.lm is lm

    # Budget: one charge per executed rollout, never above B, nothing left reserved.
    snapshot = ledger.snapshot()
    assert snapshot["reserved"] == 0 and 0 < snapshot["charged"] <= ledger.maximum
    assert snapshot["charged"] == len(calls) == len(FakeAdapter.CALLS)
    assert snapshot["charged"] <= 5 * (6 + 4)  # max_full_evals x (train + validation)

    # Seeds follow the reference rule and reach the runtime; optimization decoding.
    counts = {}
    for example_id, digest, request_seed in calls:
        k = counts.get((digest, example_id), 0)
        counts[(digest, example_id)] = k + 1
        assert request_seed == logical_request_seed(seed, cell.cell_id, "optimization/gepa", 0, example_id, digest, k)
    assert [call["seed"] for call in FakeAdapter.CALLS] == [request_seed for _, _, request_seed in calls]
    assert {call["temperature"] for call in FakeAdapter.CALLS} == {0.2}
    assert {call["model"] for call in FakeAdapter.CALLS} == {cell.task_model}

    # Reflection: common policy at the request boundary, logical seeds.
    assert lm.calls and len(artifact.reflection_requests) == len(lm.calls)
    for call, request in zip(lm.calls, artifact.reflection_requests):
        kwargs = call["kwargs"]
        assert (kwargs["temperature"], kwargs["top_p"], kwargs["max_tokens"]) == (1.0, 1.0, 48000)
        assert kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True
        assert kwargs["seed"] == request["request_seed"] and request["phase"] == "gepa_reflection"

    # Artifact contract.
    payload = artifact.to_dict()
    assert payload["method"] == "gepa" and payload["cell_id"] == cell.cell_id
    assert payload["budget_snapshot"] == ledger.snapshot() and payload["production_eligible"] is True
    assert payload["metadata"]["effective_seed"] == seed and payload["metadata"]["native_optimizer_executed"]
    assert payload["metadata"]["native_policy"] == integration.FROZEN_GEPA_POLICY.to_dict()
    incumbent = incumbent_of(artifact)
    runner.validate_bundle(incumbent)
    assert set(incumbent.roles) == set(runner.seed_bundle.roles) and incumbent.metadata["optimizer"] == "gepa"
    assert GOOD in incumbent.roles["writer"]  # native GEPA found the better writer prompt
    assert artifact.learning_curve[-1].charged_rollouts == snapshot["charged"]
    directory = runner.artifact_directory
    assert read_json(directory / "optimizer_result.json")["budget_snapshot"] == ledger.snapshot()
    assert read_json(directory / "optimizer_checkpoint_0000.json")["budget_snapshot"] == ledger.snapshot()


def test_rows_past_the_budget_are_answered_without_running(tmp_path):
    cell, runner, ledger, artifact, calls, lm, captured = _optimize(tmp_path, budget=7)
    snapshot = ledger.snapshot()
    assert snapshot["charged"] == 7 == len(calls) == len(FakeAdapter.CALLS)
    assert snapshot["reserved"] == 0 and snapshot["remaining"] == 0
    metadata = artifact.to_dict()["metadata"]
    assert metadata["stop_reason"] == "rollout_budget_spent" and metadata["rows_answered_past_budget"] > 0
    assert captured["gepa_kwargs"]["stop_callbacks"][0]._rollout_budget == 7
    assert artifact.to_dict()["budget_snapshot"] == snapshot
    runner.validate_bundle(incumbent_of(artifact))


def test_exhausted_infrastructure_fails_the_optimization_as_infrastructure(tmp_path):
    with pytest.raises(OptimizerInfrastructureFailure, match="runtime failure"):
        _optimize(tmp_path, script={"va1": ["connection"] * 3})
    assert not (tmp_path / "optimization" / "optimizer_result.json").exists()


def test_grid_cells_are_validated_exactly():
    cell = method_cell("gepa")
    integration.validate_supported_cell(cell)  # off-grid fake cell: admitted (run.py gates it)
    grid = dataclasses.replace(
        cell, task="math", topology="single", framework="langgraph", team_size=1, source_tables=(2,)
    )
    integration.validate_supported_cell(grid)
    with pytest.raises(UnsupportedGEPACell, match="outside the experiment grid"):
        integration.validate_supported_cell(
            dataclasses.replace(grid, task="toolhop", topology="independent", team_size=2)
        )
    with pytest.raises(UnsupportedGEPACell, match="framework/topology"):
        integration.validate_supported_cell(dataclasses.replace(grid, framework="crewai"))
    with pytest.raises(UnsupportedGEPACell, match="protocol_id"):
        integration.validate_supported_cell(dataclasses.replace(cell, protocol_id="other"))
    with pytest.raises(UnsupportedGEPACell, match="method"):
        integration.validate_supported_cell(method_cell("mipro"))


def test_candidate_clones_isolate_prompts_but_share_runner_and_seed_state():
    from optimizers.bridge.programs import AdapterBackedProgram
    from optimizers.protocol.rollouts import BudgetedRunner, BudgetStopRunner

    cell = method_cell("gepa")
    runner, ledger, _ = fake_runner(cell)
    shared = BudgetStopRunner(BudgetedRunner(cell=cell, runner=runner, budget=ledger))
    program = AdapterBackedProgram(integration.CommonRunnerAdapter(cell=cell, runner=shared, bundle=runner.seed_bundle))
    clone = program.deepcopy()
    assert clone._adapter is not program._adapter and clone._adapter._runner is program._adapter._runner is shared
    assert clone._adapter._call_counts is program._adapter._call_counts
    name, predictor = clone.named_predictors()[-1]
    predictor.signature = predictor.signature.with_instructions("candidate prompt")
    clone.sync_prompts_to_adapter()
    assert clone._adapter.get_prompt(name) == "candidate prompt" != program._adapter.get_prompt(name)


def _run(out: Path, *extra: str, kwargs=None) -> int:
    hooks = dataclasses.replace(fake_hooks(), optimizer_kwargs=kwargs or {"reflection_lm": FakeLM()})
    argv = [
        "--method",
        "gepa",
        "--dataset",
        "fake",
        "--topology",
        "sequential",
        "--model",
        "qwen",
        "--seed",
        "1",
        "--out",
        str(out),
        "--allow-any-cell",
        "--quiet",
        *extra,
    ]
    return run.main(argv, hooks=hooks)


def _sealed(path: Path) -> dict:
    return verify_sealed(read_json(path))


def test_protocol_job_runs_gepa_end_to_end(tmp_path):
    out = tmp_path / "gepa"
    assert _run(out) == 0
    optimization = _sealed(out / "optimization.json")
    assert optimization["status"] == "completed" and optimization["stop_reason"] == "max_full_evals"
    budget = optimization["budget"]
    charged = budget["charged"]
    assert 0 < charged <= 600 and budget["reserved"] == 0
    assert {call["temperature"] for call in FakeAdapter.CALLS[:charged]} == {0.2}
    assert {call["temperature"] for call in FakeAdapter.CALLS[charged:]} == {0.0}
    assert optimization["usage"]["task"]["model_calls"] == 2 * charged
    assert optimization["usage"]["reflection"]["model_calls"] > 0
    result = read_json(out / "optimization" / "optimizer_result.json")
    assert result["budget_snapshot"] == budget and result["method"] == "gepa"
    selection = _sealed(out / "selection.json")
    assert selection["selected_candidate"] and selection["incumbent_validation_score"] == 1.0
    assert _sealed(out / "result.json")["test"]["deployed_mean"] == 1.0
    for path in out.rglob("*"):
        if path.is_file():
            text = path.read_text(encoding="utf-8", errors="ignore")
            assert str(tmp_path) not in text and str(Path.home()) not in text and str(REPO_ROOT) not in text


def test_protocol_job_reports_infrastructure_invalid_gepa_with_the_seed(tmp_path):
    FakeAdapter.SCRIPT["va1"] = ["connection"] * 3
    out = tmp_path / "infra"
    assert _run(out) == 0
    optimization = _sealed(out / "optimization.json")
    assert optimization["status"] == "failed" and optimization["failure_kind"] == "infrastructure_invalid"
    assert optimization["budget"]["infrastructure_failures"] == 3 and optimization["budget"]["reserved"] == 0
    assert _sealed(out / "selection.json")["fallback_reason"] == "infrastructure_invalid_optimization"
    json.dumps(_sealed(out / "result.json"))
