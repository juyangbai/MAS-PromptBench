"""MIPRO port: native DSPy MIPROv2 through the protocol runner with a fake MAS and fake LMs."""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
from pathlib import Path

import pytest

from optimizers.bridge.mipro_programs import MIPROAdapterBackedProgram, MIPRORolePredict
from optimizers.protocol import REPO_ROOT, methods, run
from optimizers.protocol.artifacts import ArtifactStore, read_json
from optimizers.protocol.methods.context_fit import ContextFitLM
from optimizers.protocol.methods.dspy_bridge import LogicalSeedLM
from optimizers.protocol.methods.tests.fakes import GOOD, FakeLM, method_cell
from optimizers.protocol.schema import incumbent_of, verify_sealed
from optimizers.protocol.seeding import logical_request_seed
from optimizers.protocol.tests.fakes import FakeAdapter, fake_hooks, fake_runner

from .. import integration

# MIPRO settings of the reference method configuration.
MIPRO_SETTINGS = {
    "auto": None,
    "num_candidates": 3,
    "num_trials": 3,
    "max_bootstrapped_demos": 4,
    "max_labeled_demos": 0,
    "metric_threshold": None,
    "minibatch": False,
    "minibatch_size": 35,
    "minibatch_full_eval_steps": 5,
    "seed": 9,
    "init_temperature": 1.0,
    "view_data_batch_size": 10,
    "program_aware_proposer": True,
    "data_aware_proposer": True,
    "tip_aware_proposer": True,
    "fewshot_aware_proposer": True,
    "provide_traceback": False,
    "max_errors": None,
    "requires_permission_to_run": None,
    "dspy_version": "3.2.0",
}

# sha256 of what the program-aware proposer is shown of the reference program:
# the role signatures, MIPRORolePredict and MIPROAdapterBackedProgram.
PROPOSER_SOURCE_SHA256 = "87666fcba93364cb8b9b1a1c76100724a5a81a57839cc9e549907c9d749d3069"


def _spy(runner):
    calls = []
    native = runner.run_batch

    def run_batch(examples, bundle, seeds):
        calls.extend((str(example["id"]), bundle.digest, int(seed)) for example, seed in zip(examples, seeds))
        return native(examples, bundle, seeds)

    runner.run_batch = run_batch
    return calls


def _recording_factory(captured):
    from dspy.teleprompt import MIPROv2

    class RecordingMIPROv2(MIPROv2):
        def __init__(self, **kwargs):
            captured["constructor"] = kwargs
            super().__init__(**kwargs)

        def compile(self, student, **kwargs):
            captured["compile"] = {key: value for key, value in kwargs.items() if key not in {"trainset", "valset"}}
            return super().compile(student, **kwargs)

    return RecordingMIPROv2


def _optimize(tmp_path, *, seed=0, budget=600):
    cell = method_cell("mipro", seed=seed, budget=budget)
    runner, ledger, data = fake_runner(cell)
    runner.artifact_store = ArtifactStore(tmp_path)
    runner.artifact_directory = runner.artifact_store.named("optimization")
    calls = _spy(runner)
    lm, task_lm, captured = FakeLM(), FakeLM("fake/task", forbid_calls=True), {}
    optimizer = methods.build_optimizer(
        "mipro",
        cell,
        runner.seed_bundle,
        extra={"reflection_lm": lm, "task_lm": task_lm, "optimizer_factory": _recording_factory(captured)},
    )
    artifact = optimizer.optimize(cell, runner, ledger, list(data.rows("train")), list(data.rows("validation")))
    return cell, runner, ledger, artifact, calls, lm, task_lm, captured


def test_policy_equals_the_reference_settings():
    policy = integration.FROZEN_MIPRO_POLICY.to_dict()
    assert {key: policy[key] for key in MIPRO_SETTINGS} == MIPRO_SETTINGS
    assert methods.BUNDLE_PARAMETERS["mipro"] == "initial_bundle"
    assert methods.load_method("mipro") is integration.MIPROOptimizer
    integration.validate_frozen_policy()


@pytest.mark.parametrize("seed", [0, 1])
def test_native_mipro_runs_through_the_protocol_runner(tmp_path, seed):
    cell, runner, ledger, artifact, calls, lm, task_lm, captured = _optimize(tmp_path, seed=seed)

    # Settings reach MIPROv2 unchanged; seed = policy seed 9 + optimizer seed.
    constructor, compiled = captured["constructor"], captured["compile"]
    for key in (
        "auto",
        "num_candidates",
        "max_bootstrapped_demos",
        "max_labeled_demos",
        "init_temperature",
        "max_errors",
        "metric_threshold",
    ):
        assert constructor[key] == MIPRO_SETTINGS[key], key
    assert constructor["seed"] == compiled["seed"] == 9 + seed
    assert constructor["track_stats"] is True and constructor["log_dir"] is None and constructor["num_threads"] == 1
    for key in (
        "num_trials",
        "minibatch",
        "minibatch_size",
        "minibatch_full_eval_steps",
        "program_aware_proposer",
        "data_aware_proposer",
        "view_data_batch_size",
        "tip_aware_proposer",
        "fewshot_aware_proposer",
        "provide_traceback",
        "requires_permission_to_run",
    ):
        assert compiled[key] == MIPRO_SETTINGS[key], key
    prompt_model = constructor["prompt_model"]
    assert isinstance(prompt_model, LogicalSeedLM) and prompt_model.phase == "mipro_reflection"
    assert isinstance(prompt_model.lm, ContextFitLM) and prompt_model.lm.lm is lm
    assert constructor["task_model"] is task_lm and not task_lm.calls  # every rollout is a runner rollout

    # Budget: one charge per executed rollout, never above B, nothing left reserved.
    snapshot = ledger.snapshot()
    assert snapshot["reserved"] == 0 and 0 < snapshot["charged"] <= ledger.maximum
    assert snapshot["charged"] == len(calls) == len(FakeAdapter.CALLS)

    # Seeds follow the reference rule and reach the runtime; optimization decoding.
    counts = {}
    for example_id, digest, request_seed in calls:
        k = counts.get((digest, example_id), 0)
        counts[(digest, example_id)] = k + 1
        assert request_seed == logical_request_seed(seed, cell.cell_id, "optimization/mipro", k, example_id, digest, k)
    assert [call["seed"] for call in FakeAdapter.CALLS] == [request_seed for _, _, request_seed in calls]
    assert {call["temperature"] for call in FakeAdapter.CALLS} == {0.2}

    # Prompt model: common policy at the request boundary.
    assert lm.calls and artifact.reflection_requests
    for call in lm.calls:
        kwargs = call["kwargs"]
        assert (kwargs["temperature"], kwargs["top_p"], kwargs["max_tokens"]) == (1.0, 1.0, 48000)
        assert kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True
    assert {request["phase"] for request in artifact.reflection_requests} == {"mipro_reflection"}

    # The program-aware proposer sees the reference program, byte for byte.
    program = MIPROAdapterBackedProgram(FakeAdapter())
    signatures = [str(predictor.signature) for _, predictor in program.named_predictors()]
    sources = [inspect.getsource(MIPRORolePredict), inspect.getsource(MIPROAdapterBackedProgram)]
    prompts = [str(message.get("content")) for call in lm.calls for message in call["messages"] or ()]
    # DSPy shows the first role's signature (the role signatures share one name) and both classes.
    for text in (signatures[0], *sources):
        assert any(text in prompt for prompt in prompts), text.splitlines()[0]
    assert hashlib.sha256("\n\n".join(signatures + sources).encode()).hexdigest() == PROPOSER_SOURCE_SHA256

    # Artifact contract.
    payload = artifact.to_dict()
    assert payload["method"] == "mipro" and payload["cell_id"] == cell.cell_id
    assert payload["budget_snapshot"] == ledger.snapshot() and payload["production_eligible"] is True
    assert payload["metadata"]["effective_seed"] == 9 + seed
    assert payload["metadata"]["native_policy"] == integration.FROZEN_MIPRO_POLICY.to_dict()
    incumbent = incumbent_of(artifact)
    runner.validate_bundle(incumbent)
    assert set(incumbent.roles) == set(runner.seed_bundle.roles) and incumbent.metadata["optimizer"] == "mipro"
    assert set(incumbent.metadata["selected_demos_by_role"]) == set(incumbent.roles)
    assert GOOD in incumbent.roles["writer"]  # MIPROv2 selected the better proposed instruction
    directory = runner.artifact_directory
    assert read_json(directory / "optimizer_result.json")["budget_snapshot"] == ledger.snapshot()


def test_rows_past_the_budget_are_answered_without_running(tmp_path):
    cell, runner, ledger, artifact, calls, lm, task_lm, captured = _optimize(tmp_path, budget=9)
    snapshot = ledger.snapshot()
    assert snapshot["charged"] == 9 == len(calls) == len(FakeAdapter.CALLS)
    assert snapshot["reserved"] == 0 and snapshot["remaining"] == 0
    metadata = artifact.to_dict()["metadata"]
    assert metadata["stop_reason"] == "rollout_budget_spent" and metadata["rows_answered_past_budget"] > 0
    assert artifact.to_dict()["budget_snapshot"] == snapshot
    runner.validate_bundle(incumbent_of(artifact))


def test_cells_must_be_exact_grid_cells_when_they_claim_to_be():
    cell = method_cell("mipro")
    integration.MIPROOptimizer._validate_cell(cell)
    grid = dataclasses.replace(cell, task="hotpotqa", topology="centralized", framework="langgraph", source_tables=(3,))
    integration.MIPROOptimizer._validate_cell(grid)
    with pytest.raises(ValueError, match="outside the experiment grid"):
        integration.MIPROOptimizer._validate_cell(dataclasses.replace(grid, framework="crewai"))
    with pytest.raises(ValueError, match="protocol_id"):
        integration.MIPROOptimizer._validate_cell(dataclasses.replace(cell, protocol_id="other"))


def test_protocol_job_runs_mipro_end_to_end(tmp_path):
    hooks = dataclasses.replace(
        fake_hooks(), optimizer_kwargs={"reflection_lm": FakeLM(), "task_lm": FakeLM("fake/task", forbid_calls=True)}
    )
    out = tmp_path / "mipro"
    argv = [
        "--method",
        "mipro",
        "--dataset",
        "fake",
        "--topology",
        "sequential",
        "--model",
        "qwen",
        "--seed",
        "2",
        "--out",
        str(out),
        "--allow-any-cell",
        "--quiet",
    ]
    assert run.main(argv, hooks=hooks) == 0
    optimization = verify_sealed(read_json(out / "optimization.json"))
    assert optimization["status"] == "completed" and optimization["stop_reason"] == "num_trials_complete"
    charged = optimization["budget"]["charged"]
    assert 0 < charged <= 600 and optimization["budget"]["reserved"] == 0
    assert {call["temperature"] for call in FakeAdapter.CALLS[:charged]} == {0.2}
    assert {call["temperature"] for call in FakeAdapter.CALLS[charged:]} == {0.0}
    assert read_json(out / "optimization" / "optimizer_result.json")["budget_snapshot"] == optimization["budget"]
    selection = verify_sealed(read_json(out / "selection.json"))
    assert selection["selected_candidate"] and selection["incumbent_validation_score"] == 1.0
    for path in out.rglob("*"):
        if path.is_file():
            text = path.read_text(encoding="utf-8", errors="ignore")
            assert str(tmp_path) not in text and str(Path.home()) not in text and str(REPO_ROOT) not in text
