"""The CG-OPO loop on the protocol runner: credit, rotation, acceptance, budget and artifact contract."""

from __future__ import annotations

import copy
from dataclasses import replace

import pytest

from optimizers.bridge.adapters.module_hotpotqa import CentralizedHotpotQAAdapter
from optimizers.bridge.adapters.sequential_bfcl import SequentialBFCLAdapter
from optimizers.protocol.errors import NativeIntegrationError, RunnerContractError
from optimizers.protocol.run import Job, JobHooks, JobOptions
from optimizers.protocol.runner import execution_hook_for
from optimizers.protocol.schema import PromptBundle, content_hash

from ..integration import METHOD_SETTINGS, HiveMindOptimizer
from ..regime import plan_metadata_matches
from .fakes import (
    AnswerScorer,
    FakeReflection,
    Script,
    TelemetryHook,
    bfcl_runtime,
    fake_task_data,
    hivemind_cell,
    hotpotqa_runtime,
    protocol_runner,
)


def _centralized(budget: int):
    cell = hivemind_cell("hotpotqa", "centralized", budget=budget)
    runner, ledger, data = protocol_runner(cell, hotpotqa_runtime(cell, CentralizedHotpotQAAdapter))
    return cell, runner, ledger, list(data.rows("train")), list(data.rows("validation"))


def _assert_artifact_contract(artifact, ledger, runner) -> None:
    snapshot = ledger.snapshot()
    payload = artifact.to_dict()
    assert payload["schema"] == "mas-promptbench-native-optimizer-result"
    assert payload["method"] == "hivemind" and payload["cell_id"] == runner.cell.cell_id
    assert payload["budget_snapshot"] == snapshot and snapshot["reserved"] == 0
    assert snapshot["charged"] <= snapshot["maximum"]
    assert len(artifact.records) == len(artifact.request_ledger) == snapshot["charged"]
    assert all(record.request_seed is not None for record in artifact.records)
    grids = [row["rollout_grid"] for row in artifact.learning_curve if row["rollout_grid"] is not None]
    assert grids == list(range(0, snapshot["maximum"] + 1, 10))
    for event in artifact.events:
        assert event["event_id"] == content_hash({k: v for k, v in event.items() if k != "event_id"})
    for checkpoint in artifact.checkpoints:
        assert checkpoint["checkpoint_id"] == content_hash(
            {k: v for k, v in checkpoint.items() if k != "checkpoint_id"}
        )
    assert "optimizer_control" not in artifact.incumbent_bundle.metadata
    runner.validate_bundle(artifact.incumbent_bundle)
    assert payload["metadata"]["settings"] == METHOD_SETTINGS
    for record in artifact.records:
        assert record.metadata["runtime_metadata"]["execution_control"]["applied"] is True


def test_centralized_cycles_assign_credit_rotate_the_manager_and_accept_only_strict_gains(hotpotqa_script):
    cell, runner, ledger, training, validation = _centralized(160)
    reflection = FakeReflection()
    artifact = HiveMindOptimizer(seed_bundle=runner.seed_bundle, reflection=reflection).optimize(
        cell, runner, ledger, training, validation
    )
    # 8 coalitions x 5 train rows + 2 x 5 validation rows per cycle; a fourth cycle does not fit.
    assert artifact.native_iterations == 3 and artifact.stop_reason == "budget"
    assert ledger.snapshot()["charged"] == 150 and ledger.snapshot()["infrastructure_failures"] == 0
    history = artifact.metadata["history"]
    first, second, third = history
    assert set(first["coalition_values"]) >= {"manager_only", "reasoner_worker+retriever_worker+writer_worker"}
    for cycle in history:
        values = cycle["coalition_values"]
        assert sum(cycle["phi"].values()) == pytest.approx(
            values["reasoner_worker+retriever_worker+writer_worker"] - values["manager_only"]
        )
    # Only the retriever matters for seed prompts: the lowest-credit worker is reflected on.
    assert first["parameter_phi"]["retriever_worker"] >= max(first["parameter_phi"].values())
    assert first["target"] != "manager" and first["accepted"] and first["candidate_score"] > first["current_score"]
    assert not second["manager_cycle"] and second["accepted"] is False
    assert second["candidate_score"] == second["current_score"] == 1.0  # ties are rejected
    assert third["manager_cycle"] and third["target"] == "manager" and third["accepted"] is False
    incumbent = artifact.incumbent_bundle.roles
    changed = [role for role in incumbent if incumbent[role] != runner.seed_bundle.roles[role]]
    assert changed == [first["target"]] and "Lessons learned" in incumbent[first["target"]]
    assert [request["role"] for request in reflection.requests] == [first["target"], second["target"], "manager"]
    for request in reflection.requests:
        assert (request["temperature"], request["top_p"], request["max_output_tokens"], request["thinking"]) == (
            0.7,
            1.0,
            48000,
            True,
        )
        assert request["phase"] == "coalition_credit_and_reflection"
        assert "solves hotpotqa tasks" in request["prompt"] and "LOSING cases" in request["prompt"]
    _assert_artifact_contract(artifact, ledger, runner)
    plan = artifact.metadata["coalition_plan"]
    assert plan["hivemind_coalition_count"] == 8 and plan["hivemind_shapley_mode"] == "exact_worker_shapley"
    assert plan_metadata_matches(
        plan, ["reasoner_worker", "retriever_worker", "writer_worker"], max_coalitions=40, seed_offset=0, hm_seed=0
    )
    with pytest.raises(RunnerContractError, match="no execution hook"):
        execution_hook_for(PromptBundle(roles={"a": "x"}, metadata={"optimizer_control": {}}))


@pytest.mark.parametrize("budget,cycles", [(9, 0), (10, 1), (37, 1), (75, 1)])
def test_planned_cycles_never_exceed_the_budget(hotpotqa_script, budget, cycles):
    cell, runner, ledger, training, validation = _centralized(budget)
    artifact = HiveMindOptimizer(seed_bundle=runner.seed_bundle, reflection=FakeReflection()).optimize(
        cell, runner, ledger, training, validation
    )
    snapshot = ledger.snapshot()
    assert artifact.native_iterations == cycles and artifact.stop_reason == "budget"
    assert snapshot["charged"] <= budget and snapshot["reserved"] == 0
    if cycles:
        # Coalition batches shrink (5 -> 1 rows) until one full cycle fits.
        batch = max(b for b in range(1, 6) if 8 * b + 2 * min(5, b) <= budget)
        assert snapshot["charged"] == 8 * batch + 2 * min(5, batch)
    else:
        assert snapshot["charged"] == 0 and hotpotqa_script.calls == []
    _assert_artifact_contract(artifact, ledger, runner)


def test_noncentralized_cycle_charges_zero_call_abstentions():
    script = Script()
    cell = hivemind_cell("bfcl", "sequential", budget=90)
    runner, ledger, data = protocol_runner(cell, bfcl_runtime(cell, SequentialBFCLAdapter, script))
    reflection = FakeReflection()
    artifact = HiveMindOptimizer(
        seed_bundle=runner.seed_bundle, reflection=reflection, execution_hook=TelemetryHook(script)
    ).optimize(cell, runner, ledger, list(data.rows("train")), list(data.rows("validation")))
    snapshot = ledger.snapshot()
    assert artifact.native_iterations == 1 and snapshot["charged"] == 90 == 16 * 5 + 2 * 5
    abstentions = [record for record in artifact.records if record.usage.model_calls == 0]
    assert len(abstentions) == 5
    assert all(r.metadata["runtime_metadata"]["execution_control"]["allow_zero_model_calls"] for r in abstentions)
    history = artifact.metadata["history"][0]
    assert history["coalition_values"]["empty"] == 0.0 and not history["manager_cycle"]
    assert history["target"] in {"analyzer", "caller", "inspector", "verifier"}
    assert "solves bfcl tasks" in reflection.requests[0]["prompt"]
    _assert_artifact_contract(artifact, ledger, runner)


def test_resume_from_the_final_checkpoint_spends_nothing(hotpotqa_script):
    cell, runner, ledger, training, validation = _centralized(120)
    first = HiveMindOptimizer(seed_bundle=runner.seed_bundle, reflection=FakeReflection(), max_cycles=1).optimize(
        cell, runner, ledger, training, validation
    )
    assert first.stop_reason == "max_cycles" and first.native_iterations == 1
    calls = len(hotpotqa_script.calls)
    checkpoint = first.checkpoints[-1]
    resumed = HiveMindOptimizer(
        seed_bundle=runner.seed_bundle, reflection=FakeReflection(), max_cycles=1, resume_checkpoint=checkpoint
    ).optimize(cell, runner, ledger, training, validation)
    assert len(hotpotqa_script.calls) == calls and resumed.incumbent_bundle.digest == first.incumbent_bundle.digest
    tampered = copy.deepcopy(checkpoint)
    tampered["state"]["cycle"] = 99
    with pytest.raises(NativeIntegrationError, match="content hash"):
        HiveMindOptimizer(
            seed_bundle=runner.seed_bundle, reflection=FakeReflection(), resume_checkpoint=tampered
        ).optimize(cell, runner, ledger, training, validation)


def test_rejects_cells_outside_the_grid_and_missing_inputs(hotpotqa_script):
    cell, runner, ledger, training, validation = _centralized(60)
    optimizer = HiveMindOptimizer(seed_bundle=runner.seed_bundle, reflection=FakeReflection())
    with pytest.raises(ValueError, match="not in the experiment grid"):
        optimizer.optimize(replace(cell, team_size=8), runner, ledger, training, validation)
    with pytest.raises(NativeIntegrationError, match="non-empty"):
        optimizer.optimize(cell, runner, ledger, training, [])
    with pytest.raises(NativeIntegrationError, match="seed PromptBundle"):
        HiveMindOptimizer(reflection=FakeReflection()).optimize(cell, runner, ledger, training, validation)
    assert ledger.snapshot()["attempted"] == 0


def test_protocol_job_runs_hivemind_end_to_end(hotpotqa_script, tmp_path):
    hooks = JobHooks(
        load_task_data=fake_task_data,
        build_runtime=lambda cell: hotpotqa_runtime(cell, CentralizedHotpotQAAdapter),
        build_scorer=lambda cell: AnswerScorer(),
        optimizer_kwargs={"reflection": FakeReflection()},
        configure_environment=False,
    )
    options = JobOptions(
        method="hivemind",
        dataset="hotpotqa",
        topology="centralized",
        model="qwen",
        seed=1,
        out=tmp_path / "job",
        budget=60,
        quiet=True,
    )
    result = Job(options, hooks).run()
    assert result["status"] == "completed" and result["grid_cell"]["topology"] == "centralized"
    assert result["optimization"]["status"] == "completed" and result["optimization"]["budget"]["charged"] <= 60
    assert result["optimization"]["usage"]["reflection"]["model_calls"] == 1
    assert result["test"]["valid_for_aggregation"] and len(result["test"]["deployed_scores"]) == 2
    assert result["selection"]["selected_candidate"] in {True, False}
