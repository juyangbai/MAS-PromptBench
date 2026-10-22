"""TAVOOptimizer on the protocol runner: fixed batches, validation gate, patience, budget and artifacts."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from optimizers.protocol import REPO_ROOT, run
from optimizers.protocol import runner as protocol_runner
from optimizers.protocol.artifacts import ArtifactStore, read_json, read_jsonl
from optimizers.protocol.budget import closed_ledger
from optimizers.protocol.cells import required_cells
from optimizers.protocol.errors import UnsupportedBaselineCell
from optimizers.protocol.methods import build_optimizer, load_method
from optimizers.protocol.schema import PromptBundle, RunRecord, content_hash, incumbent_of, verify_sealed
from optimizers.protocol.tests.fakes import FakeAdapter, FakeScorer, fake_runtime, fake_task_data

from ..driver import append_overlay, strip_overlay_blocks
from ..integration import TAVOOptimizer, _messages, _ordered_roles
from ..settings import TAVOSettings
from .fakes import BAD_OVERLAY, GOOD_OVERLAY, FakeReflection, tavo_cell, tavo_runner


def _optimize(cell=None, *, overlay=GOOD_OVERLAY, store=None, directory=None, **options):
    cell = cell or tavo_cell()
    runner, budget, data = tavo_runner(cell, store=store, directory=directory)
    reflection = FakeReflection(overlay)
    optimizer = TAVOOptimizer(initial_bundle=runner.seed_bundle, reflection_client=reflection, **options)
    artifact = optimizer.optimize(cell, runner, budget, list(data.rows("train")), list(data.rows("validation")))
    return artifact, budget, reflection, runner, data


def test_registry_builds_the_optimizer_with_the_seed_bundle():
    assert load_method("tavo") is TAVOOptimizer
    cell = tavo_cell()
    runner, _, _ = tavo_runner(cell)
    optimizer = build_optimizer("tavo", cell, runner.seed_bundle, extra={"reflection_client": FakeReflection()})
    assert optimizer.initial_bundle is runner.seed_bundle and optimizer.mode == "truce-release"
    assert optimizer.trajectory_credit is True and optimizer.implementation_kind == "native_truce_release_common_runner"
    assert len(required_cells("tavo")) == 12


def test_improving_overlay_is_adopted_every_round_and_budget_is_exact():
    artifact, budget, reflection, runner, data = _optimize()
    meta = artifact.metadata
    assert budget.charged == 4 + 5 * (6 + 4) and closed_ledger(budget.snapshot())
    assert artifact.budget_snapshot == budget.snapshot()
    assert (meta["base_score"], meta["best_score"], meta["best_iteration"]) == (0.5, 1.0, 5)
    assert [len(item["attempts"]) for item in meta["iterations"]] == [1] * 5
    assert all(item["adopted"] for item in meta["iterations"])
    assert meta["stopped_reason"] is None and artifact.stop_reason == "max_outer_rounds"
    selected = artifact.selected_bundle
    overlay = meta["iterations"][-1]["attempts"][0]["overlay"]
    assert selected.roles == {role: append_overlay(text, overlay) for role, text in runner.seed_bundle.roles.items()}
    assert "GOOD" in selected.roles["writer"]
    assert strip_overlay_blocks(selected.roles["writer"]) == "Write the final answer."
    assert selected.metadata == {**runner.seed_bundle.metadata, "optimizer": "tavo", "native_mode": "truce-release"}
    assert [point.charged_rollouts for point in artifact.learning_curve] == [0, 4, 14, 24, 34, 44, 54]
    assert len(artifact.checkpoints) == 5 and artifact.checkpoints[-1]["state"]["best_iteration"] == 5
    # Fixed batches: one Random(233 + seed) stream samples train then validation.
    rng = random.Random(233)
    train_ids = [row["id"] for row in rng.sample(list(data.rows("train")), 6)]
    validation_ids = [row["id"] for row in rng.sample(list(data.rows("validation")), 4)]
    assert (meta["train_batch_ids"], meta["validation_batch_ids"], meta["rng_seed"]) == (train_ids, validation_ids, 233)
    assert meta["iterations"][0]["attempts"][0]["train_ids"] == train_ids
    # Every charged rollout reached the adapter exactly once; validation used the overlay after round 1.
    assert len(FakeAdapter.CALLS) == budget.charged
    validation_writers = [call["writer"] for call in FakeAdapter.CALLS if call["id"].startswith("va")]
    assert validation_writers[:4] == ["Write the final answer."] * 4
    assert all("GOOD" in writer for writer in validation_writers[4:])
    credit_prompt = reflection.of_kind("credit")[0]["prompt"]
    assert "Token usage: 15" in credit_prompt and "LLM Used: Qwen/Qwen3.5-9B" in credit_prompt
    assert '"source": "planner"' in credit_prompt and "Coordination Mode: ordered sequential pipeline" in credit_prompt
    requests = artifact.reflection_requests
    assert requests
    assert {(r["max_output_tokens"], r["thinking"], r["status"]) for r in requests} == {(48000, True, "success")}
    assert {r["temperature"] for r in requests} == {0.3, 0.5}
    assert len(requests) == len(reflection.requests) == meta["reflection_usage"]["n_calls"]


def test_regressing_overlay_is_retried_once_then_patience_stops_with_the_seed():
    artifact, budget, reflection, runner, _ = _optimize(overlay=BAD_OVERLAY)
    meta = artifact.metadata
    assert budget.charged == 4 + 2 * 2 * (6 + 4) and closed_ledger(budget.snapshot())
    assert meta["stopped_reason"] == "native_patience" and meta["best_iteration"] == 0
    assert [len(item["attempts"]) for item in meta["iterations"]] == [2, 2]
    assert [item["patience_counter"] for item in meta["iterations"]] == [1, 2]
    assert all(
        attempt["validation_score"] == 0.0 and not attempt["adopted"]
        for item in meta["iterations"]
        for attempt in item["attempts"]
    )
    assert artifact.selected_bundle.roles == runner.seed_bundle.roles
    # The retry trains with the rejected candidate (the train chain always advances).
    train_writers = [call["writer"] for call in FakeAdapter.CALLS if call["id"].startswith("tr")]
    assert train_writers[:6] == ["Write the final answer."] * 6 and all("BAD" in w for w in train_writers[6:])
    assert reflection.kinds().count("overlay") == 4


@pytest.mark.parametrize(
    "budget_size,overlay,charged,stop",
    [
        (7, GOOD_OVERLAY, 3, "budget_before_outer_round"),
        (13, GOOD_OVERLAY, 12, "budget_before_outer_round"),
        (30, BAD_OVERLAY, 30, "native_patience"),
        (60, BAD_OVERLAY, 44, "native_patience"),
        (600, GOOD_OVERLAY, 54, "max_outer_rounds"),
    ],
    ids=["B7", "B13", "B30", "B60", "B600"],
)
def test_budget_is_never_exceeded(budget_size, overlay, charged, stop):
    artifact, budget, _, _, _ = _optimize(tavo_cell(budget=budget_size), overlay=overlay)
    snapshot = budget.snapshot()
    assert snapshot["charged"] == charged <= budget_size and snapshot["reserved"] == 0
    assert closed_ledger(snapshot) and artifact.budget_snapshot == snapshot
    assert len(FakeAdapter.CALLS) == charged and artifact.stop_reason == stop
    if budget_size == 30:
        assert artifact.metadata["budget_truncations"] == 1  # the second retry did not fit


def test_validation_batch_size_follows_the_budget_layout():
    settings = TAVOSettings()
    assert settings.validation_batch_size(600, 6, 50) == 50
    assert settings.validation_batch_size(120, 6, 50) == 15
    assert settings.validation_batch_size(20, 6, 50) == 3
    assert settings.validation_batch_size(600, 6, 4) == 4


def test_optimizer_seed_offsets_the_batch_sampling_seed():
    artifact, _, _, _, data = _optimize(tavo_cell(seed=2))
    expected = [row["id"] for row in random.Random(235).sample(list(data.rows("train")), 6)]
    assert artifact.metadata["rng_seed"] == 235 and artifact.metadata["train_batch_ids"] == expected


def test_credit_ablation_and_validation_only_smoke(monkeypatch):
    artifact, _, reflection, _, _ = _optimize(trajectory_credit=False)
    assert "eq3" not in reflection.kinds() and artifact.metadata["equation_3_credit"] is False
    assert artifact.metadata["trajectory_credit_ablation"] == "eq3-disabled"
    monkeypatch.setenv("TAVO_CREDIT", "0")
    assert TAVOOptimizer(reflection_client=FakeReflection()).trajectory_credit is False
    smoke, budget, _, _, _ = _optimize(validation_one_cycle=True)
    assert budget.charged == 3 and smoke.production_eligible is False
    assert smoke.metadata["effective_lifecycle"] == {
        "max_iterations": 1,
        "train_batch_size": 1,
        "validation_batch_size": 1,
    }


def test_other_modes_run_end_to_end_with_their_labels():
    hybrid, _, reflection, _, _ = _optimize(mode="tavo-hybrid")
    assert hybrid.metadata["mode"] == "tavo-hybrid" and hybrid.selected_bundle.metadata["native_mode"] == "tavo-hybrid"
    assert "refine_meta" in reflection.kinds()
    paper, _, reflection, _, _ = _optimize(mode="tavo-paper-reproduction")
    assert paper.implementation_kind == "native_tavo_paper_reproduction_common_runner"
    assert "overlay" not in reflection.kinds() and "aggregate" in reflection.kinds()
    with pytest.raises(ValueError, match="unsupported TAVO mode"):
        TAVOOptimizer(reflection_client=FakeReflection(), mode="no-such-mode")


def test_cells_outside_table_6_are_rejected_before_any_rollout():
    cell = tavo_cell()
    runner, budget, data = tavo_runner(cell)
    optimizer = TAVOOptimizer(initial_bundle=runner.seed_bundle, reflection_client=FakeReflection())
    for outside in (tavo_cell(team_size=8), tavo_cell(task="math"), tavo_cell(topology="single")):
        with pytest.raises(UnsupportedBaselineCell):
            optimizer.optimize(outside, runner, budget, list(data.rows("train")), list(data.rows("validation")))
    assert budget.snapshot()["attempted"] == 0 and FakeAdapter.CALLS == []


def test_artifact_contract_and_persisted_files(tmp_path):
    store = ArtifactStore(tmp_path)
    directory = store.named("optimization")
    cell = tavo_cell()
    artifact, budget, _, runner, _ = _optimize(cell, store=store, directory=directory)
    payload = artifact.to_dict()
    assert payload["schema"] == "mas-promptbench-native-optimizer-result"
    assert (payload["method"], payload["cell_id"], payload["production_eligible"]) == ("tavo", cell.cell_id, True)
    assert payload["budget_snapshot"] == budget.snapshot() and json.dumps(payload) and content_hash(payload)
    assert incumbent_of(artifact) is artifact.selected_bundle
    runner.validate_bundle(artifact.selected_bundle)
    saved = read_json(directory / "optimizer_result.json")
    assert saved["artifact_sha256"] == content_hash({k: v for k, v in saved.items() if k != "artifact_sha256"})
    checkpoints = sorted(directory.glob("optimizer_checkpoint_*.json"))
    assert len(checkpoints) == 5
    assert read_json(checkpoints[0])["schema"] == "mas-promptbench-native-optimizer-checkpoint"
    curve = read_jsonl(directory / "learning_curve.jsonl")
    grid = [row["rollout_grid"] for row in curve if row["rollout_grid"] is not None]
    assert grid == list(range(0, 601, 10)) and curve[-1]["post_stop_carried_forward"] is True
    assert max(row["charged_rollouts"] for row in curve) == budget.charged


def test_manager_first_role_order_and_message_normalization(monkeypatch):
    cell = tavo_cell(topology="centralized")
    monkeypatch.setitem(protocol_runner._ROLE_ORDERS, ("hotpotqa", "centralized"), ("worker_a", "manager", "worker_b"))
    bundle = PromptBundle(roles={"worker_a": "a", "manager": "m", "worker_b": "b"})
    assert _ordered_roles(cell, bundle) == ["manager", "worker_a", "worker_b"]
    record = RunRecord(
        cell_id="c",
        example_id="e",
        status="success",
        score=1.0,
        messages=[
            {"type": "human", "content": "q"},
            {"name": "writer", "content": {"a": 1}, "tool_calls": [{"name": "delegate_to_x", "args": {}}, "junk"]},
            {"role": "ai", "content": None},
        ],
    )
    messages = _messages(record)
    assert [m["source"] for m in messages] == ["user", "writer", "assistant"]
    assert messages[1]["content"] == '{"a": 1}' and messages[1]["tool_calls"] == [{"name": "delegate_to_x", "args": {}}]


def test_protocol_job_runs_all_phases(tmp_path):
    out = tmp_path / "tavo"
    hooks = run.JobHooks(
        load_task_data=fake_task_data,
        build_runtime=fake_runtime,
        build_scorer=lambda cell: FakeScorer(),
        configure_environment=False,
        optimizer_kwargs={"reflection_client": FakeReflection()},
    )
    argv = [
        "--method",
        "tavo",
        "--dataset",
        "hotpotqa",
        "--topology",
        "sequential",
        "--model",
        "qwen",
        "--seed",
        "0",
        "--out",
        str(out),
        "--quiet",
    ]
    assert run.main(argv, hooks=hooks) == 0
    optimization = verify_sealed(read_json(out / "optimization.json"))
    assert optimization["status"] == "completed" and optimization["budget"]["charged"] == 54
    assert optimization["stop_reason"] == "max_outer_rounds" and optimization["usage"]["reflection"]["model_calls"] > 0
    selection = verify_sealed(read_json(out / "selection.json"))
    assert selection["selected_candidate"] and selection["incumbent_validation_score"] == 1.0
    result = verify_sealed(read_json(out / "result.json"))
    assert result["test"]["baseline_mean"] == 0.5 and result["test"]["deployed_mean"] == 1.0
    for path in out.rglob("*"):
        if path.is_file():
            text = path.read_text(encoding="utf-8", errors="ignore")
            assert str(tmp_path) not in text and str(Path.home()) not in text and str(REPO_ROOT) not in text
