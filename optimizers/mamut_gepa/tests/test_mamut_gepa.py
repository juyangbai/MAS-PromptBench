"""MAMUT-GEPA on the protocol fakes: scope, budget cap, seeds, settings and artifacts."""

from __future__ import annotations

import functools
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from optimizers.protocol import REPO_ROOT, methods, run
from optimizers.protocol.artifacts import read_json
from optimizers.protocol.budget import closed_ledger
from optimizers.protocol.cells import LLAMA_MODEL, QWEN_MODEL, required_cells
from optimizers.protocol.errors import UnsupportedBaselineCell
from optimizers.protocol.run import JobHooks
from optimizers.protocol.schema import content_hash, verify_sealed
from optimizers.protocol.seeding import reflection_seed
from optimizers.protocol.tests.fakes import FakeAdapter, FakeScorer, fake_runtime, fake_task_data

from .. import integration
from ..integration import (
    REFLECTION_TEMPLATE,
    CommonMAMUTGEPAAdapter,
    MAMUTGEPAOptimizer,
    extract_instruction,
    require_gepa_api,
)
from .fakes import MARK, FakeReflection, SharedPromptAdapter, grid_cell, grid_runner


def _optimize(cell, *, backend=None, **kwargs):
    runner, budget, data = grid_runner(cell)
    backend = backend or FakeReflection()
    optimizer = MAMUTGEPAOptimizer(seed_bundle=runner.seed_bundle, reflection=backend, **kwargs)
    artifact = optimizer.optimize(cell, runner, budget, list(data.rows("train")), list(data.rows("validation")))
    return artifact, budget, backend


def _assert_artifact_contract(artifact, budget, cell):
    snapshot = budget.snapshot()
    payload = artifact.to_dict()
    assert payload["method"] == "mamut_gepa" and payload["cell_id"] == cell.cell_id
    assert payload["budget_snapshot"] == snapshot and snapshot["reserved"] == 0 and closed_ledger(snapshot)
    assert len(artifact.records) == len(artifact.request_ledger) == snapshot["charged"] <= cell.budget
    assert all(record.request_seed is not None for record in artifact.records)
    grids = [row["rollout_grid"] for row in artifact.learning_curve if row["rollout_grid"] is not None]
    assert grids == list(range(0, cell.budget + 1, 10))
    for event in artifact.events:
        assert event["event_id"] == content_hash({k: v for k, v in event.items() if k != "event_id"})
    for checkpoint in artifact.checkpoints:
        body = {k: v for k, v in checkpoint.items() if k != "checkpoint_id"}
        assert checkpoint["checkpoint_id"] == content_hash(body)
    json.dumps(payload)


def test_scope_is_the_table_6_grid_and_other_cells_are_rejected_before_any_rollout():
    cells = required_cells("mamut_gepa")
    assert len(cells) == 12 and {cell.source_tables for cell in cells} == {(6,)}
    assert {cell.task for cell in cells} == {"hotpotqa", "lcb", "bfcl"}
    assert {cell.topology for cell in cells} == {"independent", "sequential", "centralized", "decentralized"}
    assert {(cell.framework, cell.communication, cell.team_size, cell.task_model) for cell in cells} == {
        ("langgraph", "freeform", 4, QWEN_MODEL)
    }
    assert methods.METHODS["mamut_gepa"] == "optimizers.mamut_gepa.integration:MAMUTGEPAOptimizer"
    assert "seed_bundle" in inspect.signature(MAMUTGEPAOptimizer).parameters
    out_of_scope = [
        grid_cell(framework="crewai"),  # Table 2 native framework
        grid_cell(task="math"),
        grid_cell(team_size=8),
        grid_cell(task_model=LLAMA_MODEL),
    ]
    for cell in out_of_scope:
        runner, budget, data = grid_runner(cell)
        with pytest.raises(UnsupportedBaselineCell):
            MAMUTGEPAOptimizer(seed_bundle=runner.seed_bundle, reflection=FakeReflection()).optimize(
                cell, runner, budget, list(data.rows("train")), list(data.rows("validation"))
            )
        assert budget.snapshot()["attempted"] == 0
    assert FakeAdapter.CALLS == []


def test_full_budget_never_overshoots_and_improves_the_writer():
    cell = grid_cell()
    artifact, budget, backend = _optimize(cell)
    _assert_artifact_contract(artifact, budget, cell)
    assert 0 < budget.charged <= 600
    assert artifact.stop_reason in {"metric_call_cap", "native_gepa_stop"}
    assert MARK in artifact.incumbent_bundle.roles["writer"]
    metadata = artifact.metadata
    assert metadata["accepted"] is True
    assert metadata["best_native_validation"] >= metadata["baseline_native_validation"]
    assert metadata["native_total_metric_calls"] == budget.charged
    assert metadata["proposal_calls"] == len(backend.requests) >= 1
    assert artifact.native_iterations >= 1 and len(artifact.events) >= 2


@pytest.mark.parametrize("limit,ledger", [(15, 600), (600, 40)])
def test_metric_call_cap_is_exact(limit, ledger):
    cell = grid_cell(budget=ledger)
    artifact, budget, _ = _optimize(cell, metric_call_limit=limit)
    cap = min(limit, ledger)
    assert artifact.metadata["metric_call_limit"] == cap
    assert budget.charged <= cap and artifact.stop_reason == "metric_call_cap"
    _assert_artifact_contract(artifact, budget, cell)


def test_gepa_receives_the_retained_settings(monkeypatch):
    seen = {}
    real = integration.gepa.optimize

    @functools.wraps(real)
    def spy(**kwargs):
        seen.update(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(integration.gepa, "optimize", spy)
    cell = grid_cell(seed=2)
    artifact, budget, _ = _optimize(cell)
    assert seen["seed_candidate"] == dict(artifact.seed_bundle.roles)
    assert {
        key: seen[key]
        for key in (
            "candidate_selection_strategy",
            "module_selector",
            "reflection_minibatch_size",
            "use_merge",
            "max_metric_calls",
            "seed",
            "cache_evaluation",
            "raise_on_exception",
            "display_progress_bar",
        )
    } == {
        "candidate_selection_strategy": "pareto",
        "module_selector": "round_robin",
        "reflection_minibatch_size": 3,
        "use_merge": True,
        "max_metric_calls": 600,
        "seed": 2,
        "cache_evaluation": False,
        "raise_on_exception": True,
        "display_progress_bar": False,
    }
    assert len(seen["callbacks"]) == 1 and isinstance(seen["adapter"], CommonMAMUTGEPAAdapter)
    assert [row["id"] for row in seen["trainset"]] == [f"tr{i}" for i in range(6)]
    assert [row["id"] for row in seen["valset"]] == [f"va{i}" for i in range(4)]
    assert artifact.metadata["gepa_seed"] == 2 and budget.charged <= 600


def test_reflection_requests_follow_the_template_and_common_policy():
    cell = grid_cell()
    artifact, _, backend = _optimize(cell)
    assert backend.requests
    for request in backend.requests:
        assert request["phase"] == "joint_gepa_reflection" and request["role"] in {"planner", "writer"}
        assert (request["temperature"], request["top_p"], request["max_output_tokens"], request["thinking"]) == (
            0.7,
            1.0,
            48000,
            True,
        )
        assert request["system"] is None
        prompt = request["prompt"]
        assert "===BEGIN NEW INSTRUCTION===" in prompt and "===END NEW INSTRUCTION===" in prompt
        assert "source_tagged_trajectory" in prompt and "Global CommonRunner score=" in prompt
    first = backend.requests[0]
    assert first["request_seed"] == reflection_seed(
        cell, phase="mamut_gepa_reflection", iteration=1, role=first["role"], prompt=first["prompt"]
    )
    assert "Plan the answer." in first["prompt"] or "Write the final answer." in first["prompt"]
    assert artifact.metadata["reflection"]["usage"]["model_calls"] == len(backend.requests)


def test_optimizer_seed_fixes_the_trajectory():
    def trace(seed):
        artifact, budget, backend = _optimize(grid_cell(seed=seed))
        return (
            [record.request_seed for record in artifact.records],
            artifact.incumbent_bundle.digest,
            [request["request_seed"] for request in backend.requests],
            budget.charged,
        )

    first, again, other = trace(0), trace(0), trace(1)
    assert first == again
    assert first[0] != other[0] and first[2] != other[2]


def test_extract_instruction_fallback_order():
    begin, end = "===BEGIN NEW INSTRUCTION===", "===END NEW INSTRUCTION==="
    assert extract_instruction(f"```\nfence\n```\n{begin}\n delimited \n{end}", "old") == "delimited"
    assert extract_instruction("noise ```text\nfirst\n``` more ```\nlast\n```", "old") == "last"
    assert extract_instruction(f"thinking... {begin}\nunterminated tail", "old") == "unterminated tail"
    assert extract_instruction(f"{begin}\n  \n{end}", "old") == "old"
    assert extract_instruction("no instruction at all", "old") == "old"
    assert "<curr_instructions>" in REFLECTION_TEMPLATE and "<inputs_outputs_feedback>" in REFLECTION_TEMPLATE


def test_acceptance_keeps_the_seed_unless_gepa_found_a_different_noninferior_bundle(monkeypatch):
    cell = grid_cell()
    runner, _, _ = grid_runner(cell)
    seed = dict(runner.seed_bundle.roles)
    better = {**seed, "writer": f"{MARK} writer"}
    cases = [
        (
            SimpleNamespace(
                best_candidate=better, val_aggregate_scores=[0.5, 0.75], candidates=[seed, better], total_metric_calls=0
            ),
            better,
            True,
        ),
        (
            SimpleNamespace(
                best_candidate=better, val_aggregate_scores=[0.5, 0.5], candidates=[seed, better], total_metric_calls=0
            ),
            better,
            True,
        ),
        (
            SimpleNamespace(
                best_candidate=dict(seed), val_aggregate_scores=[0.5], candidates=[seed], total_metric_calls=0
            ),
            seed,
            False,
        ),
    ]
    real = integration.gepa.optimize
    for result, expected, accepted in cases:
        monkeypatch.setattr(integration.gepa, "optimize", functools.wraps(real)(lambda result=result, **kwargs: result))
        artifact, budget, _ = _optimize(cell)
        assert dict(artifact.incumbent_bundle.roles) == expected and artifact.metadata["accepted"] is accepted
        assert artifact.stop_reason == "native_gepa_stop" and budget.charged == 0


def test_required_gepa_api_is_checked_before_any_rollout(monkeypatch):
    assert require_gepa_api() == integration.GEPA_VERSION
    monkeypatch.setattr(integration.gepa, "optimize", lambda seed_candidate, trainset, valset, adapter: None)
    cell = grid_cell()
    runner, budget, data = grid_runner(cell)
    with pytest.raises(integration.NativeIntegrationError, match="gepa==0.0.27"):
        MAMUTGEPAOptimizer(seed_bundle=runner.seed_bundle, reflection=FakeReflection()).optimize(
            cell, runner, budget, list(data.rows("train")), list(data.rows("validation"))
        )
    assert budget.snapshot()["attempted"] == 0


def test_shared_replica_prompt_receives_every_native_speakers_evidence():
    cell = grid_cell(topology="independent")
    runner, budget, data = grid_runner(cell, adapter_class=SharedPromptAdapter)
    assert tuple(runner.seed_bundle.roles) == ("writer",)
    backend = FakeReflection()
    artifact = MAMUTGEPAOptimizer(seed_bundle=runner.seed_bundle, reflection=backend, metric_call_limit=40).optimize(
        cell, runner, budget, list(data.rows("train")), list(data.rows("validation"))
    )
    assert backend.requests and {request["role"] for request in backend.requests} == {"writer"}
    prompt = backend.requests[0]["prompt"]
    assert '"native_source": "planner"' in prompt and '"native_source": "writer"' in prompt
    assert '"component": "planner"' not in prompt
    assert MARK in artifact.incumbent_bundle.roles["writer"] and budget.charged <= 40


def test_protocol_job_runs_end_to_end_with_anonymous_artifacts(tmp_path):
    backend = FakeReflection()
    hooks = JobHooks(
        load_task_data=fake_task_data,
        build_runtime=fake_runtime,
        build_scorer=lambda cell: FakeScorer(),
        configure_environment=False,
        optimizer_kwargs={"reflection": backend},
    )
    out = tmp_path / "mamut"
    argv = [
        "--method",
        "mamut_gepa",
        "--dataset",
        "hotpotqa",
        "--topology",
        "sequential",
        "--model",
        "qwen",
        "--seed",
        "1",
        "--out",
        str(out),
        "--quiet",
    ]
    assert run.main(argv, hooks=hooks) == 0
    optimization = verify_sealed(read_json(out / "optimization.json"))
    assert optimization["status"] == "completed" and optimization["budget"]["charged"] <= 600
    assert optimization["stop_reason"] in {"metric_call_cap", "native_gepa_stop"}
    assert optimization["usage"]["reflection"]["model_calls"] == len(backend.requests)
    result = verify_sealed(read_json(out / "result.json"))
    assert result["protocol_conformant"] is True and result["test"]["valid_for_aggregation"]
    payload = read_json(out / "optimization" / "optimizer_result.json")
    assert payload["metadata"]["native_state_dir"] == "optimization/native"
    assert (out / "optimization" / "native").is_dir()
    for path in out.rglob("*"):
        if path.is_file():
            text = path.read_text(encoding="utf-8", errors="ignore")
            assert str(tmp_path) not in text and str(Path.home()) not in text and str(REPO_ROOT) not in text
