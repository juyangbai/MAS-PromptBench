"""Paired seeds, uncharged evaluation, deployment selection and test gating."""

from __future__ import annotations

import pytest

from ..errors import EvaluationError, RunnerContractError
from ..evaluation import Evaluator, condition_identity, load_selection, lock_selection
from ..runner import CellExecutor
from ..schema import PromptBundle, RunRecord
from ..seeding import logical_request_seed, paired_evaluation_seed
from ..selection import assert_test_unexposed, select_for_deployment
from .fakes import FakeAdapter, FakeScorer, fake_cell, fake_runtime, fake_task_data


def _records(scores, status="success"):
    return [RunRecord(cell_id="c", example_id=f"v{i}", status=status, score=s) for i, s in enumerate(scores)]


SEED = PromptBundle(roles={"writer": "seed"})
CANDIDATE = PromptBundle(roles={"writer": "candidate"})


def test_logical_request_seeds_are_deterministic_and_offset_by_optimizer_seed():
    args = ("cell", "optimization", 0, "ex", "role", 0)
    assert logical_request_seed(0, *args) == logical_request_seed(0, *args)
    assert len({logical_request_seed(seed, *args) for seed in (0, 1, 2)}) == 3
    assert all(0 <= logical_request_seed(seed, *args) < 2**31 - 1 for seed in (0, 1, 2))
    with pytest.raises(ValueError):
        logical_request_seed(3, *args)
    condition = "a" * 64
    assert paired_evaluation_seed(condition, 1, "test", "x") == paired_evaluation_seed(condition, 1, "test", "x")
    assert paired_evaluation_seed(condition, 1, "test", "x") != paired_evaluation_seed(condition, 2, "test", "x")
    assert paired_evaluation_seed(condition, 1, "test", "x") != paired_evaluation_seed(condition, 1, "validation", "x")


@pytest.mark.parametrize(
    "seed_scores,candidate_scores,selected,reason",
    [
        ([1, 0, 1, 0], [1, 0, 1, 0], False, "validation_tie"),
        ([1, 0, 1, 0], [1, 0, 0, 0], False, "validation_regression"),
        ([1, 0, 1, 0], [1, 1, 1, 0], True, None),
    ],
)
def test_strict_improvement_selection(seed_scores, candidate_scores, selected, reason):
    decision = select_for_deployment(SEED, CANDIDATE, _records(seed_scores), _records(candidate_scores))
    assert decision.selected_candidate is selected and decision.fallback_reason == reason
    assert decision.deployed == (CANDIDATE if selected else SEED)


def test_selection_falls_back_on_invalid_or_unpaired_validation():
    broken = _records([1, 1, 1]) + [RunRecord(cell_id="c", example_id="v3", status="infrastructure_failure")]
    decision = select_for_deployment(SEED, CANDIDATE, _records([0, 0, 0, 0]), broken)
    assert decision.deployed == SEED and decision.fallback_reason.startswith("invalid_record:v3")
    unpaired = select_for_deployment(SEED, CANDIDATE, _records([0, 0]), _records([1, 1, 1]))
    assert unpaired.fallback_reason == "unpaired_validation_ids"
    with pytest.raises(ValueError):
        assert_test_unexposed({"notes": {"test_score": 1.0}})


def _evaluator(tmp_path, data):
    cell = fake_cell()
    runtime = fake_runtime(cell)
    executor = CellExecutor(cell=cell, runtime=runtime, scorer=FakeScorer(), data=data)
    seed = runtime.seed_bundle()
    condition = condition_identity(
        cell, seed, runtime_id=executor.implementation_id, scorer_id=executor.scorer_id, split_hash=data.split_hash
    )
    return Evaluator(executor=executor, condition=condition, directory=tmp_path / "evaluations"), seed


def test_paired_greedy_evaluation_is_uncharged_cached_and_retries_infrastructure(tmp_path):
    data = fake_task_data()
    evaluator, seed = _evaluator(tmp_path, data)
    candidate = PromptBundle(roles={**seed.roles, "writer": seed.roles["writer"] + " GOOD"})
    rows = data.rows("validation")
    FakeAdapter.SCRIPT["va1"] = ["connection"]
    baseline = evaluator.evaluate(seed, phase="baseline", split="validation", evaluation_seed=1, rows=rows)
    calls = len(FakeAdapter.CALLS)
    improved = evaluator.evaluate(candidate, phase="final_validation", split="validation", evaluation_seed=1, rows=rows)
    seeds_by_bundle = {}
    for call in FakeAdapter.CALLS:
        seeds_by_bundle.setdefault(call["writer"], {}).setdefault(call["id"], set()).add(call["seed"])
    assert seeds_by_bundle[seed.roles["writer"]] == seeds_by_bundle[candidate.roles["writer"]]
    assert all(call["temperature"] == 0.0 for call in FakeAdapter.CALLS)
    assert baseline.records[1].metadata["execution_attempts"] == 2 and baseline.valid
    assert baseline.usage["search_budget_charged"] == 0
    assert baseline.mean_score() == 0.5 and improved.mean_score() == 1.0
    again = evaluator.evaluate(seed, phase="baseline", split="validation", evaluation_seed=1, rows=rows)
    assert again.evaluation_id == baseline.evaluation_id and len(FakeAdapter.CALLS) == calls + len(rows)
    other_seed = evaluator.evaluate(seed, phase="baseline", split="validation", evaluation_seed=2, rows=rows)
    assert [r.request_seed for r in other_seed.records] != [r.request_seed for r in baseline.records]


def test_evaluation_resumes_from_committed_prefix(tmp_path):
    data = fake_task_data()
    evaluator, seed = _evaluator(tmp_path, data)
    rows = data.rows("validation")
    FakeAdapter.SCRIPT["va2"] = ["runtime_bug"]
    with pytest.raises(KeyError):
        evaluator.evaluate(seed, phase="baseline", split="validation", evaluation_seed=0, rows=rows)
    done = [call["id"] for call in FakeAdapter.CALLS]
    result = evaluator.evaluate(seed, phase="baseline", split="validation", evaluation_seed=0, rows=rows)
    assert done == ["va0", "va1", "va2"] and [c["id"] for c in FakeAdapter.CALLS[3:]] == ["va2", "va3"]
    assert result.complete and result.valid


def test_test_rows_and_bundles_are_locked_by_selection(tmp_path):
    data = fake_task_data()
    evaluator, seed = _evaluator(tmp_path, data)
    with pytest.raises(RunnerContractError):
        data.rows("test")
    rows = data.rows("validation")
    candidate = PromptBundle(roles={**seed.roles, "writer": seed.roles["writer"] + " GOOD"})
    baseline = evaluator.evaluate(seed, phase="baseline", split="validation", evaluation_seed=0, rows=rows)
    improved = evaluator.evaluate(candidate, phase="final_validation", split="validation", evaluation_seed=0, rows=rows)
    budget = {
        "maximum": 600,
        "charged": 0,
        "reserved": 0,
        "remaining": 600,
        "attempted": 0,
        "successful": 0,
        "semantic_failures": 0,
        "infrastructure_failures": 0,
        "retries": 0,
    }
    envelope = {"status": "completed", "cell": dict(evaluator.cell.identity), "budget": budget}
    path = tmp_path / "selection.json"
    selection = lock_selection(
        path, envelope=envelope, seed_bundle=seed, incumbent_bundle=candidate, baseline=baseline, candidate=improved
    )
    assert selection["selected_candidate"] and load_selection(path)["selection_id"] == selection["selection_id"]
    data.unlock_test(selection)
    test_rows = data.rows("test")
    other = PromptBundle(roles={**seed.roles, "writer": "unlocked"})
    with pytest.raises(EvaluationError, match="not locked"):
        evaluator.evaluate(other, phase="test", split="test", evaluation_seed=0, rows=test_rows, selection=selection)
    with pytest.raises(EvaluationError, match="requires the locked selection"):
        evaluator.evaluate(seed, phase="baseline", split="test", evaluation_seed=0, rows=test_rows)
    deployed = evaluator.evaluate(
        candidate, phase="test", split="test", evaluation_seed=0, rows=test_rows, selection=selection
    )
    assert deployed.mean_score() == 1.0
    failed = dict(envelope, status="failed", failure_kind="infrastructure_invalid")
    fallback = lock_selection(
        tmp_path / "fallback.json",
        envelope=failed,
        seed_bundle=seed,
        incumbent_bundle=candidate,
        baseline=baseline,
        candidate=improved,
    )
    assert fallback["fallback_reason"] == "infrastructure_invalid_optimization"
    assert fallback["deployed_bundle"]["bundle_sha256"] == seed.digest
    with pytest.raises(EvaluationError):
        lock_selection(
            tmp_path / "bad.json",
            envelope=dict(envelope, status="failed", failure_kind="program_failure"),
            seed_bundle=seed,
            incumbent_bundle=candidate,
            baseline=baseline,
            candidate=improved,
        )
