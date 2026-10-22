"""Budget ledger, charging and infrastructure-retry rules of the protocol runner."""

from __future__ import annotations

import pytest

from ..budget import BudgetLedger, closed_ledger
from ..errors import NativeInfrastructureExhausted, OptimizerInfrastructureFailure, RunnerContractError
from ..rollouts import BudgetedRunner
from ..session import RunnerSession
from .fakes import FakeAdapter, fake_cell, fake_runner


def test_ledger_charges_usable_outcomes_only_and_never_overshoots():
    ledger = BudgetLedger(5)
    first = ledger.reserve(3)
    ledger.commit(first, ["success", "semantic_failure", "infrastructure_failure"])
    assert ledger.snapshot()["charged"] == 2 and ledger.snapshot()["infrastructure_failures"] == 1
    trimmed = ledger.reserve(10)
    assert trimmed.size == 3
    ledger.commit_prefix(trimmed, ["success"])
    assert ledger.snapshot()["reserved"] == 0 and ledger.charged == 3
    last = ledger.reserve(2)
    ledger.commit(last, ["success", "success"])
    with pytest.raises(RuntimeError, match="optimization budget exhausted"):
        ledger.reserve(1)
    assert ledger.charged == 5 and closed_ledger(ledger.snapshot())


def test_ledger_maximum_is_the_protocol_budget():
    with pytest.raises(ValueError):
        BudgetLedger(601)


def test_success_and_semantic_failures_are_charged():
    runner, budget, data = fake_runner()
    seed = runner.seed_bundle
    good = runner.run(data.row("tr0"), seed, 11)
    wrong = runner.run(data.row("tr1"), seed, 12)
    assert (good.status, good.score) == ("success", 1.0)
    assert (wrong.status, wrong.score) == ("semantic_failure", 0.0)
    assert budget.snapshot()["charged"] == 2 and budget.snapshot()["attempted"] == 2


@pytest.mark.parametrize(
    "behavior,stage",
    [
        ("connection", "task_runtime"),
        ("bad_request", "runtime_unclassified"),
        ("sdk_pre_observation", "openai_agents_sdk"),
        ("transport_text", "task_runtime"),
        ("zero_calls", "runtime_contract"),
        ("scorer_crash", "scorer"),
    ],
)
def test_infrastructure_failures_retry_with_same_seed_and_are_not_charged(behavior, stage):
    runner, budget, data = fake_runner()
    FakeAdapter.SCRIPT["tr0"] = [behavior, behavior]
    record = runner.run(data.row("tr0"), runner.seed_bundle, 4242)
    assert record.usable and record.metadata["execution_attempts"] == 3
    assert record.metadata["infrastructure_failures"] == 2
    assert [call["seed"] for call in FakeAdapter.CALLS] == [4242, 4242, 4242]
    snap = budget.snapshot()
    assert (snap["charged"], snap["attempted"], snap["infrastructure_failures"], snap["retries"]) == (1, 3, 2, 2)


def test_exhausted_retries_return_unusable_record_uncharged():
    runner, budget, data = fake_runner()
    FakeAdapter.SCRIPT["tr0"] = ["connection"] * 3
    record = runner.run(data.row("tr0"), runner.seed_bundle, 7)
    assert record.status == "infrastructure_failure" and not record.usable
    assert record.metadata["failure_stage"] == "task_runtime" and record.metadata["execution_attempts"] == 3
    assert budget.snapshot()["charged"] == 0 and budget.snapshot()["attempted"] == 3
    with pytest.raises(OptimizerInfrastructureFailure):
        FakeAdapter.SCRIPT["tr1"] = ["connection"] * 3
        BudgetedRunner(cell=runner.cell, runner=runner, budget=budget).run(data.row("tr1"), runner.seed_bundle, 8)
    session = RunnerSession(cell=runner.cell, runner=runner, budget=budget, seed_bundle=runner.seed_bundle)
    FakeAdapter.SCRIPT["tr2"] = ["connection"] * 3
    with pytest.raises(NativeInfrastructureExhausted):
        session.run_record(data.row("tr2"))
    assert budget.snapshot()["charged"] == 0 and closed_ledger(budget.snapshot())


def test_batch_is_trimmed_at_the_cap():
    runner, budget, data = fake_runner(fake_cell(budget=4))
    rows = [data.row(item) for item in data.split_ids["train"]]
    batch = runner.run_batch(rows, runner.seed_bundle, list(range(len(rows))))
    assert (batch.requested, batch.scheduled, batch.cap_truncated) == (6, 4, 2)
    assert budget.charged == 4 and budget.reserved == 0
    with pytest.raises(RuntimeError, match="optimization budget exhausted"):
        runner.run(rows[0], runner.seed_bundle, 1)


def test_runner_rejects_test_rows_tampered_rows_and_bad_bundles_before_charging():
    runner, budget, data = fake_runner()
    with pytest.raises(RunnerContractError, match="outside the allowed splits"):
        runner.run({"id": "te0", "question": "question te0"}, runner.seed_bundle, 1)
    tampered = dict(data.row("tr0"), answer="leak")
    with pytest.raises(RunnerContractError, match="canonical row"):
        runner.run(tampered, runner.seed_bundle, 1)
    from ..schema import PromptBundle

    with pytest.raises(RunnerContractError, match="prompt roles differ"):
        runner.run(data.row("tr0"), PromptBundle(roles={"writer": "x"}), 1)
    with pytest.raises(RunnerContractError, match="no execution hook"):
        runner.run(
            data.row("tr0"),
            PromptBundle(roles=dict(runner.seed_bundle.roles), metadata={"optimizer_control": {"active_workers": []}}),
            1,
        )
    assert budget.snapshot()["attempted"] == 0 and FakeAdapter.CALLS == []


def test_rollouts_use_optimization_decoding_and_hide_scorer_fields():
    cell = fake_cell(task="bfcl")
    from .fakes import protocol_runner

    runner, _, data = protocol_runner(cell)
    runner.run(data.row("tr0"), runner.seed_bundle, 99)
    call = FakeAdapter.CALLS[-1]
    assert call["temperature"] == 0.2 and call["model"] == cell.task_model and call["seed"] == 99
    assert "ground_truth" not in call["keys"]


def test_session_seeds_are_deterministic_and_charged_once():
    runner_a, budget_a, data = fake_runner()
    runner_b, _, _ = fake_runner()
    seeds = []
    for runner in (runner_a, runner_b):
        session = RunnerSession(cell=runner.cell, runner=runner, budget=runner.budget, seed_bundle=runner.seed_bundle)
        session.set_context(iteration=3, event="probe")
        seeds.append([session.run_record(data.row(item)).request_seed for item in ("tr0", "tr1", "tr0")])
    assert seeds[0] == seeds[1] and seeds[0][0] != seeds[0][2]
    assert budget_a.charged == 3
