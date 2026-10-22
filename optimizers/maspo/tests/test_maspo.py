"""MASPO on the protocol fakes: scope, role wiring, budget, seeds, rewards and artifacts."""

from __future__ import annotations

import inspect
import json
import random
from pathlib import Path

import pytest

from optimizers.protocol import REPO_ROOT, methods, run
from optimizers.protocol.artifacts import read_json
from optimizers.protocol.budget import closed_ledger
from optimizers.protocol.cells import LLAMA_MODEL, required_cells
from optimizers.protocol.errors import NativeIntegrationError, UnsupportedBaselineCell
from optimizers.protocol.run import JobHooks
from optimizers.protocol.runner import AdapterRuntime
from optimizers.protocol.schema import content_hash, verify_sealed
from optimizers.protocol.seeding import reflection_seed
from optimizers.protocol.tests.fakes import FakeScorer, fake_task_data

from .. import search
from ..integration import MASPOOptimizer, _MASPOReflectionClient
from ..upstream import INTERMEDIATE_COMPARE_TEMPLATE, PROMPT_OPTIMIZE_TEMPLATE, TaskType
from .fakes import CENTRALIZED, MARK, SEQUENTIAL, SHARED, FakeModel, RoleEchoAdapter, grid_cell, grid_runner


@pytest.fixture(autouse=True)
def _reset_calls():
    RoleEchoAdapter.CALLS = []
    yield
    RoleEchoAdapter.CALLS = []


def _optimize(cell, adapter, *, backend=None, **kwargs):
    runner, budget, data = grid_runner(cell, adapter)
    backend = backend or FakeModel()
    artifact = MASPOOptimizer(seed_bundle=runner.seed_bundle, reflection=backend, **kwargs).optimize(
        cell, runner, budget, list(data.rows("train")), list(data.rows("validation"))
    )
    return artifact, budget, backend, data


def _assert_artifact_contract(artifact, budget, cell):
    snapshot = budget.snapshot()
    payload = artifact.to_dict()
    assert payload["method"] == "maspo" and payload["cell_id"] == cell.cell_id
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


# Scope and settings
def test_scope_is_tables_2_to_7_plus_llama_and_other_cells_are_rejected_before_any_rollout():
    cells = required_cells("maspo")
    tables = {table for cell in cells for table in cell.source_tables}
    assert len(cells) == 135 and tables == {2, 3, 4, 5, 6, 7}
    assert "single" not in {cell.topology for cell in cells}
    assert {cell.framework for cell in cells} == {"langgraph", "crewai", "autogen", "openai_agents"}
    assert len([cell for cell in cells if cell.task_model == LLAMA_MODEL]) == 12
    assert methods.METHODS["maspo"] == "optimizers.maspo.integration:MASPOOptimizer"
    assert "seed_bundle" in inspect.signature(MASPOOptimizer).parameters
    out_of_scope = [
        grid_cell(topology="single", team_size=1),
        grid_cell(topology="centralized", task="math", team_size=8),
        grid_cell(topology="centralized", task="math", task_model=LLAMA_MODEL),
    ]
    for cell in out_of_scope:
        runner, budget, data = grid_runner(cell, SHARED if cell.topology == "single" else CENTRALIZED)
        with pytest.raises(UnsupportedBaselineCell):
            MASPOOptimizer(seed_bundle=runner.seed_bundle, reflection=FakeModel()).optimize(
                cell, runner, budget, list(data.rows("train")), []
            )
        assert budget.snapshot()["attempted"] == 0
    assert RoleEchoAdapter.CALLS == []


def test_retained_settings():
    settings = MASPOOptimizer(reflection=FakeModel()).settings
    assert (
        settings.beam_width,
        settings.offspring,
        settings.minibatch,
        settings.rounds_per_turn,
        settings.maximum_search_depth,
    ) == (2, 2, 10, 3, 9)
    assert search.LOOKAHEAD_WEIGHTS == (0.4, 0.4, 0.2)
    assert (search.FALLBACK_WEIGHTS, search.SCORE_OFFSET) == ((0.7, 0.3), 0.5)
    assert (search.MIS_CAP, search.MIS_INJECT) == (3, 5)
    assert (search.PROPOSAL_TEMPERATURE, search.JUDGE_TEMPERATURE) == (0.7, 0.0)
    assert search.regime_mode(3) == "maspo-full" and search.regime_mode(1) == "maspo-budget-adapted"
    with pytest.raises(ValueError):
        MASPOOptimizer(reflection=FakeModel(), beam_width=0)
    assert len(required_cells("maspo")) == 135


# Role wiring and the full budget
def test_centralized_updates_workers_first_then_the_manager_and_spends_exactly_the_budget():
    cell = grid_cell(topology="centralized")
    artifact, budget, backend, _ = _optimize(cell, CENTRALIZED)
    metadata = artifact.metadata
    workers = ["reasoner_worker", "retriever_worker"]  # bundle order
    assert metadata["role_order"] == [*workers, "manager"] and metadata["terminal_role"] == "manager"
    assert metadata["successors"] == {"reasoner_worker": "manager", "retriever_worker": "manager", "manager": None}
    assert metadata["predecessors"] == {
        "manager": workers,
        "reasoner_worker": ["manager"],
        "retriever_worker": ["manager"],
    }
    first_turn = [step["role"] for step in metadata["steps"][:9]]
    assert first_turn == ["reasoner_worker"] * 3 + ["retriever_worker"] * 3 + ["manager"] * 3
    assert any(step.get("refresh") for step in metadata["steps"])
    assert budget.charged == 600 and artifact.stop_reason == "budget"
    _assert_artifact_contract(artifact, budget, cell)
    assert all(MARK in prompt for prompt in artifact.incumbent_bundle.roles.values())
    assert metadata["mode"] == "maspo-full" and metadata["optimizer_rng_seed"] == 0
    assert metadata["judge_calls"] > 0 and metadata["proposal_calls"] > 0
    phases = {request["phase"] for request in backend.requests}
    assert phases == {"proposal", "pairwise_judge"}
    for request in backend.requests:
        expected = 0.7 if request["phase"] == "proposal" else 0.0
        assert (request["temperature"], request["top_p"], request["max_output_tokens"], request["thinking"]) == (
            expected,
            1.0,
            48000,
            True,
        )
    proposal = next(request for request in backend.requests if request["phase"] == "proposal")
    assert proposal["prompt"].startswith("\nYou are optimizing one role's prompt in a centralized manager/worker")
    assert "PROTECTED FINAL OUTPUT CONTRACT" in proposal["prompt"]


def test_sequential_follows_the_native_stage_order_not_the_sorted_bundle():
    cell = grid_cell(topology="sequential")
    artifact, budget, backend, _ = _optimize(cell, SEQUENTIAL, maximum_search_depth=2)
    metadata = artifact.metadata
    stages = ["zeta_reader", "middle_planner", "alpha_writer"]
    assert list(artifact.seed_bundle.roles) == sorted(stages)
    assert metadata["role_order"] == stages and metadata["terminal_role"] == "alpha_writer"
    assert metadata["successors"] == {
        "zeta_reader": "middle_planner",
        "middle_planner": "alpha_writer",
        "alpha_writer": None,
    }
    assert metadata["predecessors"] == {
        "zeta_reader": [],
        "middle_planner": ["zeta_reader"],
        "alpha_writer": ["middle_planner"],
    }
    roles = [step["role"] for step in metadata["steps"] if not step.get("refresh")]
    assert roles == ["zeta_reader"] * 2 + ["middle_planner"] * 2 + ["alpha_writer"] * 2
    assert artifact.stop_reason == "maximum_search_depth" and budget.charged < 600
    proposal = next(request for request in backend.requests if request["phase"] == "proposal")
    assert "one role's prompt in a sequential multi-agent system" in proposal["prompt"]
    assert "intermediate native pipeline stage" in proposal["prompt"]
    _assert_artifact_contract(artifact, budget, cell)


def test_shared_replica_prompt_uses_global_credit_only():
    cell = grid_cell(topology="independent")
    artifact, budget, backend, _ = _optimize(cell, SHARED)
    metadata = artifact.metadata
    assert metadata["role_order"] == ["solver"] and metadata["successors"] == {"solver": None}
    assert artifact.stop_reason == "maximum_search_depth"
    assert sum(1 for step in metadata["steps"] if not step.get("refresh")) == 9
    assert metadata["judge_calls"] == 0 and {request["phase"] for request in backend.requests} == {"proposal"}
    assert "All native replicas share this prompt" in backend.requests[0]["prompt"]
    assert MARK in artifact.incumbent_bundle.roles["solver"]
    _assert_artifact_contract(artifact, budget, cell)


def test_unknown_centralized_manager_fails_closed_before_any_rollout():
    cell = grid_cell(topology="centralized")
    from .fakes import adapter_class

    runner, budget, data = grid_runner(cell, adapter_class(("lead", "worker"), "lead"))
    with pytest.raises(NativeIntegrationError, match="exact centralized manager"):
        MASPOOptimizer(seed_bundle=runner.seed_bundle, reflection=FakeModel()).optimize(
            cell, runner, budget, list(data.rows("train")), []
        )
    assert budget.snapshot()["attempted"] == 0


@pytest.mark.parametrize("ledger", [37, 125])
def test_truncated_budgets_are_never_exceeded(ledger):
    cell = grid_cell(topology="centralized", budget=ledger)
    artifact, budget, _, _ = _optimize(cell, CENTRALIZED)
    assert budget.charged == ledger and artifact.stop_reason == "budget"
    _assert_artifact_contract(artifact, budget, cell)


# Seeds
def test_optimizer_seed_drives_minibatches_and_request_seeds():
    def trace(seed):
        artifact, budget, backend, data = _optimize(
            grid_cell(topology="centralized", seed=seed, budget=200), CENTRALIZED
        )
        minibatches = [ids for step in artifact.metadata["steps"] for ids in step.get("minibatch_ids", [])]
        return (
            minibatches,
            sorted(record.request_seed for record in artifact.records),
            sorted(request["request_seed"] for request in backend.requests),
            artifact.incumbent_bundle.digest,
            budget.charged,
            data,
        )

    first, again, other = trace(0), trace(0), trace(1)
    assert first[:5] == again[:5]
    assert first[0] != other[0] and first[1] != other[1]
    rows = list(first[5].rows("train"))
    for seed, result in ((0, first), (1, other)):
        expected = [row["id"] for row in random.Random(seed).sample(rows, len(rows))]
        assert result[0][0] == expected  # first worker's first minibatch: no injected cases yet


def test_reflection_client_seeds_and_phases():
    cell = grid_cell(topology="centralized")
    backend = FakeModel()
    client = _MASPOReflectionClient(cell, backend)
    for _ in range(2):
        client.complete("<reference_prompt>\np\n</reference_prompt>", temperature=0.7, max_tokens=16384)
    client.complete("Output A:\nx\nOutput B:\ny", temperature=0.0, max_tokens=1024)
    first, second, judge = backend.requests
    assert first["request_seed"] == reflection_seed(
        cell, phase="maspo_proposal", iteration=0, role="proposal", prompt=first["prompt"]
    )
    assert second["request_seed"] != first["request_seed"]
    assert (judge["phase"], judge["role"], judge["max_output_tokens"]) == ("pairwise_judge", "pairwise_judge", 48000)
    assert client.usage["n_calls"] == 3


# Rewards, misalignment and the judge
class _StubJudge:
    """Candidate output wins locally when it contains ``win``."""

    def compare_intermediate(self, role, question, output_cand, output_base):
        return "win" in output_cand


def _run(score, role_text, successor_text=None):
    messages = [{"source": "worker", "content": role_text}]
    if successor_text is not None:
        messages.append({"source": "manager", "content": successor_text})
    return {"question": "q", "task_context": "q", "score": score, "messages": messages}


def test_evaluate_candidate_rewards_and_misalignment_mining():
    judge = _StubJudge()
    base = [_run(0.0, "x", "m"), _run(1.0, "x", "m"), _run(0.0, "x", "m"), _run(1.0, "x", "m")]
    cand = [_run(1.0, "win", "m"), _run(1.0, "win", "m win"), _run(0.0, "x", "m"), _run(0.0, "win", "m")]
    terminal = search.evaluate_candidate(judge, "manager", True, base, cand, 2)
    assert terminal["rate_global"] == 0.5 and terminal["score"] == 0.0

    mined: list = []
    items = ["i0", "i1", "i2", "i3"]
    full = search.evaluate_candidate(
        judge, "worker", False, base, cand, 2, successor_role="manager", mis_out=mined, items=items
    )
    # local: wins on 0, 1, 3, neutral 0.5 on 2; successor: 1 win, 3 neutral pairs.
    assert full["rate_local"] == 3.5 / 4 and full["rate_next"] == (1.0 + 0.5 * 3) / 4 and full["rate_global"] == 0.5
    assert full["score"] == pytest.approx(0.4 * 3.5 / 4 + 0.4 * 2.5 / 4 + 0.2 * 0.5 - 0.5)
    assert mined == [(2, "i3")]  # local win but global loss

    no_successor_output = [_run(c["score"], c["messages"][0]["content"]) for c in cand]
    fallback = search.evaluate_candidate(judge, "worker", False, base, no_successor_output, 2, successor_role="manager")
    assert fallback["rate_next"] is None
    assert fallback["score"] == pytest.approx(0.5 * 0.3 + 3.5 / 4 * 0.7 - 0.5)

    class LosingSuccessor(_StubJudge):
        def compare_intermediate(self, role, question, output_cand, output_base):
            return role != "manager" and "win" in output_cand

    mined = []
    changed = [_run(0.0, "win", "m2"), _run(0.0, "win", "m3"), _run(1.0, "win", "m4"), _run(1.0, "x", "m")]
    search.evaluate_candidate(
        LosingSuccessor(), "worker", False, base, changed, 2, successor_role="manager", mis_out=mined, items=items
    )
    assert sorted(mined) == [(0, "i1"), (1, "i0"), (1, "i2")]


def test_minibatch_injects_predecessor_misalignment_cases():
    rows = [{"id": f"tr{i}"} for i in range(12)]
    buffers = {"a": rows[:4], "b": rows[2:7], "c": rows[10:]}
    batch = search._sample_minibatch(rows, 10, random.Random(0), buffers, ["a", "b"])
    ids = [row["id"] for row in batch]
    union = {f"tr{i}" for i in range(7)}
    assert len(ids) == len(set(ids)) == 10
    assert set(ids[:5]) <= union and not set(ids[5:]) & union
    assert search._sample_minibatch(rows, 10, random.Random(0), buffers, []) == random.Random(0).sample(rows, 10)


class _ScriptedReflection:
    def __init__(self, replies):
        self.replies = list(replies)

    def complete(self, prompt, temperature=0.7, max_tokens=4096):
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def test_judge_fails_closed_and_proposals_fall_back_to_the_parent():
    verdicts = ["A", " b ", "<choose>a</choose>", "", "C", "A because", RuntimeError("down")]
    judge = search.Judge(_ScriptedReflection(verdicts), TaskType.CODE, "contract", "hotpotqa")
    results = [judge.compare_intermediate("worker", "q", "x", "y") for _ in verdicts]
    assert results == [True, False, True, False, False, False, False]
    assert judge.n_judge_calls == 6
    judge = search.Judge(
        _ScriptedReflection(["no tags", "<prompt>  </prompt>", "<prompt>new</prompt>"]),
        TaskType.CODE,
        "contract",
        "hotpotqa",
    )
    qa = {"q": {"context": "ctx", "output": "out"}}
    assert [judge.propose_new_prompt("worker", "old", qa) for _ in range(3)] == ["old", "old", "new"]
    assert (judge.n_proposal_calls, judge.n_proposal_extract_failures) == (3, 1)


def test_math_uses_the_upstream_templates():
    seen = []

    class Recorder:
        def complete(self, prompt, temperature=0.7, max_tokens=4096):
            seen.append(prompt)
            return "A" if temperature == 0 else "<prompt>x</prompt>"

    judge = search.Judge(Recorder(), TaskType.MATH, "contract", "math")
    judge.propose_new_prompt("manager", "old", {"q": {"context": "c", "output": "o"}})
    judge.compare_intermediate("manager", "q", "a", "b")
    proposal_head = PROMPT_OPTIMIZE_TEMPLATE[TaskType.MATH].split("{agent_type}")[0]
    compare_head = INTERMEDIATE_COMPARE_TEMPLATE[TaskType.MATH].split("{question}")[0]
    assert seen[0].startswith(proposal_head) and seen[1].startswith(compare_head)
    assert "Problem: q\nOutput A:\na\nOutput B:\nb" in seen[1]


def test_role_descriptions_come_from_the_centralized_catalog():
    assert search.ROLES_CATALOG.is_file()
    assert search.role_description("hotpotqa", "manager") != search._GENERIC_ROLE_DESC
    assert search.role_description("hotpotqa", "manager_r8") == search._role_descriptions("hotpotqa")["manager_r8"]
    assert search.role_description("hotpotqa", "no_such_role") == search._GENERIC_ROLE_DESC
    assert "PROTECTED FINAL OUTPUT CONTRACT" in search.dataset_contract("lcb")


# Protocol job
def test_protocol_job_runs_end_to_end_on_a_llama_cell_with_anonymous_artifacts(tmp_path):
    backend = FakeModel()
    hooks = JobHooks(
        load_task_data=fake_task_data,
        build_runtime=lambda cell: AdapterRuntime(cell, adapter_class=CENTRALIZED, capture_usage=False),
        build_scorer=lambda cell: FakeScorer(),
        configure_environment=False,
        optimizer_kwargs={"reflection": backend},
    )
    out = tmp_path / "maspo"
    argv = [
        "--method",
        "maspo",
        "--dataset",
        "hotpotqa",
        "--topology",
        "centralized",
        "--model",
        "llama",
        "--seed",
        "2",
        "--out",
        str(out),
        "--quiet",
    ]
    assert run.main(argv, hooks=hooks) == 0
    job = verify_sealed(read_json(out / "job.json"))
    assert job["cell"]["task_model"] == LLAMA_MODEL and job["protocol_conformant"] is True
    optimization = verify_sealed(read_json(out / "optimization.json"))
    assert optimization["status"] == "completed" and optimization["budget"]["charged"] == 600
    assert optimization["stop_reason"] == "budget"
    assert optimization["usage"]["reflection"]["model_calls"] == len(backend.requests)
    result = verify_sealed(read_json(out / "result.json"))
    assert result["test"]["valid_for_aggregation"]
    for path in out.rglob("*"):
        if path.is_file():
            text = path.read_text(encoding="utf-8", errors="ignore")
            assert str(tmp_path) not in text and str(Path.home()) not in text and str(REPO_ROOT) not in text
