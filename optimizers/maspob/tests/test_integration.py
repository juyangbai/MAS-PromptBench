"""MASPOBOptimizer on the protocol runner: grid scope, dependency gating and the full bandit lifecycle."""

from __future__ import annotations

import json
import socket
import sys
from pathlib import Path

import pytest

from optimizers.protocol import REPO_ROOT, methods, run
from optimizers.protocol.artifacts import read_json
from optimizers.protocol.budget import closed_ledger
from optimizers.protocol.cells import required_cells
from optimizers.protocol.errors import UnsupportedBaselineCell
from optimizers.protocol.runner import remember_role_order
from optimizers.protocol.schema import CellSpec, PromptBundle, verify_sealed
from optimizers.protocol.tests.fakes import FakeAdapter, FakeScorer, fake_runtime, fake_task_data

from .. import native
from ..integration import MASPOBOptimizer, _ordered_roles, _topology_contract
from ..regime import GNN_SETTINGS, UCB_SETTINGS
from .fakes import FakeReflectionLM, fake_embeddings, grid_cell, protocol_runner, requires_gnn


@pytest.fixture
def offline(monkeypatch):
    """Fail any network connection attempted by the code under test."""

    def refuse(*args, **kwargs):
        raise AssertionError("offline test attempted a network connection")

    monkeypatch.setattr(socket.socket, "connect", refuse)


@pytest.fixture
def no_training(monkeypatch):
    """Replace the surrogate fit (800 epochs per pull) by a recorder for long lifecycles."""
    calls = []
    _, training = native.load_model_modules()
    monkeypatch.setattr(training, "train_with_early_stopping", lambda *args, **kwargs: calls.append(kwargs))
    return calls


def _optimizer(tmp_path: Path, **kwargs) -> MASPOBOptimizer:
    kwargs.setdefault("reflection_client", FakeReflectionLM())
    kwargs.setdefault("embedding_factory", fake_embeddings)
    return MASPOBOptimizer(run_dir=tmp_path / "native", **kwargs)


def _train(data):
    return list(data.rows("train"))


# Grid scope and wiring (no torch needed)
def test_scope_is_the_twelve_table_six_cells():
    cells = required_cells("maspob")
    assert len(cells) == 12 and all(6 in cell.source_tables for cell in cells)
    assert {cell.task for cell in cells} == {"hotpotqa", "lcb", "bfcl"}
    assert {cell.topology for cell in cells} == {"independent", "sequential", "centralized", "decentralized"}
    assert {(cell.framework, cell.communication, cell.team_size, cell.task_model) for cell in cells} == {
        ("langgraph", "freeform", 4, "Qwen/Qwen3.5-9B")
    }


def test_registry_builds_the_method_with_initial_bundle_and_run_dir(tmp_path):
    seed = PromptBundle(roles={"writer": "seed"})
    optimizer = methods.build_optimizer("maspob", grid_cell(), seed, run_dir=tmp_path)
    assert isinstance(optimizer, MASPOBOptimizer)
    assert optimizer.initial_bundle is seed and optimizer.run_dir == tmp_path and optimizer.reflection_client is None


@pytest.mark.parametrize(
    "changes",
    [
        {"task": "gpqa"},
        {"framework": "crewai"},
        {"team_size": 8},
        {"communication": "structured_soft"},
        {"method": "gepa"},
        {"task_model": "meta-llama/Llama-3.1-8B-Instruct"},
    ],
)
def test_off_grid_cells_are_rejected_before_any_call(tmp_path, changes):
    fields = {
        "method": "maspob",
        "task": "hotpotqa",
        "topology": "sequential",
        "framework": "langgraph",
        "split_hash": fake_task_data().split_hash,
        **changes,
    }
    cell = CellSpec(**fields)
    runner, budget, data = protocol_runner(grid_cell())
    fake = FakeReflectionLM()
    with pytest.raises(UnsupportedBaselineCell):
        _optimizer(tmp_path, reflection_client=fake).optimize(cell, runner, budget, _train(data), [])
    assert not fake.requests and not FakeAdapter.CALLS and budget.snapshot()["attempted"] == 0


def test_missing_dependency_fails_before_any_reflection_call_or_rollout(tmp_path, monkeypatch):
    monkeypatch.setattr(native, "_MODEL_MODULES", None)
    monkeypatch.setitem(sys.modules, "torch_geometric", None)
    cell = grid_cell()
    runner, budget, data = protocol_runner(cell)
    fake = FakeReflectionLM()
    with pytest.raises(native.MASPOBDependencyError, match="pip install"):
        _optimizer(tmp_path, reflection_client=fake).optimize(cell, runner, budget, _train(data), [])
    assert not fake.requests and not FakeAdapter.CALLS and budget.snapshot()["attempted"] == 0


def test_role_order_and_surrogate_graph_follow_the_native_topology():
    remember_role_order("hotpotqa", "centralized", ("searcher_worker", "manager", "writer_worker"))
    centralized = grid_cell(topology="centralized")
    bundle = PromptBundle(roles={"manager": "m", "searcher_worker": "s", "writer_worker": "w"})
    roles = _ordered_roles(centralized, bundle)
    assert roles == ["manager", "searcher_worker", "writer_worker"]
    graph, description, protocol = _topology_contract(centralized, roles)
    assert graph == [
        {"name": "manager", "dependencies": []},
        {"name": "searcher_worker", "dependencies": ["manager"]},
        {"name": "writer_worker", "dependencies": ["manager"]},
    ]
    assert description == "centralized manager/worker star" and "delegation tools" in protocol
    remember_role_order("bfcl", "sequential", ("schema_reader", "caller", "checker"))
    sequential = grid_cell(topology="sequential", task="bfcl")
    roles = _ordered_roles(sequential, PromptBundle(roles={"caller": "c", "checker": "k", "schema_reader": "r"}))
    assert roles == ["schema_reader", "caller", "checker"]
    assert [node["dependencies"] for node in _topology_contract(sequential, roles)[0]] == [
        [],
        ["schema_reader"],
        ["caller"],
    ]
    for topology in ("independent", "decentralized"):
        graph, description, _ = _topology_contract(grid_cell(topology=topology), ["solver"])
        assert graph == [{"name": "solver", "dependencies": []}] and "shared role instruction" in description


# Bandit lifecycle (torch + torch_geometric)
@requires_gnn
def test_full_budget_lifecycle_spends_exactly_600_rollouts(tmp_path, offline, no_training):
    cell = grid_cell()
    runner, budget, data = protocol_runner(cell)
    fake = FakeReflectionLM()
    optimizer = _optimizer(tmp_path, reflection_client=fake)
    artifact = optimizer.optimize(cell, runner, budget, _train(data), list(data.rows("validation")))

    snapshot = budget.snapshot()
    assert snapshot["charged"] == snapshot["attempted"] == 600 and snapshot["remaining"] == 0
    assert closed_ledger(snapshot)
    assert artifact.budget_snapshot == snapshot and artifact.production_eligible
    history = artifact.metadata["pull_history"]
    assert len(history) == 120 and all(entry["n_items"] == 5 and not entry["truncated"] for entry in history)
    assert [entry["stage"] for entry in history] == ["pretrain"] * 5 + ["ucb"] * 115
    assert all({"pred", "uncertainty", "ucb"} <= set(entry) for entry in history[5:])
    assert len(artifact.learning_curve) == 121 and len(artifact.checkpoints) == 121
    assert artifact.metadata["effective_lifecycle"] == {"rollout_limit": 600, "pretrain_pulls": 5, "ucb_pulls": 115}
    # The surrogate is fitted after the warm-up and after every UCB pull.
    assert len(no_training) == 116 and all(
        call["max_epochs"] == 800 and call["patience"] == 200 and call["min_delta"] == 1e-10 for call in no_training
    )

    rollouts = FakeAdapter.CALLS
    assert len(rollouts) == 600 and {call["temperature"] for call in rollouts} == {0.2}
    assert {call["id"] for call in rollouts} == {row["id"] for row in data.rows("train")}
    assert len(fake.requests) == 2 * 29 and {c["temperature"] for c in fake.requests} == {0.5}

    pool = read_json(tmp_path / "native" / "maspob_prompt_pool.json")["pools"]
    assert list(pool) == ["planner", "writer"] and all(len(pool[role]) == 20 for role in pool)
    selected = artifact.selected_bundle
    best = artifact.metadata["best_indices"]
    assert selected.roles == {role: pool[role][best[index]] for index, role in enumerate(["planner", "writer"])}
    assert best in [entry["indices"] for entry in history]
    winner = max(artifact.metadata["selection_predictions"], key=lambda p: (p["posterior_mean"], -p["first_pull"]))
    assert winner["indices"] == best and selected.metadata["selection_protocol"] == "surrogate-mean"
    assert artifact.metadata["topology_graph"] == [
        {"name": "planner", "dependencies": []},
        {"name": "writer", "dependencies": ["planner"]},
    ]
    assert artifact.metadata["gnn"] == {**GNN_SETTINGS, "topology": artifact.metadata["topology_graph"]}
    assert artifact.metadata["ucb"] == UCB_SETTINGS and artifact.metadata["run_seed"] == 42
    assert artifact.metadata["generation_task"] == "hotpotqa"
    payload = json.dumps(artifact.to_dict())
    assert artifact.to_dict()["schema"] == "mas-promptbench-native-optimizer-result"
    assert str(tmp_path) not in payload and str(Path.home()) not in payload


@requires_gnn
def test_one_cycle_validation_splits_repeated_rows_across_dispatch_batches(tmp_path, offline, no_training):
    cell = grid_cell()
    runner, budget, data = protocol_runner(cell)
    artifact = _optimizer(tmp_path, validation_one_cycle=True).optimize(cell, runner, budget, _train(data)[:1], [])
    assert budget.snapshot()["charged"] == 10 and [call["id"] for call in FakeAdapter.CALLS] == ["tr0"] * 10
    assert len({call["seed"] for call in FakeAdapter.CALLS}) == 10
    assert [entry["stage"] for entry in artifact.metadata["pull_history"]] == ["pretrain", "ucb"]
    assert [entry["n_items"] for entry in artifact.metadata["pull_history"]] == [5, 5]
    assert artifact.production_eligible is False and len(artifact.checkpoints) == 3


@requires_gnn
def test_real_surrogate_training_is_deterministic_per_optimizer_seed(tmp_path, offline):
    def once(seed: int, name: str):
        cell = grid_cell(seed=seed, budget=15)
        runner, budget, data = protocol_runner(cell)
        artifact = _optimizer(tmp_path / name).optimize(cell, runner, budget, _train(data), [])
        assert budget.snapshot()["charged"] == 15
        return artifact

    first, again, other = once(0, "a"), once(0, "b"), once(1, "c")
    assert [e["stage"] for e in first.metadata["pull_history"]] == ["pretrain", "pretrain", "ucb"]
    assert first.metadata["pull_history"] == again.metadata["pull_history"]
    assert first.metadata["selection_predictions"] == again.metadata["selection_predictions"]
    assert first.selected_bundle.digest == again.selected_bundle.digest
    assert other.metadata["run_seed"] == 1042 and other.metadata["row_sampler_seed"] == 1042 + 104729
    assert [e["indices"] for e in other.metadata["pull_history"][:2]] != [
        e["indices"] for e in first.metadata["pull_history"][:2]
    ]


@requires_gnn
def test_protocol_job_runs_maspob_through_all_phases(tmp_path, offline):
    out = tmp_path / "job"
    hooks = run.JobHooks(
        load_task_data=fake_task_data,
        build_runtime=fake_runtime,
        build_scorer=lambda cell: FakeScorer(),
        configure_environment=False,
        optimizer_kwargs={"reflection_client": FakeReflectionLM(), "embedding_factory": fake_embeddings},
    )
    argv = [
        "--method",
        "maspob",
        "--dataset",
        "hotpotqa",
        "--topology",
        "sequential",
        "--model",
        "qwen",
        "--seed",
        "0",
        "--budget",
        "15",
        "--out",
        str(out),
        "--quiet",
    ]
    assert run.main(argv, hooks=hooks) == 0
    optimization = verify_sealed(read_json(out / "optimization.json"))
    assert optimization["status"] == "completed" and optimization["budget"]["charged"] == 15
    assert optimization["stop_reason"] == "rollout_budget_spent"
    assert optimization["usage"]["reflection"]["model_calls"] == 2 * 29
    assert (out / "optimization" / "native" / "maspob_prompt_pool.json").is_file()
    assert verify_sealed(read_json(out / "result.json"))["test"]["valid_for_aggregation"]
    for path in out.rglob("*"):
        if path.is_file():
            text = path.read_text(encoding="utf-8", errors="ignore")
            assert str(tmp_path) not in text and str(Path.home()) not in text and str(REPO_ROOT) not in text


def _minilm_cached() -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache

        return isinstance(try_to_load_from_cache("sentence-transformers/all-MiniLM-L6-v2", "config.json"), str)
    except Exception:
        return False


@pytest.mark.skipif(
    "sentence_transformers" in native.missing_dependencies() or not _minilm_cached(),
    reason="needs sentence-transformers and the all-MiniLM-L6-v2 weights in the Hugging Face cache",
)
def test_minilm_embeddings_are_384_dimensional_normalized_cpu_vectors(offline, monkeypatch):
    import huggingface_hub.constants
    import torch

    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", True)

    pool = {"planner": ["Plan the answer.", "Plan carefully, then answer."], "writer": ["Write the final answer."] * 2}
    embeddings, info = native.embed_pool(pool, ["planner", "writer"])
    assert [tuple(tensor.shape) for tensor in embeddings] == [(2, 384), (2, 384)]
    assert all(tensor.device.type == "cpu" and tensor.dtype == torch.float32 for tensor in embeddings)
    assert torch.allclose(embeddings[0].norm(dim=1), torch.ones(2), atol=1e-5)
    assert info["model"] == "sentence-transformers/all-MiniLM-L6-v2" and info["dim"] == 384
    assert info["max_seq_length"] == 512 and info["prompts_exceeding_max_seq"] == {"planner": 0, "writer": 0}
