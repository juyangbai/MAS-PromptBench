"""MAPROOptimizer end to end through the protocol runner: budget, model surfaces and the artifact contract."""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from pathlib import Path

import httpx
import pytest

from optimizers.protocol import REPO_ROOT, methods, run
from optimizers.protocol.artifacts import read_json
from optimizers.protocol.budget import closed_ledger
from optimizers.protocol.cells import required_cells
from optimizers.protocol.errors import NativeInfrastructureExhausted, NativeIntegrationError, UnsupportedBaselineCell
from optimizers.protocol.reflection import ReflectionClient
from optimizers.protocol.schema import CellSpec, verify_sealed
from optimizers.protocol.tests.fakes import FakeScorer, fake_task_data

from .. import integration, native
from ..integration import MAPROOptimizer, _SeededAsyncChat, _SeededProbe, task_model_client
from ..src.llm import GenerationConfig
from .fakes import MARK, MODELS, FakeBackend, TopologyAdapter, mapro_cell, mapro_runner, mapro_runtime, reset_fake

TRAIN = 6  # fake training split size


@pytest.fixture(autouse=True)
def _clean():
    reset_fake()
    yield
    reset_fake()


def _optimize(cell: CellSpec, judge: FakeBackend | None = None, **kwargs):
    runner, budget, data = mapro_runner(cell)
    reflection, judge = FakeBackend(), judge or FakeBackend()
    optimizer = MAPROOptimizer(seed_bundle=runner.seed_bundle, reflection=reflection, task_judge=judge, **kwargs)
    artifact = optimizer.optimize(cell, runner, budget, list(data.rows("train")), list(data.rows("validation")))
    return artifact, runner, budget, reflection, judge


def _assert_contract(artifact, runner, budget, cell):
    snapshot = budget.snapshot()
    payload = artifact.to_dict()
    json.dumps(payload, allow_nan=False)  # the protocol writes it with allow_nan=False and no default
    assert payload["schema"] == "mas-promptbench-native-optimizer-result"
    assert (payload["method"], payload["cell_id"], payload["protocol_id"]) == ("mapro", cell.cell_id, cell.protocol_id)
    assert payload["budget_snapshot"] == snapshot and snapshot["reserved"] == 0 and closed_ledger(snapshot)
    assert snapshot["charged"] <= budget.maximum and snapshot["charged"] % TRAIN == 0
    assert len(payload["records"]) == len(payload["request_ledger"]) == snapshot["charged"]
    assert payload["stop_reason"] in {"patience", "max_iters", "budget"}
    runner.validate_bundle(artifact.incumbent_bundle)
    assert payload["incumbent_bundle"]["bundle_sha256"] == artifact.incumbent_bundle.digest
    grid = [row["rollout_grid"] for row in payload["learning_curve"] if row["rollout_grid"] is not None]
    assert grid == list(range(0, budget.maximum + 1, 10))
    assert [event["native_event"] for event in payload["events"]][:1] == ["initial_state"]
    assert payload["events"][-1]["native_event"] == "final_state"
    assert len(payload["checkpoints"]) == len(payload["events"])
    metadata = payload["metadata"]
    assert metadata["mapro_regime"] == "mapro-adapted/pointwise+best-so-far-anchor"
    assert metadata["native_result"]["stop_reason"] == payload["stop_reason"]


@pytest.mark.parametrize(
    "topology,framework,team_size",
    [
        ("centralized", "langgraph", 4),
        ("centralized", "autogen", 4),
        ("sequential", "langgraph", 4),
        ("sequential", "crewai", 4),
        ("independent", "langgraph", 4),
        ("independent", "langgraph", 8),
        ("decentralized", "langgraph", 4),
        ("decentralized", "openai_agents", 4),
        ("decentralized", "langgraph", 2),
    ],
)
def test_mapro_runs_every_topology_through_the_protocol_runner(topology, framework, team_size):
    cell = mapro_cell(topology, framework=framework, team_size=team_size)
    artifact, runner, budget, reflection, judge = _optimize(cell)
    _assert_contract(artifact, runner, budget, cell)
    result = artifact.metadata["native_result"]
    roles = list(runner.seed_bundle.roles)
    # Seed eval + one evaluated MAP selection; then the incumbent is re-selected until patience stops.
    assert result["trajectory"] == [[0, 0.5], [1, 1.0], [2, 1.0], [3, 1.0], [4, 1.0]]
    assert (result["stop_reason"], result["iterations_run"], budget.charged) == ("patience", 4, 2 * TRAIN)
    assert result["history"][0]["assignment_idx"] == {role: 1 for role in roles}
    assert all(MARK in prompt for prompt in artifact.incumbent_bundle.roles.values())
    assert result["final_pool_sizes"] == {role: 5 for role in roles}
    assert result["tied_replicas"] == (team_size if topology in {"independent", "decentralized"} else 1)
    assert result["tied_peer_factors"] == (team_size * (team_size - 1) if topology == "decentralized" else 0)
    # Every MAS rollout is a protocol-runner call at optimization decoding on the cell's model.
    assert len(TopologyAdapter.CALLS) == budget.snapshot()["attempted"] == budget.charged
    assert all(call["temperature"] == 0.2 and call["model"] == cell.task_model for call in TopologyAdapter.CALLS)
    assert {call["id"] for call in TopologyAdapter.CALLS} == {f"tr{i}" for i in range(TRAIN)}
    # Reflection: common policy with native temperatures. Judge and probe: task-model policy.
    assert {r["phase"] for r in reflection.requests} == {"mapro_reflection"}
    assert all((r["thinking"], r["max_output_tokens"], r["top_p"]) == (True, 48000, 1.0) for r in reflection.requests)
    assert {r["temperature"] for r in reflection.requests} == {0.7}  # no failure left to blame after round 1
    assert {r["phase"] for r in judge.requests} == {"candidate_probe", "mapro_node_edge_judge"}
    for request in judge.requests:
        expected = (0.2, 0.9, 768) if request["phase"] == "candidate_probe" else (0.2, 1.0, 8)
        assert (request["temperature"], request["top_p"], request["max_output_tokens"]) == expected
        assert request["thinking"] is False


@pytest.mark.parametrize("topology", ["centralized", "sequential", "decentralized"])
def test_failures_of_the_anchor_drive_blame_and_reach_the_mutation_prompts(topology):
    TopologyAdapter.HARD_IDS = {"tr5"}
    artifact, runner, budget, reflection, _ = _optimize(mapro_cell(topology))
    result = artifact.metadata["native_result"]
    assert result["seed_train"] == 0.5 and result["best_train"] == pytest.approx(5 / 6)
    blame = [r for r in reflection.requests if r["temperature"] == 0.2]
    assert all((r["thinking"], r["max_output_tokens"], r["top_p"]) == (True, 48000, 1.0) for r in blame)
    if topology == "decentralized":  # one shared prompt: no parent to blame
        assert not blame and all(not any(item["blame_chars"].values()) for item in result["history"])
        return
    assert blame and all(any(item["blame_chars"].values()) for item in result["history"])
    assert any("omitted a needed detail" in r["output"] for r in blame)


def test_paper_settings_are_the_defaults():
    settings = MAPROOptimizer().settings
    assert (settings.candidate_count, settings.max_iterations, settings.patience, settings.epsilon) == (5, 8, 3, 0.0)
    assert (settings.scoring_batch, settings.evaluation_batch, settings.feedback_count, settings.use_demos) == (
        3,
        0,
        3,
        True,
    )
    assert native.MAPRO_LISTWISE_FLAG is False and native.MAPRO_PAPER_ANCHOR_FLAG is False


@pytest.mark.parametrize("topology", ["centralized", "decentralized"])
@pytest.mark.parametrize("maximum", [6, 7, 11, 12, 13, 17, 30, 59, 600])
def test_budget_is_never_exceeded_and_stops_before_an_unaffordable_evaluation(topology, maximum):
    # The judge prefers the newest mutation round, so every round evaluates a new assignment.
    cell = mapro_cell(topology, budget=maximum)
    artifact, runner, budget, _, judge = _optimize(cell, judge=FakeBackend(prefer_latest=True), patience=8)
    _assert_contract(artifact, runner, budget, cell)
    evaluations = min(maximum // TRAIN, 1 + 8)  # seed eval + at most max_iters assignment evaluations
    assert budget.charged == evaluations * TRAIN <= maximum
    history = artifact.metadata["native_result"]["history"]
    assert all(item["evaluated"] for item in history) and len(history) == evaluations - 1
    assert artifact.stop_reason == ("max_iters" if evaluations == 9 else "budget")


def test_budget_smaller_than_one_training_pass_is_refused_before_any_rollout():
    cell = mapro_cell("sequential", budget=TRAIN - 1)
    with pytest.raises(NativeIntegrationError, match="cannot cover the initial seed train eval"):
        _optimize(cell)
    assert TopologyAdapter.CALLS == []


def test_cells_outside_the_grid_and_frozen_regime_changes_are_refused(monkeypatch):
    runner, budget, data = mapro_runner(mapro_cell("centralized"))
    bad = CellSpec(method="mapro", task="hotpotqa", topology="centralized", framework="crewai")
    with pytest.raises(UnsupportedBaselineCell):
        MAPROOptimizer(seed_bundle=runner.seed_bundle).optimize(bad, runner, budget, list(data.rows("train")), [])
    monkeypatch.setattr(native, "MAPRO_LISTWISE_FLAG", True)
    with pytest.raises(NativeIntegrationError, match="pointwise/best-so-far"):
        MAPROOptimizer(seed_bundle=runner.seed_bundle, reflection=FakeBackend(), task_judge=FakeBackend()).optimize(
            runner.cell, runner, budget, list(data.rows("train")), []
        )
    assert budget.snapshot()["attempted"] == 0 and TopologyAdapter.CALLS == []


def test_infrastructure_exhaustion_aborts_with_a_closed_ledger():
    TopologyAdapter.FAIL_IDS = {"tr3"}
    cell = mapro_cell("centralized")
    with pytest.raises(NativeInfrastructureExhausted):
        _optimize(cell)
    calls = Counter(call["id"] for call in TopologyAdapter.CALLS)
    assert calls["tr3"] == 3 and all(count == 1 for item, count in calls.items() if item != "tr3")


def test_runs_are_deterministic_per_optimizer_seed():
    def trace(seed):
        reset_fake()
        artifact, _, budget, reflection, judge = _optimize(mapro_cell("sequential", seed=seed))
        return (
            artifact.incumbent_bundle.digest,
            sorted(call["seed"] for call in TopologyAdapter.CALLS),
            sorted(r["request_seed"] for r in reflection.requests + judge.requests),
            [row["bundle_hash"] for row in artifact.learning_curve],
            budget.charged,
        )

    first, again, other = trace(0), trace(0), trace(1)
    assert first == again
    assert first[1] != other[1] and first[2] != other[2]


class _WireTransport:
    """In-memory OpenAI-compatible server answering with the fake judge/probe policy."""

    def __init__(self):
        self.requests = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append((str(request.url), body))
        system = next((m["content"] for m in body["messages"] if m["role"] == "system"), None)
        prompt = body["messages"][-1]["content"]
        phase = "candidate_probe" if body["max_tokens"] == 768 else "mapro_node_edge_judge"
        text = FakeBackend._respond(prompt, phase, system)
        return httpx.Response(
            200,
            json={
                "id": "offline",
                "object": "chat.completion",
                "created": 0,
                "model": body["model"],
                "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": text}}],
                "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
            },
        )


def test_llama_cell_judge_and_probe_use_the_task_model_on_protocol_endpoints(monkeypatch):
    endpoints = ("http://task-a.invalid/v1", "http://task-b.invalid/v1")
    monkeypatch.setenv("TASK_ENDPOINTS", ",".join(endpoints))
    monkeypatch.setenv("TASK_MODEL", MODELS["llama"])
    wire = _WireTransport()
    monkeypatch.setattr(
        integration,
        "task_model_client",
        lambda model: task_model_client(model, http_client=httpx.Client(transport=httpx.MockTransport(wire))),
    )
    cell = mapro_cell("centralized", model="llama")
    runner, budget, data = mapro_runner(cell)
    optimizer = MAPROOptimizer(seed_bundle=runner.seed_bundle, reflection=FakeBackend())
    artifact = optimizer.optimize(cell, runner, budget, list(data.rows("train")), list(data.rows("validation")))
    _assert_contract(artifact, runner, budget, cell)
    assert isinstance(optimizer.task_judge, ReflectionClient) and optimizer.task_judge.model == MODELS["llama"]
    assert {body["model"] for _, body in wire.requests} == {MODELS["llama"]}
    assert all(
        body["chat_template_kwargs"] == {"enable_thinking": False} and isinstance(body["seed"], int)
        for _, body in wire.requests
    )
    assert {body["max_tokens"] for _, body in wire.requests} == {8, 768}
    hosts = [url.split("/v1")[0] for url, _ in wire.requests]
    assert set(hosts) == {endpoint.split("/v1")[0] for endpoint in endpoints}
    assert abs(hosts.count(hosts[0]) - len(hosts) / 2) <= 1  # round robin
    usage = optimizer.task_judge.snapshot()["usage"]
    assert usage["model_calls"] == len(wire.requests) and usage["total_tokens"] == 10 * len(wire.requests)
    assert all(MARK in prompt for prompt in artifact.incumbent_bundle.roles.values())


def test_task_model_backend_requires_a_protocol_endpoint(monkeypatch):
    monkeypatch.delenv("TASK_ENDPOINTS", raising=False)
    monkeypatch.delenv("VLLM_BASE_URL", raising=False)
    with pytest.raises(NativeIntegrationError, match="task endpoint"):
        task_model_client(MODELS["qwen"])
    monkeypatch.setenv("VLLM_BASE_URL", "http://only.invalid/v1")
    assert task_model_client(MODELS["qwen"]).base_urls == ("http://only.invalid/v1",)


def test_shims_enforce_tier_policies_over_native_generation_configs():
    cell, backend = mapro_cell("centralized"), FakeBackend()
    reflection = _SeededAsyncChat(cell, backend, "mapro_reflection", thinking=True)
    judge = _SeededAsyncChat(cell, backend, "mapro_node_edge_judge")
    asyncio.run(reflection.chat_text("rewrite", system="s", cfg=GenerationConfig(temperature=0.7, max_tokens=512)))
    asyncio.run(judge.chat_text("judge", cfg=GenerationConfig(temperature=0.2, max_tokens=8, enable_thinking=True)))
    _SeededProbe(cell, backend, 1).complete("candidate", "context")
    assert [(r["thinking"], r["max_output_tokens"], r["temperature"], r["top_p"]) for r in backend.requests] == [
        (True, 48000, 0.7, 1.0),
        (False, 8, 0.2, 1.0),
        (False, 768, 0.2, 0.9),
    ]
    repeated = _SeededAsyncChat(cell, FakeBackend(), "mapro_reflection", thinking=True)
    for _ in range(2):
        asyncio.run(repeated.chat_text("same prompt"))
    seeds = [r["request_seed"] for r in repeated.backend.requests]
    assert len(set(seeds)) == 2  # repeated identical prompts get distinct logical seeds


def test_registry_builds_mapro_with_the_seed_bundle_on_its_grid():
    cell = mapro_cell("centralized")
    seed = mapro_runtime(cell).seed_bundle()
    optimizer = methods.build_optimizer("mapro", cell, seed, extra={"reflection": FakeBackend()})
    assert isinstance(optimizer, MAPROOptimizer) and optimizer.seed_bundle is seed
    assert len(required_cells("mapro")) == 135


def test_protocol_job_runs_optimize_validate_and_test_with_anonymous_artifacts(tmp_path):
    hooks = run.JobHooks(
        load_task_data=fake_task_data,
        build_runtime=mapro_runtime,
        build_scorer=lambda cell: FakeScorer(),
        configure_environment=False,
        optimizer_kwargs={"reflection": FakeBackend(), "task_judge": FakeBackend()},
    )
    out = tmp_path / "job"
    argv = [
        "--method",
        "mapro",
        "--dataset",
        "hotpotqa",
        "--topology",
        "centralized",
        "--model",
        "qwen",
        "--seed",
        "2",
        "--out",
        str(out),
        "--quiet",
    ]
    assert run.main(argv, hooks=hooks) == 0
    optimization = verify_sealed(read_json(out / "optimization.json"))
    assert optimization["status"] == "completed" and optimization["stop_reason"] == "patience"
    assert optimization["budget"]["charged"] == 2 * TRAIN and optimization["usage"]["reflection"]["model_calls"] > 0
    result = read_json(out / "optimization" / "optimizer_result.json")
    assert result["method"] == "mapro" and result["budget_snapshot"] == optimization["budget"]
    curve = [json.loads(line) for line in (out / "optimization" / "learning_curve.jsonl").read_text().splitlines()]
    assert [row["rollout_grid"] for row in curve if row["rollout_grid"] is not None] == list(range(0, 601, 10))
    selection = verify_sealed(read_json(out / "selection.json"))
    assert selection["selected_candidate"] and selection["incumbent_validation_score"] == 1.0
    final = verify_sealed(read_json(out / "result.json"))
    assert final["protocol_conformant"] is True and final["test"]["deployed_mean"] == 1.0
    for path in out.rglob("*"):
        if path.is_file():
            text = path.read_text(encoding="utf-8", errors="ignore")
            assert str(tmp_path) not in text and str(Path.home()) not in text and str(REPO_ROOT) not in text
