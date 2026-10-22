"""Prompt pool, interface manifest, reflection client, sampler and selection (no torch needed)."""

from __future__ import annotations

import json
import math
import random

import pytest

from optimizers.protocol.reflection import LogicalTextReflectionClient, ReflectionClient
from optimizers.protocol.tests.fakes import FakeOpenAI

from .. import native
from ..regime import (
    CORE_SOURCE_SHA256,
    DEFAULTS,
    GENERATION_TASK_NAMES,
    INITIALIZATION_PROTOCOL,
    ROW_SAMPLER_SEED_SALT,
    run_seed,
)
from .fakes import FakeReflectionLM, grid_cell

DEFAULT_NUM_PROMPTS = DEFAULTS.num_prompts

SEEDS = {
    "manager": (
        "You are the manager. Delegate with delegate_to_searcher_worker, call `python_exec` when needed, "
        "and answer {question} inside <answer></answer> tags."
    ),
    "searcher_worker": "Search for evidence about {question} and report it back to the manager.",
}
ROLES = ["manager", "searcher_worker"]


def _wrapped(fake: FakeReflectionLM, seed: int = 0) -> LogicalTextReflectionClient:
    return LogicalTextReflectionClient(
        cell=grid_cell(topology="centralized", seed=seed),
        client=fake,
        method="maspob",
        default_temperature=DEFAULTS.generation_temperature,
        default_top_p=1.0,
    )


def _generate(fake, path, *, dataset="hotpotqa", generation_seed=42, inflight=4, roles=ROLES, seeds=SEEDS):
    random.seed(generation_seed)
    reflection = _wrapped(fake)
    pool, stats = native.generate_prompt_pool(
        reflection, list(roles), seeds, dataset, DEFAULT_NUM_PROMPTS, inflight, path, generation_seed=generation_seed
    )
    return pool, stats, reflection


def test_manifest_requires_the_exact_executable_interface():
    seed = SEEDS["manager"]
    manifest = native.protocol_manifest(seed)
    assert manifest["placeholders"] == ["question"] and manifest["delegations"] == ["delegate_to_searcher_worker"]
    assert manifest["roles"] == ["manager"] and manifest["tools"] == ["python_exec"]
    assert manifest["backticked_identifiers"] == ["python_exec"] and manifest["xml_tags"] == ["answer"]
    rewritten = (
        "As the manager, route work through delegate_to_searcher_worker, use `python_exec` for checks, "
        "and return {question}'s answer in <answer></answer>."
    )
    assert native.valid_variant(rewritten, seed)
    assert not native.valid_variant(rewritten.replace("{question}", "{query}"), seed)
    assert not native.valid_variant(rewritten + " Also ask delegate_to_coder_worker.", seed)
    assert not native.valid_variant("Too short {question}", seed)
    assert not native.valid_variant(rewritten + " x" * 5000, seed)


def test_pool_has_twenty_valid_variants_per_role_with_the_seed_first(tmp_path):
    fake = FakeReflectionLM()
    pool, stats, reflection = _generate(fake, tmp_path / "pool.json")
    assert set(pool) == set(ROLES) and stats["complete"] and stats["cache_status"] == "miss"
    for role in ROLES:
        assert len(pool[role]) == DEFAULT_NUM_PROMPTS == len(set(pool[role]))
        assert pool[role][0] == SEEDS[role].strip()
        assert all(native.valid_variant(text, SEEDS[role]) for text in pool[role][1:])
        assert not any(text.startswith(("```", "Here is")) for text in pool[role])  # upstream cleaning applied
    # 19 needed per role, generated with upstream's 1.5x over-generation.
    assert len(fake.requests) == 2 * 29
    sampling = {(c["temperature"], c["top_p"], c["max_output_tokens"], c["thinking"]) for c in fake.requests}
    assert sampling == {(0.5, 1.0, 48000, True)}
    assert {(r["temperature"], r["max_output_tokens"], r["thinking"], r["status"]) for r in reflection.requests} == {
        (0.5, 48000, True, "success")
    }
    payload = json.loads((tmp_path / "pool.json").read_text())
    assert payload["pools"] == pool and payload["identity"] == stats["cache_identity"]
    identity = stats["cache_identity"]
    assert identity["core_source_sha256"] == CORE_SOURCE_SHA256 and identity["generation_temperature"] == 0.5
    assert identity["generation_seed"] == 42 and identity["roles"] == ROLES and "endpoint" not in identity


def test_styles_follow_upstream_presets_then_seeded_random_draws(tmp_path):
    styles, _ = native.load_prompt_modules()
    first, second = FakeReflectionLM(), FakeReflectionLM()
    _generate(first, tmp_path / "a.json", inflight=1)
    _generate(second, tmp_path / "b.json", inflight=1)
    assert [c["prompt"] for c in first.requests] == [c["prompt"] for c in second.requests]
    manager = [c["prompt"] for c in first.requests if "'manager' role" in c["prompt"]]
    presets = [styles.generate_style_instruction(quality=name) for name in styles.PRESET_STYLE_NAMES]
    assert [any(style in prompt for style in presets) for prompt in manager] == [True] * 10 + [False] * 19
    assert all(presets[index] in manager[index] for index in range(10))
    assert "centralized manager + workers, star topology multi-agent system solving hotpotqa tasks" in manager[0]
    assert '"delegations": ["delegate_to_searcher_worker"]' in manager[0]
    third = FakeReflectionLM()
    _generate(third, tmp_path / "c.json", generation_seed=1042, inflight=1)
    assert [c["prompt"] for c in third.requests][:10] == manager[:10]
    assert [c["prompt"] for c in third.requests] != [c["prompt"] for c in first.requests]


def test_preset_priority_matches_the_reference_task_names(tmp_path):
    styles, _ = native.load_prompt_modules()
    assert GENERATION_TASK_NAMES == {"lcb": "livecodebench"}
    listed = styles.generate_style_instruction(quality=styles.PRESET_STYLE_NAMES[0])
    for dataset in ("hotpotqa", "livecodebench", "bfcl"):
        fake = FakeReflectionLM()
        _generate(fake, tmp_path / f"{dataset}.json", dataset=dataset, inflight=1, roles=["searcher_worker"])
        assert listed in fake.requests[0]["prompt"] and f"solving {dataset} tasks" in fake.requests[0]["prompt"]
    fake = FakeReflectionLM()
    _generate(fake, tmp_path / "math.json", dataset="math", inflight=1, roles=["searcher_worker"])
    assert styles.generate_style_instruction(quality="DETAILED_REASONING") in fake.requests[0]["prompt"]


def test_validated_cache_is_reused_and_a_changed_identity_regenerates(tmp_path):
    path = tmp_path / "pool.json"
    pool, _, _ = _generate(FakeReflectionLM(), path)
    reuse = FakeReflectionLM()
    cached, stats, _ = _generate(reuse, path)
    assert cached == pool and stats["cache_status"] == "validated" and stats["loaded_from_cache"] and not reuse.requests
    other = FakeReflectionLM()
    _, stats, _ = _generate(other, path, generation_seed=1042)
    assert stats["cache_status"] == "identity-mismatch" and len(other.requests) == 2 * 29


def test_invalid_or_failing_generation_fails_closed_with_a_partial_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(native.time, "sleep", lambda seconds: None)
    fake = FakeReflectionLM(invalid=True)
    with pytest.raises(RuntimeError, match="prompt pool is incomplete"):
        _generate(fake, tmp_path / "pool.json")
    assert len(fake.requests) == 2 * 29 * 3  # three attempts per job
    assert json.loads((tmp_path / "pool.json").read_text())["pools"] == {role: [SEEDS[role]] for role in ROLES}
    failing = FakeReflectionLM(error=True)
    with pytest.raises(RuntimeError, match="prompt pool is incomplete"):
        _generate(failing, tmp_path / "failing.json")


def test_reflection_client_uses_the_protocol_endpoint_and_native_temperature(monkeypatch):
    monkeypatch.setenv("REFLECTION_MODEL_BASE_URL", "http://localhost:8001/v1")
    monkeypatch.setenv("REFLECTION_MODEL_ID", "reflection-model")
    wire = FakeOpenAI(content="variant text")
    client = ReflectionClient(client=wire)
    assert client.endpoint == "http://localhost:8001/v1" and client.model == "reflection-model"
    reflection = _wrapped(client)
    assert reflection.complete("meta prompt") == "variant text"
    request = wire.bodies[0]
    assert request["model"] == "reflection-model" and request["temperature"] == 0.5 and request["top_p"] == 1.0
    assert request["max_tokens"] == 48000
    assert request["extra_body"] == {"chat_template_kwargs": {"enable_thinking": True}}
    assert request["seed"] == reflection.requests[0]["request_seed"]
    assert client.usage == {"prompt_tokens": 11, "completion_tokens": 7, "n_calls": 1}
    assert reflection.requests[0]["finish_reason"] == "stop"


def test_minibatch_sampler_rotates_shuffled_passes_and_streams_are_independent():
    combo, rows_rng, row_seed = native.rng_streams(run_seed(1))
    assert run_seed(0) == 42 and run_seed(1) == 1042 and run_seed(2) == 2042
    assert row_seed == 1042 + ROW_SAMPLER_SEED_SALT
    assert combo.random() == random.Random(1042).random() and rows_rng.random() == random.Random(row_seed).random()
    sampler = native.MinibatchSampler(list(range(6)), 5, random.Random(7))
    first, second = sampler.next(), sampler.next()
    assert len(first) == len(second) == 5 and len(set(first)) == 5
    assert sorted(first + second[:1]) == list(range(6))  # one full pass before any repeat


def test_selection_uses_posterior_mean_over_observed_combinations_with_earliest_tie_break():
    history = [
        {"pull": 1, "indices": [3, 1], "score": 1.0},
        {"pull": 2, "indices": [0, 2], "score": 0.2},
        {"pull": 3, "indices": [4, 4], "score": 0.6},
        {"pull": 4, "indices": [3, 1], "score": 0.0},
    ]
    combo_best = {(3, 1): 1.0, (0, 2): 0.2, (4, 4): 0.6}
    means = {(3, 1): 0.4, (0, 2): 0.7, (4, 4): 0.7}
    indices, raw, note, predictions = native.select_best_observed(combo_best, history, means.__getitem__)
    assert indices == [0, 2] and raw == 0.2 and "posterior mean" in note
    assert {tuple(p["indices"]): p["first_pull"] for p in predictions} == {(3, 1): 1, (0, 2): 2, (4, 4): 3}
    with pytest.raises(RuntimeError, match="posterior-mean selection failed"):
        native.select_best_observed(combo_best, history, lambda key: math.nan)
    assert native.select_best_observed({}, [], means.__getitem__)[0] == []
    anchor, protocol, _ = native.initialize_from_pretrain(history[:3], means.__getitem__)
    assert anchor == [0, 2] and protocol == INITIALIZATION_PROTOCOL
