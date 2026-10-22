"""Shapley credit (exact and Monte-Carlo), coalition plans, seeds, lessons and registered settings."""

from __future__ import annotations

import random
from itertools import combinations, permutations

import pytest

from optimizers.protocol.cells import required_cells

from ..integration import METHOD_SETTINGS, HiveMindOptimizer
from ..optimizer import exact_shapley_values, extract_lessons, metamorphose, permutation_shapley_values
from ..regime import (
    ADAPTATION_ID,
    build_coalition_plan,
    coalition_key,
    expected_plan_provenance,
    plan_metadata_matches,
    plan_provenance,
    plan_seed,
    row_order_seed,
)
from ..topology import coalition_game


def _game(agents, seed=0):
    rng = random.Random(seed)
    values = {}
    for size in range(len(agents) + 1):
        for coalition in combinations(agents, size):
            values[frozenset(coalition)] = 0.0 if not coalition else rng.random()
    return values


@pytest.mark.parametrize("agents", [("a", "b", "c"), ("w1", "w2", "w3", "w4")])
def test_exact_shapley_equals_the_permutation_definition_and_is_efficient(agents):
    values = _game(agents, seed=len(agents))
    exact = exact_shapley_values(agents, values)
    every_order = permutation_shapley_values(agents, list(permutations(agents)), values)
    assert exact == pytest.approx(every_order)
    assert sum(exact.values()) == pytest.approx(values[frozenset(agents)] - values[frozenset()])
    null_player = {coalition: values[coalition - {agents[0]}] for coalition in values}
    assert exact_shapley_values(agents, null_player)[agents[0]] == pytest.approx(0.0)


def test_additive_game_credits_each_worker_its_own_contribution():
    weights = {"a": 1.0, "b": 3.0, "c": -2.0}
    values = {frozenset(c): sum(weights[x] for x in c) for size in range(4) for c in combinations(weights, size)}
    assert exact_shapley_values(list(weights), values) == pytest.approx(weights)


def test_monte_carlo_plan_respects_the_cap_and_estimates_shapley_from_prefixes():
    workers = [f"w{i}" for i in range(6)]
    coalitions, sampled = build_coalition_plan(workers, random.Random(plan_seed(1, 0)), 40)
    assert sampled is not None and len(sampled) == (40 - 2) // (len(workers) - 1)
    assert len(coalitions) <= 40 and frozenset() in coalitions and frozenset(workers) in coalitions
    assert all(sorted(order) == sorted(workers) for order in sampled)
    assert all(frozenset(order[:i]) in coalitions for order in sampled for i in range(len(workers) + 1))
    again, sampled_again = build_coalition_plan(workers, random.Random(plan_seed(1, 0)), 40)
    assert (again, sampled_again) == (coalitions, sampled)
    assert build_coalition_plan(workers, random.Random(plan_seed(2, 0)), 40)[1] != sampled
    # Additive game: every permutation gives the exact marginal contribution.
    weights = {worker: float(index) for index, worker in enumerate(workers)}
    values = {coalition: sum(weights[w] for w in coalition) for coalition in coalitions}
    assert permutation_shapley_values(workers, sampled, values) == pytest.approx(weights)
    # Non-additive game: the estimate stays efficient on every sampled permutation.
    table = _game(workers, seed=7)
    estimate = permutation_shapley_values(workers, sampled, {c: table[c] for c in coalitions})
    assert sum(estimate.values()) == pytest.approx(table[frozenset(workers)] - table[frozenset()])
    with pytest.raises(ValueError, match="max_coalitions"):
        build_coalition_plan(workers, random.Random(0), 6)
    meta = plan_provenance(workers, coalitions, sampled, max_coalitions=40, seed_offset=1, hm_seed=0)
    assert meta["hivemind_shapley_mode"] == "monte_carlo_permutation_worker_shapley"
    assert meta["hivemind_mc_sampling_scheme"] == "iid_uniform_fixed_count"
    assert plan_metadata_matches(meta, workers, max_coalitions=40, seed_offset=1, hm_seed=0)
    assert not plan_metadata_matches(meta, workers, max_coalitions=40, seed_offset=2, hm_seed=0)


@pytest.mark.parametrize(
    "topology,roles,count",
    [
        ("centralized", ["manager", "retriever_worker", "reasoner_worker", "writer_worker"], 8),
        ("sequential", ["planner", "retriever", "reasoner", "writer"], 16),
        ("independent", ["solver"], 16),
        ("decentralized", ["debater"], 16),
    ],
)
def test_grid_games_use_the_exact_power_set(topology, roles, count):
    game = coalition_game(topology, roles, 4)
    coalitions, sampled = build_coalition_plan(list(game.players), random.Random(0), 40)
    assert sampled is None and len(coalitions) == count
    assert [len(c) for c in coalitions] == sorted(len(c) for c in coalitions)
    assert coalitions[0] == frozenset() and coalitions[-1] == frozenset(game.players)
    meta = expected_plan_provenance(list(game.players), max_coalitions=40, seed_offset=0, hm_seed=0)
    assert meta["hivemind_shapley_mode"] == "exact_worker_shapley" and meta["hivemind_coalition_count"] == count
    assert coalition_key(frozenset()) == "manager_only"
    if topology in {"independent", "decentralized"}:
        assert set(game.player_roles.values()) == {roles[0]} and game.manager is None
    invalid = ["worker_a", "worker_b"] if topology == "centralized" else [*roles, "extra", "more"]
    with pytest.raises(ValueError):
        coalition_game(topology, invalid, 4)


def test_settings_and_defaults_match_the_registered_regime():
    assert METHOD_SETTINGS["algorithm"] == "CG-OPO_coalition"
    assert METHOD_SETTINGS["fail_below"] == 0.5 and METHOD_SETTINGS["scope"] == "table_6"
    optimizer = HiveMindOptimizer(seed_bundle=None, reflection=object())
    settings = optimizer.settings
    assert (settings.coalition_batch, settings.acceptance_batch, settings.max_coalitions) == (5, 5, 40)
    assert (settings.manager_every_k, settings.max_lessons, settings.fail_below) == (3, 6, 0.5)
    assert (settings.max_cycles, settings.hm_seed) == (0, 0)
    assert plan_seed(2, 7) == 90001 * 2 + 7 and row_order_seed(2, 7) == 100003 * 2 + 7
    assert ADAPTATION_ID.startswith("hivemind-adapted/")
    assert len(required_cells("hivemind")) == 12 and settings.reflection_temperature == 0.7
    with pytest.raises(ValueError):
        HiveMindOptimizer(coalition_batch=0, reflection=object())


def test_lessons_are_parsed_and_capped_at_six():
    raw = "noise\n===BEGIN LESSONS===\n- <lesson 1> verify units\n2) cite sources\n===END LESSONS===\n"
    assert extract_lessons(raw) == "- verify units\n- cite sources"
    assert extract_lessons("1. only numbered\nprose line") == "- only numbered"
    log = [f"- lesson {i}" for i in range(8)]
    prompt = metamorphose("Base prompt.", log, 6)
    assert prompt.startswith("Base prompt.\n\n=== Lessons learned")
    assert "lesson 1\n" not in prompt and "- lesson 2" in prompt and "- lesson 7" in prompt
    assert metamorphose("Base prompt.", ["  "], 6) == "Base prompt."
