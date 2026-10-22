"""Experiment grid, registry-key mapping, learning curve and paired statistics."""

from __future__ import annotations

import pytest

from ..cells import (
    build_grid,
    find_cell,
    grid_assertions,
    parse_registry_key,
    registry_key,
    required_cells,
    runner_condition_in_grid,
    validate_cell,
)
from ..errors import UnsupportedBaselineCell
from ..learning_curve import LearningCurveRecorder
from ..reporting import (
    PairedObservation,
    exact_paired_sign_test,
    holm_adjust,
    paired_summary,
    summarize_paired_observations,
)
from ..schema import CellSpec


def test_grid_counts_and_tables():
    assert grid_assertions()["all_configurations"] == 612
    cell = find_cell("gepa", "hotpotqa", "sequential", "langgraph", "freeform", 4, "Qwen/Qwen3.5-9B")
    assert cell is not None and cell.source_tables == (3, 4, 5, 6, 7)
    assert find_cell("maspo", "math", "single", "langgraph", "freeform", 1, "Qwen/Qwen3.5-9B") is None
    assert find_cell(
        "hivemind", "bfcl", "decentralized", "langgraph", "freeform", 4, "Qwen/Qwen3.5-9B"
    ).source_tables == (6,)
    llama = [c for c in build_grid(True) if c.task_model.startswith("meta-llama/")]
    assert len(llama) == 54 and {c.method for c in llama} == {"gepa", "mipro", "mapro", "maspo"}
    assert runner_condition_in_grid("swe", "sequential", "crewai", "freeform", 4, "Qwen/Qwen3.5-9B")


@pytest.mark.parametrize(
    "args,key",
    [
        (("single", "langgraph", "freeform", 1), "single"),
        (("sequential", "crewai", "freeform", 4), "sequential_crewai"),
        (("centralized", "autogen", "freeform", 4), "centralized_autogen"),
        (("decentralized", "openai_agents", "freeform", 4), "decentralized_openai_agents"),
        (("independent", "langgraph", "freeform", 8), "independent_r8"),
        (("centralized", "langgraph", "semi_structured", 4), "centralized_communications_semi_structured"),
    ],
)
def test_registry_key_round_trip(args, key):
    assert registry_key(*args) == key
    parsed = parse_registry_key(key)
    assert parsed["topology"] == args[0]
    with pytest.raises(ValueError):
        parse_registry_key("hierarchical")


def test_baseline_cell_validation():
    assert len(required_cells("tavo")) == 12
    good = CellSpec(method="mapro", task="lcb", topology="centralized", framework="langgraph")
    validate_cell("mapro", good)
    with pytest.raises(UnsupportedBaselineCell):
        validate_cell("tavo", CellSpec(method="tavo", task="math", topology="centralized", framework="langgraph"))


def test_learning_curve_projects_committed_states_on_the_grid():
    curve = LearningCurveRecorder(maximum=40)
    curve.observe(0, "seed", 0)
    curve.observe(25, "b1", 1, 0.5)
    curve.carry_forward_after_stop()
    grid = {row["rollout_grid"]: row for row in curve.rows if row["rollout_grid"] is not None}
    assert sorted(grid) == [0, 10, 20, 30, 40]
    assert grid[20]["bundle_hash"] == "seed" and grid[30]["bundle_hash"] == "b1"
    assert grid[40]["post_stop_carried_forward"] is True


def _observations():
    rows = []
    for seed in (0, 1, 2):
        rows += [PairedObservation(seed, f"e{i}", 0.0, 1.0 if i < 3 else 0.0) for i in range(4)]
    return rows


def test_paired_summary_bootstrap_mcnemar_and_holm():
    summary = summarize_paired_observations(_observations(), bootstrap_replicates=200)
    assert summary["mean_delta_pp"] == 75.0 and summary["std_delta_pp"] == 0.0
    low, high = summary["confidence_interval_delta_pp"]
    assert low <= 75.0 <= high
    assert summary["per_seed_exact_mcnemar_p"]["0"] == exact_paired_sign_test(_observations()[:4]) == 0.25
    assert holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03}) == {"a": 0.03, "b": 0.06, "c": 0.06}
    with pytest.raises(ValueError):
        paired_summary(_observations()[:8])
