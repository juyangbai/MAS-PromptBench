"""End-to-end ``run.py`` jobs on a fake cell and ``aggregate.py`` over three seeds."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from .. import REPO_ROOT, aggregate, methods, run
from ..artifacts import read_json
from ..config import REFLECTION_MODEL_ID
from ..schema import CellSpec, verify_sealed
from .fakes import FakeAdapter, fake_hooks

TOY = "optimizers.protocol.tests.fakes:"


@pytest.fixture(autouse=True)
def _toy_methods(monkeypatch):
    monkeypatch.setitem(methods.METHODS, "toy", TOY + "ToyOptimizer")
    monkeypatch.setitem(methods.METHODS, "badtoy", TOY + "BadToyOptimizer")
    monkeypatch.setitem(methods.METHODS, "infratoy", TOY + "InfraToyOptimizer")


def _run(out: Path, method: str, seed: int = 0, *extra: str) -> int:
    argv = [
        "--method",
        method,
        "--dataset",
        "fake",
        "--topology",
        "sequential",
        "--model",
        "qwen",
        "--seed",
        str(seed),
        "--out",
        str(out),
        "--allow-any-cell",
        "--quiet",
        *extra,
    ]
    return run.main(argv, hooks=fake_hooks())


def _sealed(path: Path) -> dict:
    return verify_sealed(read_json(path))


def test_identity_job_runs_all_phases_without_budget(tmp_path):
    out = tmp_path / "identity"
    assert _run(out, "identity") == 0
    optimization = _sealed(out / "optimization.json")
    assert optimization["status"] == "completed" and optimization["budget"]["charged"] == 0
    assert optimization["stop_reason"] == "identity_no_search"
    selection = _sealed(out / "selection.json")
    assert selection["fallback_reason"] == "validation_tie" and not selection["selected_candidate"]
    result = _sealed(out / "result.json")
    assert result["test"]["valid_for_aggregation"] and result["test"]["delta_pp"] == 0.0
    assert result["protocol_conformant"] is False  # fake cell is outside the grid
    test_calls = [call for call in FakeAdapter.CALLS if call["id"].startswith("te")]
    assert len(test_calls) == 6 and all(call["temperature"] == 0.0 for call in test_calls)
    curve = (out / "optimization" / "learning_curve.jsonl").read_text().splitlines()
    assert len(curve) == 61


def test_job_records_the_reflection_model_it_calls(tmp_path, monkeypatch):
    monkeypatch.setenv("REFLECTION_MODEL_ID", "small/reflection-model")
    assert _run(tmp_path / "identity", "identity") == 0
    assert _sealed(tmp_path / "identity" / "job.json")["cell"]["reflection_model"] == "small/reflection-model"
    cell = CellSpec(method="gepa", task="hotpotqa", topology="sequential", framework="langgraph")
    assert cell.reflection_model == REFLECTION_MODEL_ID and cell.protocol_conformant
    assert not replace(cell, reflection_model="small/reflection-model").protocol_conformant


def test_toy_job_deploys_strict_improvement_and_keeps_artifacts_anonymous(tmp_path):
    out = tmp_path / "toy"
    assert _run(out, "toy", 1) == 0
    optimization = _sealed(out / "optimization.json")
    assert optimization["budget"]["charged"] == 10 and optimization["stop_reason"] == "toy_done"
    assert optimization["usage"]["task"]["model_calls"] == 20
    selection = _sealed(out / "selection.json")
    assert selection["selected_candidate"] and selection["baseline_validation_score"] == 0.5
    assert selection["incumbent_validation_score"] == 1.0
    result = _sealed(out / "result.json")
    assert result["test"]["baseline_mean"] == 0.5 and result["test"]["deployed_mean"] == 1.0
    optimization_calls = [c for c in FakeAdapter.CALLS if c["id"].startswith("tr")]
    assert optimization_calls and all(c["temperature"] == 0.2 for c in optimization_calls)
    # No absolute paths in any artifact.
    for path in out.rglob("*"):
        if path.is_file():
            text = path.read_text(encoding="utf-8", errors="ignore")
            assert str(tmp_path) not in text and str(Path.home()) not in text and str(REPO_ROOT) not in text


def test_regression_and_infrastructure_failure_keep_the_seed_bundle(tmp_path):
    assert _run(tmp_path / "bad", "badtoy") == 0
    assert _sealed(tmp_path / "bad" / "selection.json")["fallback_reason"] == "validation_regression"
    assert _run(tmp_path / "infra", "infratoy") == 0
    optimization = _sealed(tmp_path / "infra" / "optimization.json")
    assert optimization["status"] == "failed" and optimization["failure_kind"] == "infrastructure_invalid"
    # The other five train rows of the batch were usable and stay charged.
    assert optimization["budget"]["charged"] == 5 and optimization["budget"]["infrastructure_failures"] == 3
    selection = _sealed(tmp_path / "infra" / "selection.json")
    assert selection["fallback_reason"] == "infrastructure_invalid_optimization"
    assert _sealed(tmp_path / "infra" / "result.json")["test"]["delta_pp"] == 0.0


def test_phases_resume_from_saved_artifacts(tmp_path):
    out = tmp_path / "phased"
    assert _run(out, "toy", 0, "--phase", "test") == 2  # no selection yet
    assert _run(out, "toy", 0, "--phase", "validate") == 2  # no optimization yet
    assert _run(out, "toy", 0, "--phase", "optimize") == 0
    assert not (out / "selection.json").exists()
    assert _run(out, "toy", 0, "--phase", "validate") == 0
    assert not (out / "test.json").exists()
    calls_before = len(FakeAdapter.CALLS)
    assert _run(out, "toy", 0, "--phase", "test") == 0
    assert all(c["id"].startswith("te") for c in FakeAdapter.CALLS[calls_before:])
    calls_before = len(FakeAdapter.CALLS)
    assert _run(out, "toy", 0) == 0 and len(FakeAdapter.CALLS) == calls_before
    assert _run(out, "toy", 1) == 2  # the out directory belongs to seed 0


def test_off_grid_cells_need_explicit_permission(tmp_path, capsys):
    argv = [
        "--method",
        "gepa",
        "--dataset",
        "math",
        "--topology",
        "single",
        "--model",
        "qwen",
        "--seed",
        "0",
        "--out",
        str(tmp_path / "x"),
        "--framework",
        "crewai",
        "--quiet",
    ]
    assert run.main(argv, hooks=fake_hooks()) == 2
    assert "not a cell of the experiment grid" in capsys.readouterr().err
    conflict = [
        "--method",
        "gepa",
        "--dataset",
        "lcb",
        "--topology",
        "independent_r8",
        "--team-size",
        "2",
        "--model",
        "qwen",
        "--seed",
        "0",
        "--out",
        str(tmp_path / "y"),
        "--quiet",
    ]
    assert run.main(conflict, hooks=fake_hooks()) == 2


def test_aggregate_over_three_seeds(tmp_path, capsys):
    root = tmp_path / "runs"
    for seed in (0, 1, 2):
        assert _run(root / f"toy-{seed}", "toy", seed) == 0
    assert _run(root / "badtoy-0", "badtoy", 0) == 0
    out = tmp_path / "summary.json"
    assert aggregate.main([str(root), "--out", str(out), "--bootstrap", "200", "--include-nonconformant"]) == 0
    summary = json.loads(out.read_text())
    cells = {cell["cell"]["method"]: cell for cell in summary["cells"]}
    toy = cells["toy"]
    assert toy["status"] == "complete" and [row["seed"] for row in toy["per_seed"]] == [0, 1, 2]
    assert toy["mean_delta_pp"] == 50.0 and toy["std_delta_pp"] == 0.0 and toy["fallbacks"] == 0
    assert set(toy["per_seed_exact_mcnemar_p"]) == {"0", "1", "2"} and set(toy["per_seed_mcnemar_p_holm"]) == {
        "0",
        "1",
        "2",
    }
    assert cells["badtoy"]["status"] == "incomplete"
    table = capsys.readouterr().out
    assert "toy" in table and "incomplete" in table
    assert aggregate.main([str(root)]) == 1  # fake-cell jobs are non-conformant by default


def test_aggregate_names_skipped_jobs_relative_to_their_root(tmp_path, caplog):
    root = tmp_path / "runs"
    for job in ("gepa/hotpotqa/0", "mipro/hotpotqa/0"):
        (root / job).mkdir(parents=True)
        (root / job / "result.json").write_text("{")
    job_root = root / "gepa" / "hotpotqa" / "0"
    _, skipped = aggregate.load_results(aggregate.find_results([root, job_root]), roots=[root, job_root])
    assert [message.split(":")[0] for message in skipped] == ["gepa/hotpotqa/0", "mipro/hotpotqa/0"]
    _, skipped = aggregate.load_results(aggregate.find_results([job_root]), roots=[job_root])
    assert skipped[0].startswith(f"{job_root.as_posix()}: unreadable")
    assert aggregate.main([str(root)]) == 1
    assert "[aggregate] skipped mipro/hotpotqa/0: unreadable" in caplog.text
