from pathlib import Path

from core import paths


def test_repo_root_is_the_checkout():
    assert paths.REPO_ROOT == Path(__file__).resolve().parents[2]
    for name in ("core", "topologies", "teamsizes", "communications", "optimizers"):
        assert (paths.REPO_ROOT / name).is_dir()


def test_derived_directories():
    assert paths.CONFIGS_DIR == paths.REPO_ROOT / "configs"
    assert paths.PROMPTS_DIR == paths.REPO_ROOT / "configs" / "prompts"
    assert paths.BENCHMARKS_DIR == paths.REPO_ROOT / "benchmarks"
    assert paths.RESULTS_DIR == paths.REPO_ROOT / "results"
    assert (paths.PROMPTS_DIR / "single" / "gpqa" / "solver.txt").is_file()
    assert (paths.BENCHMARKS_DIR / "gpqa" / "gpqa_eval_ids.json").is_file()
