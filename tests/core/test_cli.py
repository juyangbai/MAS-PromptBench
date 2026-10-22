"""``core.cli``: the shared runner flags and how outputs are resolved."""

from pathlib import Path

import pytest

from core import batch, cli


def parse(*argv):
    return cli.build_parser("test runner").parse_args(list(argv))


def test_only_accepts_repeated_and_multi_value_forms():
    assert parse("--only", "a", "b", "--only", "c").only == ["a", "b", "c"]
    assert parse().only is None


def test_predictions_path():
    assert cli.predictions_path(parse(), Path("default.jsonl")) == Path("default.jsonl")
    assert cli.predictions_path(parse("--out-dir", "runs/x")) == Path("runs/x/predictions.jsonl")
    assert cli.predictions_path(parse("--out-dir", "runs/x", "--out", "p.jsonl")) == Path("p.jsonl")


def test_main_passes_accepted_options(tmp_path, capsys):
    calls = {}

    def load_instances(limit=None, offset=0, only=None, category="simple"):
        calls["load"] = (limit, offset, only, category)
        return [{"id": "a"}]

    def run_batch(instances, out_path=None, out_dir=None):
        calls["run"] = (instances, out_path, out_dir)
        return {}

    def add_arguments(parser):
        parser.add_argument("--category", default="simple")

    status = cli.main(
        ["--batch", "--limit", "1", "--only", "a", "--category", "multiple", "--out-dir", str(tmp_path)],
        description="test runner",
        load_instances=load_instances,
        run_batch=run_batch,
        demo=lambda: None,
        add_arguments=add_arguments,
    )
    assert status == 0
    assert calls["load"] == (None, 0, ["a"], "multiple")  # --only lifts --limit
    assert calls["run"] == ([{"id": "a"}], tmp_path / "predictions.jsonl", tmp_path)


def test_main_runs_the_demo_without_batch():
    seen = []
    status = cli.main([], description="d", load_instances=list, run_batch=dict, demo=lambda: seen.append(True))
    assert status == 0 and seen == [True]


def test_main_fails_without_instances(capsys):
    status = cli.main(["--batch"], description="d", load_instances=lambda **_: [], run_batch=dict)
    assert status == 1
    assert "no instances loaded" in capsys.readouterr().err


def test_dataset_defaults_info_mode_and_configure(capsys):
    seen = []
    report = cli.InfoMode("--smoke-dataset", lambda limit=None: {"limit": limit, "name": "ü"}, help="report")
    options = {
        "description": "d",
        "load_instances": lambda limit=None: seen.append(limit) or [],
        "run_batch": dict,
        "default_limit": 5,
        "info": report,
        "configure": lambda offset=0: seen.append(("offset", offset)),
    }
    assert cli.main(["--smoke-dataset", "--offset", "2"], **options) == 0
    assert '"limit": 5' in capsys.readouterr().out and seen == [("offset", 2)]
    assert cli.main([], **options) == 1
    assert seen[-1] == 5


def test_epilog_keeps_its_lines():
    parser = cli.build_parser("d", epilog="examples:\n  %(prog)s --limit 1")
    assert "examples:\n  " in parser.format_help()


def test_preflight_runs_before_the_demo_and_the_batch_but_not_the_info_report(capsys):
    seen = []
    options = {
        "description": "d",
        "load_instances": lambda: seen.append("load") or [{"id": "a"}],
        "run_batch": lambda instances: seen.append("run"),
        "demo": lambda: seen.append("demo"),
        "info": cli.InfoMode("--summary", lambda: {}),
        "preflight": lambda: seen.append("preflight"),
    }
    assert cli.main(["--summary"], **options) == 0 and seen == []
    assert cli.main([], **options) == 0 and seen == ["preflight", "demo"]
    seen.clear()
    assert cli.main(["--batch"], **options) == 0 and seen == ["preflight", "load", "run"]


def test_main_fails_when_every_row_failed_on_infrastructure(capsys):
    def run_batch(instances):
        raise batch.InfrastructureFailure("every row (1) failed on infrastructure")

    status = cli.main(["--batch"], description="d", load_instances=lambda: [{"id": "a"}], run_batch=run_batch)
    assert status == 1
    assert "every row (1) failed on infrastructure" in capsys.readouterr().err


def test_agents_sdk_runner_exits_with_the_install_hint_before_loading(monkeypatch):
    from topologies.decentralized.openai_agents import agents_sdk_base
    from topologies.decentralized.openai_agents.gpqa import openai_agents_gpqa as runner

    def unavailable():
        raise RuntimeError(agents_sdk_base._INSTALL_HINT)

    monkeypatch.setattr(agents_sdk_base, "load_agents_sdk", unavailable)
    monkeypatch.setattr(runner, "load_instances", lambda **_: pytest.fail("instances loaded without the SDK"))
    with pytest.raises(SystemExit) as exited:
        runner.main(["--batch", "--limit", "1"])
    assert exited.value.code == agents_sdk_base._INSTALL_HINT
