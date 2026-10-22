"""``core.tasks.apibank``: API-call parsing, upstream replay scoring, voting, prompts, records and runners."""

import importlib
import json

import pytest

from core.tasks import apibank as task

LEVEL1_ID = "level-1::AddAgenda-AddMeeting-GetUserToken-level-2-3.jsonl::0"
GOLD_CALL = "[GetUserToken(username='JohnDoe', password='pass123')]"


@pytest.fixture(scope="module")
def level1_task():
    (row,) = task.load_instances(only=[LEVEL1_ID])
    return row


def test_extract_api_call_prefers_the_last_parseable_call():
    text = "Try [Foo(a=1)] first.\nFinal: [Bar(x='[1, 2]', y=2)] and [Broken(1)]"
    assert task.extract_api_call(text) == "[Bar(x='[1, 2]', y=2)]"
    assert task.extract_api_call("[Broken(1)]") == "[Broken(1)]"
    assert task.extract_api_call("  no call  ") == "no call"


def test_parse_api_call_keyword_arguments_only():
    assert task.parse_api_call("[Bar(x='[1, 2]', y=2, z=name)]") == ("Bar", {"x": [1, 2], "y": 2, "z": "name"})
    assert task.parse_api_call("[Ping()]") == ("Ping", {})
    with pytest.raises(ValueError, match="keyword arguments only"):
        task.parse_api_call("[Bar(1)]")
    with pytest.raises(ValueError, match="no \\[ApiName"):
        task.parse_api_call("Sure, I can help with that.")


def test_level_aliases_and_names():
    assert task.normalize_level("Level-2") == "2"
    assert task.normalize_level("combined") == "all"
    assert task.benchmark_name("l3") == "API-Bank Level-3 API-call curated"
    with pytest.raises(ValueError):
        task.normalize_level("4")


def test_score_prediction_replays_the_upstream_api(level1_task):
    assert level1_task["gold_api_call"] == GOLD_CALL
    assert task.score_prediction(level1_task, GOLD_CALL)["correct"] is True
    wrong = task.score_prediction(level1_task, "[GetUserToken(username='JohnDoe', password='nope')]")
    assert (wrong["correct"], wrong["stage"]) == (False, "score")
    assert task.score_prediction(level1_task, "[AddAgenda(token='x')]")["stage"] == "api_name"
    assert task.score_prediction(level1_task, "no call")["stage"] == "parse"


def test_votes_use_parsed_calls_and_the_earliest_tie():
    calls = ["[A(x=1)]", "[A( x = 1 )]", "", "[B(y=2)]", "[B(y=2)]"]
    agents = [{"predicted_answer": call, "answer_key": task.answer_key(call)} for call in calls]
    assert task.choose_winner(agents) == 0
    assert task.vote_buckets(agents) == {task.answer_key("[A(x=1)]"): 2, task.answer_key("[B(y=2)]"): 2}
    assert task.choose_winner(agents[2:]) == 1
    assert task.choose_winner([{"predicted_answer": ""}]) is None
    assert task.answer_key("  Not A Call ") == "not a call"


def test_system_prompt_adds_the_hard_rule_and_suffix():
    prompt = task.system_prompt("no_such_topology", "solver", "my_style", "SUFFIX")
    assert prompt.startswith("You are solving API-Bank API-call tasks.")
    assert "Implementation style: my_style." in prompt
    assert prompt.endswith(task.HARD_RULE.lstrip("\n").split("\n")[-1] + "\nSUFFIX")
    real = task.system_prompt("single", "solver", "single_langgraph")
    assert task.HARD_RULE in real and "Implementation style" not in real


def test_solve_agent_scores_a_call_and_records_failures(level1_task):
    ok = task.solve_agent(
        lambda inst, **kw: {"raw": f"call: {GOLD_CALL}", "solve_s": 0.04}, level1_task, role="r", seed=3
    )
    assert list(ok)[:5] == ["role", "seed", "solve_s", "raw", "predicted_answer"]
    assert (ok["predicted_answer"], ok["correct"], ok["turns"], ok["seed"]) == (GOLD_CALL, True, 1, 3)

    def fail(inst, **kw):
        raise TimeoutError("slow")

    failed = task.solve_agent(fail, level1_task, role="r", seed=0)
    assert failed == {
        "role": "r",
        "seed": 0,
        "solve_s": 0.0,
        "error": "TimeoutError: slow",
        "raw": "",
        "messages": [],
        "telemetry": {},
    }


def test_replicas_drop_role_api_fields_and_messages(level1_task):
    def solve(inst, *, style, topology, role, seed):
        return {"raw": GOLD_CALL, "messages": [{"role": "assistant"}], "telemetry": {"n_llm_calls": 1}}

    replicas = task.solve_replicas(solve, level1_task, style="s", topology="sequential", role="verifier", n=2)
    assert [r["seed"] for r in replicas] == [0, 1]
    assert list(replicas[0]) == [
        "seed", "solve_s", "raw", "predicted_answer", "answer_key", "answer_correct", "correct", "stage", "error",
        "turns", "tool_calls", "telemetry",
    ]  # fmt: skip


def test_run_rows_writes_records_predictions_and_traces_afresh(tmp_path, level1_task):
    out = {"predicted_answer": GOLD_CALL, "correct": True, "answer_correct": 1, "turns": 1, "winner": 0}

    def run_one(instance, out_dir):
        return task.run_instance(instance, out_dir, style="s", solve=lambda: out)

    for _ in range(2):
        result = task.run_rows([level1_task], run_one, style="s", model_id="m", out_dir=tmp_path, verbose=False)
    assert result == {"n": 1, "correct": 1, "accuracy": 1.0, "style": "s"}
    records = [json.loads(line) for line in (tmp_path / "results.jsonl").read_text().splitlines()]
    assert len(records) == 1
    assert list(records[0])[:9] == [
        "id",
        "idx",
        "level",
        "file",
        "sample_id",
        "question",
        "gold_api_call",
        "gold_api_name",
        "style",
    ]
    assert records[0]["winner"] == 0 and "n_rounds" not in records[0]
    prediction = json.loads((tmp_path / "predictions.jsonl").read_text().splitlines()[0])
    assert prediction["model_name_or_path"] == "m" and prediction["predicted_answer"] == GOLD_CALL
    assert (tmp_path / "traces" / "level-1__AddAgenda-AddMeeting-GetUserToken-level-2-3.jsonl__0.json").exists()

    other = tmp_path / "elsewhere.jsonl"
    task.run_rows(
        [level1_task], run_one, style="s", model_id="m", out_dir=tmp_path, predictions=other, team_size=4, verbose=False
    )
    assert other.exists()


def test_run_replicas_rescores_the_winner_and_reports_total_failure(tmp_path, level1_task):
    winner = {"seed": 1, "predicted_answer": GOLD_CALL, "answer_key": task.answer_key(GOLD_CALL), "raw": GOLD_CALL}
    record = task.run_replicas(
        level1_task, tmp_path, style="s_r2", team_size=2, solve=lambda: [{"seed": 0, "error": "x"}, winner]
    )
    assert (record["winner"], record["correct"], record["turns"], record["n_agents"]) == (1, True, 2, 2)
    assert "raw" not in record["per_agent"][1]
    failed = task.run_replicas(
        level1_task, tmp_path, style="s_r2", team_size=1, solve=lambda: [{"seed": 0, "error": "x"}]
    )
    assert (failed["stage"], failed["error"]) == ("solve", "all API-Bank replicas failed")


def test_command_line_keeps_the_dataset_options():
    from core import cli

    parser = cli.build_parser("d", task.add_arguments, default_limit=task.DEFAULT_LIMIT, info=task.SUMMARY)
    args = parser.parse_args(["--level", "2", "--only", "a", "b"])
    assert (args.limit, args.level, args.only, args.summary) == (2, "2", ["a", "b"], False)


def test_configure_exports_the_scoring_options(monkeypatch, tmp_path):
    monkeypatch.delenv("APIBANK_CURATED_PATH", raising=False)
    monkeypatch.delenv("APIBANK_TOOLSEARCHER_SCORER", raising=False)
    task.configure(curated_path=str(tmp_path / "m.json"), toolsearcher_scorer="keyword")
    assert task.os.environ["APIBANK_CURATED_PATH"] == str(tmp_path / "m.json")
    assert task.os.environ["APIBANK_TOOLSEARCHER_SCORER"] == "keyword"


@pytest.mark.parametrize(
    ("module", "style"),
    [
        ("topologies.sequential.crewai.apibank.crewai_apibank", "sequential_crewai"),
        ("topologies.centralized.autogen.apibank.autogen_apibank", "centralized_autogen"),
        ("topologies.sequential.langgraph.apibank.langgraph_apibank", "sequential_langgraph"),
    ],
)
def test_framework_labels_are_variants_with_their_own_hooks(module, style):
    runner = importlib.import_module(module)
    assert runner.STYLE == style
    assert runner._load_prompt.__globals__ is vars(runner)


@pytest.mark.parametrize(("topology", "role"), [("sequential", "verifier"), ("centralized", "manager")])
def test_team_size_variants(topology, role):
    runner = importlib.import_module(f"teamsizes.{topology}.apibank.apibank_r8")
    assert (runner.TOPOLOGY, runner.ROLE, runner.TEAM_SIZE, runner.STYLE) == (
        topology,
        role,
        8,
        f"{topology}_apibank_r8",
    )
    assert runner.run_one.__globals__ is vars(runner)
