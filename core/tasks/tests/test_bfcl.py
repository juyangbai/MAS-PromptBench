"""``core.tasks.bfcl`` and the BFCL teams: data, prompts, schema tools, voting, AST scoring and batch files."""

import importlib
import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.utils.function_calling import convert_to_openai_tool

from core import teams
from core.tasks import bfcl as task


@pytest.fixture(scope="module")
def parallel_0():
    rows, gts = task.load_instances("parallel", only=["parallel_0"])
    return rows[0], gts[0]


def test_load_instances_slices_and_aligns_possible_answers():
    rows, gts = task.load_instances("simple", limit=2, offset=1)
    assert [row["id"] for row in rows] == [gt["id"] for gt in gts] == ["simple_1", "simple_2"]
    rows, _ = task.load_instances("simple", limit=1, only=["simple_5", "simple_3"])
    assert [row["id"] for row in rows] == ["simple_3"]


def test_command_line_defaults_and_only_lifting_the_limit(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(task, "load_instances", lambda *args: seen.append(args) or ([], []))
    for argv in (["--category", "multiple", "--only", "multiple_1"], ["--offset", "2"]):
        status = task.cli_main(argv, description="d", run_one=_run_one, model_id="m", default_out_dir=tmp_path)
        assert status == 1
    assert seen == [("multiple", None, 0, ["multiple_1"]), ("simple", task.DEFAULT_LIMIT, 2, None)]


SEQUENTIAL_TAIL = 'USER REQUEST:\n[user] Use {braces}\n\nSCHEMAS:\n[\n  {\n    "name": "f"\n  }\n]'


@pytest.fixture
def braced_instance():
    return {"question": [[{"role": "user", "content": "Use {braces}"}]], "function": [{"name": "f"}]}


@pytest.mark.parametrize("size", teams.TEAM_SIZES)
def test_sequential_langgraph_stages_show_single_braces(size, braced_instance):
    assert task.flatten_question(braced_instance["question"]) == "[user] Use {braces}"
    inputs = task.stage_inputs(braced_instance)
    texts = [stage.task.format(**inputs) for stage in teams.spec("sequential", "bfcl", size).stages]
    assert texts[-1].endswith(SEQUENTIAL_TAIL)
    assert not any("{{" in text for text in texts)


def test_sequential_crewai_tasks_show_single_braces(braced_instance):
    runner = importlib.import_module("topologies.sequential.crewai.bfcl.crewai_bfcl")
    crew_tasks = runner.build_crew().tasks
    for crew_task in crew_tasks:
        crew_task.interpolate_inputs_and_add_conversation_history(task.stage_inputs(braced_instance))
    assert crew_tasks[-1].description.endswith(SEQUENTIAL_TAIL)
    assert '[{"calculate_triangle_area": {"base": 10, "height": 5}}]' in crew_tasks[2].description
    assert not any("{{" in crew_task.description for crew_task in crew_tasks)


def test_schema_to_tool_sanitizes_parameter_names_and_maps_types():
    schema = {
        "name": "math.gcd",
        "description": "Greatest common divisor.",
        "parameters": {
            "type": "dict",
            "properties": {
                "_from": {"type": "integer", "description": "first"},
                "class": {"type": "array", "items": {"type": "float"}, "description": "second"},
                "note": {"type": "any", "description": "optional"},
            },
            "required": ["_from", "class"],
        },
    }
    tool = task.schema_to_tool(schema)
    spec = convert_to_openai_tool(tool)["function"]
    assert (spec["name"], spec["description"]) == ("math.gcd", "Greatest common divisor.")
    assert list(spec["parameters"]["properties"]) == ["from_", "class_", "note"]
    assert spec["parameters"]["properties"]["class_"]["items"] == {"type": "number"}
    assert spec["parameters"]["required"] == ["from_", "class_"]
    assert tool.invoke({"_from": 4, "class": [6.0]}) == ""  # arguments validate under the BFCL names
    assert task.parameter_names(schema) == {"from_": "_from", "class_": "class", "note": "note"}
    call = {"name": "math.gcd", "args": {"from_": 4, "class_": [6.0]}, "id": "1"}
    assert task.to_canonical([call], [schema]) == [{"math.gcd": {"_from": 4, "class": [6.0]}}]
    assert task.to_canonical([call]) == [{"math.gcd": {"from_": 4, "class_": [6.0]}}]


def test_tool_calls_get_back_the_schema_names_of_every_bfcl_function():
    renamed = set()
    for category in task.AST_CATEGORIES:
        rows, _ = task.load_instances(category)
        for row in rows:
            for schema in row["function"]:
                shown = convert_to_openai_tool(task.schema_to_tool(schema))["function"]["parameters"]
                names = list(shown.get("properties") or {})
                call = {"name": schema["name"], "args": {name: i for i, name in enumerate(names)}, "id": "0"}
                (canonical,) = task.to_canonical([call], row["function"])
                original = list((schema.get("parameters") or {}).get("properties") or {})
                assert list(canonical[schema["name"]]) == original, (row["id"], schema["name"])
                assert list(canonical[schema["name"]].values()) == list(range(len(original)))
                if names != original:
                    renamed.add(row["id"])
    assert renamed == {"simple_348", "multiple_143", "multiple_197", "parallel_multiple_9", "parallel_multiple_15"}


def test_a_correct_call_under_the_tool_names_passes_the_ast_checker():
    rows, gts = task.load_instances("simple", only=["simple_348"])
    call = {
        "name": "create_player_profile",
        "args": {"player_name": "StarPlayer", "class_": "Mage", "starting_level": 5},
    }
    model = "test-org/unregistered-model"
    task.register_model(model)

    def valid(model_output):
        return task.score_one(rows[0]["function"], model_output, gts[0]["ground_truth"], "simple", model)["valid"]

    assert valid(task.to_canonical([call], rows[0]["function"]))
    assert not valid(task.to_canonical([call]))


def test_first_tool_calls_in_canonical_form():
    messages = [
        HumanMessage("q"),
        AIMessage("", tool_calls=[{"name": "f", "args": {"a": 1}, "id": "1"}]),
        AIMessage("", tool_calls=[{"name": "g", "args": {}, "id": "2"}]),
    ]
    assert task.to_canonical(task.extract_first_tool_calls(messages)) == [{"f": {"a": 1}}]
    assert task.extract_first_tool_calls(messages[:1]) == []


def test_majority_vote_ignores_argument_and_call_order():
    a = {"agent_id": 0, "model_output": [{"f": {"x": 1, "y": 2}}, {"g": {}}]}
    b = {"agent_id": 1, "model_output": [{"g": {}}, {"f": {"y": 2, "x": 1}}]}
    c = {"agent_id": 2, "model_output": [{"h": {}}]}
    d = {"agent_id": 3, "model_output": []}
    assert task.majority_vote([c, b, a, d]) is a
    assert task.majority_vote([c, {"agent_id": 1, "model_output": [{"k": {}}]}])["agent_id"] == 1
    assert task.majority_vote([d, {"agent_id": 4, "model_output": []}]) is d
    assert task.vote_counts([c, b, a, d]) == [(task.canonical_key(a["model_output"]), 2), ('[{"h": {}}]', 1)]


def test_the_submitted_call_is_the_vote_over_canonical_calls():
    f, g = [{"f": {"x": 1}}], [{"g": {"y": 2}}]
    assert task.select_call([g, f, [{"f": {"x": 1}}], g, f]) == 1  # a majority (3 of 5)
    assert task.select_call([f, g]) == 0  # a tie: the lowest peer
    assert task.select_call([g, f, f, g]) == 0
    assert task.select_call([None, [], g]) == 2  # peers without a parseable call abstain
    assert task.select_call([None, None]) == 0


def test_decentralized_runner_submits_the_voted_call_and_scores_it_once(monkeypatch, tmp_path):
    runner = importlib.import_module("topologies.decentralized.langgraph.bfcl.langgraph_bfcl")
    gold = [{"f": {"x": 1}}]
    finals = [json.dumps(gold), '[{"g": {}}]', "no call", '[{"g": {}}]']

    class Graph:
        def invoke(self, state):
            contexts = state["contexts"]
            return {"contexts": [ctx + [AIMessage(f"```json\n{text}\n```")] for ctx, text in zip(contexts, finals)]}

    scored = []

    def score_one(function_schemas, model_output, ground_truth, category):
        scored.append(model_output)
        return {"valid": model_output == ground_truth}

    monkeypatch.setattr(runner, "N_AGENTS", len(finals))
    monkeypatch.setattr(runner, "_build_graph", Graph)
    monkeypatch.setattr(runner, "score_one", score_one)
    instance = {"id": "simple_0", "question": [[{"role": "user", "content": "q"}]], "function": [{"name": "f"}]}
    summary = runner.run_one(instance, {"ground_truth": gold}, "simple", tmp_path)
    assert (summary["winner"], summary["model_output"], summary["valid"]) == (1, [{"g": {}}], False)
    assert scored == [[{"g": {}}]]


def test_ast_scoring_keeps_dotted_names_for_a_registered_model(parallel_0):
    row, gt = parallel_0
    model = "test-org/unregistered-model"
    task.register_model(model)
    calls = [
        {"spotify.play": {"artist": "Maroon 5", "duration": 15}},
        {"spotify.play": {"artist": "Taylor Swift", "duration": 20}},
    ]
    assert task.score_one(row["function"], calls, gt["ground_truth"], "parallel", model)["valid"]
    wrong = calls[:1]
    assert not task.score_one(row["function"], wrong, gt["ground_truth"], "parallel", model)["valid"]


def test_add_verdict_records_the_report_or_the_failure():
    instance, gt = {"function": []}, {"ground_truth": []}

    def verdict(report_or_error):
        def score_one(*args):
            if isinstance(report_or_error, Exception):
                raise report_or_error
            return report_or_error

        return task.add_verdict({"model_output": []}, score_one, instance, gt, "simple")

    invalid = verdict({"valid": False, "error_type": "t", "error": ["a", "b", "c", "d"]})
    assert invalid == {"model_output": [], "valid": False, "error_type": "t", "score_error": ["a", "b", "c"]}
    assert verdict({"valid": True}) == {"model_output": [], "valid": True, "error_type": None}
    failed = verdict(KeyError("x"))
    assert failed == {"model_output": [], "valid": False, "error": "KeyError: 'x'", "stage": "score"}


def test_trace_texts():
    assert task.messages_trace([{"source": "manager", "content": "hi"}, {"content": "x"}]) == (
        "=== MANAGER ===\nhi\n\n=== ? ===\nx\n\n"
    )
    out = {"winner": 1, "per_peer": [{"peer": 0, "call": None}, {"peer": 1, "call": [{"f": {"b": 1}}], "valid": True}]}
    assert task.peer_trace(out) == (
        'winner: peer 1\n\n=== peer 0 ===\n(no call)\n\n=== peer 1 ===\n[{"f": {"b": 1}}]\n'
        "  valid=True  error_type=None\n\n"
    )


def _run_one(row, gt, category, out_dir):
    return {"id": row["id"], "category": category, "model_output": [{"f": {}}], "valid": row["id"].endswith("_0")}


def test_run_batch_writes_predictions_and_results_afresh(tmp_path):
    rows = [{"id": "simple_0"}, {"id": "simple_1"}]
    for _ in range(2):
        summary = task.run_batch(_run_one, rows, rows, "simple", tmp_path, model_id="m", verbose=False)
    assert summary == {"n": 2, "valid": 1, "valid_rate": 0.5, "category": "simple"}
    predictions = [json.loads(line) for line in (tmp_path / "predictions.jsonl").read_text().splitlines()]
    assert len(predictions) == 2
    assert predictions[0] == {
        "id": "simple_0",
        "category": "simple",
        "model_output": [{"f": {}}],
        "model_name_or_path": "m",
    }
    assert len((tmp_path / "results.jsonl").read_text().splitlines()) == 2


def test_cli_main_writes_predictions_to_out_and_results_to_the_default_dir(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(task, "load_instances", lambda category, limit, offset, only: ([{"id": f"{category}_0"}],) * 2)
    status = task.cli_main(
        ["--category", "multiple", "--out", str(tmp_path / "p.jsonl")],
        description="runner",
        run_one=_run_one,
        model_id="m",
        default_out_dir=tmp_path / "default",
    )
    assert status == 0
    assert json.loads((tmp_path / "p.jsonl").read_text())["id"] == "multiple_0"
    assert json.loads((tmp_path / "default" / "results.jsonl").read_text())["category"] == "multiple"


def test_team_specs():
    assert [s.role for s in teams.spec("sequential", "bfcl", 2).stages] == ["caller", "verifier"]
    assert teams.spec("centralized", "bfcl", 2).max_turns == 12
    r8 = teams.spec("centralized", "bfcl", 8)
    assert (r8.manager, r8.max_turns, len(r8.workers)) == ("manager_r8", 24, 7)
    assert r8.delegation_note.endswith("The seven workers are: " + ", ".join(w.role for w in r8.workers) + ".")


def test_team_size_variants_run_the_topology_runner_with_their_team():
    variant = importlib.import_module("teamsizes.centralized.bfcl.bfcl_r8")
    base = importlib.import_module("topologies.centralized.langgraph.bfcl.langgraph_bfcl")
    assert variant.TEAM.size == 8 and base.TEAM.size == 4
    assert [t.name for t in variant.DELEGATION_TOOLS] == [f"delegate_to_{w.role}" for w in variant.TEAM.workers]
    assert (variant.DEFAULT_OUT_DIR.name, base.DEFAULT_OUT_DIR.name) == (
        "bfcl_centralized_r8",
        "bfcl_centralized_langgraph",
    )
    assert variant.delegate_to_type_check_worker.invoke({"instructions": "check"}) == "check"
