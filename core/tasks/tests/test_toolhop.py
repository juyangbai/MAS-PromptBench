"""``core.tasks.toolhop``: answer scoring, the tool sandbox, the tool loop, voting, records and runners."""

import importlib
import json
from types import SimpleNamespace

import pytest

from core.tasks import toolhop as task

SOURCE = '''
import os

def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

print(add(1, 2))
'''
SAMPLE = {
    "id": 7,
    "question": "What is 2 + 3?",
    "answer": "5",
    "functions": [SOURCE, "def add(a, b):\n    return a - b\n"],
    "tools": {
        "first": {"name": "add", "description": "Add.", "parameters": {"type": "object", "properties": {}}},
        "second": {"name": "add", "description": "Subtract.", "parameters": {"type": "object", "properties": {}}},
    },
}


@pytest.fixture
def allow_exec(monkeypatch):
    monkeypatch.setenv("TOOLHOP_ALLOW_DATASET_EXEC", "1")


def test_extract_and_score_answer():
    assert task.extract_answer("so <answer> 1,234 </answer>") == "1,234"
    assert task.extract_answer(" untagged ") == "untagged"
    assert task.score_answer("1234", "<answer>1234.0</answer>")
    assert not task.score_answer("1234", "<answer>1,234</answer>")
    assert task.score_answer("Paris", "<answer>paris, France</answer>")
    assert task.score_answer("[1, 2]", "<answer>[1, 2]</answer>")
    assert not task.score_answer("Paris", "<answer>Lyon</answer>")
    assert task.score_answer("Paris", "no answer", "tool said Paris")
    assert task.answer_key(" 1,234.0 ") == "1234"


def test_votes_ignore_failed_agents_and_keep_the_earliest_tie():
    agents = [
        {"error": "x"},
        {"answer_key": "b", "predicted_answer": "B"},
        {"answer_key": "a", "predicted_answer": "A"},
    ]
    assert task.choose_winner(agents) == 1
    assert task.vote_buckets(agents) == {"b": 1, "a": 1}
    assert task.choose_winner([{"error": "x"}]) is None


def test_function_map_needs_the_exec_opt_in(monkeypatch):
    monkeypatch.delenv("TOOLHOP_ALLOW_DATASET_EXEC", raising=False)
    with pytest.raises(RuntimeError, match="TOOLHOP_ALLOW_DATASET_EXEC=1"):
        task.function_map(SAMPLE)


def test_sandbox_runs_only_function_definitions_and_renames_duplicates(allow_exec, capsys):
    functions = task.function_map(SAMPLE)
    assert set(functions) == {"add__0", "add__1"}
    assert (functions["add__0"](a=2, b=3), functions["add__1"](a=2, b=3)) == (5, -1)
    assert capsys.readouterr().out == ""
    tools = task.function_tools(SAMPLE, functions)
    assert [(tool.name, tool.description) for tool in tools] == [("add__0", "Add."), ("add__1", "Subtract.")]
    assert tools[0].call({"a": 1, "b": 1}) == 2
    with pytest.raises(ImportError, match="not allowed"):
        task._safe_import("subprocess")


def _response(content="", tool_calls=()):
    message = SimpleNamespace(
        role="assistant",
        content=content,
        tool_calls=list(tool_calls),
        model_dump=lambda exclude_none=True: {"role": "assistant", "content": content},
    )
    usage = SimpleNamespace(prompt_tokens=3, completion_tokens=2, total_tokens=5)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


def _tool_call(arguments):
    function = {"name": "add__0", "arguments": arguments}
    return SimpleNamespace(model_dump=lambda exclude_none=True: {"id": "c1", "type": "function", "function": function})


class FakeClient:
    def __init__(self, responses):
        self.requests = []
        self.responses = list(responses)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return self.responses.pop(0)


def test_tool_loop_executes_calls_until_the_answer(allow_exec):
    llm = FakeClient([_response(tool_calls=[_tool_call('{"a": 2, "b": 3}')]), _response("<answer>5</answer>")])
    out = task.tool_loop(llm, SAMPLE, task.function_map(SAMPLE), "sys", "user", model="m", role="solver", seed=4)
    assert [m["role"] for m in out["messages"]] == ["system", "user", "assistant", "tool", "assistant"]
    assert out["messages"][3]["content"] == "5"
    assert (llm.requests[0]["seed"], llm.requests[0]["model"], llm.requests[0]["tool_choice"]) == (4, "m", "auto")
    assert [t["function"]["name"] for t in llm.requests[0]["tools"]] == ["add__0", "add__1"]
    assert out["telemetry"]["n_llm_calls"] == 2 and out["telemetry"]["n_tool_calls"] == 1


def test_tool_loop_forces_an_answer_when_the_budget_runs_out(allow_exec):
    llm = FakeClient([_response(tool_calls=[_tool_call("{bad json")]), _response("<answer>5</answer>")])
    out = task.tool_loop(llm, SAMPLE, task.function_map(SAMPLE), "sys", "user", model="m", role="planner", max_turns=1)
    assert out["messages"][3]["content"].startswith("an error occurred when parsing arguments for add__0")
    assert json.loads(out["messages"][2]["tool_calls"][0]["function"]["arguments"]) == {
        "_malformed_arguments": "{bad json"
    }
    assert out["messages"][-2]["content"].startswith("The tool-call budget is exhausted.")
    assert llm.requests[1]["temperature"] == 0.0 and "tools" not in llm.requests[1]
    assert task.last_assistant_content(out["messages"]) == "<answer>5</answer>"


def test_solve_agent_records_answers_and_failures():
    messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c"}]},
        {"role": "tool", "content": "5"},
        {"role": "assistant", "content": "<answer>5</answer>"},
    ]
    ok = task.solve_agent(lambda s, **kw: {"messages": messages, "solve_s": 1.26}, SAMPLE, role="r", seed=2)
    assert (ok["predicted_answer"], ok["correct"], ok["turns"], ok["tool_calls"], ok["solve_s"]) == (
        "5",
        True,
        2,
        1,
        1.3,
    )
    assert ok["previous_tool_content"] == "5"

    def fail(sample, **kw):
        raise RuntimeError("boom")

    failed = task.solve_agent(fail, SAMPLE, role="r", seed=0)
    assert failed == {
        "role": "r",
        "seed": 0,
        "solve_s": 0.0,
        "error": "RuntimeError: boom",
        "messages": [],
        "telemetry": {},
    }


def test_run_instance_traces_and_debate_errors(tmp_path):
    out = {"predicted_answer": "5", "correct": True, "error": "debate failed", "messages": []}
    record = task.run_instance(SAMPLE, tmp_path, style="s", solve=lambda: out, debate_errors=True)
    assert (record["error"], record["stage"]) == ("debate failed", "debate")
    assert json.loads((tmp_path / "traces" / "7.json").read_text())["summary"] == record
    plain = task.run_instance(
        SAMPLE, tmp_path, style="s", solve=lambda: {**out, "messages": [{"role": "user", "content": "q"}]}
    )
    assert "error" not in plain
    assert (tmp_path / "traces" / "7.txt").read_text() == "=== USER ===\nq\n\n"


def test_run_replicas_rescores_without_tool_content(tmp_path):
    replicas = [{"seed": 0, "predicted_answer": "5", "answer_key": "5", "turns": 2, "tool_calls": 1, "messages": []}]
    record = task.run_replicas(SAMPLE, tmp_path, style="s_r1", team_size=1, solve=lambda: replicas)
    assert (record["winner"], record["answer_correct"], record["tool_calls"], record["turns"]) == (0, 1, 1, 2)
    assert "messages" not in record["per_agent"][0]
    assert (tmp_path / "traces" / "7.txt").read_text().startswith("style: s_r1\nteam_size: 1\nwinner: 0\n")


def test_run_rows_writes_afresh(tmp_path):
    def run_one(instance, out_dir):
        return {"id": instance["id"], "correct": False, "predicted_answer": "4"}

    for _ in range(2):
        result = task.run_rows([SAMPLE], run_one, style="s", model_id="m", out_dir=tmp_path, team_size=2, verbose=False)
    assert result == {"n": 1, "correct": 0, "accuracy": 0.0, "style": "s", "team_size": 2}
    predictions = [json.loads(line) for line in (tmp_path / "predictions.jsonl").read_text().splitlines()]
    assert len(predictions) == 1 and predictions[0]["question"] == SAMPLE["question"]


def test_command_line_only_lifts_the_default_limit(monkeypatch):
    seen = []
    monkeypatch.setattr(
        task, "load_instances", lambda limit=None, offset=0, only=None: seen.append((limit, only)) or []
    )
    for argv in (["--only", "1"], []):
        assert task.main(argv, description="d", run_one=None, style="s", model_id="m") == 1
    assert seen == [(None, ["1"]), (5, None)]


def test_system_prompt_fallback_names_the_style():
    prompt = task.system_prompt("no_such_topology", "solver", "my_style", "SUFFIX")
    assert prompt.startswith("You are solving ToolHop") and "Implementation style: my_style.\nSUFFIX" in prompt


@pytest.mark.parametrize(
    ("module", "style"),
    [
        ("topologies.sequential.crewai.toolhop.crewai_toolhop", "sequential_crewai"),
        ("topologies.centralized.autogen.toolhop.autogen_toolhop", "centralized_autogen"),
        ("topologies.centralized.langgraph.toolhop.langgraph_toolhop", "centralized_langgraph"),
    ],
)
def test_framework_labels_are_variants_with_their_own_hooks(module, style):
    runner = importlib.import_module(module)
    assert runner.STYLE == style
    assert runner._system_prompt.__globals__ is vars(runner)


@pytest.mark.parametrize(("topology", "role"), [("independent", "solver"), ("decentralized", "debater")])
def test_team_size_variants(topology, role):
    runner = importlib.import_module(f"teamsizes.{topology}.toolhop.toolhop_r2")
    assert (runner.TOPOLOGY, runner.ROLE, runner.TEAM_SIZE, runner.STYLE) == (
        topology,
        role,
        2,
        f"{topology}_toolhop_r2",
    )
