"""``core.tasks.lcb`` and the LCB teams: prompts, test execution, records."""

import importlib
import json

import pytest
from langchain_core.messages import AIMessage

from core import code_tasks, teams
from core.tasks import lcb as task

STDIN_SUM = "n = int(input())\nprint(n * (n + 1) // 2)"
TWO_SUM = (
    "class Solution:\n"
    "    def twoSum(self, nums, target):\n"
    "        seen = {}\n"
    "        for i, x in enumerate(nums):\n"
    "            if target - x in seen:\n"
    "                return [seen[target - x], i]\n"
    "            seen[x] = i\n"
)


def test_format_prompt_selects_the_mode():
    stdin = task.format_prompt("P")
    assert stdin == "P\n\n" + task.FORMAT_STDIN
    assert task.format_prompt("P", stdin_format=task.FORMAT_STDIN_SHORT).endswith("```python\n# YOUR CODE HERE\n```")
    functional = task.format_prompt("P", "def f(x: dict = {}):\n    ")
    assert functional.endswith("```python\ndef f(x: dict = {}):\n```")


def test_run_tests_stdin_and_functional():
    stdin = task.run_tests(STDIN_SUM, task.STDIN_DEMO_TESTS, timeout_s=10)
    assert (stdin["pass"], stdin["total"], stdin["pass_rate"]) == (4, 4, 1.0)
    assert stdin["details"][0]["mode"] == "stdin"
    functional = task.run_tests(TWO_SUM, task.FUNCTIONAL_DEMO_TESTS, timeout_s=10)
    assert functional["pass_rate"] == 1.0 and functional["details"][0]["mode"] == "functional"
    wrong = task.run_tests("print(0)", task.STDIN_DEMO_TESTS[:1], timeout_s=10)
    assert wrong["pass_rate"] == 0.0 and code_tasks.exact_match_score(wrong["pass_rate"]) == 0.0
    assert task.run_tests("print(0)", []) == code_tasks.unscored(0)


def test_compare_stdout_tolerates_decimal_formatting():
    assert task.compare_stdout("1.50 2\n", "1.5 2")
    assert not task.compare_stdout("1 2", "1 3")


def test_decode_private_tests_layers():
    import base64
    import pickle
    import zlib

    tests = [{"input": "1\n", "output": "1", "testtype": "stdin"}]
    blob = base64.b64encode(zlib.compress(pickle.dumps(json.dumps(tests)))).decode()
    assert task.decode_private_tests(blob) == tests
    assert task.decode_private_tests("") == [] and task.decode_private_tests("not base64!") == []


def test_records_and_summary():
    inst = {
        "id": "x",
        "problem": "p" * 500,
        "starter_code": "",
        "tests": [{}, {}],
        "difficulty": None,
        "platform": "atcoder",
    }
    rec = task.record(inst, None, task.test_scores(None, inst["tests"], 6), latency_s=1.0, error=None)
    assert list(rec)[:9] == "id problem starter_code predicted_code pass total pass_rate em difficulty".split()
    assert len(rec["problem"]) == 400 and rec["total"] == 2 and rec["em"] == 0.0
    solved = task.record(
        dict(inst, id="y", difficulty="easy"), "c", code_tasks.selection_scores("c", 0, 1.0), latency_s=2.0
    )
    summary = code_tasks.summarize([rec, solved])
    assert summary["n_extracted"] == 1 and summary["em"] == 0.5
    assert summary["by_difficulty"] == {"unk": {"n": 1, "em": 0.0}, "easy": {"n": 1, "em": 1.0}}
    assert "winner=0  pass_rate=1.00" in code_tasks.progress_line(1, 2, solved, [rec, solved], id_width=22)


def _fake_tests(monkeypatch, module):
    """Make ``module.run_tests`` record each program it runs; only ``STDIN_SUM`` passes."""
    ran = []

    def run_tests(code, tests, timeout_s=task.TEST_TIMEOUT_S):
        ran.append(code)
        passed = code == STDIN_SUM
        return {"pass": int(passed), "total": 1, "pass_rate": float(passed), "details": []}

    monkeypatch.setattr(module, "run_tests", run_tests)
    return ran


def test_independent_runner_submits_the_voted_program_and_runs_only_its_tests(monkeypatch):
    runner = importlib.import_module("topologies.independent.lcb.langgraph_lcb")
    codes = [STDIN_SUM, "print(0)", "print(0)\n", None]

    class Graph:
        def compile(self):
            return self

        async def ainvoke(self, state):
            return {"answers": [{"agent_id": i, "seed": i, "code": c} for i, c in reversed(list(enumerate(codes)))]}

    monkeypatch.setattr(runner, "build_graph", Graph)
    ran = _fake_tests(monkeypatch, runner)
    out = runner.solve("P", tests=[{}])
    assert (out["winner"], out["code"], out["pass_rate"]) == (1, "print(0)", 0.0)
    assert ran == ["print(0)"]
    codes[2] = "print(1)"  # a three-way tie: the lowest agent
    assert runner.solve("P")["winner"] == 0 and ran == ["print(0)"]


def test_decentralized_runner_submits_the_voted_program_and_runs_only_its_tests(monkeypatch):
    runner = importlib.import_module("topologies.decentralized.langgraph.lcb.langgraph_lcb")
    finals = ["no program", f"```python\n{STDIN_SUM}\n```", "```python\nprint(0)\n```", "```python\nprint(0)\n```"]

    class Graph:
        def invoke(self, state):
            return {"contexts": [ctx + [AIMessage(text)] for ctx, text in zip(state["contexts"], finals)]}

    monkeypatch.setattr(runner, "N_AGENTS", len(finals))
    monkeypatch.setattr(runner, "_build_graph", Graph)
    ran = _fake_tests(monkeypatch, task)
    out = runner.solve("P", tests=[{}])
    assert (out["winner"], out["code"], ran) == (2, "print(0)", ["print(0)"])
    assert [p["pass_rate"] for p in out["per_peer"]] == [None, None, 0.0, None]
    assert code_tasks.winner_pass_rate(out) == 0.0


@pytest.mark.parametrize("size", teams.TEAM_SIZES)
def test_lcb_teams(size):
    centralized = teams.spec("centralized", "lcb", size)
    assert centralized.manager == {8: "manager_r8", 10: "manager_r10"}.get(size, "manager")
    assert len(centralized.roles) == size
    sequential = teams.spec("sequential", "lcb", size)
    assert sequential.stages[-1].role == ("code_reviewer" if size >= 8 else "debugger")
    assert all("{problem_prompt}" in stage.task for stage in sequential.stages)
    assert teams.spec("decentralized", "lcb", size).recursion_limit == 18


def test_team_size_variant_runs_the_topology_runner():
    module = importlib.import_module("teamsizes.centralized.lcb.lcb_r2")
    assert module.TEAM_SIZE == 2 and module.MAX_TURNS == 13
    assert [t.name for t in module.MANAGER_TOOLS] == ["python_exec", "delegate_to_coder_worker"]
    assert module._MANAGER_TERMINATE_NUDGE.endswith("The only worker is: coder_worker.")
    assert str(module.DEFAULT_PREDICTIONS).endswith("results/lcb_centralized_r2/predictions.jsonl")
