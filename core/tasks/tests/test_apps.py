"""``core.tasks.apps`` and the APPS teams: prompts, the comparison cascade, test execution, records."""

import importlib

import pytest

from core import code_tasks, teams
from core.tasks import apps as task

SQUARE = "n = int(input())\nprint(n * n)"
TWO_SUM = (
    "class Solution:\n"
    "    def twoSum(self, nums, target):\n"
    "        seen = {}\n"
    "        for i, x in enumerate(nums):\n"
    "            if target - x in seen:\n"
    "                return (seen[target - x], i)\n"
    "            seen[x] = i\n"
)


def test_format_prompt_scaffold():
    assert task.format_prompt("Q") == (
        "QUESTION:\nQ\n\nUse Standard Input format.\n\n"
        "Enclose your final solution in a ```python``` code block.\n\nANSWER:"
    )
    call_based = task.format_prompt("Q", "def f(d={}):\n    ")
    assert "```python\ndef f(d={}):\n```\n\nUse Call-Based format." in call_based


def test_parse_input_output():
    assert task.parse_input_output('{"inputs": ["1"], "outputs": ["1"]}') == {"inputs": ["1"], "outputs": ["1"]}
    assert task.parse_input_output('{"inputs": [], "outputs": []}') is None
    assert task.parse_input_output("not json") is None and task.parse_input_output("") is None


def test_comparison_cascade():
    assert task.stdout_compare("1.0000001 2\n", "1 2")
    assert task.stdout_compare("a  \nb", "a\nb")
    assert not task.stdout_compare("1 2", "1 3")
    assert task.call_based_compare((1, 2), [1, 2])
    assert task.call_based_compare([2, 1], [1, 2])
    assert task.call_based_compare([0.33333], [1 / 3])
    assert not task.call_based_compare([1, 2], [1, 3])


def test_run_tests_stdin_and_call_based():
    stdin = task.run_tests(SQUARE, task.STDIN_DEMO_TESTS, timeout_s=10)
    assert (stdin["pass"], stdin["total"], stdin["pass_rate"]) == (4, 4, 1.0)
    assert stdin["details"][0]["mode"] == "stdin"
    call_based = task.run_tests(TWO_SUM, task.CALL_BASED_DEMO_TESTS, timeout_s=10)
    assert call_based["pass_rate"] == 1.0 and call_based["details"][0]["mode"] == "call_based"
    missing = task.run_tests("x = 1", task.CALL_BASED_DEMO_TESTS, timeout_s=10)["details"][0]
    assert missing["ok"] is False and missing["mode"] == "call_based"
    mismatch = task.run_tests("print(1)", {"inputs": ["1"], "outputs": []})
    assert mismatch["error"] == "inputs/outputs length mismatch" and mismatch["total"] == 0


def test_records_have_no_platform():
    inst = {
        "id": "7",
        "problem": "p",
        "starter_code": "",
        "input_output": {"inputs": ["1", "2"], "outputs": ["1", "4"]},
    }
    rec = task.record(inst, None, task.test_scores(None, inst["input_output"], 4), latency_s=0.0)
    assert list(rec) == "id problem starter_code predicted_code pass total pass_rate em difficulty latency_s".split()
    assert rec["total"] == 2 and code_tasks.summarize([rec])["by_difficulty"] == {"unk": {"n": 1, "em": 0.0}}


@pytest.mark.parametrize("size", teams.TEAM_SIZES)
def test_apps_teams(size):
    assert len(teams.spec("centralized", "apps", size).roles) == size
    stages = teams.spec("sequential", "apps", size).stages
    with_tools = [s.role for s in stages if s.tools]
    assert with_tools == [s.role for s in stages if s.role in ("tester", "debugger", "optimizer")]


def test_team_size_variant_runs_the_topology_runner():
    module = importlib.import_module("teamsizes.sequential.apps.apps_r10")
    assert module.TEAM_SIZE == 10 and len(module.TEAM.stages) == 10
    assert module.TEAM.stages[-1].role == "code_reviewer"
    centralized = importlib.import_module("teamsizes.centralized.apps.apps_r8")
    assert centralized._manager_system().endswith("debug_worker, review_worker.")
