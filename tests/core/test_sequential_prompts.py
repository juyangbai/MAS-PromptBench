"""Sequential stage prompts: the model reads the task's braces as they are (LangGraph templates, CrewAI tasks).

BFCL's sequential prompts are covered in ``core/tasks/tests/test_bfcl.py``.
"""

import importlib

import pytest

from core import teams

BRACED = "Print {x} for d = {}: \\frac{1}{2}."
STARTER = "def f(d: dict = {}):\n    "


class _Captured(Exception):
    """Raised by the stand-in graph with the inputs a runner's ``solve`` built."""


class _CaptureGraph:
    def invoke(self, state, config=None):
        raise _Captured(state["inputs"])


SOLVE_ARGS = {
    "gpqa": ((BRACED, ["{a}", "b", "c", "d"]), "question"),
    "hotpotqa": ((BRACED,), "question"),
    "math": ((BRACED,), "problem"),
    "lcb": ((BRACED, STARTER), "problem_prompt"),
    "apps": ((BRACED, STARTER), "problem_prompt"),
}


def _built_inputs(monkeypatch, dataset: str) -> dict:
    """The stage-template inputs the sequential LangGraph runner's ``solve`` builds for a braced task."""
    runner = importlib.import_module(f"topologies.sequential.langgraph.{dataset}.langgraph_{dataset}")
    monkeypatch.setattr(runner, "_build_llm", lambda: None)
    monkeypatch.setattr(runner, "_build_graph", lambda llm: (_CaptureGraph(), ["stage"]))
    args, _ = SOLVE_ARGS[dataset]
    with pytest.raises(_Captured) as captured:
        runner.solve(*args)
    return captured.value.args[0]


@pytest.mark.parametrize("dataset", sorted(SOLVE_ARGS))
def test_langgraph_stages_show_single_braces(monkeypatch, dataset):
    inputs = _built_inputs(monkeypatch, dataset)
    assert BRACED in inputs[SOLVE_ARGS[dataset][1]]
    if dataset in ("lcb", "apps"):
        assert STARTER.rstrip() in inputs["problem_prompt"]
    for size in teams.TEAM_SIZES:
        for stage in teams.spec("sequential", dataset, size).stages:
            text = stage.task.format(**inputs)
            assert BRACED in text and "{{" not in text, (size, stage.role)


@pytest.mark.parametrize("size", teams.TEAM_SIZES)
def test_swe_langgraph_stages_show_single_braces(size):
    for stage in teams.spec("sequential", "swe", size).stages:
        text = stage.task.format(task_brief=BRACED)
        assert BRACED in text and "{{" not in text


@pytest.mark.parametrize("dataset", ["apps", "gpqa", "hotpotqa", "lcb", "math", "swe"])
def test_crewai_tasks_and_agents_show_single_braces(dataset):
    runner = importlib.import_module(f"topologies.sequential.crewai.{dataset}.crewai_{dataset}")
    key = {"gpqa": "question", "hotpotqa": "question", "math": "problem", "swe": "task_brief"}.get(dataset)
    inputs = {key: BRACED} if key else {"problem_prompt": runner.format_prompt(BRACED, STARTER)}
    crew = runner.build_crew()
    for agent in crew.agents:
        agent.interpolate_inputs(inputs)
        assert "{{" not in agent.role + agent.goal + agent.backstory
    for crew_task in crew.tasks:
        crew_task.interpolate_inputs_and_add_conversation_history(inputs)
        assert "{{" not in crew_task.description + crew_task.expected_output
    assert BRACED in crew.tasks[0].description
