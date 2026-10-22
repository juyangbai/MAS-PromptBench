"""``core.tasks.gpqa`` and the GPQA teams: choice shuffling, letter extraction, votes, records."""

import importlib

import pytest

from core import teams
from core.tasks import gpqa as task


def test_row_id_hashes_the_question():
    assert task.row_id({"Question": "  Why?  "}, 3) == task.row_id({"Question": "Why?"}, 9)
    assert task.row_id({"Question": "Why?"}, 3).startswith("gpqa_")
    assert task.row_id({"Question": ""}, 3) == "gpqa_idx_3"


def test_shuffle_is_seeded_and_tracks_the_correct_answer():
    choices, letter = task.shuffle_choices("right", ["w1", "w2", "w3"], "0|gpqa_x")
    assert sorted(choices) == ["right", "w1", "w2", "w3"]
    assert choices[task.LETTERS.index(letter)] == "right"
    assert task.shuffle_choices("right", ["w1", "w2", "w3"], "0|gpqa_x") == (choices, letter)


def test_prompts_list_the_options():
    assert task.format_prompt("Q?", ["a", "b", "c", "d"]) == "Q?\n\nA) a\nB) b\nC) c\nD) d"
    assert task.format_agents_prompt("Q?", ["a", "b", "c", "d"]).startswith("Q?\n\nA. a\nB. b\nC. c\nD. d\n\n")
    with pytest.raises(AssertionError):
        task.format_prompt("Q?", ["a", "b"])
    peer = task.peer_review_prompt(["x", "y"], "MCQ")
    assert "\nPeer 2:\n```\ny\n```" in peer and peer.endswith("Original question:\nMCQ")


@pytest.mark.parametrize(
    ("text", "letter"),
    [
        ("Reasoning.\nAnswer: B", "B"),
        ("**Answer:** (c)", "C"),
        ("The correct option is D.", "D"),
        ("I pick\nA\n", "A"),
        ("Answer: A ... on reflection, Final answer: C", "C"),
        ("Answer: B and option C", "B"),
        ("I am not sure.", None),
    ],
)
def test_extract_answer_cascade(text, letter):
    assert task.extract_answer(text) == letter


def test_votes_break_ties_by_first_seen():
    assert task.majority_vote([{"answer": "B"}, {"answer": None}, {"answer": "A"}, {"answer": "A"}]) == "A"
    assert task.majority_vote([{"answer": "C"}, {"answer": "B"}]) == "C"
    assert task.majority_vote([{"answer": None}]) is None
    assert task.best_of_n([None, "D", "B", "B", "D"]) == "D"
    assert task.best_of_n(["E", None]) is None
    assert task.votes(["B", None, "A", "B"]) == {"B": 2, "A": 1}


def test_records_and_summary():
    inst = {"id": "gpqa_x", "question": "Q?", "choices": ["a", "b", "c", "d"], "correct_letter": "B"}
    right = task.record(inst, "B", raw="Answer: B", latency_s=1.0)
    wrong = task.record(inst, None, latency_s=2.0)
    shared = ["id", "question", "choices", "correct_letter", "predicted_letter", "correct"]
    assert list(right) == [*shared, "raw", "latency_s"]
    assert right["correct"] and not wrong["correct"]
    assert task.summarize([right, wrong]) == {
        "n": 2,
        "n_extracted": 1,
        "n_correct": 1,
        "accuracy": 0.5,
        "extracted_acc": 1.0,
    }
    assert task.per_peer_tails([{"peer": 0, "letter": "A", "raw": "x" * 400}])[0]["raw_tail"] == "x" * 300
    assert task.stage_excerpts({"solver": "y" * 900, "critic": None}) == {"solver": "y" * 800, "critic": ""}


@pytest.mark.parametrize("size", teams.TEAM_SIZES)
def test_gpqa_team_sizes(size):
    sequential = teams.spec("sequential", "gpqa", size)
    centralized = teams.spec("centralized", "gpqa", size)
    assert len(sequential.stages) == size and sequential.stages[-1].role == "verifier"
    assert all("{question}" in stage.task for stage in sequential.stages)
    assert len(centralized.workers) == size - 1
    assert centralized.max_turns == (8 if size == 2 else 16)
    for topology in ("independent", "decentralized"):
        assert teams.spec(topology, "gpqa", size).n_agents == size


def test_r10_confidence_reporter_has_no_tools(monkeypatch):
    stages = {stage.role: stage.tools for stage in teams.spec("sequential", "gpqa", 10).stages}
    assert stages["confidence_reporter"] == () and stages["verifier"] == ("calculator",)

    module = importlib.import_module("teamsizes.centralized.gpqa.gpqa_r10")
    tools = {}

    def record_worker(name, worker_tools, llm):
        tools[name] = worker_tools
        return lambda state: {}

    monkeypatch.setattr(module, "_make_worker_node", record_worker)
    module._build_graph(object())
    assert tools.pop("confidence_reporter_worker") == []
    assert all(t == [module.calculator] for t in tools.values()) and len(tools) == 8
    assert module.DEFAULT_PREDICTIONS.parent.name == "gpqa_centralized_r10"
