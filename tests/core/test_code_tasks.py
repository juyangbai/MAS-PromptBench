"""``core.code_tasks``: code extraction, python_exec, guarded calls, selection and records of LCB and APPS."""

from langchain_core.tools import tool

from core import code_tasks

TWO_SUM = (
    "class Solution:\n"
    "    def twoSum(self, nums, target):\n"
    "        seen = {}\n"
    "        for i, x in enumerate(nums):\n"
    "            if target - x in seen:\n"
    "                return (seen[target - x], i)\n"
    "            seen[x] = i\n"
)


def test_python_exec_description_and_output():
    python_exec = tool(code_tasks.make_python_exec(code_tasks.PYTHON_EXEC_DOC))
    assert python_exec.name == "python_exec"
    assert python_exec.description == code_tasks.PYTHON_EXEC_DOC.rstrip()
    assert (
        python_exec.invoke({"code": "print(input()[::-1])", "stdin": "abc"})
        == "stdout:\ncba\n\nstderr:\n\nexit_code: 0"
    )
    clipped = code_tasks.make_python_exec(code_tasks.PYTHON_EXEC_DOC, char_budget=3)("print('abcdef')")
    assert clipped.startswith("stdout:\nabc\n...<truncated tool output>...")
    assert code_tasks.PYTHON_EXEC_DOC_SHORT.startswith(
        "Execute a Python code snippet in a fresh subprocess.\n\n    Args:"
    )


def test_extract_code_takes_the_last_python_block():
    text = "```python\nprint(1)\n```\ntext\n```python\nprint(2)\n```\nTERMINATE"
    assert code_tasks.extract_code(text) == "print(2)"
    assert code_tasks.extract_code("```python\nprint(1)\n```\n```json\n{}\n```\n```\n```") == "print(1)"
    assert code_tasks.extract_code_before_terminate("```python\nprint(3)\nTERMINATE\n```") == "print(3)"
    assert code_tasks.extract_code("no code here") is None


def test_guarded_call_returns_the_result_or_the_error():
    assert code_tasks.guarded_call(TWO_SUM, "twoSum", [[2, 7], 9], 10, 0) == {"ok": True, "result": [0, 1]}
    missing = code_tasks.guarded_call("x = 1", "f", [], 10, 0)
    assert missing == {"ok": False, "error": "neither Solution.f nor f defined"}


def test_the_submitted_program_is_the_vote_over_normalized_programs():
    assert code_tasks.select_program(["print(1)", "print(2)", "print( 2)", "print(2)\n"]) == 1
    assert code_tasks.select_program(["print(1)", None, "print(2)"]) == 0  # a tie: the lowest agent
    assert code_tasks.select_program([None, "", "print(3)"]) == 2  # agents without a program abstain
    assert code_tasks.select_program([None, None]) == 0


def test_only_the_selected_peer_is_scored():
    calls = []

    def run(code, tests, timeout_s=5):
        calls.append(code)
        return {"pass_rate": 1.0}

    peers = [{"peer": 0}, {"peer": 1}]
    code_tasks.score_selected_peer(peers, 1, "good", [], run=run)
    assert calls == ["good"]
    assert peers == [
        {"peer": 0, "pass_rate": None, "resolved": None, "report": None},
        {"peer": 1, "pass_rate": 1.0, "resolved": True, "report": {"pass_rate": 1.0}},
    ]


def test_records_and_report():
    assert code_tasks.scores_of("c", {"pass": 1, "total": 2, "pass_rate": 0.5}) == {
        "pass": 1,
        "total": 2,
        "pass_rate": 0.5,
        "em": 0.0,
    }
    assert code_tasks.selection_scores(None, 0, 1.0) == {"winner": 0, "pass_rate": 1.0, "em": 0.0}
    out = {"code": "c", "winner": 1, "per_peer": [{"peer": 0, "pass_rate": None}, {"peer": 1, "pass_rate": 0.5}]}
    assert code_tasks.winner_pass_rate(out) == 0.5
    assert code_tasks.winner_pass_rate({**out, "code": None}) == 0.0
    assert code_tasks.compact_peers(out["per_peer"])[0] == {
        "peer": 0,
        "has_code": False,
        "pass_rate": None,
        "resolved": False,
    }
    replica = {"agent_id": 2, "seed": 2, "code": "c", "raw": "r", "messages": []}
    assert code_tasks.compact_replicas([replica]) == [{"agent_id": 2, "seed": 2, "code": "c"}]
    summary = {**code_tasks.summarize([{"predicted_code": "c", "em": 1.0, "difficulty": "easy"}]), "total_s": 1.0}
    report = code_tasks.banner("X", summary, metric="pass@1", difficulty_width=6)
    assert "pass@1 EM=1.000" in report and "      easy: n=  1  EM=1.000" in report
