"""``core.communication.CommPolicy`` applies one format through the shared helpers."""

import threading

import pytest

from core import communication
from core.communication import CommPolicy


def test_freeform_changes_nothing():
    policy = CommPolicy("freeform", "hotpotqa", "sequential")
    assert policy.system_prompt("You are a planner.") == "You are a planner."
    assert policy.handoff("planner", "Plan: look it up.") == "Plan: look it up."


def test_semi_structured_contract_and_handoff():
    policy = CommPolicy("semi_structured", "hotpotqa", "sequential")
    prompt = policy.system_prompt("You are a planner.")
    assert prompt == "You are a planner." + communication.communication_contract("semi_structured", "hotpotqa")
    assert policy.system_prompt(prompt) == prompt
    assert policy.handoff("planner", "Plan.").startswith("[STATUS]\ncompleted")


def test_solve_records_handoffs_and_report_metrics():
    policy = CommPolicy("structured_soft", "hotpotqa", "sequential")

    def solve(question):
        policy.handoff("planner", f"Plan for {question}")
        return {"answer": "x", "by_stage": {"planner": "Plan", "writer": "Answer: x"}}

    out = policy.solve(solve, "q")
    assert out["answer"] == "x"
    assert out["communication_inflight_handoff_count"] == 1
    assert out["communication_inflight_handoffs"][0]["role"] == "planner"
    assert out["communication_report_total"] == 2
    assert out["communication_format"] == "structured_soft"


def test_concurrent_policies_record_only_their_own_handoffs():
    turns = threading.Barrier(2)
    outputs = {}

    def run(fmt):
        policy = CommPolicy(fmt, "hotpotqa", "sequential")

        def solve():
            for step in range(3):
                turns.wait()
                policy.handoff("planner", f"Step {step}.")
            return {"by_stage": {"planner": "Plan"}}

        outputs[fmt] = policy.solve(solve)

    threads = [threading.Thread(target=run, args=(fmt,)) for fmt in ("semi_structured", "structured_soft")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    for fmt, out in outputs.items():
        assert out["communication_inflight_handoff_count"] == 3
        assert {handoff["format"] for handoff in out["communication_inflight_handoffs"]} == {fmt}
        assert out["communication_format"] == fmt


def test_unknown_format():
    with pytest.raises(ValueError):
        CommPolicy("yaml")
