"""Credit parsing, meta-knowledge, overlay construction and the three modes of one attempt."""

from __future__ import annotations

import hashlib

import pytest

from .. import settings
from ..driver import (
    active_roles_in_segment,
    append_overlay,
    build_agent_history,
    build_overlay_from_meta,
    build_result_context,
    build_result_evaluation,
    build_task_context,
    credit_assign,
    generate_meta_suggestions,
    merge_role_evaluations,
    overlay_id,
    parse_json_response,
    run_optimization_attempt,
    segment_iterations,
    strip_overlay_blocks,
)
from .fakes import FakeReflection, credit_payload, sequential_trajectory

ROLES = ["planner", "writer"]
SEEDS = {"planner": "Plan the answer.", "writer": "Write the final answer."}


def _attempt(reflection, *, mode="truce-release", credit=True, current=None, trajectories=None):
    trajectories = trajectories or [sequential_trajectory("tr0", 1.0), sequential_trajectory("tr1", 0.0)]
    seen = []

    def batch_runner(runner, metric, prompts, rows, num_threads, budget):
        seen.append(dict(prompts))
        return [dict(item) for item in trajectories]

    runner = type("Runtime", (), {"topology": "sequential"})()
    candidate, detail = run_optimization_attempt(
        runner,
        None,
        ROLES,
        SEEDS,
        current or dict(SEEDS),
        ["tr0", "tr1"],
        reflection,
        1,
        None,
        "hotpotqa",
        4,
        iteration=1,
        optimizer_mode=mode,
        batch_runner=batch_runner,
        trajectory_credit=credit,
    )
    return candidate, detail, seen


def test_parse_json_response_tolerates_fences_trailing_prose_and_garbage():
    assert parse_json_response('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_response('Result: {"a": 1}\nthen {"b": 2} and prose') == {"a": 1}
    assert parse_json_response("no json at all") == {"error": "Failed to parse response"}
    assert parse_json_response('{"a": ') == {"error": "Failed to parse response"}


def test_credit_assignment_sees_source_tagged_transcript_and_falls_back_on_failure():
    traj = sequential_trajectory("tr0", 0.0)
    history = build_agent_history("writer", ROLES, traj, SEEDS)
    assert history["agent_type"] == "pipeline_stage" and history["token_usage"] == 15
    assert history["results"] == [{"result": "final answer"}]
    assert [item["source"] for item in history["communications"]] == ["user", "planner", "writer"]
    evaluation = build_result_evaluation(traj)
    reflection = FakeReflection()
    parsed = credit_assign(reflection, history, build_task_context(traj, ROLES), build_result_context(traj), evaluation)
    assert parsed["specific_prompt_modifications"] == credit_payload("writer")["specific_prompt_modifications"]
    assert parsed["meta"]["agent_id"] == "writer"
    prompt = reflection.requests[0]["prompt"]
    assert '"source": "planner"' in prompt and "Token usage: 15" in prompt and "INCORRECT" in prompt
    assert "Team Size: 4" in prompt and "LLM Used: Qwen/Qwen3.5-9B" in prompt
    assert (reflection.requests[0]["temperature"], reflection.requests[0]["max_output_tokens"]) == (0.5, 2048)

    failed = credit_assign(FakeReflection(fail_kinds=("credit",)), history, "ctx", "result", evaluation)
    assert failed["fallback_mode"] is True and failed["overall_score"] == 5

    class Garbage(FakeReflection):
        @staticmethod
        def _credit(prompt):
            return "The agent did fine."

    garbled = credit_assign(Garbage(), history, "ctx", "result", evaluation)
    assert garbled["error"] == "Failed to parse response"
    assert merge_role_evaluations([failed, garbled]) is None
    merged = merge_role_evaluations([parsed, parsed, garbled])
    assert merged["specific_prompt_modifications"]["add_instructions"] == [
        "Verify the final answer against the question",
        "State the writer output explicitly",
    ]
    assert merged["prompt_suggestions"] == {
        "result_oriented_improvements": "Tie every step to the final answer",
        "quality_focus_additions": "Check the answer format before replying",
    }


def _instances():
    return [
        {"task_id": f"tr{i}", "agent_evaluations": {role: credit_payload(role) for role in ROLES}} for i in range(2)
    ]


def test_meta_suggestions_merge_edits_and_rank_by_salience():
    reflection = FakeReflection()
    meta = generate_meta_suggestions(reflection, _instances())
    patterns = meta["universal_patterns"]
    assert meta["training_instances_count"] == 2 and reflection.kinds() == ["salience"] * 3
    # The salience call sees every pooled add/remove/restructure suggestion.
    assert reflection.requests[0]["prompt"].count("Verify the final answer against the question") == 4
    assert "top-5" in reflection.requests[0]["prompt"] and "top-3" in reflection.requests[1]["prompt"]
    assert patterns["most_common_additions"] == [
        "Verify the final answer against the question",
        "State the planner output explicitly",
    ]
    assert patterns["most_common_removals"] == ["Remove redundant restatements of the task"]
    improvements = patterns["improvement_suggestions"]
    assert improvements["result_oriented_improvements"] == ["Tie every step to the final answer"]
    assert improvements["collaboration_optimizations"] == [] and improvements["effectiveness_enhancements"] == []


def test_meta_suggestions_fall_back_to_frequency_when_salience_is_not_json():
    meta = generate_meta_suggestions(FakeReflection(salience="broken"), _instances())
    additions = meta["universal_patterns"]["most_common_additions"]
    assert additions[0] == "Verify the final answer against the question (appeared 4 times)"
    assert "State the writer output explicitly (appeared 2 times)" in additions
    assert generate_meta_suggestions(FakeReflection(), []) == {}


def test_overlay_is_one_marked_block_that_replaces_previous_blocks():
    body = "1. Check the final answer.\n2. Keep handoffs short.\n3. Report uncertainty."
    reflection = FakeReflection(overlay=body)
    overlay = build_overlay_from_meta(
        reflection,
        {"universal_patterns": {"most_common_additions": ["x"]}},
        "hotpotqa (sequential multi-agent system with 2 optimized prompt components)",
    )
    digest = hashlib.md5(body.encode("utf-8")).hexdigest()[:10]
    assert overlay == f"<<< TRUCE_VERBALIZED_POLICY v1.0 id:{digest} >>>\n{body}\n<<< /TRUCE_VERBALIZED_POLICY >>>"
    assert overlay_id(overlay) == digest
    prompt = reflection.requests[0]["prompt"]
    assert "Output 3 to 6 concise behavioral rules." in prompt and '"most_common_additions"' in prompt
    assert reflection.requests[0]["temperature"] == 0.5

    profile = append_overlay("Seed prompt.", overlay)
    assert profile == "Seed prompt.\n\n" + overlay
    newer = overlay.replace(digest, "0123456789").replace("Report uncertainty", "Cite evidence")
    replaced = append_overlay(profile, newer)
    assert replaced.count("TRUCE_VERBALIZED_POLICY v") == 1 and replaced.endswith(newer)
    legacy = "Seed prompt.\n\n<<< MARBLE_VERBALIZED_POLICY v1 id:abc >>>\nold rule\n<<< /MARBLE_VERBALIZED_POLICY >>>"
    assert strip_overlay_blocks(legacy) == "Seed prompt."
    with pytest.raises(ValueError, match="empty"):
        build_overlay_from_meta(FakeReflection(overlay="  "), {"x": 1}, "task")


def test_centralized_context_history_and_subtrajectories():
    roles = ["manager", "worker_a", "worker_b"]
    traj = {
        "id": "c0",
        "question": "Who wrote it?",
        "topology": "centralized",
        "team_size": 4,
        "score": 0.0,
        "messages": [
            {"source": "user", "content": "Who wrote it?"},
            {
                "source": "manager",
                "content": "delegating",
                "tool_calls": [{"name": "delegate_to_worker_a", "args": {"instructions": "find the author"}}],
            },
            {"source": "worker_a", "content": "The author is X."},
            {"source": "manager", "content": "Final: X"},
        ],
    }
    context = build_task_context(traj, roles)
    assert "Coordination Mode: centralized (manager routes work" in context and "Total Iterations: 2" in context
    segments = segment_iterations(traj, "manager")
    assert [len(segment["messages"]) for segment in segments] == [3, 1]
    assert active_roles_in_segment(segments[0], roles) == ["manager", "worker_a"]
    prompts = {role: f"{role} prompt" for role in roles}
    assert build_agent_history("worker_a", roles, traj, prompts)["tasks_performed"] == [
        {"delegated_instructions": "find the author"}
    ]
    assert build_agent_history("manager", roles, traj, prompts)["agent_type"] == "manager"
    assert build_agent_history("worker_b", roles, traj, prompts) is None


def test_release_attempt_appends_one_shared_overlay_to_the_seed_prompts():
    reflection = FakeReflection()
    old_overlay = "<<< TRUCE_VERBALIZED_POLICY v1.0 id:aaaaaaaaaa >>>\nold\n<<< /TRUCE_VERBALIZED_POLICY >>>"
    current = {role: append_overlay(text, old_overlay) for role, text in SEEDS.items()}
    candidate, detail, seen = _attempt(reflection, current=current)
    assert seen == [current]  # the train batch runs with the current (newest) candidate
    overlay = detail["overlay"]
    assert overlay.startswith("<<< TRUCE_VERBALIZED_POLICY v1.0 id:") and "GOOD" in overlay
    assert candidate == {role: append_overlay(SEEDS[role], overlay) for role in ROLES}  # non-compounding
    kinds = reflection.kinds()
    assert kinds.count("eq3") == 4 and kinds.count("credit") == 4  # 2 segments and 2 roles per trajectory
    assert kinds.count("salience") == 3 and kinds.count("overlay") == 1
    assert not {"refine", "refine_meta", "aggregate"} & set(kinds)
    assert {(c["temperature"], c["max_output_tokens"]) for c in reflection.of_kind("eq3")} == {(0.3, 1536)}
    credit_prompt = reflection.of_kind("credit")[0]["prompt"]
    assert credit_prompt.index('"local_evaluations"') < credit_prompt.index('"group_chat_transcript_excerpt"')
    assert detail["trajectory_credit"]["enabled"] is True and len(detail["trajectory_credit"]["evaluations"]) == 4
    assert detail["roles_changed_from_input"] == ROLES and detail["roles_refined_from_seed"] == []
    assert detail["role_credit"]["writer"] == {"n_evals": 2, "n_parse_failures": 0, "overall_scores": [7, 7]}
    assert detail["train_mean"] == 0.5 and detail["overlay_error"] is None


def test_credit_ablation_sends_no_subtrajectory_prompts():
    reflection = FakeReflection()
    _, detail, _ = _attempt(reflection, credit=False)
    assert "eq3" not in reflection.kinds() and reflection.kinds().count("credit") == 4
    assert '"local_evaluations"' not in reflection.of_kind("credit")[0]["prompt"]
    assert detail["trajectory_credit"] == {"enabled": False, "segment_counts": {}, "evaluations": []}


def test_trajectory_credit_flag_reads_the_environment(monkeypatch):
    assert settings.trajectory_credit_enabled("1") and not settings.trajectory_credit_enabled("off")
    monkeypatch.setenv("TAVO_CREDIT", "0")
    assert settings.trajectory_credit_enabled() is False


def test_failed_overlay_keeps_the_seed_prompts():
    candidate, detail, _ = _attempt(FakeReflection(overlay=""))
    assert candidate == SEEDS and detail["overlay"] is None and "ValueError" in detail["overlay_error"]


def test_hybrid_mode_refines_each_role_from_its_seed_and_appends_the_overlay():
    reflection = FakeReflection()
    candidate, detail, _ = _attempt(reflection, mode="tavo-hybrid")
    assert reflection.kinds().count("refine_meta") == 2 and reflection.kinds().count("overlay") == 1
    assert candidate == {role: append_overlay(f"Meta-refined prompt for {role}.", detail["overlay"]) for role in ROLES}
    assert any("Write the final answer." in call["prompt"] for call in reflection.of_kind("refine_meta"))
    assert detail["roles_refined_from_seed"] == ROLES


def test_paper_mode_aggregates_per_agent_and_refines_the_previous_prompt():
    reflection = FakeReflection()
    previous = {
        role: append_overlay(
            f"Previous {role}.", "<<< TRUCE_VERBALIZED_POLICY v1.0 id:x >>>\nr\n<<< /TRUCE_VERBALIZED_POLICY >>>"
        )
        for role in ROLES
    }
    candidate, detail, _ = _attempt(reflection, mode="tavo-paper-reproduction", current=previous)
    kinds = reflection.kinds()
    assert kinds.count("aggregate") == 2 and kinds.count("refine") == 2
    assert not {"salience", "overlay", "refine_meta"} & set(kinds)
    assert candidate == {role: f"Refined prompt for {role}." for role in ROLES} and detail["overlay"] is None
    refine_prompts = [call["prompt"] for call in reflection.of_kind("refine")]
    assert any("Previous writer." in p and "VERBALIZED_POLICY" not in p for p in refine_prompts)
    # One credit job per active (agent, sub-trajectory) pair: user+planner segment and writer segment.
    assert kinds.count("credit") == 4
    assert {item["segment_index"] for item in detail["credit_evaluations"]} == {0, 1}


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="unsupported optimizer mode"):
        _attempt(FakeReflection(), mode="no-such-mode")
