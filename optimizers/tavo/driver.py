"""TRUCE / TAVO trajectory-aware verbalized prompt optimization (optimizer logic only).

The default ``truce-release`` mode follows the optimizer path enabled by the public
TRUCE release: evaluate trajectories, synthesize meta-knowledge and append one shared
verbalized-policy overlay to every role prompt. Two explicitly labelled modes are also
available: ``tavo-paper-reproduction`` aggregates suggestions per agent and
sub-trajectory and refines each agent's previous prompt (paper Algorithm 1), and
``tavo-hybrid`` refines every role from its seed prompt and appends the overlay on top.

The MARBLE engine is not ported. In every mode:

  trajectory     == the source-tagged message transcript of one full-MAS rollout
  agent profile  == a role prompt of the cell's runtime
  global score   == the dataset metric on the rollout (replaces MARBLE's ResultEvaluator)

One attempt (upstream ``optimization_pipeline.py`` main loop, ``batch_level`` strategy):

  1. run the train batch with the current candidate prompts                      [charged]
  2. Eq. 3 sub-trajectory credit, then per (example, role) trajectory-aware credit
     assignment (``AgentHistoryEvaluator.evaluate_agent_with_result_context``)
  3. aggregate meta-knowledge (``PromptRefiner.generate_meta_suggestions`` with
     ``_get_most_salient_by_llm``)
  4. apply the selected mode and return the candidate prompts

The validation gate, retry and patience live in ``integration.py``. The prompt texts
are adapted from the MIT-licensed MARBLE and TRUCE code (see ``NOTICE``).
"""

from __future__ import annotations

import datetime
import hashlib
import json
import re
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, NamedTuple

from optimizers.protocol.config import DEFAULT_TASK_MODEL
from optimizers.protocol.settings import ProtocolSettings

from .settings import trajectory_credit_enabled

# Config
OPTIMIZER_MODES = ("truce-release", "tavo-paper-reproduction", "tavo-hybrid")
DEFAULT_OPTIMIZER_MODE = "truce-release"
MODE_CLASSIFICATION = {
    "truce-release": "release-faithful TRUCE optimizer adaptation",
    "tavo-paper-reproduction": "TAVO paper-aligned optimizer adaptation",
    "tavo-hybrid": "experimental TRUCE/TAVO hybrid (role refinement plus overlay)",
}
TRUCE_RELEASE = {
    "url": "https://github.com/Bingo-W/TRUCE",
    "commit": "6428a6fc4dbca4f6e7fcebae493193c363accc8a",
    "license": "MIT",
}
BENCHMARK_ADAPTATIONS = (
    "MARBLE execution replaced by the cell's real-runner MAS",
    "global ResultEvaluator replaced by the dataset metric",
    "optimizer rollouts are charged to the protocol budget ledger",
)


def batch_mean(trajs: list[dict]) -> float:
    """Mean score of the executed trajectories (0.0 when none ran)."""
    executed = [t for t in trajs if t.get("executed")]
    if not executed:
        return 0.0
    return sum(t["score"] for t in executed) / len(executed)


# JSON parsing: AgentHistoryEvaluator._parse_json_response, with ``raw_decode`` so
# trailing prose or a second JSON object from local models does not void the result.
def parse_json_response(response_content: str) -> dict:
    """The JSON object in a model reply (fenced or bare), else ``{"error": ...}``."""
    try:
        content = response_content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        json_start = content.find("{")
        if json_start >= 0:
            obj, _ = json.JSONDecoder().raw_decode(content[json_start:])
            return obj
        raise ValueError("No valid JSON found in response")
    except Exception:
        return {"error": "Failed to parse response"}


def _now() -> str:
    # Upstream stamps meta-knowledge with the local wall clock; it becomes part of the overlay prompt.
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# Trajectory -> TAVO structures. The dataset metric replaces MARBLE's ResultEvaluator;
# the transcript excerpt rides inside result_evaluation, whose 12000-character window
# is where the credit-assignment prompt receives process/trajectory context.
def build_result_evaluation(traj: dict) -> dict:
    """MARBLE-style result evaluation of one trajectory: verdict, score and a transcript excerpt."""
    verdict = "CORRECT" if traj["score"] > 0 else "INCORRECT"
    excerpt = [
        {"source": m.get("source"), "content": (m.get("content") or "")[:500]} for m in (traj["messages"] or [])[-24:]
    ]
    return {
        "evaluation_type": "dataset_metric (replaces MARBLE ResultEvaluator in this port)",
        "global_score": traj["score"],
        "verdict": verdict,
        "final_answer": str(traj.get("answer"))[:800] if traj.get("answer") is not None else None,
        "rollout_error": traj.get("error"),
        "group_chat_transcript_excerpt": excerpt,
    }


def build_result_context(traj: dict) -> str:
    """The overall-result paragraph given to credit assignment."""
    verdict = "CORRECT" if traj["score"] > 0 else "INCORRECT"
    parts = [
        f"**Overall Final Result Score**: {traj['score']:.2f}/1.0 "
        "(official dataset metric on the system's final answer; 1.0 = fully correct)",
        f"**Result Verdict**: {verdict}",
    ]
    if traj.get("error"):
        parts.append(f"**Rollout Error**: {str(traj['error'])[:400]}")
    return "\n".join(parts)


def _manager_role(roles: list[str] | tuple[str, ...] | None, msgs: list[dict]) -> str:
    """Resolve the concrete manager id (e.g. ``manager_r8``), never a fixed alias."""
    if roles:
        return roles[0]
    return next((str(m.get("source")) for m in msgs if str(m.get("source") or "").startswith("manager")), "manager")


def build_task_context(traj: dict, roles: list[str] | tuple[str, ...] | None = None) -> str:
    """Describe the executing topology (``BatchTaskOptimizer`` task context, five fields)."""
    msgs = traj["messages"] or []
    topology = traj.get("topology", "centralized")
    team_size = int(
        traj.get("team_size")
        or (
            len(roles)
            if roles
            else len({str(m.get("source")) for m in msgs if m.get("source") not in (None, "user", "tool", "assistant")})
        )
    )
    if topology == "centralized":
        manager_role = _manager_role(roles, msgs)
        coordination = (
            f"centralized ({manager_role} routes work via delegate_to_<worker> tool calls; "
            "workers report back to the manager)"
        )
        planning = "manager-directed delegation"
        iterations = sum(1 for m in msgs if m.get("source") == manager_role)
    else:
        descriptions = {
            "sequential": ("ordered sequential pipeline", "fixed stage-to-stage handoffs"),
            "independent": (
                "independent replicas using a shared prompt",
                "independent execution followed by the native answer aggregation",
            ),
            "decentralized": ("decentralized peers using a shared prompt", "native peer communication and consensus"),
        }
        if topology not in descriptions:
            raise ValueError(f"unsupported TAVO topology: {topology}")
        coordination, planning = descriptions[topology]
        iterations = len(segment_iterations(traj))
    env_task_model = ProtocolSettings.from_env().task_model
    task_model = traj.get("task_model") or (DEFAULT_TASK_MODEL if env_task_model is None else env_task_model)
    return f"""
Task: {traj["question"][:800]}
Coordination Mode: {coordination}
Planning Method: {planning}
Total Iterations: {iterations}
Team Size: {team_size}
LLM Used: {task_model}
"""


def build_agent_history(role: str, roles: list[str], traj: dict, prompts: dict[str, str]) -> dict | None:
    """Per-agent history (upstream ``extract_agent_history_from_jsonl``).

    tasks_performed = delegation payloads targeting the role (the top task for the
    manager or a pipeline stage), communications = the full shared transcript,
    results = the role's own outputs. Roles absent from the trajectory are skipped.
    """
    msgs = traj["messages"] or []
    topology = traj.get("topology", "centralized")
    if topology in {"independent", "decentralized"} and len(roles) == 1:
        # The optimized prompt is shared; keep every replica's own source tag.
        own = [m for m in msgs if m.get("source") not in (None, "user", "tool", "system")]
    else:
        own = [m for m in msgs if m.get("source") == role]
    manager_role = roles[0] if topology == "centralized" else None
    if topology != "centralized":
        if not own:
            return None
        tasks = [{"task": traj["question"][:600], "topology": topology}]
    elif role == manager_role:
        tasks = [{"task": traj["question"][:600]}]
    else:
        tasks = []
        for m in msgs:
            for call in m.get("tool_calls") or []:
                if (call.get("name") or "") == f"delegate_to_{role}":
                    args = call.get("args") or {}
                    payload = args.get("instructions", args)
                    tasks.append({"delegated_instructions": str(payload)[:600]})
        if not tasks and own:
            tasks = [{"task": "(activated in the group chat without an explicit delegation payload)"}]
        if not tasks and not own:
            return None
    comms = [{"source": m.get("source"), "content": (m.get("content") or "")[:300]} for m in msgs][:40]
    results = [{"result": (m.get("content") or "")[:500]} for m in own[-5:]]
    return {
        "agent_id": role,
        "agent_type": (
            ("manager" if role == manager_role else "worker")
            if topology == "centralized"
            else {
                "sequential": "pipeline_stage",
                "independent": "shared_replica_prompt",
                "decentralized": "shared_peer_prompt",
            }[topology]
        ),
        "profile": prompts[role],
        "original_system_prompt": prompts[role],
        "tasks_performed": tasks,
        "communications": comms,
        "results": results,
        "token_usage": int((traj.get("telemetry") or {}).get("total_tokens") or 0),
    }


# Eq. 3 sub-trajectory credit (``ResultEvaluator.evaluate_local_iterations`` with the
# ``local_iteration_evaluation`` template). A sub-trajectory is one manager turn plus
# everything until the next manager turn (centralized), else one participant turn
# plus its tool replies.
_EQ3_PROMPT = """Please evaluate the quality and progress of the current iteration in this multi-agent collaboration.

**Current Iteration Context:**
Iteration Number: {iteration_number}
Iteration Content: {iteration_content}

**Task Context:**
Task Background: {task_context}
Previous Iterations: {previous_iterations}

**Evaluation of final result:**
{global_evaluation}

**Evaluation Focus:**
- Whether the current iteration has advanced the task
- Whether the collaboration between agents is effective
- Whether the output meets the expectations of the current stage

**Evaluation Criteria:**

- **iteration_progress**: Progress made in current iteration
  Scoring: 1-10 points, evaluating advancement towards task completion

- **agent_coordination**: Agent coordination in current iteration
  Scoring: 1-10 points, evaluating collaboration effectiveness

- **stage_appropriateness**: Evaluate whether the output meets the current stage
  Scoring: 1-10 points, evaluating appropriateness of actions for current stage

Please return the evaluation results in JSON format:
{{
    "iteration_progress": {{
        "score": score(1-10),
        "analysis": "progress analysis",
        "key_achievements": ["key achievement 1", "key achievement 2"]
    }},
    "agent_coordination": {{
        "score": score(1-10),
        "analysis": "coordination analysis",
        "coordination_highlights": ["coordination highlight 1", "coordination highlight 2"],
        "coordination_issues": ["coordination issue 1", "coordination issue 2"]
    }},
    "stage_appropriateness": {{
        "score": score(1-10),
        "analysis": "stage appropriateness analysis",
        "alignment_evidence": ["alignment evidence 1", "alignment evidence 2"]
    }},
    "iteration_summary": {{
        "overall_score": average score,
        "main_contributions": ["main contribution 1", "main contribution 2"],
        "areas_for_improvement": ["area for improvement 1", "area for improvement 2"],
        "next_iteration_suggestions": ["next iteration suggestion 1", "next iteration suggestion 2"]
    }}
}}"""


def segment_iterations(traj: dict, manager_role: str | None = None) -> list[dict]:
    """Split the shared transcript into sub-trajectories."""
    msgs = traj.get("messages") or []
    if traj.get("topology", "centralized") != "centralized":
        # A native participant turn plus its following tool replies is the observable
        # local iteration; no synthetic manager or round is invented.
        segments = []
        current = []
        seen_participant = False
        for message in msgs:
            source = message.get("source")
            participant = source not in (None, "user", "tool", "system")
            if participant and seen_participant and current:
                segments.append({"messages": current})
                current = []
            current.append(dict(message))
            seen_participant = seen_participant or participant
        if current:
            segments.append({"messages": current})
        return segments
    manager_role = manager_role or _manager_role(None, msgs)
    segments: list[dict] = []
    current: list[dict] = []
    seen_manager = False
    for m in msgs:
        is_manager = m.get("source") == manager_role
        # The prepended user task stays with the first manager turn as context.
        if is_manager and seen_manager and current:
            segments.append({"messages": current})
            current = []
        current.append(
            {
                "source": m.get("source"),
                "content": (m.get("content") or "")[:400],
                "tool_calls": [
                    {"name": c.get("name"), "args": c.get("args") or {}}
                    for c in (m.get("tool_calls") or [])
                    if isinstance(c, dict)
                ],
            }
        )
        seen_manager = seen_manager or is_manager
    if current:
        segments.append({"messages": current})
    return segments


def evaluate_iteration_credit(
    reflection: Any, iteration_number: int, segment: dict, previous: list[dict], task_context: str, result_context: str
) -> dict:
    """Eq. 3 credit F_t of one sub-trajectory, given the earlier segments."""
    prompt = _EQ3_PROMPT.format(
        iteration_number=iteration_number,
        iteration_content=json.dumps(segment, ensure_ascii=False)[:800],
        task_context=task_context,
        previous_iterations=json.dumps(previous, ensure_ascii=False)[:500],
        global_evaluation=result_context,
    )
    # Upstream parameters: temperature 0.3, max_token_num 1536. Guarded like upstream:
    # one failed credit call must not abort the attempt.
    try:
        out = reflection.complete(prompt, temperature=0.3, max_tokens=1536)
        ev = parse_json_response(out)
    except Exception as exc:
        ev = {"error": str(exc)[:300], "evaluation_type": "local"}
    if not isinstance(ev, dict):
        ev = {"error": "non-dict credit response", "evaluation_type": "local"}
    ev["meta"] = {"evaluation_type": "local", "iteration_number": iteration_number}
    return ev


# Credit assignment: AgentHistoryEvaluator.evaluate_agent_with_result_context (prompt verbatim).
def credit_assign(
    reflection: Any, agent_history: dict, task_context: str, result_context: str, result_evaluation: dict
) -> dict:
    """Trajectory-aware credit and prompt suggestions for one agent of one rollout."""
    evaluation_prompt = f"""
Please analyze the following Agent's performance and provide improvement recommendations based on final result quality:

**Task Context:**
{task_context}

**Final Result Quality Assessment:**
{result_context}

**Full Result Evaluation:**
{json.dumps(result_evaluation, ensure_ascii=False, indent=2)[:12000]}

**Agent Information:**
- Agent ID: {agent_history.get("agent_id")}
- Agent Type: {agent_history.get("agent_type")}
- Profile: {agent_history.get("profile")}

**Current System Prompt:**
{agent_history.get("original_system_prompt", "No system prompt available")}

**Agent Historical Performance:**
- Number of tasks executed: {len(agent_history.get("tasks_performed", []))}
- Number of communications: {len(agent_history.get("communications", []))}
- Token usage: {agent_history.get("token_usage", 0)}

**Detailed Task History:**
{json.dumps(agent_history.get("tasks_performed", []), ensure_ascii=False, indent=2)[:1000]}...

**Communication History:**
{json.dumps(agent_history.get("communications", []), ensure_ascii=False, indent=2)[:1000]}...

**Result History:**
{json.dumps(agent_history.get("results", []), ensure_ascii=False, indent=2)[:1000]}...

**Result Quality-Based Deep Analysis Requirements:**

1. **Result-Oriented Assessment**: Analyze whether this agent's contributions are effective based on final result quality
2. **Causality Analysis**: Analyze the causal relationship between this agent's behavior and final result quality
3. **Impact Verification**: Whether this agent's work had positive impact on final results
4. **Problem Attribution**: If result quality is poor, whether this agent is one of the influencing factors
5. **Targeted Improvement**: Provide targeted improvement recommendations based on result quality issues

**Important**: Please judge agent performance effectiveness by combining result quality assessment, not just process analysis.

Please return evaluation results in JSON format:
{{
    "overall_score": 1-10,
    "result_oriented_analysis": {{
        "contribution_to_final_result": "Analysis of this agent's specific contribution to final result",
        "effectiveness_rating": 1-10,
        "impact_on_quality": "Analysis of impact on result quality"
    }},
    "strengths": ["Strength 1 based on result verification", "Strength 2 based on result verification", ...],
    "weaknesses": ["Weakness 1 affecting result quality", "Weakness 2 affecting result quality", ...],
    "causality_analysis": {{
        "positive_contributions": ["Behavior 1 promoting result quality", "Behavior 2 promoting result quality"],
        "negative_impacts": ["Behavior 1 damaging result quality", "Behavior 2 damaging result quality"],
        "missed_opportunities": ["Missed opportunity 1 to improve results", "Missed opportunity 2 to improve results"]
    }},
    "prompt_suggestions": {{
        "result_oriented_improvements": "Prompt improvement suggestions based on result quality issues",
        "effectiveness_enhancements": "Prompt modifications to improve actual effectiveness",
        "quality_focus_additions": "Prompt additions to enhance result quality awareness",
        "collaboration_optimizations": "Prompt adjustments to optimize collaboration effects"
    }},
    "targeted_recommendations": {{
        "immediate_fixes": ["Immediate improvement 1", "Immediate improvement 2"],
        "strategic_improvements": ["Strategic improvement 1", "Strategic improvement 2"],
        "quality_assurance_measures": ["Quality assurance measure 1", "Quality assurance measure 2"]
    }},
    "specific_prompt_modifications": {{
        "add_instructions": ["Specific instruction 1 to add", "Specific instruction 2 to add"],
        "remove_content": ["Content 1 to remove", "Content 2 to remove"],
        "restructure_suggestions": ["The content to be modified 1(and the way to modify)", "The content to be modified 2(and the way to modify)"]
    }}
}}
"""
    try:
        lm_out = reflection.complete(evaluation_prompt, temperature=0.5, max_tokens=2048)
        evaluation_result = parse_json_response(lm_out)
        evaluation_result["meta"] = {
            "evaluation_type": "agent_with_result_context",
            "result_context_used": True,
            "agent_id": agent_history.get("agent_id"),
        }
        return evaluation_result
    except Exception as exc:  # upstream fallback
        return {
            "overall_score": 5,
            "error": str(exc),
            "evaluation_type": "agent_with_result_context",
            "fallback_mode": True,
        }


# Meta-knowledge: PromptRefiner.generate_meta_suggestions, _get_most_salient_by_llm
# (prompt verbatim) and _get_most_frequent.
def _get_most_frequent(items_list: list[str], top_k: int = 5) -> list[str]:
    if not items_list:
        return []
    counter = Counter(items_list)
    frequent_items = []
    for item, count in counter.most_common():
        if len(item.strip()) > 10 and count >= 2:
            frequent_items.append(f"{item} (appeared {count} times)")
            if len(frequent_items) >= top_k:
                break
    return frequent_items


def _get_most_salient_by_llm(reflection: Any, items_list: list[str], top_k: int = 5) -> list[str]:
    try:
        items = [s.strip() for s in items_list if isinstance(s, str) and len(s.strip()) > 0]
        if not items:
            return []
        joined = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(items))
        if len(joined) > 12000:
            joined = joined[:12000]
        prompt = f"""
You are given a list of short rules/suggestions (may be redundant or semantically similar).
Task: (1) normalize/merge near-duplicates; (2) rank by (estimated frequency + practical importance);
(3) return the top-{top_k} representative and concise items.

Input items (one per line with index):
{joined}

Return STRICT JSON only:
{{
  "top": [
    {{
      "text": "representative concise rule",
      "support_examples_idx": [1,5,9],
      "support_count": 3,
      "importance": 0
    }}
  ]
}}
"""
        content = reflection.complete(prompt, temperature=0.5, max_tokens=1024).strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        try:
            data = json.loads(content)
        except Exception:
            left = content.find("{")
            right = content.rfind("}") + 1
            if left >= 0 and right > left:
                data = json.loads(content[left:right])
            else:
                raise
        tops = data.get("top", []) if isinstance(data, dict) else []
        results = []
        for t in tops[:top_k]:
            if isinstance(t, dict):
                text = (t.get("text") or "").strip()
                if text:
                    results.append(text)
        if not results:
            return _get_most_frequent(items_list, top_k=top_k)
        return results
    except Exception:
        return _get_most_frequent(items_list, top_k=top_k)


def generate_meta_suggestions(reflection: Any, training_instances: list[dict]) -> dict:
    """Aggregate per-agent suggestions into the most salient meta-knowledge."""
    if not training_instances:
        return {}
    all_suggestions: dict[str, list[str]] = {
        "add_instructions": [],
        "remove_content": [],
        "restructure_suggestions": [],
        "result_oriented_improvements": [],
        "collaboration_optimizations": [],
        "effectiveness_enhancements": [],
        "quality_focus_additions": [],
    }
    for instance in training_instances:
        agent_evaluations = instance.get("agent_evaluations", {})
        for evaluation in agent_evaluations.values():
            modifications = evaluation.get("specific_prompt_modifications", {}) or {}
            if modifications.get("add_instructions"):
                all_suggestions["add_instructions"].extend(str(x) for x in modifications["add_instructions"])
            if modifications.get("remove_content"):
                all_suggestions["remove_content"].extend(str(x) for x in modifications["remove_content"])
            if modifications.get("restructure_suggestions"):
                all_suggestions["restructure_suggestions"].extend(
                    str(x) for x in modifications["restructure_suggestions"]
                )
            prompt_suggestions = evaluation.get("prompt_suggestions", {}) or {}
            for key in [
                "result_oriented_improvements",
                "collaboration_optimizations",
                "effectiveness_enhancements",
                "quality_focus_additions",
            ]:
                suggestion = prompt_suggestions.get(key, "")
                if suggestion and suggestion != "N/A":
                    all_suggestions[key].append(str(suggestion))
    return {
        "training_instances_count": len(training_instances),
        "generated_time": _now(),
        "universal_patterns": {
            "most_common_additions": _get_most_salient_by_llm(reflection, all_suggestions["add_instructions"], top_k=5),
            "most_common_removals": _get_most_salient_by_llm(reflection, all_suggestions["remove_content"], top_k=3),
            "most_common_restructure_suggestions": _get_most_salient_by_llm(
                reflection, all_suggestions["restructure_suggestions"], top_k=3
            ),
            "improvement_suggestions": {
                "result_oriented_improvements": list(set(all_suggestions["result_oriented_improvements"]))[:5],
                "collaboration_optimizations": list(set(all_suggestions["collaboration_optimizations"]))[:5],
                "effectiveness_enhancements": list(set(all_suggestions["effectiveness_enhancements"]))[:5],
                "quality_focus_additions": list(set(all_suggestions["quality_focus_additions"]))[:5],
            },
        },
    }


# Verbalized-policy overlay: public TRUCE OverlayMixin (prompt verbatim). Both the
# TRUCE markers and the legacy MARBLE markers are recognized when replacing a block.
_OVERLAY_BEGIN_RE = re.compile(r"<<<\s*(?:TRUCE|MARBLE)_VERBALIZED_POLICY\s+v[^\s]+\s+id:([^\s]+)\s*>>>")
_OVERLAY_BLOCK_RE = re.compile(
    r"<<<\s*(?:TRUCE|MARBLE)_VERBALIZED_POLICY[\s\S]*?"
    r"<<<\s*/(?:TRUCE|MARBLE)_VERBALIZED_POLICY\s*>>>",
    re.MULTILINE,
)


def _compute_short_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:10]


def overlay_id(text: str) -> str | None:
    """Version id of the overlay block in a prompt, or None."""
    match = _OVERLAY_BEGIN_RE.search(text or "")
    return match.group(1) if match else None


def strip_overlay_blocks(content: str) -> str:
    """The prompt without any overlay block."""
    return re.sub(_OVERLAY_BLOCK_RE, "", content).strip()


def build_overlay_from_meta(
    reflection: Any, meta: dict, task_type: str, base_overlay: str | None = None, overlay_version: str = "1.0"
) -> str:
    """Write the shared verbalized-policy overlay from meta-knowledge."""
    prompt = f"""
You are to synthesize a reusable TRUCE prompt-rule overlay to append to every agent profile.

Context:
- Task type: {task_type}
- Meta-knowledge (JSON):
{json.dumps(meta, ensure_ascii=False, indent=2)[:4000]}

Requirements:
1. Output 3 to 6 concise behavioral rules.
2. Make each rule actionable and specific to improving final task quality.
3. Preserve existing agent roles and coordination structure.
4. Cover result orientation, collaboration, quality checks, and common failure modes when supported by the meta-knowledge.
5. Do not include explanations, citations, JSON, or markdown headings.
6. If a previous overlay exists, refine it instead of repeating it:
{(base_overlay or "N/A")[:1000]}
"""
    overlay_body = reflection.complete(prompt, temperature=0.5, max_tokens=2048).strip()
    if not overlay_body:
        raise ValueError("Overlay generation returned empty content")
    return (
        f"<<< TRUCE_VERBALIZED_POLICY v{overlay_version} id:{_compute_short_hash(overlay_body)} >>>\n"
        + overlay_body
        + "\n<<< /TRUCE_VERBALIZED_POLICY >>>"
    )


def append_overlay(profile: str, overlay_text: str) -> str:
    # force_replace semantics: remove any old overlay block, then append the new one.
    """Replace any overlay block in a profile with ``overlay_text``."""
    profile = strip_overlay_blocks(profile)
    sep = "\n\n" if profile and not profile.endswith("\n\n") else ""
    return f"{profile}{sep}{overlay_text}"


# Per-role refinement (hybrid and paper modes): PromptRefiner.refine_with_meta_knowledge,
# _format_instance_suggestions and the refine_agent_prompt fallback (prompts verbatim).
def _strip_code_fence(text: str) -> str:
    """Unwrap a single whole-output code fence (robustness addition, not upstream)."""
    t = text.strip()
    m = re.fullmatch(r"```[^\n]*\n(.*?)\n?```", t, re.DOTALL)
    return m.group(1).strip() if m else t


def _format_instance_suggestions(suggestions: dict | None) -> str:
    if not suggestions:
        return "None"
    formatted = []
    modifications = suggestions.get("specific_prompt_modifications", {}) or {}
    if modifications.get("add_instructions"):
        formatted.append(f"Add Instructions: {', '.join(str(x) for x in modifications['add_instructions'])}")
    if modifications.get("remove_content"):
        formatted.append(f"Remove Content: {', '.join(str(x) for x in modifications['remove_content'])}")
    prompt_suggestions = suggestions.get("prompt_suggestions", {}) or {}
    for key, value in prompt_suggestions.items():
        if value and value != "N/A":
            formatted.append(f"{key}: {value}")
    return "\n".join(formatted) if formatted else "None"


def refine_agent_prompt(
    reflection: Any, agent_id: str, original_prompt: str, evaluation_suggestions: dict, task_context: str
) -> str:
    """Paper-mode refinement of one agent's prompt from its suggestions."""
    prompt_suggestions = evaluation_suggestions.get("prompt_suggestions", {}) or {}
    specific_modifications = evaluation_suggestions.get("specific_prompt_modifications", {}) or {}
    strengths = [str(s) for s in evaluation_suggestions.get("strengths", []) or []]
    weaknesses = [str(w) for w in evaluation_suggestions.get("weaknesses", []) or []]
    optimization_prompt = f"""
Please optimize the Agent's System Prompt based on the following evaluation suggestions:

**Task Context:**
{task_context}

**Agent ID:** {agent_id}

**Current System Prompt:**
{original_prompt}

**Evaluation Results:**
- Strengths: {", ".join(strengths)}
- Weaknesses: {", ".join(weaknesses)}

**Opinions on modifications to different preference directions:**
- Result-Oriented Improvements: {prompt_suggestions.get("result_oriented_improvements", "N/A")}
- Effectiveness Enhancements: {prompt_suggestions.get("effectiveness_enhancements", "N/A")}
- Quality Focus Additions: {prompt_suggestions.get("quality_focus_additions", "N/A")}
- Collaboration Optimizations: {prompt_suggestions.get("collaboration_optimizations", "N/A")}

**Specific Modification Instructions:**
- Add Instructions: {specific_modifications.get("add_instructions", [])}
- Remove Content: {specific_modifications.get("remove_content", [])}
- Restructure Suggestions: {specific_modifications.get("restructure_suggestions", "N/A")}

**Requirements:**
1. Maintain the core role positioning of the original prompt
2. Make specific improvements based on suggestions
3. Ensure the new prompt is clear, specific, and actionable
4. Keep appropriate length, avoid being too verbose
5. Return the complete optimized prompt text directly, without other explanations

Please generate the optimized System Prompt:
"""
    try:
        out = reflection.complete(optimization_prompt, temperature=0.5, max_tokens=2048)
        refined = _strip_code_fence(out) if out else ""
        return refined or original_prompt
    except Exception:
        return original_prompt


def refine_with_meta_knowledge(
    reflection: Any,
    agent_id: str,
    original_prompt: str,
    task_context: str,
    meta: dict,
    instance_suggestions: dict | None = None,
) -> str:
    """Hybrid-mode refinement of one agent's seed prompt from meta-knowledge."""
    if not meta:
        if instance_suggestions:
            return refine_agent_prompt(reflection, agent_id, original_prompt, instance_suggestions, task_context)
        return original_prompt
    universal_patterns = meta.get("universal_patterns", {})
    imp = universal_patterns.get("improvement_suggestions", {})
    enhanced_prompt = f"""
Please optimize the Agent's System Prompt using meta-learned knowledge from {meta.get("training_instances_count", 0)} training instances:

**Task Context:**
{task_context}

**Agent ID:** {agent_id}

**Current System Prompt:**
{original_prompt}

**Meta-Learned Knowledge:**

Most Common Effective Additions:
{chr(10).join(f"- {addition}" for addition in universal_patterns.get("most_common_additions", []))}

Common Improvement Areas:
- Result-oriented improvements: {"; ".join(imp.get("result_oriented_improvements", [])[:2])}
- Collaboration optimizations: {"; ".join(imp.get("collaboration_optimizations", [])[:2])}
- Effectiveness enhancements: {"; ".join(imp.get("effectiveness_enhancements", [])[:2])}

**Instance-Specific Suggestions:**
{_format_instance_suggestions(instance_suggestions) if instance_suggestions else "None provided - rely on meta-knowledge"}

**Requirements:**
1. Apply relevant meta-learned improvements to this agent
2. Integrate instance-specific suggestions where they complement meta-knowledge
3. Maintain the core role and expertise of the original prompt
4. Return the complete optimized prompt text directly

Please generate the optimized System Prompt:
"""
    try:
        out = reflection.complete(enhanced_prompt, temperature=0.5, max_tokens=2048)
        refined = _strip_code_fence(out) if out else ""
        return refined or original_prompt
    except Exception:
        if instance_suggestions:
            return refine_agent_prompt(reflection, agent_id, original_prompt, instance_suggestions, task_context)
        return original_prompt


def merge_role_evaluations(evals: list[dict]) -> dict | None:
    """Deterministically merge one role's credit evaluations into one suggestion set."""
    valid = [e for e in evals if isinstance(e, dict) and not e.get("fallback_mode") and "error" not in e]
    if not valid:
        return None

    def _dedup(seq: list[str], cap: int) -> list[str]:
        seen, out = set(), []
        for s in seq:
            s = str(s).strip()
            if s and s not in seen:
                seen.add(s)
                out.append(s)
            if len(out) >= cap:
                break
        return out

    add_instructions, remove_content, restructure = [], [], []
    prompt_suggestion_parts: dict[str, list[str]] = {}
    strengths, weaknesses = [], []
    for e in valid:
        mods = e.get("specific_prompt_modifications", {}) or {}
        add_instructions.extend(mods.get("add_instructions") or [])
        remove_content.extend(mods.get("remove_content") or [])
        restructure.extend(mods.get("restructure_suggestions") or [])
        for key, value in (e.get("prompt_suggestions", {}) or {}).items():
            if value and value != "N/A":
                prompt_suggestion_parts.setdefault(key, []).append(str(value))
        strengths.extend(e.get("strengths") or [])
        weaknesses.extend(e.get("weaknesses") or [])
    return {
        "specific_prompt_modifications": {
            "add_instructions": _dedup(add_instructions, 6),
            "remove_content": _dedup(remove_content, 4),
            "restructure_suggestions": _dedup(restructure, 4),
        },
        "prompt_suggestions": {k: "; ".join(_dedup(v, 3))[:600] for k, v in prompt_suggestion_parts.items()},
        "strengths": _dedup(strengths, 5),
        "weaknesses": _dedup(weaknesses, 5),
    }


def active_roles_in_segment(segment: dict, roles: list[str]) -> list[str]:
    """Roles that acted or were explicitly delegated to in one sub-trajectory."""
    if not roles:
        return []
    sources = {m.get("source") for m in segment.get("messages", [])}
    delegated: set[str] = set()
    for message in segment.get("messages", []):
        for call in message.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            name = str(call.get("name") or "")
            if name.startswith("delegate_to_"):
                delegated.add(name.removeprefix("delegate_to_"))
    manager_role = roles[0]
    return [
        role
        for role in roles
        if role in sources or role in delegated or (role == manager_role and manager_role in sources)
    ]


def aggregate_role_evaluations_llm(reflection: Any, role: str, evaluations: list[dict]) -> dict | None:
    """Paper per-agent aggregation of edits across tasks and sub-trajectories.

    The released overlay path does not execute this step, so it is isolated to the
    ``tavo-paper-reproduction`` mode; the deterministic merge is the failure fallback.
    """
    fallback = merge_role_evaluations(evaluations)
    valid = [e for e in evaluations if isinstance(e, dict) and "error" not in e and not e.get("fallback_mode")]
    if not valid:
        return fallback
    prompt = f"""
Aggregate the following trajectory-aware prompt edits for agent {role} across tasks and
subtrajectories. Resolve conflicts, remove duplicates, and retain only generalizable,
actionable changes for this agent. Return STRICT JSON only with this schema:
{{
  "specific_prompt_modifications": {{
    "add_instructions": ["..."],
    "remove_content": ["..."],
    "restructure_suggestions": ["..."]
  }},
  "prompt_suggestions": {{
    "result_oriented_improvements": "...",
    "effectiveness_enhancements": "...",
    "quality_focus_additions": "...",
    "collaboration_optimizations": "..."
  }},
  "strengths": ["..."],
  "weaknesses": ["..."]
}}

Agent-specific evaluations:
{json.dumps(valid, ensure_ascii=False, indent=2)[:12000]}
"""
    try:
        result = parse_json_response(reflection.complete(prompt, temperature=0.5, max_tokens=2048))
        if not isinstance(result, dict) or result.get("error"):
            return fallback
        if not (result.get("specific_prompt_modifications") or result.get("prompt_suggestions")):
            return fallback
        return result
    except Exception:
        return fallback


# One optimization attempt: train batch -> credit -> meta-knowledge -> candidate.
BatchRunner = Callable[..., list[dict]]


class _TrajectoryContext(NamedTuple):
    """What every credit prompt about one trajectory quotes."""

    task_context: str
    result_evaluation: dict
    result_context: str


class _CreditJob(NamedTuple):
    """One credit assignment: an agent's history in one trajectory (or one of its sub-trajectories)."""

    traj_index: int
    segment_index: int | None
    role: str
    history: dict
    task_context: str
    result_context: str
    result_evaluation: dict


class _Credit(NamedTuple):
    """The credit evaluation of one :class:`_CreditJob`."""

    traj_index: int
    segment_index: int | None
    role: str
    evaluation: dict


@dataclass(frozen=True)
class _Proposal:
    """The candidate prompts of one attempt and how the selected mode produced them."""

    candidate: dict[str, str]
    refined: dict[str, str]
    role_suggestions: dict[str, dict | None]
    meta: dict = field(default_factory=dict)
    overlay: str | None = None
    overlay_error: str | None = None


def run_optimization_attempt(
    runner: Any,
    plain_metric: Callable[[Any, Any], float] | None,
    roles: list[str],
    seed_prompts: dict[str, str],
    current_prompts: dict[str, str],
    train_rows: list[Any],
    reflection: Any,
    num_threads: int,
    budget: Any,
    dataset: str,
    refl_inflight: int,
    iteration: int | None = None,
    optimizer_mode: str = DEFAULT_OPTIMIZER_MODE,
    *,
    batch_runner: BatchRunner,
    trajectory_credit: bool | None = None,
) -> tuple[dict[str, str], dict]:
    """Run one native attempt and return ``(candidate_prompts, detail)``.

    ``batch_runner(runner, plain_metric, prompts, rows, num_threads, budget)``
    executes (and charges) the train batch and returns trajectory dicts. The
    executed trajectories get credit (Eq. 3 sub-trajectory credit when enabled,
    then per-agent credit assignment) and the selected mode turns the credit
    into candidate prompts.
    """
    if optimizer_mode not in OPTIMIZER_MODES:
        raise ValueError(f"unsupported optimizer mode: {optimizer_mode!r}")
    credit_enabled = trajectory_credit_enabled() if trajectory_credit is None else bool(trajectory_credit)
    manager_role = roles[0]
    train_trajs = batch_runner(runner, plain_metric, current_prompts, train_rows, num_threads, budget)
    executed = [t for t in train_trajs if t.get("executed")]

    local_evals_by_traj: dict[int, list] = {}
    if credit_enabled and executed:
        local_evals_by_traj = _subtrajectory_credit(reflection, executed, roles, manager_role, refl_inflight)
    jobs, contexts = _credit_jobs(executed, roles, current_prompts, local_evals_by_traj, credit_enabled, optimizer_mode)
    credits = _assign_credit(reflection, jobs, refl_inflight)
    training_instances = [
        {
            "task_id": traj["id"],
            "task_context": contexts[ti].task_context,
            "result_evaluation": contexts[ti].result_evaluation,
            "agent_evaluations": {credit.role: credit.evaluation for credit in credits if credit.traj_index == ti},
            "global_score": traj["score"],
        }
        for ti, traj in enumerate(executed)
    ]

    topology = getattr(runner, "topology", "centralized")
    task_type = (
        f"{dataset} (centralized {len(roles)}-role manager/worker multi-agent system)"
        if topology == "centralized"
        else f"{dataset} ({topology} multi-agent system with {len(roles)} optimized prompt components)"
    )
    if optimizer_mode in ("truce-release", "tavo-hybrid"):
        proposal = _release_proposal(
            reflection, roles, seed_prompts, training_instances, credits, task_type, optimizer_mode, refl_inflight
        )
    else:
        proposal = _paper_proposal(reflection, roles, seed_prompts, current_prompts, credits, task_type, refl_inflight)

    detail = {
        "train_ids": [t["id"] for t in train_trajs],
        "train_scores": [t["score"] for t in train_trajs],
        "train_mean": batch_mean(train_trajs),
        "train_executed": len(executed),
        "optimizer_mode": optimizer_mode,
        "manager_role": manager_role,
        "team_size": len(roles),
        "role_credit": {
            role: {
                "n_evals": sum(1 for (_, _, r, ev) in credits if r == role),
                "n_parse_failures": sum(
                    1 for (_, _, r, ev) in credits if r == role and ("error" in ev or ev.get("fallback_mode"))
                ),
                "overall_scores": [ev.get("overall_score") for (_, _, r, ev) in credits if r == role],
            }
            for role in roles
        },
        "credit_evaluations": [
            {"train_id": executed[ti]["id"], "segment_index": segment_index, "role": role, "evaluation": ev}
            for (ti, segment_index, role, ev) in credits
        ],
        "trajectory_credit": {
            "enabled": credit_enabled,
            "segment_counts": {str(traj["id"]): len(segment_iterations(traj, manager_role)) for traj in executed}
            if credit_enabled
            else {},
            "evaluations": [
                {"train_id": executed[ti]["id"], "iteration_number": k, "evaluation": ev}
                for ti, evals in sorted(local_evals_by_traj.items())
                for k, ev in sorted(evals, key=lambda t: t[0])
            ],
        },
        "meta_suggestions": proposal.meta,
        "overlay": proposal.overlay,
        "overlay_error": proposal.overlay_error,
        "role_instance_suggestions": proposal.role_suggestions,
        "refined_prompts": proposal.refined,
        "candidate_prompts": proposal.candidate,
        "roles_refined_from_seed": [r for r in roles if proposal.refined[r].strip() != seed_prompts[r].strip()],
        "roles_changed_from_input": [r for r in roles if proposal.candidate[r].strip() != current_prompts[r].strip()],
    }
    return proposal.candidate, detail


def _subtrajectory_credit(
    reflection: Any, executed: list[dict], roles: list[str], manager_role: str, refl_inflight: int
) -> dict[int, list[tuple[int, dict]]]:
    """Eq. 3 credit F_t of every sub-trajectory: ``{trajectory index: [(segment index, evaluation)]}``."""
    seg_jobs = []
    for ti, traj in enumerate(executed):
        segs = segment_iterations(traj, manager_role)
        for k, seg in enumerate(segs):
            seg_jobs.append((ti, k, seg, segs[:k], traj))

    def run_seg(job):
        ti, k, seg, prev, traj = job
        ev = evaluate_iteration_credit(
            reflection, k, seg, prev, build_task_context(traj, roles), build_result_context(traj)
        )
        return ti, k, ev

    local_evals_by_traj: dict[int, list[tuple[int, dict]]] = {}
    if seg_jobs:
        with ThreadPoolExecutor(max_workers=min(refl_inflight, len(seg_jobs))) as ex:
            for ti, k, ev in ex.map(run_seg, seg_jobs):
                local_evals_by_traj.setdefault(ti, []).append((k, ev))
    return local_evals_by_traj


def _credit_jobs(
    executed: list[dict],
    roles: list[str],
    current_prompts: dict[str, str],
    local_evals_by_traj: dict[int, list[tuple[int, dict]]],
    credit_enabled: bool,
    optimizer_mode: str,
) -> tuple[list[_CreditJob], list[_TrajectoryContext]]:
    """The credit-assignment jobs, and the context of each executed trajectory.

    The release and hybrid modes assess every agent over the whole trajectory;
    the paper mode assesses every (agent, sub-trajectory) pair the agent was
    active in.
    """
    jobs: list[_CreditJob] = []
    contexts = []
    for ti, traj in enumerate(executed):
        context = _trajectory_context(traj, roles, local_evals_by_traj.get(ti, []), credit_enabled)
        contexts.append(context)
        if optimizer_mode == "tavo-paper-reproduction":
            local_by_index = dict(local_evals_by_traj.get(ti, []))
            jobs.extend(_segment_credit_jobs(ti, traj, context, roles, current_prompts, local_by_index))
            continue
        for role in roles:
            history = build_agent_history(role, roles, traj, current_prompts)
            if history is not None:
                jobs.append(
                    _CreditJob(
                        ti, None, role, history, context.task_context, context.result_context, context.result_evaluation
                    )
                )
    return jobs, contexts


def _trajectory_context(
    traj: dict, roles: list[str], local_evals: list[tuple[int, dict]], credit_enabled: bool
) -> _TrajectoryContext:
    """The task context, result evaluation and result context of one trajectory.

    With trajectory credit, the local (Eq. 3) evaluations go into the result
    evaluation before the transcript excerpt, so the 12000-character window of
    the credit prompt cannot truncate them.
    """
    task_context = build_task_context(traj, roles)
    result_evaluation = build_result_evaluation(traj)
    result_context = build_result_context(traj)
    if credit_enabled:
        excerpt = result_evaluation.pop("group_chat_transcript_excerpt", None)
        result_evaluation["local_evaluations"] = [ev for _, ev in sorted(local_evals, key=lambda t: t[0])]
        if excerpt is not None:
            result_evaluation["group_chat_transcript_excerpt"] = excerpt
    return _TrajectoryContext(task_context, result_evaluation, result_context)


def _segment_credit_jobs(
    ti: int,
    traj: dict,
    context: _TrajectoryContext,
    roles: list[str],
    current_prompts: dict[str, str],
    local_by_index: dict[int, dict],
) -> list[_CreditJob]:
    """Paper mode: one job per (agent, sub-trajectory) pair of trajectory ``ti`` the agent was active in.

    A job sees only its segment's messages, and its result evaluation adds the
    segment and the segment's local (Eq. 3) evaluation when there is one.
    """
    jobs = []
    for segment_index, segment in enumerate(segment_iterations(traj, roles[0])):
        segment_traj = dict(traj)
        segment_traj["messages"] = segment.get("messages", [])
        segment_evaluation = dict(context.result_evaluation)
        segment_evaluation["subtrajectory"] = segment
        if segment_index in local_by_index:
            segment_evaluation["local_evaluation"] = local_by_index[segment_index]
        for role in active_roles_in_segment(segment, roles):
            history = build_agent_history(role, roles, segment_traj, current_prompts)
            if history is not None:
                jobs.append(
                    _CreditJob(
                        ti,
                        segment_index,
                        role,
                        history,
                        context.task_context,
                        context.result_context,
                        segment_evaluation,
                    )
                )
    return jobs


def _assign_credit(reflection: Any, jobs: list[_CreditJob], refl_inflight: int) -> list[_Credit]:
    """Trajectory-aware credit assignment of every job, in job order."""

    def run_credit(job: _CreditJob) -> _Credit:
        evaluation = credit_assign(reflection, job.history, job.task_context, job.result_context, job.result_evaluation)
        return _Credit(job.traj_index, job.segment_index, job.role, evaluation)

    if not jobs:
        return []
    with ThreadPoolExecutor(max_workers=min(refl_inflight, len(jobs))) as ex:
        return list(ex.map(run_credit, jobs))


def _release_proposal(
    reflection: Any,
    roles: list[str],
    seed_prompts: dict[str, str],
    training_instances: list[dict],
    credits: list[_Credit],
    task_type: str,
    optimizer_mode: str,
    refl_inflight: int,
) -> _Proposal:
    """``truce-release`` and ``tavo-hybrid``: meta-knowledge and one shared overlay on every role.

    The release keeps the seed prompts as the base profiles (the public
    pipeline's default overlay mode); the hybrid first refines every role from
    its seed prompt with the meta-knowledge and its own suggestions.
    """
    meta = generate_meta_suggestions(reflection, training_instances)
    role_suggestions = {
        role: merge_role_evaluations([credit.evaluation for credit in credits if credit.role == role]) for role in roles
    }

    def run_overlay() -> tuple[str | None, str | None]:
        try:
            if meta:
                return build_overlay_from_meta(reflection, meta, task_type), None
        except Exception as exc:
            return None, f"{type(exc).__name__}: {exc}"[:300]
        return None, None

    def run_hybrid_refine(role: str) -> tuple[str, str]:
        return role, refine_with_meta_knowledge(
            reflection,
            role,
            seed_prompts[role],
            f"Task Type: {task_type}",
            meta,
            instance_suggestions=role_suggestions[role],
        )

    if optimizer_mode == "truce-release":
        overlay_text, overlay_error = run_overlay()
        refined = dict(seed_prompts)
    else:
        with ThreadPoolExecutor(max_workers=min(refl_inflight, 1 + len(roles))) as ex:
            overlay_future = ex.submit(run_overlay)
            refine_futures = [ex.submit(run_hybrid_refine, role) for role in roles]
            refined = dict(f.result() for f in refine_futures)
            overlay_text, overlay_error = overlay_future.result()

    candidate = {}
    for role in roles:
        text = strip_overlay_blocks(refined[role]) or seed_prompts[role]
        candidate[role] = append_overlay(text, overlay_text) if overlay_text else text
    return _Proposal(candidate, refined, role_suggestions, meta, overlay_text, overlay_error)


def _paper_proposal(
    reflection: Any,
    roles: list[str],
    seed_prompts: dict[str, str],
    current_prompts: dict[str, str],
    credits: list[_Credit],
    task_type: str,
    refl_inflight: int,
) -> _Proposal:
    """``tavo-paper-reproduction`` (paper Algorithm 1): per-agent aggregation, then refinement.

    Every agent's edits are aggregated across tasks and sub-trajectories and
    refine that agent's previous prompt; no shared overlay is mixed in.
    """

    def aggregate_for_role(role: str) -> tuple[str, dict | None]:
        evals = [credit.evaluation for credit in credits if credit.role == role]
        return role, aggregate_role_evaluations_llm(reflection, role, evals)

    with ThreadPoolExecutor(max_workers=min(refl_inflight, max(1, len(roles)))) as ex:
        role_suggestions = dict(ex.map(aggregate_for_role, roles))

    def run_paper_refine(role: str) -> tuple[str, str]:
        previous = strip_overlay_blocks(current_prompts[role]) or seed_prompts[role]
        suggestions = role_suggestions[role]
        if not suggestions:
            return role, previous
        return role, refine_agent_prompt(reflection, role, previous, suggestions, f"Task Type: {task_type}")

    with ThreadPoolExecutor(max_workers=min(refl_inflight, max(1, len(roles)))) as ex:
        refined = dict(ex.map(run_paper_refine, roles))
    return _Proposal(dict(refined), refined, role_suggestions)


__all__ = [
    "BENCHMARK_ADAPTATIONS",
    "DEFAULT_OPTIMIZER_MODE",
    "MODE_CLASSIFICATION",
    "OPTIMIZER_MODES",
    "TRUCE_RELEASE",
    "active_roles_in_segment",
    "aggregate_role_evaluations_llm",
    "append_overlay",
    "batch_mean",
    "build_agent_history",
    "build_overlay_from_meta",
    "build_result_context",
    "build_result_evaluation",
    "build_task_context",
    "credit_assign",
    "evaluate_iteration_credit",
    "generate_meta_suggestions",
    "merge_role_evaluations",
    "overlay_id",
    "parse_json_response",
    "refine_agent_prompt",
    "refine_with_meta_knowledge",
    "run_optimization_attempt",
    "segment_iterations",
    "strip_overlay_blocks",
]
