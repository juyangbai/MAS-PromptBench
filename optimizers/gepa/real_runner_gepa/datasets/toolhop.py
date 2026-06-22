"""ToolHop dataset bridge for real-runner GEPA."""
from __future__ import annotations

import random
from collections import Counter
from pathlib import Path
from typing import Any

import dspy

from real_runner_gepa.datasets.frozen_splits import (
    apply_signal_split_if_requested,
    signal_manifest_metadata,
)
from real_runner_gepa.datasets.split_utils import real_eval_ids


REPO_ROOT = Path(__file__).resolve().parents[4]
PROFILE_FEATURE_WEIGHTS = {
    "answer_type": 4.0,
    "previous_answer_type": 3.0,
    "hop_count": 2.0,
    "question_word_bucket": 1.5,
    "duplicate_tool_names": 0.75,
    "domain": 0.5,
}


def load_all() -> list[dspy.Example]:
    """Load ToolHop rows through the shared benchmark loader."""
    from topologies.single.toolhop import langgraph_toolhop as toolhop_common

    rows = toolhop_common.load_instances()
    examples: list[dspy.Example] = []
    for row in rows:
        rid = str(row.get("id"))
        examples.append(
            dspy.Example(
                id=rid,
                question=row.get("question") or "",
                tools=row.get("tools") or {},
                functions=row.get("functions") or [],
                task_instance=row,
                answer=str(row.get("answer", "")),
                raw={"id": rid},
            ).with_inputs("task_instance")
        )
    return examples


def _norm(value: Any, default: str = "unknown") -> str:
    text = str(value if value is not None else "").strip().lower()
    return text or default


def _question_word_bucket(question: str) -> str:
    words = len(str(question or "").split())
    if words <= 18:
        return "<=18"
    if words <= 24:
        return "19-24"
    if words <= 30:
        return "25-30"
    return "31+"


def _features(example: dspy.Example) -> dict[str, Any]:
    row = getattr(example, "task_instance", {}) or {}
    tools = row.get("tools") or {}
    tool_names = [
        str(schema.get("name"))
        for schema in tools.values()
        if isinstance(schema, dict) and schema.get("name")
    ]
    sub_task = row.get("sub_task") or {}
    return {
        "answer_type": _norm(row.get("answer_type")),
        "previous_answer_type": _norm(row.get("previous_answer_type")),
        "hop_count": len(sub_task) if isinstance(sub_task, dict) else len(tools),
        "question_word_bucket": _question_word_bucket(str(row.get("question") or "")),
        "duplicate_tool_names": len(tool_names) != len(set(tool_names)),
        "domain": _norm(row.get("domain")),
    }


def split_profile_counts(rows: list[dspy.Example]) -> dict[str, dict]:
    features = [_features(row) for row in rows]
    return {
        name: dict(Counter(row[name] for row in features))
        for name in PROFILE_FEATURE_WEIGHTS
    }


def _largest_remainder_counts(values: list[Any], size: int) -> Counter:
    counts = Counter(values)
    total = sum(counts.values())
    if size <= 0 or total <= 0:
        return Counter()
    quotas = {key: counts[key] * size / total for key in counts}
    allocated = Counter({key: int(quotas[key]) for key in counts})
    remaining = size - sum(allocated.values())
    ranked = sorted(
        counts,
        key=lambda key: (
            quotas[key] - int(quotas[key]),
            quotas[key],
            str(key),
        ),
        reverse=True,
    )
    for key in ranked[:remaining]:
        allocated[key] += 1
    return allocated


def _target_profile(reference: list[dspy.Example], size: int) -> dict[str, Counter]:
    features = [_features(example) for example in reference]
    return {
        name: _largest_remainder_counts([row[name] for row in features], size)
        for name in PROFILE_FEATURE_WEIGHTS
    }


def _profile_cost(counts: dict[str, Counter], target: dict[str, Counter]) -> float:
    cost = 0.0
    for name, weight in PROFILE_FEATURE_WEIGHTS.items():
        target_counts = target.get(name, Counter())
        observed = counts.get(name, Counter())
        keys = set(target_counts) | set(observed)
        denom = max(sum(target_counts.values()), 1)
        cost += weight * sum(
            abs(observed.get(key, 0) - target_counts.get(key, 0))
            for key in keys
        ) / denom
    return cost


def _select_profile_matched(
    pool: list[dspy.Example],
    reference: list[dspy.Example],
    size: int,
    rng: random.Random,
) -> list[dspy.Example]:
    if size <= 0:
        return []
    if len(pool) < size:
        raise ValueError(f"ToolHop profile split needs {size} examples; got {len(pool)}")
    target = _target_profile(reference or pool, size)
    selected: list[dspy.Example] = []
    selected_ids: set[str] = set()
    counts: dict[str, Counter] = {name: Counter() for name in PROFILE_FEATURE_WEIGHTS}
    seen_domains: set[str] = set()
    feature_by_id = {str(candidate.id): _features(candidate) for candidate in pool}

    for _ in range(size):
        remaining = [candidate for candidate in pool if str(candidate.id) not in selected_ids]
        candidate_pool = remaining
        for primary_name in ("answer_type", "previous_answer_type", "hop_count", "question_word_bucket"):
            deficits = {
                value
                for value, desired in target.get(primary_name, Counter()).items()
                if counts[primary_name][value] < desired
            }
            if not deficits:
                continue
            filtered = [
                candidate
                for candidate in candidate_pool
                if feature_by_id[str(candidate.id)][primary_name] in deficits
            ]
            if filtered:
                candidate_pool = filtered

        best: tuple[float, float, str, dspy.Example, dict[str, Any]] | None = None
        for candidate in candidate_pool:
            cid = str(candidate.id)
            features = feature_by_id[cid]
            trial = {name: Counter(value) for name, value in counts.items()}
            for name, value in features.items():
                trial[name][value] += 1
            cost = _profile_cost(trial, target)
            domain_bonus = 0.03 if features["domain"] not in seen_domains else 0.0
            tie = rng.random() * 1e-6
            key = (cost - domain_bonus + tie, -domain_bonus, cid, candidate, features)
            if best is None or key < best:
                best = key
        if best is None:
            break
        _, _, _, chosen, chosen_features = best
        selected.append(chosen)
        selected_ids.add(str(chosen.id))
        seen_domains.add(str(chosen_features["domain"]))
        for name, value in chosen_features.items():
            counts[name][value] += 1

    rng.shuffle(selected)
    return selected


def train_val_split(
    examples: list[dspy.Example],
    train_size: int,
    val_size: int,
    seed: int = 0,
    offset: int = 0,
) -> tuple[list[dspy.Example], list[dspy.Example]]:
    """Profile-aware split excluding ToolHop's protected 0..99 report IDs."""
    frozen = apply_signal_split_if_requested("toolhop", examples, train_size, val_size, seed, offset)
    if frozen is not None:
        return frozen

    if train_size < 0 or val_size < 0:
        raise ValueError(f"split sizes must be non-negative; got train={train_size} val={val_size}")
    excluded = real_eval_ids("toolhop")
    reference = [example for example in examples if str(example.id) in excluded]
    pool = [example for example in examples if str(example.id) not in excluded]
    rng = random.Random(seed)
    rng.shuffle(pool)
    pool = pool[offset:]
    need = train_size + val_size
    if len(pool) < need:
        raise ValueError(
            f"ToolHop needs {need} examples after excluding {len(excluded)} protected eval IDs "
            f"and offset={offset}; got {len(pool)}"
        )
    train = _select_profile_matched(pool, reference, train_size, rng)
    train_ids = {str(row.id) for row in train}
    val_pool = [example for example in pool if str(example.id) not in train_ids]
    val = _select_profile_matched(val_pool, reference, val_size, rng)
    overlap = {str(row.id) for row in train} & {str(row.id) for row in val}
    if overlap:
        raise ValueError(f"ToolHop train/val overlap: {sorted(overlap)[:5]}")
    return train, val


def split_manifest_metadata(
    train_size: int,
    val_size: int,
    seed: int = 0,
    offset: int = 0,
) -> dict[str, Any] | None:
    return signal_manifest_metadata("toolhop", train_size, val_size, seed, offset)


def _prediction_text(prediction: Any) -> str:
    for attr in ("answer", "predicted_answer", "raw"):
        value = getattr(prediction, attr, None)
        if value is not None and str(value).strip():
            return str(value)
    return ""


def _trace_agent_text(pred_trace: Any) -> str:
    if not pred_trace:
        return ""
    parts: list[str] = []
    for _, _, outputs in pred_trace:
        trace = getattr(outputs, "agent_trace", None)
        if trace:
            parts.append(str(trace))
    return "\n".join(parts)


def metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
    """DSPy-compatible ToolHop final-answer metric."""
    from topologies.single.toolhop import langgraph_toolhop as toolhop_common

    pred_text = _prediction_text(prediction)
    gold = str(getattr(example, "answer", ""))
    prev_tool_content = str(
        getattr(prediction, "scoring_prev_tool_content", "")
        or getattr(prediction, "previous_tool_content", "")
        or ""
    )
    final_text_correct = toolhop_common.score_answer(gold, pred_text, "")
    correct = final_text_correct or toolhop_common.score_answer(gold, pred_text, prev_tool_content)
    score = 1.0 if correct else 0.0
    role = pred_name or "program"
    trace_text = _trace_agent_text(pred_trace) or getattr(prediction, "agent_trace", "")
    if correct:
        route = "final answer" if final_text_correct else "selected agent tool observation"
        feedback = (
            f"Correct ToolHop answer for role {role} via {route}. "
            f"Gold answer: {gold}. Predicted answer: {pred_text}."
        )
    else:
        feedback = (
            f"ToolHop failure for role {role}.\n"
            f"Gold answer: {gold}\n"
            f"Predicted answer: {pred_text or '<empty>'}\n"
            f"Real-runner trace:\n{str(trace_text)[:1800]}\n"
            "Actionable fix: call the dynamic tools needed for each hop, preserve intermediate "
            "values exactly, and end with one short final answer wrapped as <answer>...</answer>."
        )
    return dspy.Prediction(score=score, feedback=feedback)
