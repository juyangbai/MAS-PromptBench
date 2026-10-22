"""Validation-only, strict-improvement deployment selection."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .schema import PromptBundle, RunRecord


@dataclass(frozen=True)
class SelectionDecision:
    """Which bundle is deployed and why."""

    deployed: PromptBundle
    seed_score: float | None
    candidate_score: float | None
    delta: float | None
    selected_candidate: bool
    fallback_reason: str | None
    paired_example_ids: tuple[str, ...]


def _score_map(records: Iterable[RunRecord]) -> tuple[dict[str, float], str | None]:
    values: dict[str, float] = {}
    for record in records:
        if record.example_id in values:
            return {}, f"duplicate_example:{record.example_id}"
        if not record.usable or record.score is None:
            return {}, f"invalid_record:{record.example_id}:{record.status}"
        score = float(record.score)
        if not 0.0 <= score <= 1.0:
            return {}, f"score_out_of_range:{record.example_id}"
        values[record.example_id] = score
    if not values:
        return {}, "empty_validation"
    return values, None


def select_for_deployment(
    seed_bundle: PromptBundle,
    candidate_bundle: PromptBundle,
    seed_validation: Iterable[RunRecord],
    candidate_validation: Iterable[RunRecord],
) -> SelectionDecision:
    """Deploy the candidate only for a complete, paired, strictly better validation.

    A tie, a regression, an unusable record or unpaired IDs keep the seed.
    """
    seed, seed_error = _score_map(seed_validation)
    candidate, candidate_error = _score_map(candidate_validation)
    if seed_error or candidate_error:
        reason = seed_error or candidate_error
        return SelectionDecision(seed_bundle, None, None, None, False, reason, ())
    if set(seed) != set(candidate):
        return SelectionDecision(seed_bundle, None, None, None, False, "unpaired_validation_ids", ())
    ordered = tuple(sorted(seed))
    seed_score = sum(seed[key] for key in ordered) / len(ordered)
    candidate_score = sum(candidate[key] for key in ordered) / len(ordered)
    delta = candidate_score - seed_score
    if candidate_score > seed_score:
        return SelectionDecision(candidate_bundle, seed_score, candidate_score, delta, True, None, ordered)
    reason = "validation_tie" if candidate_score == seed_score else "validation_regression"
    return SelectionDecision(seed_bundle, seed_score, candidate_score, delta, False, reason, ordered)


FORBIDDEN_SELECTION_KEYS = frozenset(
    {
        "test_ids",
        "test_examples",
        "test_score",
        "held_out_test",
        "test_predictions",
        "official_evaluation_ids",
    }
)


def assert_test_unexposed(metadata: dict) -> None:
    """Reject candidate-selection metadata that contains protected test material."""

    def walk(value, path: tuple[str, ...] = ()) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if str(key).strip().lower() in FORBIDDEN_SELECTION_KEYS:
                    raise ValueError("test data exposed during selection at " + ".".join(path + (str(key),)))
                walk(child, path + (str(key),))
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                walk(child, path + (str(index),))

    walk(metadata)


__all__ = ["FORBIDDEN_SELECTION_KEYS", "SelectionDecision", "assert_test_unexposed", "select_for_deployment"]
