"""``core.tasks.math``: answer extraction, Hendrycks equivalence and aggregation."""

from core.tasks import math as task


def test_extract_boxed_counts_nested_braces():
    assert task.extract_boxed("so \\boxed{1} then \\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"
    assert task.extract_boxed("no box") is None
    assert task.extract_boxed("\\boxed{unbalanced") is None


def test_is_equiv_normalizes_latex():
    assert task.is_equiv("\\frac12", "0.5")
    assert task.is_equiv("\\dfrac{1}{2}", "1/2")
    assert not task.is_equiv("2", "3")
    assert task.score(None, "2") == 0.0


def test_majority_over_equivalence_buckets():
    assert task.majority_vote([{"answer": "1/2"}, {"answer": "3"}, {"answer": "0.5"}]) == "1/2"
    assert task.best_of_n(["3", None, "", "\\frac{1}{2}", "0.5"]) == "\\frac{1}{2}"
    assert task.majority_vote([{"answer": None}]) is None
