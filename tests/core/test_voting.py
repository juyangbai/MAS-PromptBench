"""``core.voting``: the gold-free majority vote that selects an ensemble's submission."""

from core import voting


def test_the_largest_bucket_wins_and_its_lowest_member_is_selected():
    assert voting.majority(["a", "b", "b", "c"]) == 1
    assert voting.majority(["a", "b", "a", "b", "b"]) == 1
    assert voting.majority(["c", "a", "b", "a"]) == 1


def test_ties_go_to_the_bucket_holding_the_lowest_member():
    assert voting.majority(["b", "a"]) == 0
    assert voting.majority(["a", "b", "b", "a"]) == 0
    assert voting.majority([None, "x", "y", "y", "x"]) == 1


def test_empty_keys_abstain_and_an_all_abstaining_ensemble_selects_the_first_member():
    assert voting.majority([None, "", "a", None, ""]) == 2
    assert voting.majority(["", "a", "", "", "b", "b"]) == 4
    assert voting.majority([None, "", None]) == 0


def test_texts_vote_whitespace_normalized():
    assert voting.normalized("  def f():\n\treturn 1\n\n") == "def f(): return 1"
    assert voting.normalized(None) == ""
    programs = ["print(1)", "x = 1\nprint(x)", "x = 1\n\nprint(x)  \n", "   ", None]
    assert voting.majority_text(programs) == 1
    assert voting.majority_text(["   ", None, "\n"]) == 0
