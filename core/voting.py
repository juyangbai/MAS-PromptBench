"""Gold-free selection of an ensemble's submission: a majority vote over the members' final outputs.

Each member (an independent replica or a decentralized peer, in agent order)
casts one vote: the dataset's key of its final output, e.g. the whitespace-
normalized program or patch, or the canonical BFCL call list. Members without
an output (empty or unparseable) abstain. The largest bucket of equal keys wins,
ties go to the bucket holding the lowest member, and that bucket's lowest member
is selected; when every member abstains, the first member is. Gold labels and
tests never enter the vote, so only the selected output is scored.
"""

from __future__ import annotations

from collections.abc import Sequence


def normalized(text: str | None) -> str:
    """``text`` with every whitespace run collapsed to one space and the ends stripped ("" for None)."""
    return " ".join((text or "").split())


def majority(keys: Sequence[str | None]) -> int:
    """Index of the selected member, given each member's vote key (None or "" abstains)."""
    buckets: dict[str, list[int]] = {}
    for index, key in enumerate(keys):
        if key:
            buckets.setdefault(key, []).append(index)
    if not buckets:
        return 0
    return max(buckets.values(), key=lambda members: (len(members), -members[0]))[0]


def majority_text(texts: Sequence[str | None]) -> int:
    """:func:`majority` over the whitespace-normalized texts (programs, patches)."""
    return majority([normalized(text) for text in texts])
