"""Robust extraction of a [0,1] score from a judge model's text output."""

from __future__ import annotations

import re

_FLOAT = re.compile(r"-?\d+(?:\.\d+)?")
_LIST_LINE = re.compile(
    r"^(?:[^\w\-]|-(?=\s))*(?:\d{1,2}[.)]\s+)?(?:candidate\s*(?P<idx>\d+)\s*[:.)\-]?\s*)?\**\s*"
    r"(?P<val>-?\d+(?:\.\d+)?)\s*(?:/\s*1(?:\.0+)?)?\s*\**\s*\.?\s*$",
    re.I,
)


def parse_score(text: str, default: float = 0.5) -> float:
    """First number in a judge reply, rescaled into [0, 1] (``default`` when none parses)."""
    if not text:
        return default
    m = _FLOAT.findall(text)
    if not m:
        return default
    try:
        val = float(m[0])
    except ValueError:
        return default
    # Some judges answer on a 0-100 or 0-10 scale despite instructions.
    if val > 1.0:
        if val <= 10.0:
            val /= 10.0
        elif val <= 100.0:
            val /= 100.0
        else:
            val = 1.0
    return float(min(1.0, max(0.0, val)))


def parse_score_list(text: str, k: int, default: float = 0.5) -> list[float]:
    """Parse the listwise judge's output: one score per line, k lines expected
    (paper Fig. 4: "return exactly a score each line corresponding to the
    prompt's original position"). Falls back to all floats in reading order,
    pads with `default` when short; each value rescaled/clamped like parse_score."""
    # strict pass: lines that are exactly one score (optionally decorated as
    # "- 0.62", "1. 0.62", "Candidate 3: 0.62", "**0.62**", "0.62/1.00"). A preamble
    # like "Here are the 3 scores:" must NOT contribute (it shifted every score by one).
    strict: list[tuple[int | None, float]] = []
    for line in (text or "").splitlines():
        m = _LIST_LINE.match(line.strip())
        if m:
            idx = int(m.group("idx")) if m.group("idx") else None
            strict.append((idx, parse_score(m.group("val"), default)))
    if len(strict) >= k > 0:
        head = strict[:k]
        idxs = [i for i, _ in head]
        if all(i is not None for i in idxs) and sorted(idxs) == list(range(1, k + 1)):
            # judge answered "Candidate i: score" in RANKED order -> place by index
            placed = [default] * k
            for i, v in head:
                placed[i - 1] = v
            return placed
        return [v for _, v in head]

    def _pick(tokens: list[str]) -> str | None:
        # scores are two-decimal per the template; prefer decimal-form floats so a
        # prose line like "Candidate 1: 0.7" yields 0.7, not the candidate index
        dec = [t for t in tokens if "." in t]
        return (dec or tokens or [None])[0]

    vals: list[float] = []
    if text:
        for line in text.splitlines():
            tok = _pick(_FLOAT.findall(line))
            if tok is not None:
                vals.append(parse_score(tok, default))
            if len(vals) >= k:
                break
        if len(vals) < k:  # judge ignored line structure: take floats in order
            toks = _FLOAT.findall(text)
            dec = [t for t in toks if "." in t]
            flat = [parse_score(t, default) for t in (dec if len(dec) >= k else toks)]
            if len(flat) >= len(vals):
                vals = flat[:k]
    return (vals + [default] * k)[:k]
