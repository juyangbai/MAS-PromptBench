"""Golden behavior-snapshot tests.

Each cell re-runs one fixed instance (or a static snapshot) against the fake
LLM server and must reproduce the recorded golden exactly: every request
body sent to the model (messages, tools, decoding parameters), the runner's
reduced result and its own score.

    PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider tests/golden -q
    PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider tests/golden -q -m golden_fast
    PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider tests/golden -q -k 'registry and hotpotqa'

Regenerate after an intended behavior change: ``python -m tests.golden.record --force``.
"""

from __future__ import annotations

import os

import pytest

from tests.golden import cells as cellmod
from tests.golden import prereqs, render, store

CELLS = cellmod.all_cells()
GOLDENS = store.load_all(cellmod.AREAS)
DIFF_LINES = int(os.environ.get("GOLDEN_DIFF_LINES", "400"))


def _params():
    for cell in CELLS:
        marks = [pytest.mark.golden]
        if cell["fast"]:
            marks.append(pytest.mark.golden_fast)
        yield pytest.param(cell["id"], id=cell["id"], marks=marks)


@pytest.mark.golden_fast
def test_inventory():
    """Every runner / registry key still exists and nothing new is unrecorded."""
    if not GOLDENS:
        pytest.fail("no goldens recorded; run `python -m tests.golden.record`")
    live = {cell["id"] for cell in CELLS}
    recorded = set(GOLDENS)
    missing = sorted(recorded - live)
    new = sorted(live - recorded)
    message = []
    if missing:
        message.append(f"{len(missing)} recorded cell(s) no longer exist: {missing[:20]}")
    if new:
        message.append(f"{len(new)} cell(s) have no golden (record them): {new[:20]}")
    assert not message, "\n".join(message)


@pytest.mark.parametrize("cell_id", list(_params()))
def test_cell(cell_id, golden_missing, golden_results):
    if cell_id in golden_missing:
        pytest.skip(prereqs.message(golden_missing[cell_id]))
    expected = GOLDENS.get(cell_id)
    if expected is None:
        pytest.fail(f"{cell_id}: no golden recorded (python -m tests.golden.record --force --only '{cell_id}')")
    result = golden_results[cell_id]
    if result.get("harness_failure"):
        tail = "\n".join(result["meta"].get("log_tail") or [])
        pytest.fail(f"{cell_id}: worker {result['snapshot']['status']}\n{tail}")
    actual = result["snapshot"]
    if actual != expected:
        max_lines = DIFF_LINES if DIFF_LINES > 0 else 10**9
        pytest.fail(
            f"{cell_id}: behavior differs from golden ({render.summary(expected, actual)})\n"
            f"{render.diff(expected, actual, max_lines=max_lines)}",
            pytrace=False,
        )
