"""pytest wiring for the golden behavior snapshots.

All selected cells are executed once per session, in parallel worker
processes (``GOLDEN_WORKERS`` overrides the pool size), before the first
comparison runs; each ``test_cell[...]`` then only diffs its snapshot.
Cells whose local Hugging Face cache entries are missing (``prereqs``) are
skipped, not run.
"""

from __future__ import annotations

import time

import pytest

from tests.golden import harness, prereqs

_STATS: dict = {}


def pytest_configure(config):
    config.addinivalue_line("markers", "golden: golden behavior-snapshot cell (tests/golden)")
    config.addinivalue_line("markers", "golden_fast: small representative subset of the golden cells (~1 min)")


def _selected_cells(session) -> list[str]:
    ids = []
    for item in session.items:
        callspec = getattr(item, "callspec", None)
        if callspec is not None and "cell_id" in callspec.params and item.get_closest_marker("golden"):
            ids.append(callspec.params["cell_id"])
    return list(dict.fromkeys(ids))


@pytest.fixture(scope="session")
def golden_missing(request) -> dict[str, tuple[str, ...]]:
    """Selected cells that cannot run here, with their missing cache entries."""
    from tests.golden import cells as cellmod

    index = cellmod.cells_by_id()
    missing = prereqs.missing(index[cid] for cid in _selected_cells(request.session))
    _STATS["missing"] = missing
    return missing


@pytest.fixture(scope="session")
def golden_results(request, tmp_path_factory, golden_missing):
    from tests.golden import cells as cellmod
    from tests.golden import store

    ids = [cid for cid in _selected_cells(request.session) if cid not in golden_missing]
    session = tmp_path_factory.mktemp("golden-session")
    started = time.time()
    results = harness.run_cells(ids, session_dir=session) if ids else {}
    first = time.time() - started
    # Cells that differ are re-run (low parallelism, up to 2x): a behavior
    # change reproduces, a load-induced subprocess timeout does not.
    goldens = store.load_all(cellmod.AREAS)
    reruns = harness.settle(results, {cid: goldens[cid] for cid in ids if cid in goldens}, session_dir=session)
    _STATS.update(cells=len(ids), seconds=time.time() - started, first_pass=first, reruns=reruns)
    return results


def pytest_terminal_summary(terminalreporter):
    missing = _STATS.get("missing")
    if missing:
        entries = [entry for absent in missing.values() for entry in absent]
        terminalreporter.write_line(f"{prereqs.message(entries)} ({len(missing)} cell(s) skipped)")
    if "cells" not in _STATS:
        return
    terminalreporter.write_line(
        f"golden: executed {_STATS['cells']} cell(s) in {_STATS['seconds']:.0f}s "
        f"(first pass {_STATS['first_pass']:.0f}s, {harness.default_workers()} parallel workers)"
    )
    reruns = _STATS.get("reruns") or {}
    flaky = {cid: h for cid, h in reruns.items() if h and h[-1] == "match"}
    if flaky:
        terminalreporter.write_line(f"golden: {len(flaky)} cell(s) matched only on re-run (load-sensitive):")
        for cid, history in sorted(flaky.items()):
            terminalreporter.write_line(f"  {cid}: attempts {history[1:]}")
            terminalreporter.write_line("    " + history[0].replace("\n", "\n    "))
