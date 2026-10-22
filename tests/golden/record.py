"""Record (regenerate) golden snapshots from the current code.

    python -m tests.golden.record                 # all cells; refuses to overwrite
    python -m tests.golden.record --force         # overwrite existing goldens
    python -m tests.golden.record --force --only 'registry/gepa/hotpotqa/*' --only 'cli/*'
    python -m tests.golden.record --list          # print the cell inventory

Cells run in parallel worker processes (``--workers`` or GOLDEN_WORKERS).
With ``--only`` the selected cells are merged into the existing golden files.
A cell whose worker crashed or timed out is never written.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

from tests.golden import cells as cellmod
from tests.golden import harness, prereqs, render, store

MANIFEST = store.DATA_DIR / "manifest.json"


def select(patterns: list[str] | None) -> list[str]:
    ids = [cell["id"] for cell in cellmod.all_cells()]
    if not patterns:
        return ids
    chosen = [cid for cid in ids if any(fnmatch.fnmatchcase(cid, p) for p in patterns)]
    if not chosen:
        raise SystemExit(f"no cell matches {patterns}")
    return chosen


def write_manifest(goldens: dict[str, dict], seconds: float | None) -> None:
    index = cellmod.cells_by_id()
    per_area = Counter(store.area_of(cid) for cid in goldens)
    status = Counter(snap.get("status") for snap in goldens.values())
    not_ok = {
        cid: snap.get("error") or snap.get("status")
        for cid, snap in sorted(goldens.items())
        if snap.get("status") != "ok"
    }
    partial = {cid: snap.get("skipped") for cid, snap in sorted(goldens.items()) if snap.get("skipped")}
    by_error: dict[str, list[str]] = defaultdict(list)
    for cid, error in not_ok.items():
        by_error[str(error)].append(cid)
    sizes = {p.name: p.stat().st_size for p in sorted(store.DATA_DIR.glob("*.json.gz"))}
    manifest = {
        "cells": len(goldens),
        "per_area": dict(sorted(per_area.items())),
        "status": dict(sorted(status.items())),
        "fast_cells": sorted(cid for cid in goldens if index.get(cid, {}).get("fast")),
        "not_ok": [{"error": error, "cells": ids} for error, ids in sorted(by_error.items())],
        "partially_skipped": partial,
        "agents_sdk": (
            "decentralized/openai_agents turns use the harness stand-in (tests/golden/sdk_fake.py); "
            "the OpenAI Agents SDK request format itself is not covered"
        ),
        "golden_bytes": sizes,
        "golden_bytes_total": sum(sizes.values()),
        "last_full_record_seconds": round(seconds, 1) if seconds is not None else None,
        "record_command": "python -m tests.golden.record --force",
    }
    MANIFEST.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="overwrite existing golden cells")
    parser.add_argument("--only", action="append", help="glob over cell ids (repeatable)")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=harness.DEFAULT_TIMEOUT_S, help="per-cell timeout (s)")
    parser.add_argument("--list", action="store_true", help="list cell ids and exit")
    parser.add_argument("--keep-tmp", action="store_true", help="keep worker temp dirs (debugging)")
    parser.add_argument(
        "--no-verify",
        dest="verify",
        action="store_false",
        help="skip the reproduction pass (faster, but a load-induced flake could be recorded)",
    )
    args = parser.parse_args(argv)

    if args.list:
        for cell in cellmod.all_cells():
            print(cell["id"], "[fast]" if cell["fast"] else "")
        return 0

    ids = select(args.only)
    index = cellmod.cells_by_id()
    absent = prereqs.missing(index[cid] for cid in ids)
    if absent:
        print(prereqs.message(entry for entries in absent.values() for entry in entries), file=sys.stderr)
        return 2
    areas = sorted({store.area_of(cid) for cid in ids})
    existing = store.load_all(cellmod.AREAS)
    clash = [cid for cid in ids if cid in existing]
    if clash and not args.force:
        print(
            f"refusing to overwrite {len(clash)} existing golden cell(s) (e.g. {clash[0]}); pass --force",
            file=sys.stderr,
        )
        return 2

    started = time.time()
    failures: list[str] = []

    def progress(cid: str, result: dict, done: int, total: int) -> None:
        snap = result["snapshot"]
        mark = "ok" if snap.get("status") == "ok" else snap.get("status")
        if result.get("harness_failure"):
            failures.append(cid)
        print(f"[{done:>4}/{total}] {mark:<8} {result['meta'].get('wall_seconds', 0):>6.1f}s  {cid}", flush=True)

    with tempfile.TemporaryDirectory(prefix="golden-session-") as session:
        results = harness.run_cells(
            ids,
            workers=args.workers,
            timeout_s=args.timeout,
            progress=progress,
            keep_tmp=args.keep_tmp,
            session_dir=Path(session),
        )
        unstable: list[str] = []
        if args.verify and not failures:
            # Second pass: a golden is only written if an independent re-run
            # reproduces it; disagreements are settled at low parallelism by
            # majority of three runs.
            print("verification pass ...", flush=True)
            second = harness.run_cells(
                ids, workers=args.workers, timeout_s=args.timeout, session_dir=Path(session), with_dependencies=False
            )
            differ = [
                cid
                for cid in ids
                if second[cid].get("harness_failure") or second[cid]["snapshot"] != results[cid]["snapshot"]
            ]
            if differ:
                print(f"{len(differ)} cell(s) differed between passes; settling: {differ[:10]}", flush=True)
                for cid in differ[:10]:
                    if not second[cid].get("harness_failure"):
                        print(
                            f"--- {cid} (pass 1 vs pass 2)\n"
                            + render.diff(results[cid]["snapshot"], second[cid]["snapshot"], max_lines=30),
                            flush=True,
                        )
                third = harness.run_cells(
                    differ,
                    workers=min(4, len(differ)),
                    timeout_s=args.timeout,
                    session_dir=Path(session),
                    with_dependencies=False,
                )
                for cid in differ:
                    if third[cid].get("harness_failure"):
                        unstable.append(cid)
                    elif third[cid]["snapshot"] == second[cid]["snapshot"]:
                        results[cid] = second[cid]
                    elif third[cid]["snapshot"] != results[cid]["snapshot"]:
                        unstable.append(cid)
    seconds = time.time() - started
    if unstable:
        print(f"UNSTABLE cells (three runs disagree), nothing written: {unstable}", file=sys.stderr)
        return 1
    if failures:
        for cid in failures:
            print(
                f"HARNESS FAILURE {cid}:\n  " + "\n  ".join(results[cid]["meta"].get("log_tail") or []), file=sys.stderr
            )
        print(f"{len(failures)} cell(s) crashed or timed out; nothing written", file=sys.stderr)
        return 1

    merged = dict(existing)
    for cid in ids:
        merged[cid] = results[cid]["snapshot"]
    # Drop goldens of cells that no longer exist (full re-record only).
    if not args.only:
        live = {cell["id"] for cell in cellmod.all_cells()}
        merged = {cid: snap for cid, snap in merged.items() if cid in live}
    for area in cellmod.AREAS:
        snaps = {cid: snap for cid, snap in merged.items() if store.area_of(cid) == area}
        if snaps and (area in areas or not args.only):
            path = store.write_area(area, snaps)
            print(f"wrote {path.relative_to(cellmod.REPO)} ({len(snaps)} cells, {path.stat().st_size / 1e6:.2f} MB)")
    previous = json.loads(MANIFEST.read_text()).get("last_full_record_seconds") if MANIFEST.exists() else None
    write_manifest(merged, seconds if not args.only else previous)
    not_ok = sum(1 for cid in ids if results[cid]["snapshot"].get("status") != "ok")
    print(f"recorded {len(ids)} cell(s) in {seconds:.0f}s ({not_ok} recorded with status != ok; see manifest.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
