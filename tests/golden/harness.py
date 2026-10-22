"""Parent-side orchestration: run cells in parallel worker processes.

Each cell gets its own process, temp dir (HOME, TMPDIR, outputs) and a fixed
environment, so results do not depend on the caller's shell, on other cells
or on the machine's network.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterable
from pathlib import Path

from tests.golden import cells as cellmod

REPO = cellmod.REPO
DEFAULT_TIMEOUT_S = int(os.environ.get("GOLDEN_CELL_TIMEOUT", "900"))

# Decoding / seed protocol pinned for every recording (README "Decoding").
FIXED_ENV = {
    "MODEL_ID": "Qwen/Qwen3.5-9B",
    "TASK_MODEL": "Qwen/Qwen3.5-9B",
    "REFL_MODEL": "Qwen/Qwen3.5-122B-A10B-FP8",
    "TASK_MODEL_TEMPERATURE": "0.0",
    "TASK_MODEL_TOP_P": "0.9",
    "TASK_MODEL_MAX_TOKENS": "32768",
    "REQUEST_SEED": "0",
    "TASK_TEMP_EVAL": "0.0",
    "TASK_TEMP_OPTIMIZE": "0.2",
    "OPENAI_API_KEY": "golden-key",
    "TOOLHOP_ALLOW_DATASET_EXEC": "1",
    "ALLOW_TOY_SWE_WORKDIR": "1",
    # hermetic / offline
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_DATASETS_DISABLE_PROGRESS_BARS": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "CREWAI_DISABLE_TELEMETRY": "true",
    "CREWAI_DISABLE_TRACKING": "true",
    "CREWAI_TRACING_ENABLED": "false",
    "OTEL_SDK_DISABLED": "true",
    "LITELLM_LOCAL_MODEL_COST_MAP": "True",
    "LITELLM_TELEMETRY": "False",
    "ANONYMIZED_TELEMETRY": "False",
    "DO_NOT_TRACK": "1",
    "GIT_ALLOW_PROTOCOL": "file",
    "GIT_TERMINAL_PROMPT": "0",
    "HTTP_PROXY": "http://127.0.0.1:9",
    "HTTPS_PROXY": "http://127.0.0.1:9",
    "ALL_PROXY": "http://127.0.0.1:9",
    "http_proxy": "http://127.0.0.1:9",
    "https_proxy": "http://127.0.0.1:9",
    "NO_PROXY": "127.0.0.1,localhost",
    "no_proxy": "127.0.0.1,localhost",
    "PYTHONHASHSEED": "0",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONUNBUFFERED": "1",
    "TZ": "UTC",
    "LC_ALL": "C.UTF-8",
    "LANG": "C.UTF-8",
}
_PASSTHROUGH = ("PATH", "TERM", "SHELL", "LD_LIBRARY_PATH", "CONDA_PREFIX", "SSL_CERT_FILE")


def real_home() -> str:
    return os.environ.get("GOLDEN_REAL_HOME") or str(Path("~").expanduser())


def hf_home() -> str:
    """The Hugging Face cache the cell workers read (offline)."""
    return os.environ.get("HF_HOME") or str(Path(real_home()) / ".cache" / "huggingface")


def cell_env(cell_tmp: Path, session_dir: Path) -> dict[str, str]:
    env = {k: os.environ[k] for k in _PASSTHROUGH if k in os.environ}
    env["PATH"] = os.pathsep.join([str(Path(sys.executable).parent), env.get("PATH", "")])
    env.update(FIXED_ENV)
    home = cell_tmp / "home"
    tmp = cell_tmp / "tmp"
    home.mkdir(parents=True, exist_ok=True)
    tmp.mkdir(parents=True, exist_ok=True)
    env.update(
        {
            "HOME": str(home),
            "TMPDIR": str(tmp),
            "TEMP": str(tmp),
            "TMP": str(tmp),
            "HF_HOME": hf_home(),
            "XDG_CACHE_HOME": str(home / ".cache"),
            "XDG_DATA_HOME": str(home / ".local" / "share"),
            "GOLDEN_REAL_HOME": real_home(),
            "GOLDEN_CELL_TMP": str(cell_tmp),
            "GOLDEN_SESSION_DIR": str(session_dir),
            "SWE_WORK_ROOT": str(cell_tmp / "swe_work"),
            "PYTHONPATH": str(REPO),
        }
    )
    if os.environ.get("HF_DATASETS_CACHE"):
        env["HF_DATASETS_CACHE"] = os.environ["HF_DATASETS_CACHE"]
    return env


def default_workers() -> int:
    if os.environ.get("GOLDEN_WORKERS"):
        return max(1, int(os.environ["GOLDEN_WORKERS"]))
    return max(2, min(32, (os.cpu_count() or 4) // 2))


def _exec_heavy(cell: dict) -> bool:
    """Cells whose scorer runs many timed subprocesses (APPS: 20 tests, 4 s
    timeout each) are throttled so machine load cannot turn a pass into a
    timeout."""
    return cell.get("dataset") == "apps"


EXEC_HEAVY_LIMIT = int(os.environ.get("GOLDEN_EXEC_WORKERS", "6"))
RERUN_MAX = int(os.environ.get("GOLDEN_RERUN_MAX", "25"))


def _with_dependencies(ids: Iterable[str], index: dict[str, dict]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()

    def visit(cid: str) -> None:
        if cid in seen:
            return
        seen.add(cid)
        for dep in index[cid].get("depends") or []:
            visit(dep)
        ordered.append(cid)

    for cid in ids:
        visit(cid)
    return ordered


_COST_HINT = {"dataset": 0, "method": 0, "scorer": 1, "cli": 1, "registry": 2, "runner": 2, "communication": 2}


def run_cells(
    ids: Iterable[str],
    *,
    workers: int | None = None,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    progress: Callable[[str, dict, int, int], None] | None = None,
    keep_tmp: bool = False,
    session_dir: Path | None = None,
    with_dependencies: bool = True,
) -> dict[str, dict]:
    """Run cells (plus dependencies) and return ``{id: {"snapshot", "meta"}}``.

    ``session_dir`` holds artifacts shared between cells (cached optimizer
    examples); pass the same directory to a later call with
    ``with_dependencies=False`` to re-run cells without their dependencies.
    """
    index = cellmod.cells_by_id()
    wanted = list(dict.fromkeys(ids))
    unknown = [cid for cid in wanted if cid not in index]
    if unknown:
        raise KeyError(f"unknown golden cells: {unknown[:5]}")
    order = _with_dependencies(wanted, index) if with_dependencies else wanted
    # Dependencies first, then the slow kinds, then stable id order.
    priority = {
        cid: (
            0 if any(cid in (index[o].get("depends") or []) for o in order) else 1,
            _COST_HINT.get(index[cid]["kind"], 3),
            cid,
        )
        for cid in order
    }
    pending = sorted(order, key=lambda cid: priority[cid])
    workers = workers or default_workers()
    root = Path(tempfile.mkdtemp(prefix="golden-"))
    session = Path(session_dir) if session_dir is not None else root / "session"
    session.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}
    running: dict[str, tuple[subprocess.Popen, float, Path]] = {}
    done: set[str] = set()
    total = len(order)
    try:
        while pending or running:
            launched = True
            while launched and pending and len(running) < workers:
                launched = False
                heavy = sum(1 for rid in running if _exec_heavy(index[rid]))
                for cid in pending:
                    if _exec_heavy(index[cid]) and heavy >= EXEC_HEAVY_LIMIT:
                        continue
                    deps = (index[cid].get("depends") or []) if with_dependencies else []
                    if all(dep in done for dep in deps):
                        pending.remove(cid)
                        cell_tmp = root / cid.replace("/", "__")
                        cell_tmp.mkdir(parents=True)
                        out = cell_tmp / "snapshot.json"
                        log = (cell_tmp / "worker.log").open("w")
                        proc = subprocess.Popen(
                            [sys.executable, "-m", "tests.golden.worker", "--cell", cid, "--out", str(out)],
                            cwd=str(cell_tmp),
                            env=cell_env(cell_tmp, session),
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL,
                            start_new_session=True,
                        )
                        log.close()
                        running[cid] = (proc, time.time(), cell_tmp)
                        launched = True
                        break
            time.sleep(0.05)
            for cid, (proc, started, cell_tmp) in list(running.items()):
                status = proc.poll()
                elapsed = time.time() - started
                if status is None and elapsed < timeout_s:
                    continue
                if status is None:
                    _kill(proc)
                result = _collect(cid, cell_tmp, status, elapsed, timed_out=status is None)
                results[cid] = result
                done.add(cid)
                del running[cid]
                if not keep_tmp:
                    shutil.rmtree(cell_tmp, ignore_errors=True)
                if progress:
                    progress(cid, result, len(done), total)
    finally:
        for proc, _, _ in running.values():
            _kill(proc)
        if not keep_tmp:
            shutil.rmtree(root, ignore_errors=True)
    return results


def _kill(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except Exception:
        proc.kill()
    proc.wait()


def _collect(cid: str, cell_tmp: Path, status, elapsed: float, timed_out: bool) -> dict:
    import json

    out = cell_tmp / "snapshot.json"
    if out.exists() and not timed_out:
        result = json.loads(out.read_text())
        result.setdefault("meta", {})["wall_seconds"] = round(elapsed, 2)
        return result
    log = cell_tmp / "worker.log"
    tail = log.read_text(errors="replace").splitlines()[-25:] if log.exists() else []
    state = "harness_timeout" if timed_out else f"harness_crash(exit={status})"
    return {
        "snapshot": {"id": cid, "status": state},
        "meta": {"wall_seconds": round(elapsed, 2), "log_tail": tail},
        "harness_failure": True,
    }


def settle(
    results: dict[str, dict],
    reference: dict[str, dict],
    *,
    session_dir: Path | None = None,
    attempts: int = 2,
    workers: int = 4,
) -> dict[str, list[str]]:
    """Re-run, at low parallelism, cells whose snapshot differs from ``reference``.

    A behavior change reproduces on every run; a difference caused by machine
    load (a timed subprocess in a scorer) does not. ``results`` is updated in
    place with the first re-run that matches; the returned mapping lists, per
    re-run cell, the first-pass diff and the outcome of each attempt
    ("match" / "differs" / "crash").
    """
    from tests.golden import render

    history: dict[str, list[str]] = {}
    pending = [
        cid
        for cid, res in results.items()
        if cid in reference and (res.get("harness_failure") or res["snapshot"] != reference[cid])
    ]
    for cid in pending:
        snap = results[cid]["snapshot"]
        first = "crash" if results[cid].get("harness_failure") else render.diff(reference[cid], snap, max_lines=12)
        history[cid] = [f"first pass: {first}"]
    if len(pending) > RERUN_MAX:
        # Many differing cells means a systematic change, not machine load.
        return {cid: history[cid] + ["not re-run"] for cid in pending}
    for _ in range(attempts):
        if not pending:
            break
        again = run_cells(
            pending, workers=min(workers, len(pending)), session_dir=session_dir, with_dependencies=session_dir is None
        )
        still = []
        for cid in pending:
            res = again[cid]
            if res.get("harness_failure"):
                history.setdefault(cid, []).append("crash")
                still.append(cid)
            elif res["snapshot"] == reference[cid]:
                history.setdefault(cid, []).append("match")
                results[cid] = res
            else:
                history.setdefault(cid, []).append("differs")
                results[cid] = res
                still.append(cid)
        pending = still
    return history
