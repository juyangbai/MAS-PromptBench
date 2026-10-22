"""Sandboxed shell tool for SWE-bench agents.

Model-issued shell commands run inside the instance's SWE-bench Singularity
image (the same per-instance SIF used for evaluation) with networking
disabled, a clean environment and no home directory. Only the agent's own
worktree is mounted, writable at /workspace, with its .git metadata mounted
read-only. Output size, wall-clock time, memory and open files are bounded.

Runners register each cloned worktree with the image of its instance via
`register_worktree`; `shell_exec` then resolves the image from the worktree
path. Set SWE_SHELL_SANDBOX=0 to run commands directly on the host instead
(unsafe; only for debugging without Singularity).
"""

from __future__ import annotations

import os
import resource
import signal
import subprocess
import threading
from collections.abc import Callable
from pathlib import Path

MAX_TOOL_OUTPUT_BYTES = 256 * 1024
MAX_SHELL_COMMAND_CHARS = 32 * 1024
MAX_SHELL_TIMEOUT_S = 120
MAX_MEMORY_BYTES = 8 * 1024 * 1024 * 1024
MAX_OPEN_FILES = 2048

ImageSource = Path | str | Callable[[], Path | str]

_IMAGES: dict[Path, ImageSource] = {}
_LOCK = threading.Lock()


def sandbox_enabled() -> bool:
    return os.environ.get("SWE_SHELL_SANDBOX", "1").strip().lower() not in {"0", "false", "no", "off"}


def register_worktree(workdir: Path | str, image: ImageSource) -> None:
    """Bind a worktree to its instance image (a path, or a zero-arg callable
    that returns the path and is resolved on the first shell call)."""
    with _LOCK:
        _IMAGES[Path(workdir).resolve()] = image


def unregister_worktree(workdir: Path | str) -> None:
    with _LOCK:
        _IMAGES.pop(Path(workdir).resolve(), None)


def _image_for(repo: Path) -> Path:
    with _LOCK:
        for candidate in (repo, *repo.parents):
            source = _IMAGES.get(candidate)
            if source is None:
                continue
            if callable(source):
                source = source()
                _IMAGES[candidate] = source
            return Path(source)
    raise RuntimeError(f"no SWE image registered for worktree {repo}")


def singularity_prefix(image: Path | str, *, writable_tmpfs: bool) -> list[str]:
    command = [
        "singularity",
        "exec",
        "--containall",
        "--cleanenv",
        "--no-home",
        # Keep the image's own timezone; host localtime changes date-based tests.
        "--no-mount",
        "/etc/localtime",
        "--net",
        "--network",
        "none",
    ]
    if writable_tmpfs:
        command.append("--writable-tmpfs")
    command.append(str(image))
    return command


def _limit_launcher() -> None:
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def _kill_group(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def host_shell_exec(workdir: Path | str, command: str, timeout_s: int = 60) -> str:
    """Unsandboxed fallback, used only when SWE_SHELL_SANDBOX=0."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=str(workdir),
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        return f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}\nexit_code: {result.returncode}"
    except subprocess.TimeoutExpired:
        return f"ERROR: command exceeded {timeout_s}s timeout"
    except Exception as e:
        return f"ERROR: {e}"


def sandboxed_shell_exec(workdir: Path | str, command: str, timeout_s: int = 60) -> str:
    """Run one shell tool call inside the worktree's networkless SWE image."""
    try:
        repo = Path(workdir).resolve()
        if not repo.is_dir() or not (repo / ".git").is_dir():
            raise RuntimeError(f"invalid SWE worktree: {repo}")
        image = _image_for(repo)
        if not isinstance(command, str) or len(command) > MAX_SHELL_COMMAND_CHARS:
            raise ValueError("shell command is not a bounded string")
        timeout = min(MAX_SHELL_TIMEOUT_S, max(1, int(timeout_s)))
        command_line = singularity_prefix(image, writable_tmpfs=True)
        image_index = command_line.index(str(image))
        command_line[image_index:image_index] = [
            "--bind",
            f"{repo}:/workspace:rw",
            "--bind",
            f"{repo / '.git'}:/workspace/.git:ro",
        ]
        inner = (
            "set -uo pipefail; "
            f"ulimit -v {MAX_MEMORY_BYTES // 1024}; "
            f"ulimit -n {MAX_OPEN_FILES}; "
            "ulimit -c 0; export HOME=/tmp/home TMPDIR=/tmp PYTHONHASHSEED=0; "
            'mkdir -p "$HOME"; cd /workspace; '
            'exec /bin/bash --noprofile --norc -c "$1"'
        )
        command_line.extend(["bash", "-c", inner, "swe-tool", command])
        process = subprocess.Popen(
            command_line,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={"PATH": os.environ.get("PATH", "")},
            cwd="/tmp",
            start_new_session=True,
            preexec_fn=_limit_launcher,
        )
        stdout = bytearray()
        stderr = bytearray()
        exceeded = threading.Event()
        lock = threading.Lock()

        def drain(stream, target: bytearray) -> None:
            for chunk in iter(lambda: stream.read(65536), b""):
                with lock:
                    remaining = MAX_TOOL_OUTPUT_BYTES - len(stdout) - len(stderr)
                    if remaining > 0:
                        target.extend(chunk[:remaining])
                    if len(chunk) > remaining:
                        exceeded.set()
                        _kill_group(process)

        assert process.stdout is not None and process.stderr is not None
        threads = [
            threading.Thread(target=drain, args=(process.stdout, stdout), daemon=True),
            threading.Thread(target=drain, args=(process.stderr, stderr), daemon=True),
        ]
        for thread in threads:
            thread.start()
        try:
            returncode = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_group(process)
            process.wait()
            return f"ERROR: command exceeded {timeout}s timeout"
        finally:
            for thread in threads:
                thread.join(timeout=5)
        if exceeded.is_set():
            return f"ERROR: command exceeded {MAX_TOOL_OUTPUT_BYTES} output bytes"
        out = bytes(stdout).decode("utf-8", errors="replace")
        err = bytes(stderr).decode("utf-8", errors="replace")
        return f"stdout:\n{out}\nstderr:\n{err}\nexit_code: {returncode}"
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"


def shell_exec(workdir: Path | str, command: str, timeout_s: int = 60) -> str:
    """The SWE `shell_exec` tool body shared by every SWE runner."""
    if sandbox_enabled():
        return sandboxed_shell_exec(workdir, command, timeout_s)
    return host_shell_exec(workdir, command, timeout_s)
