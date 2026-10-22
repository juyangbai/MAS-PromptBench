"""Atomic, root-confined JSON artifact storage for one job directory.

Artifacts never record absolute paths or host names; paths inside payloads
are relative to the job directory and error text is passed through
:func:`scrub_text`.
"""

from __future__ import annotations

import fcntl
import json
import os
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .errors import ArtifactError
from .schema import canonical_json


def scrub_text(text: str) -> str:
    """Replace machine-specific absolute path prefixes in error/traceback text."""
    from . import REPO_ROOT

    replacements = [
        (str(REPO_ROOT), "<repo>"),
        (sys.prefix, "<python>"),
        (sys.base_prefix, "<python>"),
        (os.path.expanduser("~"), "~"),
    ]
    for old, new in sorted(replacements, key=lambda item: -len(item[0])):
        if old and old not in {"/", "~"}:
            text = text.replace(old, new)
    return text


def atomic_write_json(path: Path, value: Any, *, indent: int | None = None) -> Path:
    """Write ``value`` as JSON via a temporary file and an atomic rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (
        json.dumps(value, sort_keys=True, ensure_ascii=False, indent=indent, allow_nan=False)
        if indent is not None
        else canonical_json(value)
    ) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path


def read_json(path: Path) -> Any:
    """Parse one JSON artifact (``ArtifactError`` when unreadable)."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"cannot read artifact {path.name}: {exc}") from exc


def append_jsonl(path: Path, value: Any) -> Path:
    """Append one JSON line and flush it to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return path


def read_jsonl(path: Path) -> list[Any]:
    """All rows of a JSONL artifact (empty when it does not exist)."""
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@contextmanager
def exclusive(path: Path) -> Iterator[None]:
    """Process-level exclusive lock on ``path`` (created if missing)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


class ArtifactStore:
    """Write JSON/JSONL artifacts below one root directory."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _inside(self, path: Path) -> Path:
        resolved = Path(path).resolve()
        if resolved != self.root and self.root not in resolved.parents:
            raise ArtifactError("artifact path escapes the store root")
        return resolved

    def named(self, *parts: str) -> Path:
        """Subdirectory ``root/<parts...>`` (created if missing)."""
        path = self._inside(self.root.joinpath(*parts))
        path.mkdir(parents=True, exist_ok=True)
        return path

    def relative(self, path: Path) -> str:
        """``path`` relative to the store root, as recorded in artifacts."""
        return self._inside(path).relative_to(self.root).as_posix()

    def write_json(self, directory: Path, name: str, value: Any) -> Path:
        """Atomically write ``directory/name`` (a simple file name inside the store)."""
        if "/" in name or name in {"", ".", ".."}:
            raise ValueError("artifact name must be a simple filename")
        return atomic_write_json(self._inside(Path(directory) / name), value)

    def append_jsonl(self, directory: Path, name: str, value: Any) -> Path:
        """Append one row to ``directory/name`` (a simple file name inside the store)."""
        if "/" in name or name in {"", ".", ".."}:
            raise ValueError("artifact name must be a simple filename")
        return append_jsonl(self._inside(Path(directory) / name), value)


__all__ = ["ArtifactStore", "append_jsonl", "atomic_write_json", "exclusive", "read_json", "read_jsonl", "scrub_text"]
