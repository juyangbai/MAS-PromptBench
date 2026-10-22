"""Readable line rendering + unified diff of golden snapshots."""

from __future__ import annotations

import difflib
import json
from typing import Any


def _scalar(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def lines(value: Any, path: str = "") -> list[str]:
    """Flatten to ``path: value`` lines; multi-line strings become ``|`` blocks."""
    out: list[str] = []
    if isinstance(value, dict):
        if not value:
            out.append(f"{path or '.'}: {{}}")
        for key in sorted(value):
            out.extend(lines(value[key], f"{path}.{key}" if path else str(key)))
    elif isinstance(value, list):
        if not value:
            out.append(f"{path or '.'}: []")
        for index, item in enumerate(value):
            out.extend(lines(item, f"{path}[{index}]"))
    elif isinstance(value, str) and "\n" in value:
        out.append(f"{path}: |")
        out.extend(f"    | {line}" for line in value.split("\n"))
    else:
        out.append(f"{path or '.'}: {_scalar(value)}")
    return out


def diff(expected: Any, actual: Any, *, max_lines: int = 400, context: int = 3) -> str:
    text = list(difflib.unified_diff(lines(expected), lines(actual), "golden", "current", n=context, lineterm=""))
    if max_lines > 0 and len(text) > max_lines:
        hidden = len(text) - max_lines
        text = text[:max_lines] + [f"... ({hidden} more diff lines; rerun with GOLDEN_DIFF_LINES=0 for all)"]
    return "\n".join(text)


def summary(expected: dict, actual: dict) -> str:
    """One-line hints: which top-level fields differ, request counts."""
    keys = sorted(set(expected) | set(actual))
    changed = [k for k in keys if expected.get(k) != actual.get(k)]
    parts = [f"changed fields: {', '.join(changed) or '-'}"]
    if "requests" in expected or "requests" in actual:
        parts.append(
            f"requests golden={len(expected.get('requests') or [])} current={len(actual.get('requests') or [])}"
        )
    if expected.get("status") != actual.get("status"):
        parts.append(f"status golden={expected.get('status')} current={actual.get('status')}")
    return "; ".join(parts)
