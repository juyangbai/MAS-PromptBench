"""Code-block extraction shared by the code-generation tasks."""
from __future__ import annotations

import ast
import re


_PY_FENCE_RE = re.compile(r"```(?:python|py)\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_BARE_FENCE_RE = re.compile(r"```\s*\n(.*?)```", re.DOTALL)


def extract_python_code(text: str) -> str | None:
    """Return the last non-empty fenced block that looks like Python code.

    LCB prompts ask for a final fenced Python block, but
    communications/structured_soft can add JSON or prose fences. Labeled
    Python blocks win, and empty or prose-only fences are skipped so a
    trailing empty fence can't hide a valid program earlier in the reply.
    """
    raw = text or ""
    for blocks in (_PY_FENCE_RE.findall(raw), _BARE_FENCE_RE.findall(raw)):
        for block in reversed(blocks):
            candidate = block.strip()
            if _is_python_code(candidate):
                return candidate
    return None


def _is_python_code(candidate: str) -> bool:
    if not candidate:
        return False
    stripped = candidate.lstrip()
    if stripped.startswith(("{", "[")):
        return False
    try:
        ast.parse(candidate)
    except SyntaxError:
        return False
    return True
