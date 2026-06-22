"""Robust code-block extraction helpers shared by code-generation tasks."""
from __future__ import annotations

import ast
import re


_PY_FENCE_RE = re.compile(r"```(?:python|py)\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_BARE_FENCE_RE = re.compile(r"```\s*\n(.*?)```", re.DOTALL)


def extract_python_code(text: str) -> str | None:
    """Return the last non-empty fenced block that is plausibly Python code.

    LCB prompts require a final fenced Python block, but communications/structured_soft can
    add extra JSON/prose fences. Prefer explicitly labeled Python blocks and
    ignore empty or prose-only blocks so a trailing empty fence does not erase a
    valid program emitted earlier in the response.
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
