"""Parse BFCL function calls from model text into canonical `{name: args}` dicts.

Shared by every BFCL runner that reads calls from fenced JSON (sequential,
centralized, decentralized), so all topologies are scored on the same parse.
Runners that use native tool calls (single, independent) already produce the
canonical shape.
"""

from __future__ import annotations

import json
import re

_FENCED_RE = re.compile(r"```(?:\w*)\s*([\s\S]*?)\s*```")

# Common non-canonical shapes chat models emit for one call.
_NAME_ARGS_PAIRS = (
    ("fn_name", "args"),
    ("name", "arguments"),
    ("function_name", "arguments"),
    ("function", "arguments"),
)


def normalize_call(call: dict) -> dict:
    """Rewrite `{"name": f, "arguments": {...}}`-style dicts to `{f: {...}}`."""
    for name_key, args_key in _NAME_ARGS_PAIRS:
        if (
            name_key in call
            and args_key in call
            and isinstance(call[name_key], str)
            and isinstance(call[args_key], dict)
        ):
            return {call[name_key]: call[args_key]}
    return call


def extract_canonical(text: str) -> list[dict] | None:
    """Return the last fenced JSON list-of-dicts in `text`, normalized.

    Returns None if no fenced block parses to a non-empty list of dicts.
    A trailing TERMINATE marker is removed first so it cannot break a fence.
    """
    text = re.sub(r"\bTERMINATE\b", "", text or "")
    candidates = [m.group(1) for m in _FENCED_RE.finditer(text)]
    for candidate in reversed(candidates):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, list) and parsed and all(isinstance(x, dict) for x in parsed):
            return [normalize_call(call) for call in parsed]
    return None
