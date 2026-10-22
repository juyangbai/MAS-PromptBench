"""Make captured values JSON-safe, stable and anonymous.

* absolute paths (repository root, Python prefix, temp/home dirs) become
  placeholders; random temp-dir names, UUIDs, memory addresses, ISO
  timestamps and the fake server's port are masked;
* timing fields (``*_s``, ``latency*``, ``elapsed*`` ...) are dropped;
* arbitrary objects (pydantic models, dataclasses, LangChain messages) are
  converted to plain data.
"""

from __future__ import annotations

import dataclasses
import getpass
import hashlib
import json
import math
import os
import re
import socket
import sys
from pathlib import Path, PurePath
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]

_VOLATILE_KEY_RE = re.compile(
    r"^(latency.*|elapsed.*|duration.*|.*_seconds|.*timestamp.*|created_at|updated_at|started_at|"
    r"finished_at|start_time|end_time|wall.*|t0|t1|time|timing.*)$",
    re.IGNORECASE,
)
_UUID_RE = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
_ADDR_RE = re.compile(r"0x[0-9a-fA-F]{6,16}")
_ISO_TS_RE = re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?")
_TMPNAME_RE = re.compile(r"(?<=/)tmp[a-z0-9_]{8}(?=/|\b)")
_PORT_RE = re.compile(r"(127\.0\.0\.1|localhost):\d{2,5}")
_PID_RE = re.compile(r"\bpid[=: ]\s*\d+", re.IGNORECASE)
# Timing values embedded in serialized text, e.g. 'solve_s=0.14' or '"latency_s": 0.3'.
_TIMING_IN_TEXT_RE = re.compile(
    r'(\b[A-Za-z_]*(?:_s|_secs?|_seconds|latency|elapsed|duration)\b"?\s*[:=]\s*)-?\d+(?:\.\d+)?'
)

_replacements: list[tuple[str, str]] = []
LONG_STRING_LIMIT = 4000


def _norm(path: str) -> str:
    return path.rstrip("/") if path not in ("", "/") else path


def configure(extra: dict[str, str] | None = None) -> None:
    """Install path placeholders (longest first)."""
    pairs: dict[str, str] = {}
    for path, label in (extra or {}).items():
        if path:
            pairs[_norm(str(path))] = label
            try:
                pairs[_norm(str(Path(path).resolve()))] = label
            except OSError:
                pass
    pairs[_norm(str(REPO_ROOT))] = "<repo>"
    pairs[_norm(os.path.realpath(REPO_ROOT))] = "<repo>"
    for prefix in {sys.prefix, sys.base_prefix, sys.exec_prefix}:
        pairs[_norm(prefix)] = "<python>"
    real_home = os.environ.get("GOLDEN_REAL_HOME") or str(Path("~").expanduser())
    pairs[_norm(real_home)] = "<home>"
    _replacements[:] = sorted(pairs.items(), key=lambda kv: -len(kv[0]))


def scrub_text(text: str) -> str:
    if not _replacements:
        configure()
    out = text
    for old, new in _replacements:
        if old and old in out:
            out = out.replace(old, new)
    out = _PORT_RE.sub("<fake-llm>", out)
    out = _UUID_RE.sub("<uuid>", out)
    out = _ADDR_RE.sub("<addr>", out)
    out = _ISO_TS_RE.sub("<timestamp>", out)
    out = _TMPNAME_RE.sub("<tmpname>", out)
    out = _PID_RE.sub("pid=<pid>", out)
    out = _TIMING_IN_TEXT_RE.sub(r"\1<t>", out)
    for ident in _identity_strings():
        if ident and ident in out:
            out = out.replace(ident, "<user>")
    return out


_IDENTITY: list[str] | None = None


def _identity_strings() -> list[str]:
    global _IDENTITY
    if _IDENTITY is None:
        values = set()
        try:
            values.add(getpass.getuser())
        except Exception:
            pass
        try:
            values.add(socket.gethostname().split(".")[0])
        except Exception:
            pass
        _IDENTITY = sorted((v for v in values if v and len(v) >= 4), key=len, reverse=True)
    return _IDENTITY


def digest(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()[:16]


def to_data(value: Any, *, depth: int = 0, long_limit: int | None = None) -> Any:
    """Convert to JSON-safe data, scrubbing strings and dropping timing keys."""
    if depth > 60:
        return "<max-depth>"
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return "<nan>"
        if math.isinf(value):
            return "<inf>" if value > 0 else "<-inf>"
        return round(value, 6)
    if isinstance(value, str):
        text = scrub_text(value)
        if long_limit is not None and len(text) > long_limit:
            return f"<long-text sha1={digest(text)} len={len(text)}>"
        return text
    if isinstance(value, bytes):
        return f"<bytes sha1={hashlib.sha1(value).hexdigest()[:16]} len={len(value)}>"
    if isinstance(value, PurePath):
        return scrub_text(str(value))
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            skey = scrub_text(str(key))
            if _VOLATILE_KEY_RE.match(skey) or (
                skey.endswith("_s") and isinstance(item, (int, float)) and not isinstance(item, bool)
            ):
                continue
            out[skey] = to_data(item, depth=depth + 1, long_limit=long_limit)
        return out
    if isinstance(value, (list, tuple)):
        return [to_data(item, depth=depth + 1, long_limit=long_limit) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [to_data(item, depth=depth + 1, long_limit=long_limit) for item in value]
        return sorted(items, key=lambda x: json.dumps(x, sort_keys=True, default=str))
    if isinstance(value, BaseException):
        return {"exception": type(value).__name__, "message": scrub_text(str(value))}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return to_data(
            {f.name: getattr(value, f.name) for f in dataclasses.fields(value)}, depth=depth + 1, long_limit=long_limit
        )
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            data = dump(mode="json")
        except Exception:
            try:
                data = dump()
            except Exception:
                data = None
        if data is not None:
            return to_data(
                {"__type__": type(value).__name__, **data} if isinstance(data, dict) else data,
                depth=depth + 1,
                long_limit=long_limit,
            )
    to_dict = getattr(value, "toDict", None)
    if callable(to_dict):
        try:
            return to_data(to_dict(), depth=depth + 1, long_limit=long_limit)
        except Exception:
            pass
    if hasattr(value, "__dict__") and not isinstance(value, type):
        data = {k: v for k, v in vars(value).items() if not k.startswith("_")}
        return to_data({"__type__": type(value).__name__, **data}, depth=depth + 1, long_limit=long_limit)
    return scrub_text(repr(value))


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False)
