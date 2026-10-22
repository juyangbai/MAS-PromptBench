"""Golden storage: one gzip-compressed JSON file per area under ``data/``.

Long strings that repeat inside a cell (system prompts, tool schemas,
conversation prefixes) are interned into a per-cell string table on disk and
expanded again on load, so stored goldens stay small while every request is
kept in full.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parent / "data"
FORMAT_VERSION = 1
INTERN_MIN = 160
_REF = "\u0000ref:"


def area_of(cell_id: str) -> str:
    return cell_id.split("/", 1)[0]


def path_for(area: str) -> Path:
    return DATA_DIR / f"{area}.json.gz"


def _intern(value: Any, table: dict[str, int], strings: list[str]) -> Any:
    if isinstance(value, str):
        if len(value) < INTERN_MIN:
            return value
        if value not in table:
            table[value] = len(strings)
            strings.append(value)
        return f"{_REF}{table[value]}"
    if isinstance(value, list):
        return [_intern(v, table, strings) for v in value]
    if isinstance(value, dict):
        return {k: _intern(v, table, strings) for k, v in value.items()}
    return value


def _expand(value: Any, strings: list[str]) -> Any:
    if isinstance(value, str):
        if value.startswith(_REF):
            return strings[int(value[len(_REF) :])]
        return value
    if isinstance(value, list):
        return [_expand(v, strings) for v in value]
    if isinstance(value, dict):
        return {k: _expand(v, strings) for k, v in value.items()}
    return value


def pack(snapshot: dict) -> dict:
    strings: list[str] = []
    body = _intern(snapshot, {}, strings)
    return {"strings": strings, "body": body} if strings else {"body": body}


def unpack(packed: dict) -> dict:
    return _expand(packed["body"], packed.get("strings") or [])


def load_area(area: str) -> dict[str, dict]:
    path = path_for(area)
    if not path.exists():
        return {}
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        payload = json.load(fh)
    if payload.get("format") != FORMAT_VERSION:
        raise RuntimeError(f"{path.name}: unsupported golden format {payload.get('format')!r}")
    return {cid: unpack(packed) for cid, packed in payload["cells"].items()}


def load_all(areas) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for area in areas:
        out.update(load_area(area))
    return out


def write_area(area: str, snapshots: dict[str, dict]) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": FORMAT_VERSION,
        "area": area,
        "cells": {cid: pack(snapshots[cid]) for cid in sorted(snapshots)},
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    path = path_for(area)
    tmp = path.with_suffix(".tmp")
    # mtime=0 keeps the gzip header free of timestamps (byte-stable files).
    with (
        open(tmp, "wb") as raw_fh,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw_fh, mtime=0, compresslevel=9) as fh,
    ):
        fh.write(raw)
    tmp.replace(path)
    return path
