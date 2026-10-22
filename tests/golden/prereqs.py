"""Local Hugging Face cache entries the golden cells need, checked offline.

Cells read their datasets from the local Hugging Face cache (offline mode),
the method cells count reflection prompts with the reflection model's
tokenizer, and MASPOB embeds prompts with MiniLM. A cell whose entries are
missing cannot reproduce its golden, so it is skipped with one message
instead of failing with a diff. Only the file system is inspected.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

from tests.golden import harness

# Cell dataset -> Hugging Face dataset repo (API-Bank is replayed from the repository).
DATASET_REPOS = {
    "apps": "codeparrot/apps",
    "bfcl": "gorilla-llm/Berkeley-Function-Calling-Leaderboard",
    "gpqa": "Idavidrein/gpqa",
    "hotpotqa": "hotpot_qa",
    "lcb": "livecodebench/code_generation_lite",
    "math": "qwedsacf/competition_math",
    "swe": "princeton-nlp/SWE-bench_Verified",
    "toolhop": "bytedance-research/ToolHop",
}
REFLECTION_TOKENIZER = "Qwen/Qwen3.5-122B-A10B-FP8"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Families whose cells never load dataset rows.
NO_DATASET_AREAS = {"prompts"}
README = "tests/golden/README.md"


def requirements(cell: dict) -> tuple[str, ...]:
    """Cache entries one cell needs, as ``kind:repo`` strings."""
    needs = []
    repo = DATASET_REPOS.get(cell.get("dataset") or "")
    if repo and cell["area"] not in NO_DATASET_AREAS:
        needs.append(f"dataset:{repo}")
    if cell["area"] == "methods":
        needs.append(f"tokenizer:{REFLECTION_TOKENIZER}")
        if cell["id"].split("/")[1] == "maspob":
            needs.append(f"model:{EMBEDDING_MODEL}")
    return tuple(needs)


def _snapshots(hub: Path, kind: str, repo: str) -> list[Path]:
    root = hub / f"{kind}s--{repo.replace('/', '--')}" / "snapshots"
    return sorted(root.iterdir()) if root.is_dir() else []


def present(entry: str) -> bool:
    """Whether one ``kind:repo`` entry is in the cache the cell workers use."""
    kind, repo = entry.split(":", 1)
    home = Path(harness.hf_home())
    hub = home / "hub"
    if kind == "dataset":
        if _snapshots(hub, "dataset", repo):
            return True
        processed = Path(os.environ.get("HF_DATASETS_CACHE") or home / "datasets")
        name = repo.replace("/", "___").lower()
        return processed.is_dir() and any(p.name.lower() == name for p in processed.iterdir())
    if kind == "tokenizer":
        return any((snap / "tokenizer.json").is_file() for snap in _snapshots(hub, "model", repo))
    return any((snap / "config.json").is_file() for snap in _snapshots(hub, "model", repo))


def missing(cells: Iterable[dict]) -> dict[str, tuple[str, ...]]:
    """``{cell id: missing entries}`` for every cell that cannot run here."""
    found: dict[str, bool] = {}
    result = {}
    for cell in cells:
        for entry in requirements(cell):
            if entry not in found:
                found[entry] = present(entry)
        absent = tuple(entry for entry in requirements(cell) if not found[entry])
        if absent:
            result[cell["id"]] = absent
    return result


def message(entries: Iterable[str]) -> str:
    """The one skip message."""
    return f"golden: missing local HF cache entries: {', '.join(sorted(set(entries)))} — see {README}"
