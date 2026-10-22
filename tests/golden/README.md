# Golden behavior snapshots

The **golden suite** replays 790 recorded cells (requests, decoding, outputs, scores, method artifacts) against a **local fake model server**.

## Overview

| Command | What it does |
|---|---|
| `python -m pytest -p no:cacheprovider tests/golden -q` | replays every cell and compares it with its snapshot in `data/` |
| `python -m tests.golden.record` | re-records (see its `--help` and [Command-Line Flags](../../docs/content/reference/cli.md#tests)) |

Full suite (unit tests + golden snapshots) and lint, from the repository root, with `pytest` and `ruff` from the `dev` extra (`pip install -e ".[dev]"`):

```bash
GOLDEN_WORKERS=16 HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 TOOLHOP_ALLOW_DATASET_EXEC=1 \
  python -m pytest -q -p no:cacheprovider \
  tests/golden tests/core core/tasks/tests optimizers          # unit tests + golden snapshots
ruff check . && ruff format --check .                          # lint + format
```

---

## Prerequisites

Everything runs **offline**: no network calls, and the unit tests use fake models too. The cells read these entries from the local Hugging Face cache (`HF_HOME`, or `HF_DATASETS_CACHE` for processed datasets):

| Entry | Needed by |
|---|---|
| datasets `codeparrot/apps`, `gorilla-llm/Berkeley-Function-Calling-Leaderboard`, `Idavidrein/gpqa`, `hotpot_qa`, `livecodebench/code_generation_lite`, `qwedsacf/competition_math`, `princeton-nlp/SWE-bench_Verified`, `bytedance-research/ToolHop` | every cell on that dataset except the `prompts/*` cells (API-Bank needs none) |
| tokenizer `Qwen/Qwen3.5-122B-A10B-FP8` | `methods/*` (reflection prompts are counted with it) |
| model `sentence-transformers/all-MiniLM-L6-v2` | `methods/maspob/*` |

Populate the cache once, while online, with the dataset loaders and `transformers` / `sentence-transformers`. A cell whose entries are missing is skipped with `golden: missing local HF cache entries: ...`, and `record` refuses to record it.
