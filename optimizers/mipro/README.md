# MIPRO Prompt Optimization

DSPy **MIPROv2** (instruction + demo search) over the **real** topology runners. MIPRO selects both role instructions and few-shot demos and re-scores by running the actual multi-agent runner, so improvements transfer directly back to the benchmark.

Part of [optimizers/](../README.md). The seed prompts in `configs/prompts/` are **read-only input** — MIPRO never modifies `configs/`.

## Overview

MIPRO runs against any `(topology, dataset)` pair — all 5 topologies × 9 datasets (`apibank apps bfcl gpqa hotpotqa lcb math swe toolhop`). It searches over candidate instructions and few-shot demos, scoring each by running the real runner, and writes the compiled prompts (plus demo summaries under `compiled_demos/`) to `results/mipro/` (gitignored).

### Directory layout

```
mipro/
├── run_mipro.sh                    # sweep launcher (one process per pair)
├── real_runner_mipro/              # the bridge
│   ├── registry.py                     # (topology, dataset) → adapter
│   ├── protocol.py  programs.py  mipro_programs.py  lm.py  output_contracts.py
│   ├── adapters/                       # per-pair prompt adapters
│   ├── datasets/                       # per-dataset loaders + split / eval-id logic
│   └── pilots/run_mipro_dataset.py     # generic per-pair entrypoint
└── docs/ADDING_PAIR.md             # how to add a new pair
```

---

## How it works

The bridge (`real_runner_mipro/`) exposes each `(topology, dataset)` pair's per-role prompts as DSPy-mutable predictors; MIPRO renders its selected instructions and demos into those prompts and runs the real runner to score each candidate.

MIPRO writes the optimized prompts (and demo summaries under `compiled_demos/`) to its `--out` directory under `results/mipro/` (gitignored); `configs/` is never modified.

---

## Usage

### Run one optimization

```bash
cd optimizers/mipro
export MIPRO_REFL_ENDPOINT=http://localhost:8000/v1     # proposal model
export MIPRO_TASK_ENDPOINTS=http://localhost:8000/v1    # comma-separated task models
python -m real_runner_mipro.pilots.run_mipro_dataset \
  --dataset math --topology single \
  --train-size 25 --val-size 25 --num-candidates 3 --num-trials 3 \
  --out results/mipro/single_math
```

### Run a sweep

`run_mipro.sh` runs MIPRO on each `(dataset, topology)` pair as its own process:

```bash
DATASETS="math gpqa" TOPOLOGIES="single centralized" bash optimizers/mipro/run_mipro.sh
```

### Settings

Set via environment variables; results land in `results/mipro/<topology>_<dataset>/`.

| Variable | Purpose |
|---|---|
| `MIPRO_REFL_ENDPOINT` | proposal-model endpoint |
| `MIPRO_TASK_ENDPOINTS` | comma-separated task-model endpoints |
| `DATASETS` / `TOPOLOGIES` | sweep grid |
| `TRAIN_SIZE` / `VAL_SIZE` | split sizes |
| `NUM_CANDIDATES` | instruction / demo candidates per round |
| `NUM_TRIALS` | optimization trials |
| `NUM_THREADS` | eval parallelism |
| `OUT_ROOT` | results root |

Add `MIPRO_EXCLUDE_REAL_EVAL_IDS=1` to hold the reported eval IDs (`benchmarks/<dataset>/<dataset>_eval_ids.json`) out of the train/val pool.

---

## Adding a pair

See [`docs/ADDING_PAIR.md`](docs/ADDING_PAIR.md). In short:

1. Add a dataset loader under `real_runner_mipro/datasets/` (copy an existing one).
2. Add an adapter under `real_runner_mipro/adapters/` (copy the closest existing module).
3. Register the pair in `real_runner_mipro/registry.py`.
4. Smoke-test with a minimal run (`--train-size 1 --val-size 1 --num-candidates 1 --num-trials 1`) and check the artifacts under `--out`.
