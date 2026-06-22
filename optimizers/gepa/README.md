# GEPA Prompt Optimization

**GEPA** applies reflective prompt evolution over the **real** topology runners: it mutates each pair's per-role prompts and re-scores by running the actual multi-agent runner — not a mirrored copy of it — so improvements transfer directly back to the benchmark.

Part of [optimizers/](../README.md). The seed prompts in `configs/prompts/` are **read-only input** — GEPA never modifies `configs/`.

## Overview

GEPA runs against any `(topology, dataset)` pair — all 5 topologies × 9 datasets (`apibank apps bfcl gpqa hotpotqa lcb math swe toolhop`). It proposes new role instructions from execution traces, keeps the candidate that scores higher on a held-out validation split, and writes the compiled prompts to `results/gepa/` (gitignored).

### Directory layout

```
gepa/
├── run_gepa.sh                     # sweep launcher (one process per pair)
├── real_runner_gepa/               # the bridge
│   ├── registry.py                     # (topology, dataset) → adapter
│   ├── protocol.py  programs.py  lm.py  output_contracts.py  early_stop.py
│   ├── adapters/                       # per-pair prompt adapters
│   ├── datasets/                       # per-dataset loaders + split / eval-id logic
│   └── pilots/run_gepa_dataset.py      # generic per-pair entrypoint
├── docs/ADDING_PAIR.md             # how to add a new pair
├── templates/                      # adapter + dataset skeletons
└── scripts/smoke_new_adapters.py   # adapter-wiring smoke test
```

---

## How it works

The bridge (`real_runner_gepa/`) follows DSPy's optimizer boundary without re-implementing the agent framework:

1. A `RealRunnerAdapter` owns one `(topology, dataset)` pair and exposes its mutable role prompts via `roles()` / `get_prompt()` / `set_prompt()`.
2. `AdapterBackedProgram` registers one DSPy predictor per mutable role, so GEPA discovers and mutates the role instructions through `named_predictors()`.
3. `forward()` syncs the candidate prompts into the adapter, runs the real runner once, and emits per-role execution traces in the format GEPA reads.

GEPA writes the optimized (compiled) prompts and scores to its `--out` directory under `results/gepa/` (gitignored); `configs/` is never modified.

---

## Usage

### Run one optimization

```bash
cd optimizers/gepa
export VLLM_BASE_URL=http://localhost:8000/v1        # task model (the runner)
export GEPA_REFL_ENDPOINT=http://localhost:8000/v1   # reflection model
python -m real_runner_gepa.pilots.run_gepa_dataset \
  --dataset math --topology single \
  --train-size 25 --val-size 25 --max-full-evals 5 \
  --out results/gepa/single_math
```

### Run a sweep

`run_gepa.sh` runs GEPA on each `(dataset, topology)` pair as its own process:

```bash
DATASETS="math gpqa" TOPOLOGIES="single centralized" bash optimizers/gepa/run_gepa.sh
```

### Settings

Set via environment variables; results land in `results/gepa/<topology>_<dataset>/`.

| Variable | Purpose |
|---|---|
| `GEPA_REFL_ENDPOINT` | reflection-model endpoint |
| `DATASETS` / `TOPOLOGIES` | sweep grid |
| `TRAIN_SIZE` / `VAL_SIZE` | split sizes |
| `MAX_FULL_EVALS` | optimization budget |
| `REFLECTION_MINIBATCH_SIZE` | rows per reflection step |
| `N_AGENTS` / `N_ROUNDS` | multi-agent shape (centralized / decentralized) |
| `COMPONENT_SELECTOR` | which role(s) to mutate |
| `EARLY_STOP_PATIENCE` | stop after N non-improving evals |
| `NUM_THREADS` | eval parallelism |
| `OUT_ROOT` | results root |

Add `GEPA_EXCLUDE_REAL_EVAL_IDS=1` to hold the reported eval IDs (`benchmarks/<dataset>/<dataset>_eval_ids.json`) out of the train/val pool.

---

## Adding a pair

See [`docs/ADDING_PAIR.md`](docs/ADDING_PAIR.md). In short:

1. Add a dataset loader under `real_runner_gepa/datasets/` (copy `templates/dataset_template.py`).
2. Add an adapter under `real_runner_gepa/adapters/` (copy `templates/adapter_template.py`).
3. Register the pair in `real_runner_gepa/registry.py`.
4. Smoke-test the wiring:

```bash
cd optimizers/gepa && python scripts/smoke_new_adapters.py
```
