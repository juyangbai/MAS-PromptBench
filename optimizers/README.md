# Prompt Optimization

Two prompt optimizers — **GEPA** and **MIPRO** — improve the seed prompts in `configs/prompts/` by running the **real** topology runners and re-scoring. Each mutates a pair's per-role prompts and measures the gain on the actual multi-agent runner — not a mirrored copy of it — so improvements transfer directly back to the benchmark.

These workspaces are **optional** — they are not required to run the base benchmark (see [topologies/](../topologies/)). Every `(topology, dataset)` pair from `topologies/` is an optimizer target.

## Overview

| Optimizer | Method | Workspace | Run interface |
|---|---|---|---|
| **GEPA** | reflective prompt evolution | [`gepa/`](gepa/) | [`gepa/README.md`](gepa/README.md) |
| **MIPRO** | MIPROv2 (instruction + few-shot example search) | [`mipro/`](mipro/) | [`mipro/README.md`](mipro/README.md) |

The two workspaces are **separate on purpose** — their dataset and LM internals have diverged, so merging them would change optimization behavior. Each ships a self-contained real-runner bridge, a sweep launcher, and its own README.

### Directory layout

```
optimizers/
├── gepa/                       # GEPA — reflective prompt evolution
│   ├── run_gepa.sh                 # sweep launcher
│   ├── real_runner_gepa/           # the bridge (adapters/ datasets/ registry.py …)
│   │   └── pilots/run_gepa_dataset.py   # generic per-pair entrypoint
│   ├── docs/  templates/  scripts/
│   └── README.md
└── mipro/                      # MIPROv2 — instruction + few-shot example search
    ├── run_mipro.sh                # sweep launcher
    ├── real_runner_mipro/          # the bridge
    │   └── pilots/run_mipro_dataset.py
    └── README.md
```

Path pattern: `optimizers/<optimizer>/real_runner_<optimizer>/` is the bridge; `run_<optimizer>.sh` is the sweep launcher. Runtime `cache/` and `results/` are gitignored.

---

## How it works

Each optimizer wraps the topology runners in a **small adapter layer** that plugs them into the optimizer's interface without re-implementing the agent framework:

1. An **adapter** owns one `(topology, dataset)` pair and exposes its per-role prompts as mutable predictors (`roles()` / `get_prompt()` / `set_prompt()`).
2. A **program** registers one predictor per mutable role, so the optimizer discovers and mutates the role instructions through `named_predictors()`.
3. `forward()` syncs the candidate prompts into the adapter, runs the **actual** runner once, and emits optimizer-readable traces per role.

Because every candidate is scored by running the real multi-agent runner, the resulting prompts transfer back to the benchmark unchanged.

---

## Usage

### Point at an endpoint

Both optimizers need a reflection/proposal model in addition to the task model:

```bash
export VLLM_BASE_URL=http://localhost:8000/v1        # task model (the runner)
export GEPA_REFL_ENDPOINT=http://localhost:8000/v1   # GEPA reflection model
export MIPRO_REFL_ENDPOINT=http://localhost:8000/v1  # MIPRO proposal model
```

### Run one optimization

```bash
# GEPA — from optimizers/gepa/
python -m real_runner_gepa.pilots.run_gepa_dataset \
  --dataset math --topology single --train-size 25 --val-size 25 \
  --max-full-evals 5 --out results/gepa/single_math

# MIPRO — from optimizers/mipro/
python -m real_runner_mipro.pilots.run_mipro_dataset \
  --dataset math --topology single --train-size 25 --val-size 25 \
  --num-candidates 3 --num-trials 3 --out results/mipro/single_math
```

### Run a sweep

Each launcher runs one process per `(dataset, topology)` pair:

```bash
DATASETS="math gpqa" TOPOLOGIES="single centralized" bash gepa/run_gepa.sh
DATASETS="math gpqa" TOPOLOGIES="single centralized" bash mipro/run_mipro.sh
```

See each workspace's README for the full set of environment settings.

---

## Inputs and outputs

- **Input** — `configs/prompts/<topology>/<dataset>/<role>.txt`, the seed prompts. **Read-only**: neither optimizer modifies `configs/`.
- **Output** — compiled prompts and scores under `results/{gepa,mipro}/<topology>_<dataset>/` (**gitignored**). Optimized prompts do **not** ship; the repo ships only the seeds.
- **Eval protection** — train/val splits draw from `benchmarks/<dataset>/<dataset>_eval_ids.json`, so optimization never trains on reported eval IDs (enable with `GEPA_EXCLUDE_REAL_EVAL_IDS=1` / `MIPRO_EXCLUDE_REAL_EVAL_IDS=1`).
