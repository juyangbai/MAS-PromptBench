# Prompt Optimization

Eight prompt optimizers — **GEPA**, **MIPRO**, **MAPRO**, **MASPO**, **HiveMind**, **MAMUT-GEPA**, **MASPOB** and **TAVO** — improve the seed prompts in `configs/prompts/` by running the **real** topology runners and re-scoring. Each mutates a pair's per-role prompts and measures the gain on the actual multi-agent runner — not a mirrored copy of it — so an optimized prompt runs unchanged in the benchmark.

This part is **optional** — the base benchmark runs without it (see [topologies/](../topologies/)). Every `(topology, dataset)` pair is an optimizer target, including the [`teamsizes/`](../teamsizes/) and [`communications/`](../communications/) variants.

## Overview

| Method | Key | Approach | Code | Cells |
|---|---|---|---|---|
| **GEPA** | `gepa` | reflective prompt evolution (DSPy) | [`gepa/`](gepa/) | all (Tables 2–7) + Llama |
| **MIPRO** | `mipro` | MIPROv2 instruction + few-shot search (DSPy) | [`mipro/`](mipro/) | all (Tables 2–7) + Llama |
| **MAPRO** | `mapro` | per-role prompt pools, max-product belief propagation, blame-driven mutation | [`mapro/`](mapro/) | all (Tables 2–7) + Llama |
| **MASPO** | `maspo` | role-wise evolutionary beam search with pairwise judging | [`maspo/`](maspo/) | all (Tables 2–7) + Llama |
| **HiveMind** | `hivemind` | coalition (Shapley) credit, lesson-based refinement of the lowest-credit role | [`hivemind/`](hivemind/) | Table 6 |
| **MAMUT-GEPA** | `mamut_gepa` | one joint GEPA search over all role prompts | [`mamut_gepa/`](mamut_gepa/) | Table 6 |
| **MASPOB** | `maspob` | prompt-variant bandit with a GATv2 surrogate (LinUCB) | [`maspob/`](maspob/) | Table 6 |
| **TAVO** | `tavo` | trajectory credit assignment + shared verbalized-policy overlay | [`tavo/`](tavo/) | Table 6 |

All eight share one [run protocol](protocol/README.md) — 600 rollouts per job, deployment only if strictly better than the seeds on validation, scoring on a held-out test split — each with its method's upstream settings. MASPOB also needs `torch`, `torch_geometric` and `sentence-transformers` (CPU is enough).

### Directory layout

```
optimizers/
├── gepa/                       # GEPA
├── mipro/                      # MIPRO
├── mapro/                      # MAPRO
├── maspo/                      # MASPO
├── hivemind/                   # HiveMind
├── mamut_gepa/                 # MAMUT-GEPA
├── maspob/                     # MASPOB
├── tavo/                       # TAVO
├── protocol/                   # the shared run protocol
│   └── methods/                    # method registry, identity, shared DSPy plumbing
└── bridge/                     # the shared real-runner bridge
```

---

## How it works

Every rollout runs through the shared [real-runner bridge](bridge/README.md), a **small adapter layer** that wraps the topology runners instead of re-implementing them: an adapter exposes a pair's per-role prompts, the optimizer rewrites them, and every candidate is scored on the **actual** runner ([how it works](bridge/README.md#how-it-works)).

The protocol scores LiveCodeBench, APPS and SWE-bench with cheaper checks than the topology runners; see [Scoring](protocol/README.md#scoring).

---

## Usage

```bash
# from the repository root
export TASK_ENDPOINTS=http://localhost:8000/v1             # task model (or --task-endpoints)
export REFLECTION_MODEL_BASE_URL=http://localhost:8200/v1  # reflection model

python -m optimizers.protocol.run --method mapro --dataset hotpotqa --topology centralized \
    --model qwen --seed 0 --out runs/mapro/hotpotqa/centralized/qwen/0
python -m optimizers.protocol.aggregate runs/ --out runs/summary.json
```

`--method` is a key of the [Overview](#overview) table or `identity` (the seed prompts). Every option is in the protocol's [Usage](protocol/README.md#usage), every method knob in its [Settings](protocol/README.md#settings).

---

## Inputs and outputs

- **Input** — `configs/prompts/<topology>/<dataset>/<role>.txt`, the seed prompts. **Read-only**: no optimizer modifies `configs/`.
- **Splits** — fixed train / validation / test ids in `benchmarks/<dataset>/<dataset>_splits.json`; `test` is the reported evaluation set, so optimization never sees a reported instance.
- **Output** — job artifacts under `--out` (`runs/` is **gitignored**); the repository ships only the seeds.
