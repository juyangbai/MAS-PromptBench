# Team-Size

This MAS-PromptBench study measures how **team size `r`** — the number of agents in a multi-agent system — affects each `(topology, dataset)` pair. It mirrors the LangGraph variants in [`topologies/`](../topologies/README.md), swept over `r ∈ {2, 4, 8, 10}`.

Like the base pairs, the HotpotQA, LCB, BFCL, API-Bank and ToolHop team sizes are **optimizer targets** for the prompt optimizers in [`optimizers/`](../optimizers/README.md).

## Overview

`r` is swept over the **4 multi-agent topologies** — `single` is excluded (one agent has no team size).

| Topology | What `r` sizes |
|---|---|
| `sequential` | length of the pipeline (r-stage chain) |
| `centralized` | 1 manager + (r−1) workers |
| `decentralized` | r peer debaters × 2 rounds |
| `independent` | r parallel replicas |

### Directory layout

```
teamsizes/
├── output_contracts.py          # per-dataset final-answer contracts
├── apibank_common.py            # API-Bank team sizes: majority vote over r replicas
├── toolhop_common.py            # ToolHop team sizes: majority vote over r replicas
├── centralized/<ds>/<ds>_r{2,4,8,10}.py
├── decentralized/<ds>/<ds>_r{2,4,8,10}.py
├── sequential/<ds>/<ds>_r{2,4,8,10}.py
└── independent/
    ├── langgraph_base.py        # LangGraph fan-out / fan-in template
    └── <ds>/<ds>_r{2,4,8,10}.py
```

Path pattern: `teamsizes/<topology>/<dataset>/<dataset>_r{2,4,8,10}.py`. Each module is a few lines: it runs the `topologies/` LangGraph runner with the size-`r` team of [`configs/teams/<dataset>.yaml`](../configs/teams/README.md), or, for API-Bank and ToolHop, r seeded replicas of the answering role and a majority vote.


---

## Team sizes

`r=4` is the baseline — it mirrors the `topologies/` LangGraph design. `r=2` trims to the essential roles, while `r=8` and `r=10` add specialist roles on top of `r=4`.

| Topology | r=2 | r=4 (baseline) | r=8 | r=10 |
|---|---|---|---|---|
| `sequential` | 2-stage pipeline | 4-stage | 8-stage | 10-stage |
| `centralized` | 1 manager + 1 worker | 1 manager + 3 workers | 1 manager + 7 workers | 1 manager + 9 workers |
| `decentralized` | 2 peers × 2 rounds | 4 peers × 2 | 8 peers × 2 | 10 peers × 2 |
| `independent` | 2 replicas | 4 replicas | 8 replicas | 10 replicas |

**Role expansion** (full per-(topology, dataset, r) list in [`configs/prompts/roles.yaml`](../configs/prompts/roles.yaml)):

- **r=2** — the most essential tool-using stage + the agent that produces the final answer (preserves required tools while halving the team).
- **r=8** — r=4 + 4 new specialist roles per dataset (e.g. `requirements_parser`, `algorithm_designer`).
- **r=10** — r=8 + 2 more specialist roles (e.g. `optimizer`, `regression_checker`).

---

## Usage

### Point at an endpoint

```bash
export VLLM_BASE_URL=http://localhost:8000/v1   # any OpenAI-compatible server
export MODEL_ID=Qwen/Qwen3.5-9B
```

### Run a baseline

Without `--batch`, GPQA, HotpotQA, MATH, LiveCodeBench and APPS pairs run a canned demo, and BFCL, SWE-bench, API-Bank and ToolHop pairs a small batch:

```bash
python -m teamsizes.centralized.math.math_r4                     # canned demo
python -m teamsizes.centralized.math.math_r4 --batch --limit 10  # real batch
```

`toolhop` requires `TOOLHOP_ALLOW_DATASET_EXEC=1`. Per-dataset setup and scoring: [`benchmarks/README.md`](../benchmarks/README.md); command line: [`topologies/README.md`](../topologies/README.md#cli-flags).

### Output

`scripts/run_teamsizes.sh` writes each cell under `results/teamsizes/r{N}/<topology>_<dataset>/` (BFCL: one sub-folder per AST category); a runner run directly uses its own default under `results/` (e.g. `results/math_centralized_r8/`):

```
├── predictions.jsonl             # one JSON line per instance
├── results.jsonl                 # BFCL, SWE-bench, API-Bank and ToolHop only
├── traces/                       # per-instance; BFCL, SWE-bench, API-Bank and ToolHop only
└── (SWE only) patches/<instance_id>.diff
```

---
