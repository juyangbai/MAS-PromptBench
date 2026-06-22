# Team-Size

This MAS-PromptBench study measures how **team size `r`** — the number of agents in a multi-agent system — affects each `(topology, dataset)` pair. It mirrors the LangGraph variants in [`topologies/`](../topologies/README.md), swept over `r ∈ {2, 4, 8, 10}`.

Like the base pairs, these are **optimizer targets** — GEPA and MIPRO mutate the per-role prompts in `configs/prompts/` and re-run a pair to measure improvement.

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
├── apibank_common.py            # apibank team-size wrapper (N replicas + majority vote)
├── toolhop_common.py            # toolhop team-size wrapper
├── centralized/<ds>/<ds>_r{2,4,8,10}.py
├── decentralized/<ds>/<ds>_r{2,4,8,10}.py
├── sequential/<ds>/<ds>_r{2,4,8,10}.py
└── independent/
    ├── langgraph_base.py        # shared fan-out / fan-in scaffold
    └── <ds>/<ds>_r{2,4,8,10}.py
```

Path pattern: `teamsizes/<topology>/<dataset>/<dataset>_r{2,4,8,10}.py`.


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

Every pair ships a no-arg smoke demo and a `--batch` mode:

```bash
python teamsizes/centralized/math/math_r4.py                     # smoke demo
python teamsizes/centralized/math/math_r4.py --batch --limit 10  # real batch
```

`toolhop` requires `TOOLHOP_ALLOW_DATASET_EXEC=1`. Per-dataset setup and scoring are documented once in [`topologies/README.md`](../topologies/README.md#3-datasets).

### Output

Each run writes under `results/teamsizes_r{N}/<dataset>/<topology>_r{N}/`:

```
├── predictions.jsonl    # per-row prediction payload
├── results.jsonl        # per-row metrics + telemetry
├── traces/<idx>.txt     # multi-stage agent transcripts
├── (BFCL only) <category>/
└── (SWE only) shard_<i>/patches/<iid>.diff
```

---

