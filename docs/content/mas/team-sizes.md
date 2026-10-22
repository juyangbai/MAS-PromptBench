# Team Sizes

Each multi-agent topology also runs with 2, 4, 8 and 10 agents. This page shows what grows with the team in each topology, and how to run and optimize every size.
{ .lede }

<div class="facts" markdown>
<div><span>Sizes</span>n = 2, 4, 8, 10</div>
<div><span>Topologies</span>4</div>
<div><span>Datasets</span>9</div>
<div><span>Runners</span>144</div>
</div>

## What n changes

Single is excluded: one agent has no team size. For the other four topologies, `n = 4` is the baseline: the team of the LangGraph runners in `topologies/`.

| Topology | What n sets | n = 2 | n = 4 | n = 8 | n = 10 |
| --- | --- | --- | --- | --- | --- |
| [Independent](independent.md) | parallel replicas | 2 | 4 | 8 | 10 |
| [Sequential](sequential.md) | pipeline stages | 2 | 4 | 8 | 10 |
| [Centralized](centralized.md) | 1 manager + (n − 1) workers | 1 + 1 | 1 + 3 | 1 + 7 | 1 + 9 |
| [Decentralized](decentralized.md) | debating peers, always 2 rounds | 2 × 2 | 4 × 2 | 8 × 2 | 10 × 2 |

For seven datasets, every size of the team is declared in the team spec, [`configs/teams/<dataset>.yaml`](https://github.com/juyangbai/MAS-PromptBench/tree/main/configs/teams); ToolHop and API-Bank work differently (see below). Tools, aggregation rules and the Decentralized round count stay as in the baseline. What grows is the set of agents:

- **Independent and Decentralized** add copies of the one shared prompt (`solver`, `coder`, `caller` or `patcher`; `debater`).
- **Sequential and Centralized** add new roles. At `n = 2` the team keeps the essential tool-using role and the role that writes the answer. At `n = 8` and `n = 10` it adds specialist roles defined in [`configs/prompts/roles.yaml`](https://github.com/juyangbai/MAS-PromptBench/blob/main/configs/prompts/roles.yaml), whose seed prompts sit beside the baseline ones in `configs/prompts/<topology>/<dataset>/`.
- **Centralized** swaps in `manager_r8.txt` or `manager_r10.txt` at `n = 8` and `n = 10`; those prompts name all 7 or 9 workers. At `n = 2` the manager's turn cap is halved (18 to 9 on HotpotQA).

For example, the HotpotQA Sequential pipeline at each size:

| n | Stages, in order |
| --- | --- |
| 2 | `retriever`, `writer` |
| 4 | `planner`, `retriever`, `reasoner`, `writer` |
| 8 | `query_decomposer`, `planner`, `searcher`, `retriever`, `evidence_filter`, `reasoner`, `citation_compiler`, `writer` |
| 10 | the 8 above plus `entity_disambiguator` (after `searcher`) and `answer_simplifier` (last) |

!!! warning "Shell variables override n"
    Apart from ToolHop and API-Bank, the Independent and Decentralized runners read their size from `INDEPENDENT_N_AGENTS` or `DECENTRALIZED_N_AGENTS` first and only fall back to the n of the team spec. Unset both before running team-size cells.

!!! note "ToolHop and API-Bank"
    These two datasets have no team spec. For them every team-size runner goes through a shared wrapper (`teamsizes/toolhop_common.py`, `teamsizes/apibank_common.py`). It runs n seeded copies of the topology's answering role (`solver`, `verifier`, `manager` or `debater`) and majority-votes their answers, for all four topologies. The optimizer adapters for these team sizes run the real topology at size n instead.

## Files

Runners follow `teamsizes/<topology>/<dataset>/<dataset>_r<N>.py`, for every topology, all nine datasets and N in 2, 4, 8, 10; for example, `teamsizes/centralized/hotpotqa/hotpotqa_r8.py`. Each file is a few lines: it runs the topology's LangGraph runner from `topologies/` with the size-N team preset (`core/variant.py`), or the ToolHop and API-Bank wrapper.

## Run a cell

Run from the repository root. Team-size runners take the same command line as the topology runners. Their default output folders differ by runner, and some only print scores, so pass `--out-dir`:

=== "GPQA, HotpotQA, MATH, LCB, APPS"

    ```bash
    python -m teamsizes.centralized.hotpotqa.hotpotqa_r8 --batch --limit 100 \
      --out-dir results/teamsizes/r8/centralized_hotpotqa
    ```

=== "BFCL"

    ```bash
    # always a batch, one category at a time
    python -m teamsizes.sequential.bfcl.bfcl_r4 --category simple --limit 100 \
      --out-dir results/teamsizes/r4/sequential_bfcl/simple
    ```

=== "ToolHop, API-Bank"

    ```bash
    export TOOLHOP_ALLOW_DATASET_EXEC=1
    python -m teamsizes.decentralized.toolhop.toolhop_r10 --limit 100 \
      --out-dir results/teamsizes/r10/decentralized_toolhop
    ```

Without `--batch`, the GPQA, HotpotQA, MATH, LiveCodeBench and APPS runners play a built-in smoke demo. SWE-bench cells take the [SWE-bench options](../tasks/swe-bench.md#flags).

## Run the sweep

`scripts/run_teamsizes.sh` loops over `RVALUES`, `TOPOLOGIES` and `DATASETS` (all values by default) and runs each cell as its own process:

```bash title="Sweep two sizes of Centralized on HotpotQA"
RVALUES="2 10" TOPOLOGIES="centralized" DATASETS="hotpotqa" \
  bash scripts/run_teamsizes.sh
```

It writes each cell to `results/teamsizes/r<N>/<topology>_<dataset>/` (`OUT_ROOT` changes the root; BFCL gets one subfolder per category). BFCL and SWE-bench run their eval IDs; the other datasets use the same limits as the topology sweep.

## Optimize a size

Pass `--team-size N` to the run protocol, or use the key `<topology>_r<N>`. The bridge has team-size adapters for HotpotQA, LiveCodeBench, BFCL, ToolHop and API-Bank; the experiment grid covers HotpotQA, LiveCodeBench and BFCL with GEPA, MIPRO, MAPRO and MASPO, and other cells need `--allow-any-cell`. Decentralized jobs keep 2 rounds at every size. For example, MIPRO on a Centralized team of 8:

```bash title="MIPRO on Centralized, 8 agents · HotpotQA"
python -m optimizers.protocol.run --method mipro --dataset hotpotqa \
  --topology centralized --team-size 8 \
  --model qwen --seed 0 --out runs/mipro/hotpotqa/centralized_r8/qwen/0
```

See [Run an Optimizer](../optimizers/running.md).
