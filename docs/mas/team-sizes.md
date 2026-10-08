# Team Sizes

Each multi-agent topology also runs with 2, 4, 8 and 10 agents. This page shows what grows with the team in each topology, how to run every size and how team size changes the effect of prompt optimization.
{ .lede }

<div class="facts" markdown>
<div><span>Sizes</span>n = 2, 4, 8, 10</div>
<div><span>Topologies</span>4</div>
<div><span>Datasets</span>9</div>
<div><span>Runners</span>144</div>
</div>

## What n changes

Single is excluded: one agent has no team size. For the other four topologies, `n = 4` is the baseline and mirrors the LangGraph design of the runners in `topologies/`.

| Topology | What n sets | n = 2 | n = 4 | n = 8 | n = 10 |
| --- | --- | --- | --- | --- | --- |
| [Independent](independent.md) | parallel replicas | 2 | 4 | 8 | 10 |
| [Sequential](sequential.md) | pipeline stages | 2 | 4 | 8 | 10 |
| [Centralized](centralized.md) | 1 manager + (n − 1) workers | 1 + 1 | 1 + 3 | 1 + 7 | 1 + 9 |
| [Decentralized](decentralized.md) | debating peers, always 2 rounds | 2 × 2 | 4 × 2 | 8 × 2 | 10 × 2 |

Tools, aggregation rules and the Decentralized round count stay as in the baseline. What grows is the set of agents:

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
    Apart from ToolHop and API-Bank, the Independent and Decentralized runners read their size from `INDEPENDENT_N_AGENTS` or `DECENTRALIZED_N_AGENTS` and only fall back to the n in the file name. Unset both before running team-size cells.

!!! note "ToolHop and API-Bank"
    For these two datasets every team-size runner goes through a shared wrapper (`teamsizes/toolhop_common.py`, `teamsizes/apibank_common.py`). It runs n seeded copies of the topology's final role (`solver`, `verifier`, `manager` or `debater`) and majority-votes their answers, for all four topologies. The optimizer adapters for `<topology>_r<N>` on these datasets build the real topology with the first n roles instead.

## Files

Runners follow `teamsizes/<topology>/<dataset>/<dataset>_r<N>.py`, for every topology, all nine datasets and N in 2, 4, 8, 10. For example, `teamsizes/centralized/hotpotqa/hotpotqa_r8.py`.

## Run a cell

Run from the repository root. The runners' default output folders are inconsistent: some omit the team size, and several runners (all Independent cells for GPQA, HotpotQA, MATH, LiveCodeBench and APPS, plus Decentralized MATH and Sequential APPS) only print scores. Always pass an output path that names the size.

=== "GPQA, HotpotQA, MATH, LCB, APPS"

    ```bash
    python -m teamsizes.centralized.hotpotqa.hotpotqa_r8 --batch --limit 100 \
      --out results/teamsizes_r8/hotpotqa/centralized_r8/predictions.jsonl
    ```

=== "BFCL, SWE-bench"

    ```bash
    # Always a batch: no --batch flag
    python -m teamsizes.sequential.bfcl.bfcl_r4 --limit 100 \
      --out-dir results/teamsizes_r4/bfcl/sequential_r4
    ```

=== "ToolHop, API-Bank"

    ```bash
    export TOOLHOP_ALLOW_DATASET_EXEC=1
    python -m teamsizes.decentralized.toolhop.toolhop_r10 --limit 100 \
      --out-dir results/teamsizes_r10/toolhop/decentralized_r10
    ```

Without arguments, the GPQA, HotpotQA, MATH, LiveCodeBench and APPS runners play a built-in smoke demo.

## Run the sweep

`scripts/run_teamsizes.sh` loops over `RVALUES`, `TOPOLOGIES` and `DATASETS` (all values by default) and runs each cell as its own process with the launcher limits:

```bash title="Sweep two sizes of Centralized on HotpotQA"
RVALUES="2 10" TOPOLOGIES="centralized" DATASETS="hotpotqa" \
  bash scripts/run_teamsizes.sh
```

The script passes only `--batch --limit`. BFCL and SWE-bench cells fail on `--batch`, and the other cells write to each runner's default folder or, for the print-only runners above, nowhere. To keep results, run the cells you need with explicit paths as above.

## Optimize a size

The optimizer topology names are `<topology>_r<N>`, for example `centralized_r8` or `decentralized_r2`. They exist only for HotpotQA, LiveCodeBench, ToolHop and API-Bank. The pilot takes the agent count from the name; for Decentralized, pass `--n-rounds 2` to keep the baseline's two rounds (the pilot default is 1).

```bash title="GEPA on Centralized with 8 agents, HotpotQA"
cd optimizers/gepa
python -m real_runner_gepa.pilots.run_gepa_dataset --dataset hotpotqa --topology centralized_r8 \
  --train-size 25 --val-size 25 --max-full-evals 5 --out results/gepa/centralized_r8_hotpotqa
```
