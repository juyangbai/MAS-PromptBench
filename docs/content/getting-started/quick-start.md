# Quick Start

Score a multi-agent system with its seed prompts, let an optimizer improve them, and read the gain \( \Delta \). The example is the Centralized (LangGraph) team on HotpotQA, a cell every one of the eight optimizers covers.
{ .lede }

<div class="facts" markdown>
<div><span>Needs</span>The conda environment</div>
<div><span>Models</span>Task and reflection endpoints</div>
<div><span>Run from</span>Repository root</div>
<div><span>Optimizer</span>Any of the eight</div>
</div>

Finish [Installation](installation.md) and [Connect a Model](connect-a-model.md) first.

## 1. Export the model variables

```bash title="Runners and optimizers"
export VLLM_BASE_URL=http://localhost:8000/v1              # task model, for the runners
export MODEL_ID=Qwen/Qwen3.5-9B

export TASK_ENDPOINTS=$VLLM_BASE_URL                       # task model, for the optimizer
export REFLECTION_MODEL_BASE_URL=http://localhost:8200/v1  # reflection model
```

## 2. Run a baseline

Start with the smoke demo, then a real batch. Runners are started as modules from the repository root.

```bash title="Centralized · HotpotQA"
# smoke demo: one built-in question, no dataset download
python -m topologies.centralized.langgraph.hotpotqa.langgraph_hotpotqa

# the 100 evaluation questions, one record per question
python -m topologies.centralized.langgraph.hotpotqa.langgraph_hotpotqa --batch --limit 100 \
  --out-dir results/quickstart/centralized_hotpotqa
```

The batch writes `predictions.jsonl` to `--out-dir` and ends with the exact-match and F1 scores. The first 100 HotpotQA rows are exactly the evaluation IDs, so `--limit 100` scores the same set every topology reports. Each [task page](../tasks/index.md) lists its flags and how to select its evaluation IDs.

## 3. Optimize the prompts

All eight optimizers run through one command. Pick a method key and start a job:

```bash title="One optimizer job"
METHOD=gepa   # or mipro, mapro, maspo, hivemind, mamut_gepa, maspob, tavo
python -m optimizers.protocol.run --method $METHOD --dataset hotpotqa --topology centralized \
  --model qwen --seed 0 --out runs/$METHOD/hotpotqa/centralized/qwen/0
```

The job runs three phases:

1. **Optimize**: the method searches for better role prompts with 600 usable rollouts of the real runner on the fixed `train` and `validation` splits.
2. **Validate**: seed and best prompts run on the whole validation split. The best prompts are deployed only if they score strictly higher; otherwise the seeds stay.
3. **Test**: seed and deployed prompts run on the held-out `test` split, the same 100 questions as the baseline.

A full job takes many model calls, and progress goes to stderr. If a job stops during validation or test, rerun the same command to resume; an interrupted optimization phase needs a fresh `--out`. [Run an Optimizer](../optimizers/running.md) covers budgets, phases and the method settings.

## 4. Read the result

The job prints one JSON line when it ends:

| Field | Meaning |
| --- | --- |
| `baseline_mean` | Test score of the seed prompts, as a fraction |
| `deployed_mean` | Test score of the deployed prompts |
| `delta_pp` | \( \Delta \), in percentage points |
| `fallback_reason` | `null` when the optimized prompts were deployed; else why the seeds stayed, such as `validation_tie` or `validation_regression` |
| `valid_for_aggregation` | `false` if a test record was unusable (an infrastructure failure); the scores are then `null` |

The same values, with per-example scores, are in `result.json` under `--out`. Run seeds 1 and 2 the same way, then pair the three seeds per cell:

```bash title="Summarize seeds 0 to 2"
python -m optimizers.protocol.aggregate runs/ --out runs/summary.json
```

[Read Run Outputs](../evaluation/outputs.md) explains every file a job writes.

## 5. Try another cell

The same steps work for any cell. The runner module picks the baseline; the protocol flags pick the same cell for the optimizer.

=== "Sequential (CrewAI) · BFCL"

    ```bash
    python -m topologies.sequential.crewai.bfcl.crewai_bfcl --category simple --limit 20 \
      --out-dir results/quickstart/sequential_crewai_bfcl_simple

    python -m optimizers.protocol.run --method $METHOD --dataset bfcl \
      --topology sequential --framework crewai \
      --model qwen --seed 0 --out runs/$METHOD/bfcl/sequential_crewai/qwen/0
    ```

=== "Team of 8 · HotpotQA"

    ```bash
    python -m teamsizes.centralized.hotpotqa.hotpotqa_r8 --batch --limit 100 \
      --out-dir results/quickstart/centralized_hotpotqa_r8

    python -m optimizers.protocol.run --method $METHOD --dataset hotpotqa \
      --topology centralized --team-size 8 \
      --model qwen --seed 0 --out runs/$METHOD/hotpotqa/centralized_r8/qwen/0
    ```

=== "Structured messages · LiveCodeBench"

    ```bash
    python -m communications.sequential.lcb.lcb_structured_soft --batch --limit 50 \
      --out-dir results/quickstart/sequential_lcb_structured_soft

    python -m optimizers.protocol.run --method $METHOD --dataset lcb \
      --topology sequential --communication structured_soft \
      --model qwen --seed 0 --out runs/$METHOD/lcb/sequential_structured_soft/qwen/0
    ```

!!! note "The experiment grid"
    The protocol refuses a cell outside its experiment grid unless you pass `--allow-any-cell`, and such a job is reported as non-conformant. The three cells above are in the grid for GEPA, MIPRO, MAPRO and MASPO. HiveMind, MAMUT-GEPA, MASPOB and TAVO cover the four multi-agent LangGraph teams of HotpotQA, LiveCodeBench and BFCL, with freeform messages and four agents.

## Where to go next

<div class="cards" markdown>

- [Tasks](../tasks/index.md)
  Scorers, data and flags for each of the nine datasets.
- [Workflow Topologies](../mas/topologies.md)
  How the five topologies route work between agents.
- [Run an Optimizer](../optimizers/running.md)
  Budgets, phases and settings for the eight optimizers.
- [Command-Line Flags](../reference/cli.md)
  Every runner and run-protocol flag, with defaults.

</div>
