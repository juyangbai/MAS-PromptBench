# Quick Start

Score a system with its seed prompts, let GEPA optimize them, and read the gain \( \Delta \). The example is the single-agent MATH cell, then the same steps for a multi-agent cell.
{ .lede }

<div class="facts" markdown>
<div><span>Needs</span>The conda environment</div>
<div><span>Model</span>Any OpenAI-compatible endpoint</div>
<div><span>Run from</span>Repository root</div>
<div><span>Optimizer</span>GEPA</div>
</div>

Finish [Installation](installation.md) and [Connect a Model](connect-a-model.md) first.

## 1. Export the model variables

```bash title="Runners and optimizers"
export VLLM_BASE_URL=http://localhost:8000/v1   # or your provider's URL
export MODEL_ID=Qwen/Qwen3.5-9B
export OPENAI_API_KEY=EMPTY                      # your key, for a hosted API

export GEPA_TASK_ENDPOINTS=$VLLM_BASE_URL        # the optimizers' task model
export GEPA_REFL_ENDPOINT=$VLLM_BASE_URL         # GEPA's reflection model
export TASK_MODEL=$MODEL_ID REFL_MODEL=$MODEL_ID
```

## 2. Run a baseline

Start with the smoke demo, then a real batch. Runners are started as modules from the repository root.

```bash title="Single agent · MATH"
# smoke demo: one built-in problem, no dataset download
python -m topologies.single.math.langgraph_math

# batch on 100 problems, predictions saved as JSONL
python -m topologies.single.math.langgraph_math --batch --limit 100 \
  --out results/topologies_baseline/single_math/predictions.jsonl
```

!!! tip "Keep the predictions"
    For GPQA, HotpotQA, MATH, LiveCodeBench and APPS, a batch without `--out` only prints its scores. BFCL, SWE-bench, ToolHop and API-Bank take `--out-dir` instead. Each task page lists its exact flags.

## 3. Optimize the prompts

GEPA runs the same runner on a train split, rewrites the prompt from the traces, and keeps the new prompt only if it scores at least as well on the validation split.

```bash title="GEPA · Single · MATH"
cd optimizers/gepa
python -m real_runner_gepa.pilots.run_gepa_dataset \
  --dataset math --topology single \
  --train-size 25 --val-size 25 --max-full-evals 5 \
  --out results/gepa/single_math
```

Progress is written to `status.json` in the output folder while it runs. The frozen eval IDs are excluded from the splits by default.

## 4. Read the result

```bash title="Print the gain"
python - <<'EOF'
import json
m = json.load(open("results/gepa/single_math/meta.json"))
print(f"baseline {m['baseline_score']:.3f}  optimized {m['compiled_score']:.3f}  "
      f"delta {100 * m['delta']:+.1f} pp  kept: {m['selected_prompt_source']}")
EOF
```

Scores are fractions; multiply by 100 for percentage points. If the compiled prompt loses on validation, `selected_prompt_source` is `baseline` and the seed prompt is kept. The optimized prompts are in `compiled/`, one text file per role. [Read the Results](../evaluation/results.md) explains every file.

## 5. Try a multi-agent cell

The same four steps work for any cell. Here are two multi-agent cells: Sequential (CrewAI) on BFCL and Independent on MATH.

=== "Sequential · BFCL"

    ```bash
    # baseline: BFCL always runs a batch and takes --out-dir
    python -m topologies.sequential.crewai.bfcl.crewai_bfcl --limit 100 \
      --out-dir results/topologies_baseline/sequential_crewai_bfcl

    # optimize
    cd optimizers/gepa
    python -m real_runner_gepa.pilots.run_gepa_dataset \
      --dataset bfcl --topology sequential_crewai \
      --train-size 25 --val-size 25 --max-full-evals 5 \
      --out results/gepa/sequential_crewai_bfcl
    ```

=== "Independent · MATH"

    ```bash
    # baseline: four replicas, outputs aggregated
    python -m topologies.independent.math.langgraph_math --batch --limit 100 \
      --out results/topologies_baseline/independent_math/predictions.jsonl

    # optimize: match the runner's four agents
    cd optimizers/gepa
    python -m real_runner_gepa.pilots.run_gepa_dataset \
      --dataset math --topology independent --n-agents 4 \
      --train-size 25 --val-size 25 --max-full-evals 5 \
      --out results/gepa/independent_math
    ```

!!! warning "Match the team the runner uses"
    The optimizer pilots default to `--n-agents 2 --n-rounds 1`, which they apply to Independent and Decentralized cells. The runners use 4 agents, and Decentralized runs 2 rounds. Pass `--n-agents 4` (and `--n-rounds 2` for Decentralized) to optimize the same team you evaluate.

## Where to go next

<div class="cards" markdown>

- [Tasks](../tasks/index.md)
  Scorers, data and flags for each of the nine datasets.
- [Workflow Topologies](../mas/topologies.md)
  How the five topologies route work between agents.
- [Run an Optimizer](../optimizers/running.md)
  Sweeps, budgets and topology names for GEPA and MIPRO.
- [Command-Line Flags](../reference/cli.md)
  Every runner and optimizer flag, with defaults.

</div>
