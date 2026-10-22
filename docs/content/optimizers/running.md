# Run an Optimizer

This page takes one cell from served endpoints to a finished `result.json`, then pairs the three optimizer seeds into a summary. The commands are the same for all eight methods; only `--method` changes.
{ .lede }

## Before you start

A job talks to two models: the task model, which runs every agent, and the reflection model, which the method uses to propose prompts. Serve both with the scripts in `models/` ([Connect a Model](../getting-started/connect-a-model.md)), then point the job at them from the repository root:

```bash title="Endpoints"
export TASK_ENDPOINTS=http://localhost:8000/v1,http://localhost:8001/v1   # task model replicas
export REFLECTION_MODEL_BASE_URL=http://localhost:8200/v1                 # reflection model
```

- `TASK_ENDPOINTS` takes a comma-separated list, used round robin; `--task-endpoints` overrides it, and `VLLM_BASE_URL` is the fallback. Every endpoint must serve the task model under its model ID. The job sets `MODEL_ID` from `--model` itself.
- `REFLECTION_MODEL_BASE_URL` defaults to `http://localhost:8200/v1`, the port of `models/serve_qwen3_5_122b.sh`. A reflection model other than `Qwen/Qwen3.5-122B-A10B-FP8` (`REFLECTION_MODEL_ID`) makes the job non-conformant.
- Cells on the OpenAI Agents SDK need its [isolated install](../reference/environment.md#openai-agents-sdk). The job restarts itself with the SDK first on `PYTHONPATH`, and exits with status 2 when the SDK is unusable.

## Choose a cell

A cell is a dataset plus a runtime: topology, framework, communication format, team size and task model. `--topology` takes a base topology, refined by the other flags, or a registry key that names the whole runtime:

| Runtime | Flags |
| --- | --- |
| Centralized, LangGraph, 4 agents | `--topology centralized` |
| Sequential on CrewAI | `--topology sequential_crewai`, or `--topology sequential --framework crewai` |
| Decentralized on the OpenAI Agents SDK | `--topology decentralized_openai_agents` |
| Independent with 8 agents | `--topology independent_r8`, or `--topology independent --team-size 8` |
| Centralized with structured reports | `--topology centralized_communications_structured_soft` |
| Llama as the task model | `--model llama` (`meta-llama/Llama-3.1-8B-Instruct`; `qwen` is `Qwen/Qwen3.5-9B`) |

Team-size and communication runtimes exist for HotpotQA, LiveCodeBench, BFCL, API-Bank and ToolHop; [Command-Line Flags](../reference/cli.md#run-protocol) lists every key. A cell outside a method's part of the experiment grid stops with an error unless you pass `--allow-any-cell`; [Optimizers](index.md#the-eight-methods) says which cells each method covers.

## Run a job

```bash title="One job"
python -m optimizers.protocol.run --method mapro --dataset hotpotqa --topology centralized \
  --model qwen --seed 0 --out runs/mapro/hotpotqa/centralized/qwen/0
```

The job runs its three phases in order: optimization on the 600-rollout budget, final validation of the seed and incumbent bundles, then the test split. Progress goes to stderr, including a line every 10 charged rollouts with the charged, attempted and infrastructure-failure counts. Stdout carries one JSON line with `status`, `cell_id`, `baseline_mean`, `deployed_mean`, `delta_pp`, `valid_for_aggregation` and `fallback_reason`. A job, evaluation or installation error exits with status 2.

The `--out` folder holds `job.json`, `optimization.json`, `selection.json`, `test.json` and `result.json`; [Read Run Outputs](../evaluation/outputs.md) explains each file.

### Phases and resuming

`--phase` runs one phase from the saved artifacts: `optimize`, `validate` (needs `optimization.json`) or `test` (needs `selection.json`). The default, `all`, runs whatever is missing, so rerunning a finished job prints its saved summary.

- `job.json` seals the job's identity. Reusing an `--out` folder for another job, or after the code or data changed, stops with an error.
- An interrupted optimization is never restarted silently: start again with a fresh `--out`.
- Evaluations are journaled per item and resume where they stopped.

## Smoke runs

A `--budget` below 600 makes a non-conformant smoke run, which aggregation skips by default. Methods charge rollouts before they propose anything, so a small budget can end a job before any search. The smallest budget that scores one proposed candidate, with 150 train and 50 validation rows:

| Method | Smallest budget | With less |
| --- | --- | --- |
| `identity` | 1 | not applicable |
| `gepa` | 56 (106 to validate a winner) | `rollout_budget_spent`, seed kept |
| `mipro` | bootstrap + 100: 105 to 400 | unrun rows fail; `rollout_budget_spent` |
| `mamut_gepa` | 56 (106) | `metric_call_cap`, seed kept |
| `mapro` | 2 × train: 300 | `budget` after the seed pass; an error below one pass |
| `hivemind` | 18 (centralized 10) with 1-row batches; 90 (50) at full size | no cycle, `budget`, seed kept |
| `maspo` | 20 (30 for both candidates) | unrun rows score 0; `budget` |
| `maspob` | 10 for one LinUCB pull (30 after 5 warm-ups) | warm-up only; `rollout_budget_spent` |
| `tavo` | 12 | `budget_before_outer_round`, seed kept; fails below 3 |

GPQA (48 train rows) and SWE-bench (146) shift the budgets that scale with the training split. The [run protocol README](https://github.com/juyangbai/MAS-PromptBench/blob/main/optimizers/protocol/README.md#smoke-budgets) shows what each method charges.

```bash title="Smoke-test one method on one cell"
python -m optimizers.protocol.run --method gepa --dataset math --topology single \
  --model qwen --seed 0 --budget 56 --phase optimize --out runs/smoke/gepa_math_single
```

`--phase optimize` stops before the uncharged evaluations, which always run on the full validation and test splits; check that `job.json`, `optimization.json` and `optimization/optimizer_result.json` exist.

## Run the three seeds

A reported cell needs optimizer seeds 0, 1 and 2. Run jobs in parallel as separate processes; rollouts inside one process are serialized.

```bash title="Three seeds"
for seed in 0 1 2; do
  python -m optimizers.protocol.run --method gepa --dataset hotpotqa --topology centralized \
    --model qwen --seed "$seed" --evaluation-cache runs/evaluations \
    --out "runs/gepa/hotpotqa/centralized/qwen/$seed" &
done
wait
```

Evaluations are content-addressed and do not depend on the method, so a shared `--evaluation-cache` lets jobs of different methods on the same cell and seed reuse one seed-bundle evaluation. Concurrent jobs lock each evaluation they write.

## Aggregate the seeds

```bash title="Paired summary"
python -m optimizers.protocol.aggregate runs/ --out runs/summary.json
```

`aggregate` finds every `result.json` under the given roots, groups the jobs by cell and prints a tab-separated table to stdout:

| Column | Content |
| --- | --- |
| `method`, `task`, `runtime`, `model` | The cell; `runtime` is its registry key. |
| `seeds` | `0,1,2` for a complete cell. |
| `base`, `deployed` | Test means of the seed and deployed bundles, averaged over seeds. |
| `delta_pp`, `std_pp` | Mean and standard deviation of the per-seed gain, in percentage points. |
| `ci95_pp` | Seed-stratified paired bootstrap 95% interval of the gain. |
| `fallbacks` | Seeds that kept the seed bundle, out of 3. |
| `min_p_holm` | Smallest Holm-adjusted per-seed exact McNemar p-value, for binary scores. |

One detail line per seed follows each cell. A cell with a missing seed or infrastructure-invalid test data prints as `incomplete` with the reason. Skipped files, such as non-conformant jobs, are logged as warnings.

- `--family` sets the Holm family: `task` (dataset × model, the default), `method` or `none`.
- `--bootstrap` sets the replicates (10,000 by default).
- `--include-nonconformant` adds smoke-budget and off-grid jobs.

`--out` writes the full JSON summary; see [Aggregate summary](../evaluation/outputs.md#aggregate-summary).

## Spread the load

Every client an adapter builds takes the next entry of `TASK_ENDPOINTS`, round robin and thread-safe, so a job spreads its rollouts across replicas without sharding. `models/serve_qwen3_5_9b.sh` starts one replica per GPU on consecutive ports from 8000 (`serve_llama3_1_8b.sh` from 8100), so list each port. The reflection model is one endpoint; `serve_qwen3_5_122b.sh` serves it on port 8200 with tensor parallelism over 4 GPUs.
