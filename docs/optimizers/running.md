# Run an Optimizer

This page takes one `(topology, dataset)` cell from endpoints to a finished `meta.json`, then scales up to sweeps. The commands are the same for GEPA and MIPRO apart from the module name and the budget flags.
{ .lede }

## Before you start

Both pilots run from their workspace folder, `optimizers/gepa/` or `optimizers/mipro/`. A relative `--out` path resolves from there, so `--out results/gepa/single_math` writes to `optimizers/gepa/results/gepa/single_math/`. The `results/` and `cache/` folders inside each workspace are gitignored.

!!! warning "DSPy version"
    `environment.yml` pins `dspy>=2.6,<3`, but no DSPy 2.x release ships `dspy.teleprompt.GEPA`, which the GEPA pilot imports. If that import fails, install a DSPy 3 release with the `optuna` extra that DSPy 3's MIPROv2 needs: `pip install -U "dspy[optuna]>=3"`.

## Export the endpoints

Each optimizer needs a task endpoint for the agents and a reflection or proposal endpoint for the optimizer model. The optimizers ignore `VLLM_BASE_URL`, and their built-in endpoint defaults are fixed ports and hostnames, so set both explicitly.

=== "GEPA"

    ```bash
    export GEPA_TASK_ENDPOINTS=http://localhost:8000/v1,http://localhost:8001/v1
    export GEPA_REFL_ENDPOINT=http://localhost:9000/v1
    export TASK_MODEL=Qwen/Qwen3.5-9B
    export REFL_MODEL=Qwen/Qwen3.5-122B-A10B-FP8
    export GEPA_EXCLUDE_REAL_EVAL_IDS=1   # already the default; keeps intent explicit
    ```

=== "MIPRO"

    ```bash
    export MIPRO_TASK_ENDPOINTS=http://localhost:8000/v1,http://localhost:8001/v1
    export MIPRO_REFL_ENDPOINT=http://localhost:9000/v1
    export MIPRO_TASK_MODEL=Qwen/Qwen3.5-9B
    export MIPRO_REFL_MODEL=Qwen/Qwen3.5-122B-A10B-FP8
    export MIPRO_EXCLUDE_REAL_EVAL_IDS=1
    ```

For a hosted API, set `OPENAI_API_KEY`; the same key is sent to both endpoints. If `MODEL_ID` is set in your shell, the agents request that model name instead of `TASK_MODEL`.

## Choose a topology name

A cell is a dataset plus a topology name from the optimizer registry:

| Topology name | Datasets | What runs |
| --- | --- | --- |
| `single`, `independent`, `sequential`, `centralized`, `decentralized` | all nine | The LangGraph topology runner |
| `sequential_crewai`, `centralized_autogen`, `decentralized_openai` | all nine | The CrewAI, AutoGen or OpenAI SDK runner |
| `<base>_r<N>` | `hotpotqa`, `lcb`, `toolhop`, `apibank` | [Team-size](../mas/team-sizes.md) variant with N agents |
| `<base>_communications_<format>` | `hotpotqa`, `lcb`, `toolhop`, `apibank` | [Communication-protocol](../mas/communication-protocols.md) variant |

Here `<base>` is `independent`, `sequential`, `centralized` or `decentralized`, `<N>` is 2, 4, 8 or 10, and `<format>` is `freeform`, `semi_structured` or `structured_soft` (Freeform, Semi-structured, Structured). There is no `single_r<N>`. Team-size and communication runners exist for more datasets, but the optimizers register these variants for the four datasets above only. Print the exact list for a dataset:

```bash title="List topology names"
cd optimizers/gepa
python -c "from real_runner_gepa.registry import topologies; print(topologies('lcb'))"
```

An unregistered name stops the pilot with `unknown topology ... for dataset ...`.

## Run the cell

```bash title="One GEPA cell"
cd optimizers/gepa
python -m real_runner_gepa.pilots.run_gepa_dataset \
  --dataset math --topology single --train-size 25 --val-size 25 \
  --max-full-evals 5 --out results/gepa/single_math
```

For MIPRO, run `python -m real_runner_mipro.pilots.run_mipro_dataset` from `optimizers/mipro` with `--num-candidates 3 --num-trials 3` in place of `--max-full-evals`. For Independent and Decentralized cells add `--n-agents 4 --n-rounds 2` to match the runner defaults. [GEPA](gepa.md#flags) and [MIPRO](mipro.md#flags) list every flag.

## Watch status.json

The pilot rewrites `status.json` at each phase and prints the same payload to stdout as a `{"status": ...}` line:

```bash
watch -n 30 cat results/gepa/single_math/status.json
```

The `phase` field moves through `started`, `loading_dataset`, `baseline_eval_started`, `baseline_eval_done`, `gepa_compile_started`, `gepa_compile_done` (MIPRO: `mipro_compile_started`, `mipro_compile_done`), `compiled_eval_started` and `complete`. A crash writes `failed` with `error_type`, `error` and `traceback`. The search itself runs between the two `compile` phases and takes most of the time.

## Read meta.json

```bash title="Headline numbers"
python -c "import json; m = json.load(open('results/gepa/single_math/meta.json')); \
print({k: m[k] for k in ('baseline_score', 'compiled_score', 'delta', \
'selected_prompt_source', 'selection_reason')})"
```

Scores are fractions of the validation rows. The optimized prompts are in `compiled/<role>.txt`. [Read the Results](../evaluation/results.md) explains every field.

## Run a sweep

Each launcher loops over `DATASETS` × `TOPOLOGIES`, runs one pilot process per cell in sequence, and writes to `$OUT_ROOT/<topology>_<dataset>`. The scripts `cd` into their workspace, so `OUT_ROOT` is relative to `optimizers/gepa/` or `optimizers/mipro/`. A failing cell, including an unregistered topology name, prints `skipped/failed` and the loop continues.

```bash title="Team-size sweep with GEPA"
DATASETS="hotpotqa lcb" \
TOPOLOGIES="centralized_r2 centralized_r4 centralized_r8 centralized_r10" \
  bash optimizers/gepa/run_gepa.sh
```

=== "run_gepa.sh"

    | Variable | Default | Passed as |
    | --- | --- | --- |
    | `GEPA_REFL_ENDPOINT` | `http://localhost:8000/v1` | exported |
    | `DATASETS` | `bfcl gpqa hotpotqa math apps lcb swe apibank toolhop` | `--dataset` |
    | `TOPOLOGIES` | the eight base names | `--topology` |
    | `TRAIN_SIZE` | `25` | `--train-size` |
    | `VAL_SIZE` | `25` | `--val-size` |
    | `MAX_FULL_EVALS` | `5` | `--max-full-evals` |
    | `REFLECTION_MINIBATCH_SIZE` | `3` | `--reflection-minibatch-size` |
    | `NUM_THREADS` | `4` | `--num-threads` |
    | `N_AGENTS` | `4` | `--n-agents` |
    | `N_ROUNDS` | `2` | `--n-rounds` |
    | `COMPONENT_SELECTOR` | `round_robin` | `--component-selector` |
    | `EARLY_STOP_PATIENCE` | `3` | `--early-stop-patience` |
    | `OUT_ROOT` | `results/gepa` | `--out` prefix |

    The script always adds `--skip-perfect-score`. It does not set `GEPA_TASK_ENDPOINTS`, so export it first.

=== "run_mipro.sh"

    | Variable | Default | Passed as |
    | --- | --- | --- |
    | `MIPRO_REFL_ENDPOINT` | `http://localhost:8000/v1` | exported |
    | `MIPRO_TASK_ENDPOINTS` | `http://localhost:8000/v1` | exported |
    | `DATASETS` | `bfcl gpqa hotpotqa math apps lcb swe apibank toolhop` | `--dataset` |
    | `TOPOLOGIES` | the eight base names | `--topology` |
    | `TRAIN_SIZE` | `25` | `--train-size` |
    | `VAL_SIZE` | `25` | `--val-size` |
    | `NUM_CANDIDATES` | `3` | `--num-candidates` |
    | `NUM_TRIALS` | `3` | `--num-trials` |
    | `NUM_THREADS` | `4` | `--num-threads` |
    | `OUT_ROOT` | `results/mipro` | `--out` prefix |

    Because the script exports both MIPRO endpoint variables, the fallback to the GEPA variables never applies in a sweep. It does not pass `--n-agents` or `--n-rounds`, so Independent and Decentralized cells run with 2 agents and 1 round.

## Use several task endpoints

`GEPA_TASK_ENDPOINTS` and `MIPRO_TASK_ENDPOINTS` take a comma-separated list. Every time an adapter builds an agent client it takes the next endpoint in the list, round robin and thread-safe, and DSPy's task model rotates the same way per call. Load spreads across replicas without sharding. All endpoints must serve the same model under the same name. `models/serve_qwen3_5_9b.sh` starts one replica per GPU on consecutive ports from 8000 (`VLLM_BASE_PORT`), so list each port. The reflection or proposal model is a single endpoint; `models/serve_qwen3_5_122b.sh` also defaults to port 8000, so move it with `VLLM_PORT` or run it on another host.

## Compute tips

- **Count runs, not calls.** Every metric call is one full multi-agent run. A GEPA cell at the sweep setting allows up to 5 × (25 + 25) = 250 runs plus 50 for the pilot's two validation passes. A MIPRO trial scores the whole validation split unless you pass `--minibatch`.
- **Threads help most on HotpotQA and LCB.** Their adapters load a private copy of the runner module per call, so `--num-threads` runs examples in parallel. The APPS, MATH, SWE, API-Bank and ToolHop adapters hold a lock on the runner module while an example runs, so extra threads there mostly wait.
- **Check before you search.** GEPA's `--baseline-only` scores the seed prompts on validation and exits, which confirms the endpoints, the wiring and the starting score.
