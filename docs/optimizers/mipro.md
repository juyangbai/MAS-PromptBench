# MIPRO

MIPRO searches over two things at once: the instruction text of each role and a set of few-shot demonstrations for it. This page explains how DSPy's MIPROv2 runs over the real topology runners, lists every flag of the pilot with its default, and shows a worked command.
{ .lede }

<div class="facts" markdown>
<div><span>Entry point</span>run_mipro_dataset</div>
<div><span>Changes</span>Instructions + demos</div>
<div><span>Sweep split</span>25 train / 25 val</div>
<div><span>Sweep budget</span>3 candidates, 3 trials</div>
</div>

## How the search runs

The pilot [`run_mipro_dataset.py`](https://github.com/juyangbai/MAS-PromptBench/blob/main/optimizers/mipro/real_runner_mipro/pilots/run_mipro_dataset.py) loads the dataset, draws the train and validation splits, scores the seed prompts on validation, then calls `MIPROv2.compile(seed_program, trainset=train, valset=val)`. DSPy's MIPROv2 works in three steps, and every candidate it scores is a full run of the real multi-agent system.

1. **Bootstrap demos.** It runs the program on training rows and keeps traces whose metric score passes as candidate demo sets, up to `--max-bootstrapped-demos` per set. `--max-labeled-demos` adds raw training examples; its pilot default is 0.
2. **Propose instructions.** The proposal model writes `--num-candidates` instructions per role. Four toggles decide what it sees: the program (`--program-aware-proposer`), a summary of the training data (`--data-aware-proposer`, built from batches of `--view-data-batch-size` rows), a random prompting tip (`--tip-aware-proposer`) and the bootstrapped demos (`--fewshot-aware-proposer`).
3. **Search combinations.** For `--num-trials` trials, an Optuna TPE sampler picks one instruction and one demo set per role, and the program is scored on the validation split. With `--minibatch`, trials use `--minibatch-size` rows and the best candidate gets a full validation pass every `--minibatch-full-eval-steps` trials. The best combination is returned.

MIPRO uses only the metric's numeric score. The feedback text that GEPA reflects on is ignored.

### How demos reach the runner

The topology runners read plain system prompts, not DSPy demo objects. The MIPRO bridge ([`mipro_programs.py`](https://github.com/juyangbai/MAS-PromptBench/blob/main/optimizers/mipro/real_runner_mipro/mipro_programs.py)) therefore renders each role's selected demos into its prompt, under a `### MIPROv2 Selected Demonstrations` heading, as one JSON block per demo. Long values are trimmed: strings to at most 1,400 characters (head and tail kept), dictionaries to 12 keys and lists to 6 items. The prompt the runner executes is the instruction plus these blocks.

### Data summary compaction

For LCB, the data summary the proposal model reads is built from compacted rows: the problem is cut to 2,500 characters, starter code to 1,000, and hidden tests are replaced by a count. Execution and scoring still use the full rows. `meta.json` records whether this happened in `dataset_summary_compacted`.

### Budget

Without `--auto`, `--num-candidates` and `--num-trials` both default to 3. `--auto light|medium|heavy` uses DSPy's preset budget instead; passing `--num-candidates` or `--num-trials` together with `--auto` stops the pilot with an error. There is no early stopping.

## Flags

All flags of `python -m real_runner_mipro.pilots.run_mipro_dataset`, with defaults from its argument parser:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--dataset` | required | One of `apibank apps bfcl gpqa hotpotqa lcb math swe toolhop` |
| `--topology` | required | A name from `registry.topologies(dataset)`; see [topology names](running.md#choose-a-topology-name) |
| `--train-size` | `1` | Training rows |
| `--val-size` | `1` | Validation rows |
| `--offset` | `0` | Rows dropped from the pool before the split |
| `--split-seed` | `0` | Seed for the split shuffle |
| `--n-agents` | `2` | Agents for `independent`, `decentralized`, `decentralized_openai` and the independent/decentralized communication variants |
| `--n-rounds` | `1` | Debate rounds for the decentralized variants |
| `--auto` | none | `light`, `medium` or `heavy` |
| `--num-candidates` | `3` without `--auto` | Instruction and demo-set candidates per role |
| `--num-trials` | `3` without `--auto` | Search trials |
| `--max-bootstrapped-demos` | `4` | Bootstrapped demos per set |
| `--max-labeled-demos` | `0` | Raw training examples per set |
| `--metric-threshold` | none | Minimum score for a bootstrapped trace to become a demo |
| `--minibatch` | off | Score trials on validation minibatches |
| `--minibatch-size` | `35` | Rows per minibatch |
| `--minibatch-full-eval-steps` | `5` | Trials between full validation passes |
| `--num-threads` | `1` | Parallel evaluation threads |
| `--seed` | `9` | Optimizer seed |
| `--init-temperature` | `1.0` | Proposal sampling temperature |
| `--view-data-batch-size` | `10` | Rows per batch in the data summary |
| `--max-errors` | none | Errors tolerated before aborting; none uses DSPy's setting |
| `--[no-]program-aware-proposer` | on | Show the program to the proposer |
| `--[no-]data-aware-proposer` | on | Show the data summary |
| `--[no-]tip-aware-proposer` | on | Add a prompting tip |
| `--[no-]fewshot-aware-proposer` | on | Show bootstrapped demos |
| `--provide-traceback` | off | Include tracebacks of failed runs in DSPy's logs |
| `--verbose` | off | Verbose DSPy output |
| `--out` | `optimizers/mipro/results/<topology>_<dataset>_real_mipro` | Output folder; a relative path resolves from your working directory |

The MIPRO pilot has no `--baseline-only` and no `--early-stop-patience`. Team-size names such as `sequential_r4` take their agent count from the name.

!!! warning "Match the runner shape"
    The topology runners default to 4 agents and 2 debate rounds, but this pilot defaults to 2 and 1, and `run_mipro.sh` does not override them. Pass `--n-agents 4 --n-rounds 2` for Independent and Decentralized cells.

## Environment variables

Each MIPRO variable falls back to its GEPA counterpart, so one set of exports can serve both optimizers.

| Variable | Fallback | Default | Effect |
| --- | --- | --- | --- |
| `MIPRO_TASK_ENDPOINTS` | `GEPA_TASK_ENDPOINTS` | `localhost` ports 15000 to 15007 | Comma-separated agent endpoints |
| `MIPRO_REFL_ENDPOINT` | `GEPA_REFL_ENDPOINT` | `http://localhost:15000/v1` | Proposal-model endpoint |
| `MIPRO_TASK_MODEL` | `TASK_MODEL` | `Qwen/Qwen3.5-9B` | Agent model name |
| `MIPRO_REFL_MODEL` | `REFL_MODEL` | `Qwen/Qwen3.5-122B-A10B-FP8` | Proposal model name |
| `MIPRO_EXCLUDE_REAL_EVAL_IDS` | `GEPA_EXCLUDE_REAL_EVAL_IDS` | on when unset | `0`, `false`, `no` or `off` puts eval IDs back in the pool |
| `MIPRO_DATA_SUMMARY_COMPACT_DATASETS` | `MIPRO_REFLECTION_COMPACT_DATASETS`, then `GEPA_REFLECTION_COMPACT_DATASETS` | `lcb` | Datasets with a compacted data summary; `*` for all |
| `MODEL_ID` | none | unset | When set, overrides the model name the agents request |
| `OPENAI_API_KEY` | none | `EMPTY` | Key for both endpoints |

`REAL_RUNNER_TASK_MAX_TOKENS`, `REAL_RUNNER_FAIL_ON_ADAPTER_ERROR`, `LCB_COMPILED_PROMPT_CHAR_LIMIT`, `LCB_COMPILED_PROMPT_HEAD_CHARS` and `BFCL_CATEGORY` behave as on the [GEPA page](gepa.md#environment-variables). The proposal model samples at temperature 1.0 with up to 48,000 output tokens.

## Outputs

The `--out` folder holds `status.json`, `baseline_val.jsonl`, `compiled_val.jsonl`, `optimized_val.jsonl`, `meta.json`, `compiled/<role>.txt` (instruction plus rendered demos), `compiled_raw/<role>.txt`, `compiled_demos/<role>.json` (`{"role": ..., "demos": [...]}`) and MIPRO's log folder `mipro_state/`. `meta.json` adds `compiled_demo_files` and `selected_demo_counts`. [Read the Results](../evaluation/results.md) covers every file.

## Worked command

```bash title="MIPRO on Decentralized MATH"
cd optimizers/mipro
export MIPRO_TASK_ENDPOINTS=http://localhost:8000/v1,http://localhost:8001/v1
export MIPRO_REFL_ENDPOINT=http://localhost:9000/v1
python -m real_runner_mipro.pilots.run_mipro_dataset \
  --dataset math --topology decentralized \
  --train-size 25 --val-size 25 --num-candidates 3 --num-trials 3 \
  --num-threads 4 --n-agents 4 --n-rounds 2 \
  --out results/mipro/decentralized_math
```

Results land in `optimizers/mipro/results/mipro/decentralized_math/`. For a wiring check, use `--train-size 1 --val-size 1 --num-candidates 1 --num-trials 1` and inspect the files under `--out`.
