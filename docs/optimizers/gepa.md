# GEPA

GEPA rewrites role prompts by reflection: it runs the multi-agent system on a few training rows, shows a reflection model what each role did and how the scorer judged it, and asks for a better instruction. This page covers how the loop runs in this code, every flag of the pilot, and the files it writes.
{ .lede }

<div class="facts" markdown>
<div><span>Entry point</span>run_gepa_dataset</div>
<div><span>Changes</span>Role instructions</div>
<div><span>Sweep split</span>25 train / 25 val</div>
<div><span>Sweep budget</span>5 full evals</div>
</div>

## How the loop runs

The pilot [`run_gepa_dataset.py`](https://github.com/juyangbai/MAS-PromptBench/blob/main/optimizers/gepa/real_runner_gepa/pilots/run_gepa_dataset.py) does four things in order:

1. Loads the dataset, draws the train and validation splits, and builds the program for the chosen topology.
2. Scores the seed prompts on the validation split and writes `baseline_val.jsonl`.
3. Calls DSPy's `GEPA.compile(seed_program, trainset=train, valset=val)`. Each step samples a minibatch of training rows, runs the full multi-agent system on them, sends the traces to the reflection model, and gets back a new instruction for the selected role or roles. A candidate that beats its parent on that minibatch is then scored on the whole validation split, and GEPA returns the candidate with the best validation score.
4. Re-scores the best candidate on the validation split, applies the [acceptance rule](index.md#trainval-split-and-acceptance), and writes `meta.json`.

### What the reflection model sees

The system runs once per example, and the bridge records a trace for every role from that one run. For each selected role and each minibatch row, the reflection model receives the role's current instruction plus:

- **Inputs**: the task instance. For LCB the instance is compacted first: the problem is cut to 7,000 characters (head and tail), starter code to 3,000, and the hidden tests are replaced by a count.
- **Generated outputs**: the final answer, tool calls, winner and vote summary, and a role-specific trace. In the shared module adapter that trace is the role name, the winner, the selected answer and up to 1,200 characters of that role's own output.
- **Feedback**: the text the dataset metric returns. For a wrong MATH answer it names the extracted answer, the gold answer, the problem and a trace excerpt; each dataset writes its own.

The reflection model is `REFL_MODEL` at `GEPA_REFL_ENDPOINT`, sampled at temperature 1.0 with up to 48,000 output tokens.

### Component selection

`--component-selector all` (the pilot default) asks for new instructions for every role at each step. `round_robin` (the sweep default) updates one role per step and cycles through the roles in order.

### Minibatches and budget

`--reflection-minibatch-size` sets how many training rows feed one reflection step. `--max-full-evals N` is converted by DSPy into a budget of `N × (train + val)` metric calls, and each metric call is one full multi-agent run. With 25 train rows, 25 validation rows and `N = 5`, that is up to 250 runs, plus 50 more for the pilot's own baseline and compiled evaluations. `--auto light|medium|heavy` uses DSPy's preset budget instead and overrides `--max-full-evals`.

### Early stopping and perfect scores

With `--early-stop-patience P` above 0, a stopper halts the search after `P` full validation evaluations in a row that fail to raise the best validation score. 0 disables it. `--skip-perfect-score` skips reflection on any minibatch where every row already scores 1.0.

## Flags

All flags of `python -m real_runner_gepa.pilots.run_gepa_dataset`, with defaults from its argument parser:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--dataset` | required | One of `apibank apps bfcl gpqa hotpotqa lcb math swe toolhop` |
| `--topology` | required | A name from `registry.topologies(dataset)`; see [topology names](running.md#choose-a-topology-name) |
| `--train-size` | `1` | Training rows |
| `--val-size` | `1` | Validation rows |
| `--offset` | `0` | Rows dropped from the pool before the split |
| `--split-seed` | `0` | Seed for the split shuffle |
| `--n-agents` | `2` | Agents for `independent`, `decentralized`, `decentralized_openai` and the independent/decentralized communication variants |
| `--n-rounds` | `1` | Debate rounds for the decentralized variants, including `decentralized_r<N>` |
| `--auto` | none | `light`, `medium` or `heavy`; replaces `--max-full-evals` |
| `--max-full-evals` | `1` | Budget in full train+val passes |
| `--reflection-minibatch-size` | `1` | Training rows per reflection step |
| `--num-threads` | `1` | Parallel evaluation threads |
| `--early-stop-patience` | `0` | Non-improving full evaluations before stopping; 0 is off |
| `--component-selector` | `all` | `all` or `round_robin` |
| `--skip-perfect-score` | off | Skip reflection on all-perfect minibatches |
| `--baseline-only` | off | Score the seed prompts on validation, write `meta.json`, stop |
| `--out` | `optimizers/gepa/results/<topology>_<dataset>_real_gepa` | Output folder; a relative path resolves from your working directory |

Team-size names such as `centralized_r8` take their agent count from the name, not from `--n-agents`.

!!! warning "Match the runner shape"
    The pilot defaults are smoke-test values. The topology runners default to 4 agents and 2 debate rounds, and so does `run_gepa.sh`. Pass `--n-agents 4 --n-rounds 2` on direct runs so the optimized prompts are tuned for the same team you evaluate.

## Environment variables

| Variable | Default | Effect |
| --- | --- | --- |
| `GEPA_TASK_ENDPOINTS` | built-in list of localhost ports | Comma-separated endpoints for the agents; always set it |
| `GEPA_REFL_ENDPOINT` | `http://localhost:8000/v1` | Reflection-model endpoint |
| `TASK_MODEL` | `Qwen/Qwen3.5-9B` | Agent model name |
| `REFL_MODEL` | `Qwen/Qwen3.5-122B-A10B-FP8` | Reflection model name |
| `MODEL_ID` | unset | When set, overrides the model name the agents request |
| `OPENAI_API_KEY` | `EMPTY` | Key for both endpoints |
| `GEPA_EXCLUDE_REAL_EVAL_IDS` | on when unset | `0`, `false`, `no` or `off` puts eval IDs back in the pool |
| `GEPA_REFLECTION_COMPACT_DATASETS` | `lcb` | Datasets whose traces are compacted for reflection; `*` for all |
| `REAL_RUNNER_TASK_MAX_TOKENS` | `4096` | Completion cap for agent clients built by the module adapters |
| `REAL_RUNNER_FAIL_ON_ADAPTER_ERROR` | `1` | `0` turns any adapter exception into a failed example instead of aborting |
| `LCB_COMPILED_PROMPT_CHAR_LIMIT` | `24000` | Length cap on compiled LCB prompts before the compiled validation run |
| `LCB_COMPILED_PROMPT_HEAD_CHARS` | `6000` | Characters kept from the start when that cap applies |
| `BFCL_CATEGORY` | `simple` | BFCL subset loaded for optimization |

Connection errors and timeouts inside a run always become a failed example with an `ERROR:` answer rather than stopping the pair. The optimizer does not read `VLLM_BASE_URL`; it points each runner at the next entry of `GEPA_TASK_ENDPOINTS`.

## Outputs

The `--out` folder holds `status.json`, `baseline_val.jsonl`, `compiled_val.jsonl`, `optimized_val.jsonl`, `meta.json`, `compiled/<role>.txt`, `compiled_raw/<role>.txt` and GEPA's log folder `gepa_state/`. The pilot also sets `DSPY_CACHEDIR` to `optimizers/gepa/cache/<topology>_<dataset>/`. [Read the Results](../evaluation/results.md) describes every file and field.

## Worked command

This runs one cell with the same settings `run_gepa.sh` uses:

```bash title="GEPA on Centralized HotpotQA"
cd optimizers/gepa
export GEPA_TASK_ENDPOINTS=http://localhost:8000/v1,http://localhost:8001/v1
export GEPA_REFL_ENDPOINT=http://localhost:9000/v1
python -m real_runner_gepa.pilots.run_gepa_dataset \
  --dataset hotpotqa --topology centralized \
  --train-size 25 --val-size 25 --max-full-evals 5 \
  --reflection-minibatch-size 3 --component-selector round_robin \
  --early-stop-patience 3 --skip-perfect-score \
  --num-threads 4 --n-agents 4 --n-rounds 2 \
  --out results/gepa/centralized_hotpotqa
```

Results land in `optimizers/gepa/results/gepa/centralized_hotpotqa/`. Add `--baseline-only` first if you want to check the wiring and the seed score before paying for the search. Check the DSPy version before the first run; see [Run an Optimizer](running.md#before-you-start).
