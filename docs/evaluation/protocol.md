# Evaluation Protocol

This page fixes what a score means: which instances are reported, how optimization data is kept apart from them, how a before/after pair is measured, and which settings stay constant across runs.
{ .lede }

## Frozen eval-ID manifests

Every dataset has a manifest at `benchmarks/<dataset>/<dataset>_eval_ids.json` with the fields `dataset`, `sample`, `n`, `source` and `ids`. The IDs are the instances behind reported scores, so every topology, team size and communication format is scored on the same set.

| Dataset | Eval IDs | Sample |
| --- | ---: | --- |
| `gpqa` | 100 | report 100 |
| `hotpotqa` | 100 | report 100 |
| `math` | 100 | report 100 |
| `lcb` | 50 | report 50 |
| `apps` | 50 | ids 0..49 |
| `swe` | 30 | balanced_30: 15 `<15 min fix`, 15 `15 min - 1 hour` |
| `bfcl` | 25 | 10 simple, 5 multiple, 5 parallel, 5 parallel_multiple |
| `apibank` | 100 | 33 Level 1, 33 Level 2, 34 Level 3 |
| `toolhop` | 100 | dataset ids 0..99 |
| **Total** | **655** | |

Restrict a baseline run to the manifest with `--only`:

```bash title="Run HotpotQA on its eval IDs"
IDS=$(python -c "import json; \
print(' '.join(json.load(open('benchmarks/hotpotqa/hotpotqa_eval_ids.json'))['ids']))")
python -m topologies.single.hotpotqa.langgraph_hotpotqa --batch --only $IDS \
  --out results/topologies_baseline/single_hotpotqa/predictions.jsonl
```

The GPQA, HotpotQA, MATH, LCB and APPS runners take `--only` as one list. The BFCL, SWE-bench, ToolHop, API-Bank and communication runners take it once per ID (`--only a --only b`). The API-Bank runners also load their manifest by default.

## Train/val splits for optimization

The optimizers draw train and validation rows from each dataset's loader in `datasets/<dataset>.py`. The generic split in [`split_utils.py`](https://github.com/juyangbai/MAS-PromptBench/blob/main/optimizers/gepa/real_runner_gepa/datasets/split_utils.py) does this:

1. Drop every ID in the eval manifest, when exclusion is on.
2. Drop the first `--offset` rows.
3. Shuffle with `random.Random(--split-seed)`.
4. Take the first `--train-size` rows as train and the next `--val-size` rows as validation.

If the pool is too small the pilot stops with an error. Three datasets use their own rules. API-Bank balances train and validation across Levels 1 to 3. ToolHop picks rows whose profile matches the eval set. SWE-bench interleaves rows round robin by repository. API-Bank and ToolHop always exclude their eval IDs, whatever the variable says. For those two, GEPA also looks for a frozen `benchmarks/<dataset>/<dataset>_gepa25_signal_split.json` when you ask for 25/25 with seed 0 and offset 0; none ship, so the seeded split is used.

Exclusion is controlled by `GEPA_EXCLUDE_REAL_EVAL_IDS`, and by `MIPRO_EXCLUDE_REAL_EVAL_IDS` (which falls back to the GEPA variable) for MIPRO. In the code it is **on when the variable is unset**; only `0`, `false`, `no` or `off` turn it off. The optimizer READMEs tell you to add `=1` to enable it; in the code that only restates the default. Set it to `1` explicitly in your scripts, never to `0` for a reported run, and check `meta.json`: `exclude_real_eval_ids` should be `true` and `train_real_eval_overlap` and `val_real_eval_overlap` should be empty lists.

## The paired measurement

The pilot measures the seed prompts and the optimized prompts on the same validation rows, with the same model, endpoints, sampling, topology and scorer. Only the role prompts differ.

1. Score the seed prompts on validation and write `baseline_val.jsonl`.
2. Run the optimizer with `trainset=train, valset=val`.
3. Score the compiled prompts on the same validation rows and write `compiled_val.jsonl`.
4. Apply the acceptance rule and write the chosen side to `optimized_val.jsonl`.

`delta = compiled_score - baseline_score` is the mean of the per-row differences, a direct estimate of the gain \( \Delta \) defined on the [Optimizers](../optimizers/index.md#what-gets-optimized) page.

!!! warning "Validation is also the selection set"
    GEPA and MIPRO choose their best candidate by its validation score, and the pilot reports the delta on those same rows. With 25 rows, one row is 4 percentage points. The pilots stop at this validation measurement; a held-out number needs the selected prompts scored on the frozen eval IDs.

## Sampling settings

The optimizer workspaces define their sampling in `lm.py`:

| Setting | Agents | Reflection / proposal |
| --- | --- | --- |
| `temperature` | 0.2 | 1.0 |
| `top_p` | 0.9 | not set |
| `seed` | 0 | not set |
| `max_tokens` | 1024 in `task_sampling()`; 4096 for agent clients built by the module adapters (`REAL_RUNNER_TASK_MAX_TOKENS`) | 48000 |
| Extra body | `repetition_penalty` 1.05, thinking disabled through `chat_template_kwargs` | none |

The agent clients built by the adapters reuse temperature, top_p, seed and the extra body. The topology runners use the same values and set their own `max_tokens` (2048 in the Single HotpotQA runner, for example).

## Scorers

Each task family uses its official or community-standard scorer in the runners. During optimization the metric comes from the optimizer's `datasets/<dataset>.py`, which calls the runner's scoring code where it can and uses a cheaper check for code execution and patches.

| Family | Dataset | Runner scorer | Optimization metric |
| --- | --- | --- | --- |
| Reasoning | `gpqa` | Letter match | Same |
| | `hotpotqa` | Official EM and token F1 | EM (F1 appears in feedback) |
| | `math` | Hendrycks `is_equiv` on `\boxed{}` | Same, via the Single MATH runner |
| Coding | `lcb` | pass@1, stdin and functional tests | 1 if the first 3 tests all pass |
| | `apps` | pass@1, stdin and call-based tests | 1 if the first 3 tests all pass |
| | `swe` | `FAIL_TO_PASS` and `PASS_TO_PASS` tests | 1 if the output holds a non-trivial unified diff |
| Tool-calling | `bfcl` | Official `bfcl_eval` AST checker | Same checker |
| | `apibank` | Official API-Bank harness | The runner's `score_prediction` |
| | `toolhop` | Exact or normalized answer match | The runner's `score_answer` |

Every metric also returns feedback text, which GEPA reads and MIPRO ignores. Task details are on the [task pages](../tasks/index.md).

## Output contracts

[`topologies/output_contracts.py`](https://github.com/juyangbai/MAS-PromptBench/blob/main/topologies/output_contracts.py) fixes the final answer format per dataset. The role that emits the answer gets a `PROTECTED FINAL OUTPUT CONTRACT` block before its prompt and a reminder after it. That role is the solver-type role in Single and Independent, the `debater` in Decentralized, the last stage in Sequential (for example `writer` for HotpotQA) and the manager plus answer-producing workers in Centralized.

| Dataset | Required ending |
| --- | --- |
| `gpqa` | `Answer: A` to `Answer: D` |
| `hotpotqa` | `Answer: <short-form>` |
| `math` | A `\boxed{...}` line first and last |
| `apps`, `lcb` | One fenced `python` block |
| `swe` | One fenced `diff` block |
| `bfcl` | One fenced `json` list of function calls |
| `apibank` | One bracketed call, `[ApiName(arg='value')]` |
| `toolhop` | `<answer>...</answer>` |

The contract is not an optimization target. The adapters attach it at execution time, outside the text the optimizer edits, so compiled prompts never contain it and cannot remove it. The bridges carry their own copy with fuller wording (`real_runner_*/output_contracts.py`, version 2, recorded as `output_contract_version` in `meta.json`); the runners use version 1.

## Telemetry

[`topologies/telemetry.py`](https://github.com/juyangbai/MAS-PromptBench/blob/main/topologies/telemetry.py) gives every runner record the same five counters: `prompt_tokens`, `completion_tokens`, `total_tokens`, `n_llm_calls` and `n_tool_calls`. One extractor exists per framework:

- **LangGraph**: sums usage over the AI messages; Independent sums across all replicas.
- **CrewAI**: reads the crew's `usage_metrics`; it does not track tool calls, so `n_tool_calls` stays 0.
- **AutoGen**: sums `models_usage` per message and counts tool-call request events.
- **OpenAI SDK**: accumulates usage per call; the fallback path counts calls only and leaves tokens at 0.

Counts are totals for the row across every agent and every round, so they grow with team size and debate rounds. `normalize()` forces the five-key shape. The optimizer's `*_val.jsonl` records do not carry these counters, so compare cost with baseline runs.
