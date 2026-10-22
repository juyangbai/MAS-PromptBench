# Evaluation Protocol

This page fixes what a score means: which instances are reported, how optimization data is kept apart from them, how a seed and an optimized prompt set are compared, and which settings stay constant across methods and runs.
{ .lede }

<div class="facts" markdown>
<div><span>Protocol</span>mas-promptbench</div>
<div><span>Budget</span>600 usable rollouts</div>
<div><span>Selection</span>Strictly better on validation</div>
<div><span>Evaluation IDs</span>730</div>
</div>

## Evaluation IDs

Every dataset has a manifest at `benchmarks/<dataset>/<dataset>_eval_ids.json` with the fields `dataset`, `sample`, `n`, `source` and `ids`. The IDs are the instances behind reported scores, so every topology, team size, communication format and optimizer is scored on the same set.

| Dataset | IDs | Sample |
| --- | ---: | --- |
| `gpqa` | 100 | report 100 |
| `hotpotqa` | 100 | report 100 |
| `math` | 100 | report 100 (Precalculus, Level 5) |
| `lcb` | 50 | report 50 |
| `apps` | 50 | dataset IDs 0 to 49 |
| `swe` | 30 | a fixed 30-instance sample of SWE-bench Verified |
| `bfcl` | 100 | 40 simple, 20 multiple, 20 parallel, 20 parallel_multiple |
| `apibank` | 100 | 33 Level 1, 33 Level 2, 34 Level 3 |
| `toolhop` | 100 | dataset IDs 0 to 99 |
| **Total** | **730** | |

Restrict a baseline run to the manifest with `--only`:

```bash title="Run HotpotQA on its evaluation IDs"
IDS=$(python -c "import json; \
print(' '.join(json.load(open('benchmarks/hotpotqa/hotpotqa_eval_ids.json'))['ids']))")
python -m topologies.single.hotpotqa.langgraph_hotpotqa --batch --only $IDS \
  --out results/topologies_baseline/single_hotpotqa/predictions.jsonl
```

The sweep launchers pass the BFCL and SWE-bench IDs with `--only` and a `--limit` for the other datasets. The API-Bank runners load their manifest by default.

## Fixed splits

The optimizers never draw their own splits. Each dataset ships `benchmarks/<dataset>/<dataset>_splits.json` with fixed, ordered `train`, `validation` and `test` ID lists:

| Dataset | Train | Validation | Test | Protocol metric |
| --- | ---: | ---: | ---: | --- |
| `gpqa` | 48 | 50 | 100 | `accuracy` |
| `hotpotqa` | 150 | 50 | 100 | `exact_match` |
| `math` | 150 | 50 | 100 | `accuracy` |
| `lcb` | 150 | 50 | 50 | `pass_at_1` |
| `apps` | 150 | 50 | 50 | `pass_at_1` |
| `swe` | 146 | 50 | 30 | `resolved_rate` |
| `bfcl` | 150 | 50 | 100 | `accuracy` |
| `toolhop` | 150 | 50 | 100 | `accuracy` |
| `apibank` | 150 | 50 | 100 | `accuracy` |

`test` is the evaluation-ID set, in the same order. `train` and `validation` are drawn with `split_seed` 0 from the remaining pool and never overlap it, so optimization never sees a reported instance. A job's identity includes a hash of the splits, and test rows are released only after the selection is sealed.

## One job, three phases

A job is one method, one cell and one optimizer seed in {0, 1, 2}. A cell is (task, topology, framework, communication format, team size, task model).

1. **Optimize.** The method gets the train and validation rows, the frozen seed bundle and a budget of 600 usable full-system rollouts, and returns its incumbent bundle.
2. **Final validation.** Uncharged and greedy. The seed and the incumbent run on the full validation split with paired per-item request seeds. The incumbent is deployed only if its validation mean is strictly higher; a tie, a regression, an unusable record or an infrastructure-invalid optimization keeps the seed bundle. The decision is sealed in `selection.json` before any test row is loaded.
3. **Test.** Uncharged and greedy. The seed and deployed bundles run on the test split with the same paired seeds. The gain is `delta_pp = 100 × (deployed mean − seed mean)`.

[Read Run Outputs](outputs.md) lists the fallback reasons and every artifact.

### Budget and charging

A rollout is one complete multi-agent execution of one row, scored by the dataset metric.

- **Usable** rollouts end in `success` or `semantic_failure`, which includes wrong or malformed answers. Only these are charged.
- **Infrastructure failures** are an exception before a scored observation (connection error, timeout, 5xx, BadRequest), transport-error text, no model call, or a scorer exception. They are never charged, are retried twice with the same request seed, and then return unusable.
- Batches reserve budget before dispatch and are trimmed at the cap, so a job never overshoots 600.
- A `--budget` other than 600, an off-grid cell or another reflection model makes a job non-conformant; aggregation skips it by default.

## Decoding and seeds

| Setting | Optimization rollouts | Final validation and test | Reflection |
| --- | --- | --- | --- |
| Model | the task model | the task model | `Qwen/Qwen3.5-122B-A10B-FP8` |
| Temperature | 0.2 | 0.0 | 1.0 unless the method sets it |
| Top-p | 0.9 | 0.9 | 1.0 unless the method sets it |
| Max output tokens | 32,768 | 32,768 | 48,000 |
| Thinking | off | off | on |

The task model is `Qwen/Qwen3.5-9B` (`--model qwen`) or `meta-llama/Llama-3.1-8B-Instruct` (`--model llama`). The runners also send `repetition_penalty` 1.05.

Every request carries a seed. Optimization and reflection requests get a logical seed hashed from the optimizer seed, the cell, the phase and the request's place in the search (iteration, row, role, turn), offset by 0, 1000 or 2000 for optimizer seed 0, 1 or 2; an infrastructure retry reuses it. Evaluation items get a paired seed that depends on the runtime condition, the optimizer seed, the split and the row, but not on the bundle or the method. The seed and deployed bundles therefore see the same seed on each row, and a deployed seed bundle reproduces the baseline exactly.

## Scorers

Every rollout in every phase is scored 0 or 1 by `optimizers.bridge.datasets.<dataset>.metric`, and a split's score is the mean. LiveCodeBench, APPS and SWE-bench use cheaper checks than the topology runners report:

| Dataset | Protocol metric | Topology runners |
| --- | --- | --- |
| `gpqa` | Option letter equals the gold letter | Same |
| `hotpotqa` | Official exact match | Exact match and token F1 |
| `math` | Hendrycks `is_equiv` of the last `\boxed{}` | Same |
| `bfcl` | `bfcl_eval` AST checker | Same |
| `apibank` | The runners' API-Bank replay check | Same |
| `toolhop` | ToolHop's matcher on the final answer, else on the selected agent's last tool result | Same |
| `lcb` | Passes the first 3 private tests | All private tests |
| `apps` | Passes the first 3 tests | The first 20 tests |
| `swe` | Structural check only: a unified diff with a header and at least one changed line; not applied, no tests run | Every `FAIL_TO_PASS` and `PASS_TO_PASS` test passes in the instance's SWE-bench image |

The bridge's APPS loader keeps only `interview` problems; all APPS evaluation IDs are `interview`. Every metric also returns feedback text, which reflective methods read. Task details are on the [task pages](../tasks/index.md).

## Output contracts

`core/output_contracts.py` fixes the final answer format per dataset. The role that emits the answer gets a `PROTECTED FINAL OUTPUT CONTRACT` block before its prompt and a reminder after it. That role is the solver-type role in Single and Independent, the `debater` in Decentralized, the last stage in Sequential (for example `writer` for HotpotQA) and the manager plus the answer-producing workers in Centralized.

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

The contract is not an optimization target. The bridge adapters attach their own copy, `optimizers/bridge/output_contracts.py` (version 2, with fuller wording), at execution time and outside the text a method edits, so optimized prompts never contain it and cannot remove it. The runners use version 1.

## Telemetry

`core/telemetry.py` gives every runner record the same five counters: `prompt_tokens`, `completion_tokens`, `total_tokens`, `n_llm_calls` and `n_tool_calls`.

- **LangGraph**: sums usage over the AI messages; Independent sums across all replicas.
- **CrewAI**: reads the crew's `usage_metrics`; it does not track tool calls, so `n_tool_calls` stays 0.
- **AutoGen**: sums `models_usage` per message and counts tool-call request events.
- **OpenAI Agents SDK**: the debate engine in `agents_sdk_base.py` sums the usage of every SDK run.

Counts are totals for the row across every agent and round, so they grow with team size and debate rounds. Protocol records carry their own usage per component (task, judge, reflection) in the job artifacts.

## Statistics

`optimizers.protocol.aggregate` pairs the three optimizer seeds of each cell. Per seed it reports the seed and deployed test means and their difference; across seeds, the mean and standard deviation of the difference, a seed-stratified paired bootstrap 95% interval (10,000 replicates) and, for binary scores, an exact McNemar test per seed with Holm correction inside a family (dataset × model by default). A cell is complete only when all three seeds have valid test data. See [Aggregate the seeds](../optimizers/running.md#aggregate-the-seeds).
