# Read Run Outputs

Every optimizer job leaves a folder of JSON artifacts, aggregation leaves one summary, and every runner batch leaves JSONL records. This page says what each file holds, where the optimized prompts are, and how the keep-or-reject decision is recorded.
{ .lede }

## Job folder

`python -m optimizers.protocol.run --out <dir>` writes:

```text
runs/gepa/hotpotqa/centralized/qwen/0/
├── job.json                       # job identity, seed bundle, evaluation condition
├── optimization/
│   ├── attempts.jsonl             # every execution attempt, retries included
│   ├── records.jsonl              # every charged rollout, with the budget after it
│   ├── bundle-<sha256>.json       # each bundle the method ran
│   ├── optimizer_result.json      # the method's result and incumbent
│   ├── native/                    # the method's scratch folder
│   └── dspy_cache/                # the job's DSPy disk cache
├── optimization.json              # status, budget, usage, incumbent bundle
├── evaluations/<evaluation id>.json   # uncharged evaluations, content-addressed
├── selection.json                 # the sealed deployment decision
├── test.json                      # paired test scores
└── result.json                    # the job summary
```

Methods add their own files under `optimization/`, such as `learning_curve.jsonl` and checkpoints. An evaluation in progress is journaled as `evaluations/<evaluation id>.partial.jsonl`, and `.lock` files guard concurrent writers; `--evaluation-cache` moves `evaluations/` to a shared folder.

The top-level files and the evaluations are sealed: each carries a `sha256` of its own content, checked when the protocol reads it back, and a `schema` such as `mas-promptbench-job-result`. Paths inside are relative to the job folder, and no artifact records an absolute path or a host name.

## result.json

The job summary that aggregation reads:

| Field | Content |
| --- | --- |
| `status`, `protocol_id`, `protocol_conformant` | `completed`; `mas-promptbench`; whether the budget, the cell and the reflection model follow the protocol. |
| `cell`, `cell_id` | The full job identity (method, task, runtime, budget, optimizer seed, reflection model, protocol and split hashes) and its short hash. |
| `grid_cell`, `registry_key`, `source_tables` | The grid configuration, the bridge key of its runtime, and the experiment tables that include it. |
| `optimizer_seed`, `condition_id` | The seed, and the hash of the method-independent evaluation condition. |
| `optimization` | `status`, `failure_kind`, `stop_reason`, `budget`, `usage` and `elapsed_seconds` of the optimization phase. |
| `selection` | `selection_id`, `selected_candidate`, `fallback_reason`, `baseline_validation_score`, `incumbent_validation_score`, `validation_delta`. |
| `test` | `example_ids`, `baseline_scores`, `deployed_scores`, `baseline_mean`, `deployed_mean`, `delta_pp` and `valid_for_aggregation`. |

Scores are fractions in [0, 1]; `delta_pp` is already in percentage points, 100 × (deployed − baseline). It is `null`, and `valid_for_aggregation` false, when either test evaluation has an unusable record, one that still failed on infrastructure after its retries.

## selection.json

The deployment decision, sealed before any test row is loaded. It holds the `seed_bundle`, the `incumbent_bundle` and the `deployed_bundle`, the validation scores of both candidates, `validation_delta`, the IDs of both validation evaluations, `ordered_validation_ids`, `selected_candidate` and `fallback_reason`:

| `fallback_reason` | Meaning |
| --- | --- |
| `null` | The incumbent was strictly better on validation and is deployed. |
| `validation_tie` | Same validation mean; the seed bundle is kept. |
| `validation_regression` | Lower validation mean; the seed bundle is kept. |
| `infrastructure_invalid_optimization` | The optimization failed on exhausted infrastructure; the seed bundle is kept. |
| `invalid_incumbent_bundle` | The incumbent could not be executed by the cell's runtime. |
| `invalid_record:<id>:<status>`, `duplicate_example:<id>`, `score_out_of_range:<id>`, `unpaired_validation_ids`, `empty_validation` | The validation records could not be compared. |

Every bundle is stored as `roles` (role name to prompt text), `demos`, `metadata` and `bundle_sha256`. The deployed prompts are `deployed_bundle.roles`; MIPRO's demos are rendered into the role prompts when they run.

```bash title="Print the deployed prompts"
python -c "import json; s = json.load(open('runs/gepa/hotpotqa/centralized/qwen/0/selection.json')); \
[print(f'## {role}\n{text}\n') for role, text in s['deployed_bundle']['roles'].items()]"
```

## optimization.json and test.json

`optimization.json` is the envelope of the optimization phase:

- `status` is `completed` or `failed`. A failure has `error_type`, `error` and a `failure_kind`: `infrastructure_invalid` (reported with the seed bundle), or `contract_failure`, `program_failure` or `ambiguous_external` (the job stops).
- `budget` counts `maximum`, `charged`, `remaining`, `reserved`, `attempted`, `successful`, `semantic_failures`, `infrastructure_failures` and `retries`.
- `usage` totals model calls and tokens per component: `task`, `judge` and `reflection`.
- `stop_reason` says why the method stopped, for example `rollout_budget_spent`, `budget` or `max_full_evals`.
- `seed_bundle`, `incumbent_bundle`, `training_ids`, `validation_ids` and `optimizer_artifact` (the path of `optimizer_result.json`) complete it.

`test.json` holds the per-row test data behind `result.json`: `example_ids`, `baseline_scores`, `deployed_scores`, `baseline_status`, `deployed_status`, both means, `delta_pp`, `usage` and the IDs of the two test evaluations.

## Aggregate summary

`python -m optimizers.protocol.aggregate ... --out summary.json` writes `schema`, `protocol_id`, `family`, `bootstrap_replicates` and one entry per cell in `cells`:

| Field | Content |
| --- | --- |
| `cell`, `registry_key`, `source_tables` | Method, task, topology, framework, communication, team size and task model. |
| `status` | `complete`, or `incomplete` with `problems` (missing seeds, duplicate seeds, infrastructure-invalid test data). |
| `seeds_present`, `per_seed_job`, `fallbacks` | Seeds found; per seed the selection, fallback reason, validation scores, charged rollouts, stop reason and status; seeds that kept the seed bundle. |
| `per_seed` | Per seed `n`, `baseline`, `deployed` and `delta_pp`. |
| `mean_delta_pp`, `std_delta_pp` | Mean and standard deviation of the per-seed gains. |
| `macro_baseline`, `macro_deployed` | Seed and deployed test means averaged over seeds. |
| `confidence_interval_delta_pp` | Seed-stratified paired bootstrap 95% interval. |
| `per_seed_exact_mcnemar_p`, `per_seed_mcnemar_p_holm`, `holm_family` | Exact McNemar p-values for binary scores, Holm-adjusted within the family. |

```python title="Load complete cells from a summary"
import json

summary = json.load(open("runs/summary.json"))
for cell in summary["cells"]:
    c = cell["cell"]
    if cell["status"] != "complete":
        print(c["method"], c["task"], cell["registry_key"], "incomplete:", cell["problems"])
        continue
    low, high = cell["confidence_interval_delta_pp"]
    print(f"{c['method']:<11}{c['task']:<9}{cell['registry_key']:<44}"
          f"{100 * cell['macro_baseline']:6.1f}{100 * cell['macro_deployed']:6.1f}"
          f"{cell['mean_delta_pp']:+7.2f} [{low:+.2f}, {high:+.2f}]")
```

## Runner outputs

A runner batch writes one JSON record per instance as it completes, and empties its files when the batch starts.

| Datasets | Output files |
| --- | --- |
| `gpqa`, `hotpotqa`, `math`, `lcb`, `apps` | `predictions.jsonl`, or the `--out` file |
| `bfcl` | `predictions.jsonl`, `results.jsonl`, `traces/<id>.txt` |
| `swe` | `predictions.jsonl`, `results.jsonl`, `patches/<id>.diff`, `traces/<id>.txt`, `eval_logs/` |
| `apibank`, `toolhop` | `results.jsonl` (records), `predictions.jsonl` (predictions), `traces/` |
| communication pairs | `results/communications_baseline/<topology>_<dataset>_<format>/results.jsonl` (BFCL: one folder per `--category`) unless `--out` or `--out-dir` is given |

A record holds the instance `id`, the gold and predicted answers and the dataset's score fields (for example `em` and `f1` for HotpotQA), followed by the runner's fields: usually `latency_s`, the five [telemetry](protocol.md#telemetry) counters and `error`. Some runners write a default file under `results/` and others only print scores, so pass `--out` or `--out-dir`. `scripts/run_topologies.sh` writes to `results/topologies_baseline/<topology>[_<framework>]_<dataset>/`, and `run_teamsizes.sh` to `results/teamsizes/r<r>/<topology>_<dataset>/`.
