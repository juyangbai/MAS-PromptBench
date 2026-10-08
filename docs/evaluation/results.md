# Read the Results

Every optimizer run leaves a folder of JSON and text files, and every baseline run leaves JSONL records. This page says what each file holds, how the keep-or-reject decision is recorded, and how to turn scores into the percentage-point gains the paper reports.
{ .lede }

## Optimizer run folder

```text
optimizers/gepa/results/gepa/centralized_hotpotqa/
├── status.json            # current phase, rewritten at each step
├── baseline_val.jsonl     # seed prompts, one record per validation row
├── compiled_val.jsonl     # compiled prompts, same rows
├── optimized_val.jsonl    # copy of whichever side was kept
├── meta.json              # everything about the run, written last
├── compiled/<role>.txt    # optimized prompt per role, cleaned
├── compiled_raw/<role>.txt
└── gepa_state/            # GEPA's own log folder
```

A MIPRO folder has `mipro_state/` instead of `gepa_state/`, plus `compiled_demos/<role>.json`. Without `--out`, the folder is `optimizers/<optimizer>/results/<topology>_<dataset>_real_<optimizer>/`.

## status.json

The pilot overwrites `status.json` at each phase. It always holds `phase` and `updated_at`, plus the fields of that phase:

| `phase` | Extra fields |
| --- | --- |
| `started` | `dataset`, `topology`, `train_size`, `val_size`, `num_threads`; GEPA adds `max_full_evals`, MIPRO adds `auto`, `num_candidates`, `num_trials` |
| `loading_dataset` | none |
| `baseline_eval_started` | `train_ids`, `val_ids`, level, repo and profile counts, `split_manifest`, `protected_eval_sample`, `adapter_roles` |
| `baseline_eval_done` | `baseline_score` |
| `gepa_compile_started` / `mipro_compile_started` | none |
| `gepa_compile_done` / `mipro_compile_done` | `prompt_sanitizer` |
| `compiled_eval_started` | `prompt_sanitizer` |
| `complete` | `baseline_score`, `compiled_score`; MIPRO adds `accept_compiled_prompt`, `selected_prompt_source`; GEPA `--baseline-only` has `baseline_only` instead of `compiled_score` |
| `failed` | `error_type`, `error`, `traceback` |

## Validation records

Each line of `baseline_val.jsonl`, `compiled_val.jsonl` and `optimized_val.jsonl` is one validation row:

| Field | Content |
| --- | --- |
| `id` | Instance ID |
| `score` | Metric score for the row, 0 to 1 |
| `latency_s` | Wall time for the full multi-agent run |
| `answer` | Final answer text the metric saw |
| `winner`, `vote_summary` | Winning agent and vote buckets, where the topology votes |
| `agent_trace` | Role-level trace text |
| `communication_*` | Parse and report fields, on communication-protocol cells |

GEPA records can also carry `predicted_answer`, `runner_correct`, `runner_answer_correct`, `scoring_prev_tool_content` and `previous_tool_content` when the adapter returns them.

## compiled/ and compiled_raw/

`compiled_raw/<role>.txt` is each role's instruction exactly as it was evaluated in the compiled run. `compiled/<role>.txt` is the same text with any reasoning preamble removed: everything up to the last `</think>`, and the code fence around the prompt that follows it. `compiled_prompt_files` in `meta.json` lists the `compiled/` files. Neither file contains the output contract, which is attached at run time. For MIPRO both files include the rendered demonstrations, and `compiled_demos/<role>.json` stores the demos as `{"role": ..., "demos": [...]}`. For ToolHop and LCB, a sanitizer edits the prompts before the compiled run; `compiled_prompt_sanitizer` in `meta.json` lists what changed.

## meta.json

| Group | Fields |
| --- | --- |
| Identity | `cell`, `mode` (`real-runner-gepa`, `real-runner-gepa-baseline-only` or `real-runner-mipro-v2`), `dataset`, `topology` |
| Split | `train_size`, `val_size`, `offset`, `split_seed`, `actual_train_ids`, `actual_val_ids`, `actual_{train,val}_level_counts`, `actual_{train,val}_repo_counts`, `actual_{train,val}_profile_counts`, `split_manifest`, `protected_eval_sample` |
| Eval-ID protection | `exclude_real_eval_ids`, `excluded_real_eval_id_count`, `excluded_real_eval_ids`, `train_real_eval_overlap`, `val_real_eval_overlap` |
| Prompts | `adapter_roles`, `compiled_prompt_files`, `compiled_prompt_quality`, `compiled_prompt_sanitizer`; MIPRO adds `compiled_demo_files`, `selected_demo_counts` |
| GEPA settings | `n_agents`, `n_rounds`, `auto`, `max_full_evals`, `reflection_minibatch_size`, `num_threads`, `component_selector`, `skip_perfect_score`, `gepa_log_dir` |
| MIPRO settings | `n_agents`, `n_rounds`, `auto`, `num_candidates`, `num_trials`, `max_bootstrapped_demos`, `max_labeled_demos`, `metric_threshold`, `minibatch`, `minibatch_size`, `minibatch_full_eval_steps`, `num_threads`, `seed`, `init_temperature`, `view_data_batch_size`, `program_aware_proposer`, `data_aware_proposer`, `tip_aware_proposer`, `fewshot_aware_proposer`, `dataset_summary_compacted`, `mipro_log_dir` |
| Environment | `output_contracts`, `output_contract_version`, `dspy_cache_dir`, `dspy_version`, `task_model`, `reflection_model`, `task_endpoints`, `reflection_endpoint` |
| Scores | `baseline_score`, `compiled_score`, `delta`, `accept_compiled_prompt`, `selected_prompt_source`, `selection_reason`, `acceptance_diagnostics`, `baseline_failure_summary`, `compiled_failure_summary`, `baseline_records`, `compiled_records` |
| Team-size cells | `teamsizes_enabled`, `base_topology`, `team_size`, `teamsizes_module` |
| Communication cells | `communications_enabled`, `base_topology`, `communications_format`, `communications_module`, `communications_base_module` |

A failure summary holds `n`, `zero_score_count`, `zero_score_ids`, `infra_error_count`, `infra_error_ids` and `model_failure_count`. A GEPA `--baseline-only` run writes a shorter file with `compiled_score` and `delta` set to `null`. `n_agents` and `n_rounds` record the flags even on topologies that ignore them.

## Acceptance and `selection_reason`

`delta` is always `compiled_score - baseline_score`, whether or not the compiled prompts were kept. The decision is in `accept_compiled_prompt` and `selected_prompt_source` (`compiled` or `baseline`), and `acceptance_diagnostics.policy` names the rule applied.

| `selection_reason` | Kept |
| --- | --- |
| `compiled_score >= baseline_score` | compiled |
| `compiled_score < baseline_score; keep baseline prompt for final eval` | baseline |
| `compiled validation has infra errors; keep baseline prompt for final eval` | baseline |
| `toolhop compiled_score > baseline_score` | compiled |
| `toolhop compiled_score tied baseline with row-level improvements and no regressions` | compiled |
| `toolhop compiled_score < baseline_score; keep baseline prompt for final eval` | baseline |
| `toolhop compiled_score tied baseline but regressed validation rows; keep baseline prompt` | baseline |
| `toolhop compiled_score tied baseline with no row-level gain; keep baseline prompt` | baseline |
| `toolhop compiled prompt failed anti-memorization quality checks; keep baseline prompt` | baseline |
| `baseline-only run; no compiled prompt produced` | baseline |

An infrastructure error is any compiled record whose trace or answer contains a known failure marker, such as a connection error, a context-length error or a Python traceback.

## Baseline runner outputs

| Runner family | Output flag | Files |
| --- | --- | --- |
| `gpqa`, `hotpotqa`, `math`, `lcb`, `apps` | `--out <file.jsonl>` | One JSONL record per instance; nothing is saved without `--out` |
| `bfcl`, `swe` | `--out-dir <dir>` | `predictions.jsonl`, `results.jsonl`, `traces/`; SWE-bench adds `patches/<id>.diff`. The default folder differs per runner, for example `results/bfcl` for Single |
| `toolhop`, `apibank` | `--out-dir <dir>` | `predictions.jsonl`, `results.jsonl`, `traces/`; default `results/<dataset>/<topology>_<framework>/` |
| Communication runners | `--out <file.jsonl>` | Default `results/communications_baseline/<topology>_<dataset>_<format>/results.jsonl` |

`scripts/run_topologies.sh` writes the first family to `results/topologies_baseline/<topology>_<dataset>/predictions.jsonl`, with the framework in the topology part (for example `centralized_autogen_hotpotqa`). Team-size runners follow their topology's family, but their default paths don't include the team size, so always pass an explicit path.

## Percentage points

Scores are fractions in [0, 1]. Multiply by 100 for percent and report a gain as the difference in percentage points (pp). On 25 validation rows one row is 4 pp.

## Load every run

This reads every `meta.json` under the GEPA sweep root and prints the baseline, the score of the prompts that were kept, and both deltas in pp. Run it from the repository root; it needs only the standard library.

```python title="summarize_gepa.py"
import json
from pathlib import Path

ROOT = Path("optimizers/gepa/results/gepa")  # OUT_ROOT used by run_gepa.sh

rows = []
for path in sorted(ROOT.glob("*/meta.json")):
    meta = json.loads(path.read_text())
    if meta.get("compiled_score") is None:  # --baseline-only run
        continue
    base = 100 * meta["baseline_score"]
    kept = 100 * (meta["compiled_score"] if meta["accept_compiled_prompt"]
                  else meta["baseline_score"])
    rows.append((meta["topology"], meta["dataset"], base, kept,
                 kept - base, 100 * meta["delta"]))

print(f"{'topology':<46}{'dataset':<10}{'base':>7}{'opt':>7}{'Δ pp':>8}{'raw Δ':>8}")
for topo, ds, base, opt, gain, raw in rows:
    print(f"{topo:<46}{ds:<10}{base:7.1f}{opt:7.1f}{gain:+8.1f}{raw:+8.1f}")
```

`Δ pp` is the gain of the prompts the pilot kept, so it is never negative. `raw Δ` is the compiled prompts' change, which shows regressions the acceptance rule caught. For MIPRO, point `ROOT` at `optimizers/mipro/results/mipro`.
