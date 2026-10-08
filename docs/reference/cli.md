# Command-Line Flags

Every flag accepted by the benchmark's runners, optimizer pilots and sweep scripts, with its type and default as defined in the code. Runner flags differ by dataset family, so find your dataset first.
{ .lede }

## Invoke a runner

Run runners as modules from the repository root, the way the sweep scripts do:

```bash title="Module form"
python -m topologies.single.hotpotqa.langgraph_hotpotqa --batch --limit 100 \
  --out results/topologies_baseline/single_hotpotqa/predictions.jsonl
```

The file-path form needs the repository root on `PYTHONPATH`, because most runners import `topologies.*` (or `teamsizes.*`) before they add the root to `sys.path`:

```bash title="File-path form"
PYTHONPATH=. python topologies/single/hotpotqa/langgraph_hotpotqa.py --batch --limit 100
```

Module paths follow the file layout: `topologies.<topology>[.<framework>].<dataset>.<framework>_<dataset>`, `teamsizes.<topology>.<dataset>.<dataset>_r<N>` and `communications.<topology>.<dataset>.<dataset>_<format>`. Every runner also accepts `--help`.

## Flag families at a glance

| Family | Batch switch | Output flag | `--only` style | Run with no flags |
| --- | --- | --- | --- | --- |
| `gpqa`, `hotpotqa`, `math`, `lcb`, `apps` | `--batch` | `--out FILE` (no default) | `--only a b c` | canned demo |
| `bfcl` | none; `--batch` is an error | `--out-dir DIR` | `--only a --only b` | batch of 5 |
| `swe` | none; `--batch` is an error | `--out-dir DIR` | `--only a --only b` | batch of 2 |
| `toolhop` | `--batch` accepted and ignored | `--out-dir DIR` | `--only a --only b` | batch of 5 |
| `apibank` | `--batch` accepted and ignored | `--out-dir DIR` | `--only a --only b` | batch of 2 |
| communication protocols | `--batch` accepted and ignored | `--out FILE` | `--only a --only b` | full batch |

`--only` comes in two styles. In the first family it is declared with `nargs="*"`, so you list IDs after one flag. Everywhere else it uses `action="append"`, so you repeat the flag once per ID.

## gpqa, hotpotqa, math, lcb, apps

All eight topology variants of these datasets share these flags.

| Flag | Type | Default | Meaning |
| --- | --- | --- | --- |
| `--batch` | switch | off | Run the dataset. Without it the runner solves one built-in example and prints the trace. |
| `--limit` | int | none (all rows) | Maximum rows, applied after `--only` and `--offset`. |
| `--offset` | int | `0` | Skip this many rows. |
| `--only` | list of IDs | none | Keep only these instance IDs. Space-separated after one `--only`. |
| `--out` | path | none | Per-instance JSONL file, overwritten on each run. Without it, a batch only prints scores. |

Dataset-specific flags:

| Flag | Datasets | Type | Default | Meaning |
| --- | --- | --- | --- | --- |
| `--shuffle-seed` | gpqa | int | `0` | Seed for the per-row shuffling of answer choices. |
| `--difficulty` | lcb | `easy`, `medium`, `hard` | none | Keep one LiveCodeBench difficulty tier. |
| `--platform` | lcb (LangGraph sequential, centralized and decentralized only) | `codeforces`, `leetcode`, `atcoder` | none | Keep one source platform. |
| `--difficulty` | apps | `introductory`, `interview`, `competition` | none | Keep one APPS difficulty tier. |
| `--max-tests-per-row` | apps | int | `20` | Cap the tests run per problem; `-1` runs all. |

```bash title="Score the first 50 LiveCodeBench rows"
python -m topologies.centralized.langgraph.lcb.langgraph_lcb --batch --limit 50 \
  --out results/topologies_baseline/centralized_langgraph_lcb/predictions.jsonl
```

## bfcl

BFCL runners always run a batch and have no `--batch` flag.

| Flag | Type | Default | Meaning |
| --- | --- | --- | --- |
| `--category` | `simple`, `multiple`, `parallel`, `parallel_multiple` | `simple` | BFCL AST subset to evaluate. |
| `--limit` | int | `5` | Maximum rows. Ignored when `--only` is given. |
| `--offset` | int | `0` | Skip this many rows. |
| `--only` | ID, repeatable | none | Keep only these instance IDs, such as `simple_0`. |
| `--out-dir` | path | per variant, see [Default output folders](#default-output-folders) | Folder for `predictions.jsonl`, `results.jsonl` and `traces/`. |

```bash title="Two BFCL instances"
python -m topologies.single.bfcl.langgraph_bfcl --only simple_0 --only simple_1 \
  --out-dir results/bfcl_single_check
```

## swe

SWE-bench runners always run a batch and have no `--batch` flag.

| Flag | Type | Default | Meaning |
| --- | --- | --- | --- |
| `--subset` | str | `test` | Hugging Face split of SWE-bench Verified. |
| `--limit` | int | `2` | Maximum instances. Ignored when `--only` is given. |
| `--offset` | int | `0` | Skip this many instances. |
| `--only` | ID, repeatable | none | Keep only these instance IDs, such as `astropy__astropy-12907`. |
| `--workdir-root` | path | per variant, under `~` | Root for per-instance repository clones. |
| `--out-dir` | path | per variant | Folder for `predictions.jsonl` and `results.jsonl`. |
| `--eval` | `local`, `singularity`, `none` | `local` (single); `singularity` (all others) | Test backend: pytest on the host, pytest inside the per-instance Singularity image, or skip and keep only the patches. Only the single runner offers `local`. |
| `--skip-eval` | switch | off | Same as `--eval none`. Centralized and decentralized LangGraph runners only. |
| `--keep-workdirs` | switch | off | Keep the per-instance clones after solving. |

## toolhop

ToolHop runners always run a batch. They execute dataset-provided Python tools, so they refuse to solve unless `TOOLHOP_ALLOW_DATASET_EXEC=1` is set.

| Flag | Type | Default | Meaning |
| --- | --- | --- | --- |
| `--batch` | switch | off | Accepted for uniformity; has no effect. |
| `--limit` | int | `5` | Maximum rows. Ignored when `--only` is given. |
| `--offset` | int | `0` | Skip this many rows. |
| `--only` | ID, repeatable | none | Keep only these dataset IDs. |
| `--out-dir` | path | `results/toolhop/<style>` | Folder for `predictions.jsonl`, `results.jsonl` and `traces/`. |
| `--smoke-dataset` | switch | off | Load and validate ToolHop, print a summary and exit, without model calls or tool execution. |

## apibank

API-Bank runners always run a batch. At the default level `all` they read the task list from `benchmarks/apibank/apibank_eval_ids.json`; other levels are built from the vendored API-Bank source.

| Flag | Type | Default | Meaning |
| --- | --- | --- | --- |
| `--batch` | switch | off | Accepted for uniformity; has no effect. |
| `--limit` | int | `2` | Maximum rows. Applied after `--only`, so raise it when you pass more than two IDs. |
| `--offset` | int | `0` | Skip this many rows. |
| `--only` | ID, repeatable | none | Keep only these task IDs. |
| `--out-dir` | path | `results/apibank/<style>` | Folder for `predictions.jsonl`, `results.jsonl` and traces. |
| `--summary` | switch | off | Print a dataset summary and exit without calling the model. |
| `--level` | `all`, `1`, `2`, `3` | none, which falls back to `APIBANK_LEVEL` (`all`) | API-Bank level slice. |
| `--curated-path` | path | none | Use this curated manifest; sets `APIBANK_CURATED_PATH`. |
| `--toolsearcher-scorer` | `official`, `upstream`, `keyword` | none: `official` for levels `all`, 2 and 3, `keyword` for level 1 | ToolSearcher scorer; sets `APIBANK_TOOLSEARCHER_SCORER`. |

!!! warning "Some runners append"
    BFCL, ToolHop and API-Bank runners open `predictions.jsonl` and `results.jsonl` in append mode, so a second run into the same `--out-dir` adds rows to the first. Use a fresh folder per run. SWE-bench runners empty both files when a batch starts, and the `--out` file of the first family is overwritten.

## Default output folders

Paths are relative to the repository root. `<style>` is the value used by ToolHop and API-Bank.

| Variant | `bfcl --out-dir` | `swe --out-dir` | `swe --workdir-root` | `<style>` |
| --- | --- | --- | --- | --- |
| single | `results/bfcl` | `results/swe_bench` | `~/swe_work` | `single_langgraph` |
| independent | `results/bfcl_independent` | `results/swe_bench_independent` | `~/swe_work_independent` | `independent_langgraph` |
| sequential, LangGraph | `results/bfcl_sequential_langgraph` | `results/swe_bench_sequential_langgraph` | `~/swe_work_sequential_langgraph` | `sequential_langgraph` |
| sequential, CrewAI | `results/bfcl_sequential` | `results/swe_bench_sequential` | `~/swe_work_sequential` | `sequential_crewai` |
| centralized, LangGraph | `results/bfcl_centralized_langgraph` | `results/swe_bench_centralized_langgraph` | `~/swe_work_centralized_langgraph` | `centralized_langgraph` |
| centralized, AutoGen | `results/bfcl_centralized` | `results/swe_bench_centralized` | `~/swe_work_centralized` | `centralized_autogen` |
| decentralized, LangGraph | `results/bfcl_decentralized_langgraph` | `results/swe_bench_decentralized_langgraph` | `~/swe_work_decentralized_langgraph` | `decentralized_langgraph` |
| decentralized, OpenAI SDK | `results/bfcl_decentralized` | `results/swe_bench_decentralized` | `~/swe_work_decentralized` | `decentralized_openai` |

## Communication-protocol runners

The 60 runners under `communications/` share one CLI, `cli_main` in `communications/communication_formats.py`. Topologies are `independent`, `sequential`, `centralized` and `decentralized`; datasets `hotpotqa`, `lcb`, `toolhop`, `apibank` and `swe`; formats `freeform`, `semi_structured` and `structured_soft`. Each wraps the LangGraph runner of its topology and dataset.

| Flag | Type | Default | Meaning |
| --- | --- | --- | --- |
| `--batch` | switch | off | Accepted; the runner always runs a batch. |
| `--limit` | int | none (all rows) | Maximum rows. |
| `--offset` | int | `0` | Skip this many rows. |
| `--only` | ID, repeatable | none | Keep only these instance IDs. |
| `--out` | path | `results/communications_baseline/<topology>_<dataset>_<format>/results.jsonl` | Output JSONL file. |

```bash title="One protocol cell"
python -m communications.centralized.hotpotqa.hotpotqa_structured_soft --batch --limit 100
```

ToolHop cells still need `TOOLHOP_ALLOW_DATASET_EXEC=1`. See [Communication Protocols](../mas/communication-protocols.md).

## Team-size runners

The 144 runners under `teamsizes/<topology>/<dataset>/<dataset>_r<N>.py` (N is 2, 4, 8 or 10) take the same flags, types and defaults as the topology runners for their dataset, including `--skip-eval` on the centralized and decentralized SWE-bench runners and `--platform` on the non-independent LCB runners. Only the default folders differ:

| Dataset family | Default folders |
| --- | --- |
| `gpqa`, `hotpotqa`, `math`, `lcb`, `apps` | no output file; pass `--out` |
| `bfcl` | `results/bfcl_<topology>_r<N>` for N = 2, 8, 10; at N = 4 the LangGraph topology runner's folder |
| `swe` | `results/swe_bench_<topology>_r<N>` and `~/swe_work_<topology>_r<N>` for N = 2, 8, 10; at N = 4 the LangGraph topology runner's folders |
| `toolhop`, `apibank` | `results/<dataset>/<topology>_<dataset>_r<N>` |

Pass an explicit `--out` or `--out-dir` that names the team size, such as `results/teamsizes_r4/hotpotqa/centralized_r4/`, so runs at different sizes never share a file. See [Team Sizes](../mas/team-sizes.md).

## Optimizer pilots

Run the pilots from their workspace folder (`optimizers/gepa` or `optimizers/mipro`). The default `--out` sits under that workspace's `results/` folder; a relative `--out` is resolved against your current folder. The defaults are smoke-test sized. [GEPA](../optimizers/gepa.md) and [MIPRO](../optimizers/mipro.md) explain each setting.

| Flag | GEPA default | MIPRO default | Meaning |
| --- | --- | --- | --- |
| `--dataset` | required | required | Registered dataset name. |
| `--topology` | required | required | Optimizer topology name for that dataset. |
| `--train-size`, `--val-size` | `1`, `1` | `1`, `1` | Split sizes. |
| `--offset`, `--split-seed` | `0`, `0` | `0`, `0` | Split offset and shuffle seed. |
| `--n-agents`, `--n-rounds` | `2`, `1` | `2`, `1` | Agents for independent and decentralized pairs; debate rounds for decentralized pairs. |
| `--auto` | none | none | `light`, `medium` or `heavy` budget preset. |
| `--num-threads` | `1` | `1` | Parallel evaluation threads. |
| `--out` | `results/<topology>_<dataset>_real_gepa` | `results/<topology>_<dataset>_real_mipro` | Output folder. |
| `--max-full-evals` | `1` | — | GEPA budget; replaced by `--auto` when set. |
| `--reflection-minibatch-size` | `1` | — | Rows per reflection step. |
| `--component-selector` | `all` | — | `all` or `round_robin`: which role prompts to mutate per step. |
| `--early-stop-patience` | `0` (off) | — | Stop after this many full evaluations without improvement. |
| `--skip-perfect-score` | off | — | Passed to DSPy's GEPA as `skip_perfect_score`. |
| `--baseline-only` | off | — | Score the seed prompts and stop. |
| `--num-candidates`, `--num-trials` | — | `3`, `3` | MIPRO search size; must be omitted when `--auto` is set. |
| `--max-bootstrapped-demos`, `--max-labeled-demos` | — | `4`, `0` | Few-shot demo limits. |
| `--minibatch`, `--minibatch-size`, `--minibatch-full-eval-steps` | — | off, `35`, `5` | Minibatch evaluation. |
| `--seed`, `--init-temperature` | — | `9`, `1.0` | Passed to MIPROv2. |
| `--metric-threshold`, `--max-errors` | — | none, none | Passed to MIPROv2. |
| `--view-data-batch-size` | — | `10` | Passed to MIPROv2's `compile`. |
| `--program-aware-proposer`, `--data-aware-proposer`, `--tip-aware-proposer`, `--fewshot-aware-proposer` | — | on (each has a `--no-` form) | Which inputs MIPROv2's instruction proposer uses. |
| `--provide-traceback`, `--verbose` | — | off, off | Passed to MIPROv2. |

The sweep launchers pass larger values. `optimizers/gepa/run_gepa.sh` uses 25/25 splits, `--max-full-evals 5`, `--reflection-minibatch-size 3`, `--num-threads 4`, `--n-agents 4`, `--n-rounds 2`, `--component-selector round_robin`, `--early-stop-patience 3` and `--skip-perfect-score`. `optimizers/mipro/run_mipro.sh` uses 25/25 splits, 3 candidates, 3 trials and 4 threads. Their environment knobs are listed in [Environment Variables](environment.md).

## Sweep scripts

The three baseline sweeps in `scripts/` run one process per cell against one endpoint. Each changes into the repository root first.

| Variable | `run_topologies.sh` | `run_communications.sh` | `run_teamsizes.sh` |
| --- | --- | --- | --- |
| `VLLM_BASE_URL` | `http://localhost:8000/v1` | same | same |
| `MODEL_ID` | `Qwen/Qwen3.5-9B` | same | same |
| `TOOLHOP_ALLOW_DATASET_EXEC` | `1` | `1` | `1` |
| `DATASETS` | all nine | `hotpotqa lcb toolhop apibank swe` | all nine |
| `TOPOLOGIES` | not read; the eight variants are fixed in `TOPOS` | `independent sequential centralized decentralized` | `independent sequential centralized decentralized` |
| `FORMATS` | not read | `freeform semi_structured structured_soft` | not read |
| `RVALUES` | not read | not read | `2 4 8 10` |
| `OUT_ROOT` | `results/topologies_baseline` | not read | not read |

Row limits are fixed in each script's `LIMIT` table: gpqa 100, hotpotqa 100, math 100, lcb 50, apps 50, bfcl 100, swe 30, apibank 100, toolhop 100, and 50 for any dataset not listed. The header comments mention a `<DATASET>_LIMIT` override, but no script reads one; edit the table instead.

What each script passes:

- `run_topologies.sh` runs `--batch --limit N --out $OUT_ROOT/<variant>_<dataset>/predictions.jsonl`, except for apibank and toolhop, which get `--batch --limit N` and write to their default folders.
- `run_communications.sh` runs `--batch --limit N` and relies on the default `--out`.
- `run_teamsizes.sh` runs `--batch --limit N` with no output flag.

!!! warning "Known launcher gaps"
    BFCL and SWE-bench runners reject `--batch` (and `--out`), so those cells fail in `run_topologies.sh` and `run_teamsizes.sh`; run them by hand with `--out-dir`. Because `run_teamsizes.sh` passes no `--out`, its gpqa, hotpotqa, math, lcb and apps cells print scores but save no predictions.
