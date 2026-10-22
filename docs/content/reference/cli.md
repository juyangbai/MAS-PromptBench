# Command-Line Flags

Every command-line option of the runners, the run protocol, the aggregation, the prompt generator and the golden recorder, with its default in the code. Environment variables are on [Environment Variables](environment.md).
{ .lede }

## Invoke a runner

Run every runner as a module from the repository root:

```bash title="Module form"
python -m topologies.single.hotpotqa.langgraph_hotpotqa --batch --limit 100 \
  --out results/topologies_baseline/single_hotpotqa/predictions.jsonl
```

The file-path form (`python topologies/...py`) fails with `ModuleNotFoundError: No module named 'core'` unless the repository root is on `PYTHONPATH`. Module paths follow the file layout:

| Runner family | Module |
| --- | --- |
| Topology | `topologies.<topology>[.<framework>].<dataset>.<framework>_<dataset>` |
| Team size | `teamsizes.<topology>.<dataset>.<dataset>_r<r>` |
| Communication format | `communications.<topology>.<dataset>.<dataset>_<format>` |

## Runner options

Every runner in `topologies/`, `teamsizes/` and `communications/` takes the same options from `core/cli.py`:

| Option | Default | Meaning |
| --- | --- | --- |
| `--batch` | off | Run the benchmark slice. Without it, a runner with a canned demo runs the demo; the BFCL, SWE-bench, API-Bank, ToolHop and communication runners have none and run a batch. |
| `--limit N` | per dataset (below) | At most N instances, after `--offset`. |
| `--offset K` | `0` | Skip the first K instances. |
| `--only ID ...` | none | Only these IDs. List several after one flag or repeat it; it lifts `--limit`. |
| `--out-dir DIR` | none | Write `DIR/predictions.jsonl` and the run's other artifacts, such as `traces/`. |
| `--out PATH` | none, or a runner default under `results/` | The predictions JSONL, instead of `DIR/predictions.jsonl`. |

Each dataset adds its options through `add_arguments` in `core/tasks/<dataset>.py`:

| Dataset | Options | Default `--limit` |
| --- | --- | --- |
| `apibank` | `--level` (`all`, `1`, `2`, `3`), `--curated-path`, `--toolsearcher-scorer {official,upstream,keyword}`, `--summary` (JSON report, no model call) | 2 |
| `apps` | `--difficulty {introductory,interview,competition}`, `--max-tests-per-row` (20; `<= 0` keeps all) | none |
| `bfcl` | `--category {simple,multiple,parallel,parallel_multiple}` (`simple`) | 5 |
| `gpqa` | `--shuffle-seed` (0, the option order every runner uses) | none |
| `lcb` | `--difficulty {easy,medium,hard}`; `--platform {codeforces,leetcode,atcoder}` on the topology and team-size runners that offer it | none |
| `swe` | `--subset` (`test`), `--workdir-root DIR`, `--eval` (`singularity`; `local` for `single`; `none` collects patches only), `--skip-eval`, `--keep-workdirs` | 2 |
| `toolhop` | `--smoke-dataset` (load and validate only) | 5 |
| `hotpotqa`, `math` | none | none |

- Communication pairs have no default `--limit`. `--help` shows a runner's exact options.
- Every output file is emptied when a batch starts, so a run never mixes with an earlier one. Pass `--out` or `--out-dir` to keep predictions: some runners write a default file under `results/`, others only print scores.
- The exit status is 1 when no instance loads, or when every row failed on an endpoint infrastructure error (`INFRASTRUCTURE_ERRORS` in `core/batch.py`; the records are still written), so a sweep marks the cell failed.

```bash title="Two BFCL instances"
python -m topologies.single.bfcl.langgraph_bfcl --only simple_0 simple_1 \
  --out-dir results/bfcl_single_check
```

The sweep launchers in `scripts/` take no options; their [environment variables](environment.md#sweep-launchers) choose the grid.

## Run protocol

`python -m optimizers.protocol.run` runs one optimizer job; [Run an Optimizer](../optimizers/running.md) walks through it.

| Option | Default | Meaning |
| --- | --- | --- |
| `--method` | required | `gepa`, `mipro`, `mapro`, `maspo`, `hivemind`, `mamut_gepa`, `maspob`, `tavo` or `identity` (the seed prompts). |
| `--dataset` | required | `apibank`, `apps`, `bfcl`, `gpqa`, `hotpotqa`, `lcb`, `math`, `swe` or `toolhop`. |
| `--topology` | required | A base topology or a registry key such as `sequential_crewai`, `independent_r8` or `centralized_communications_structured_soft`. |
| `--framework` | `langgraph` | `langgraph`, `crewai`, `autogen` or `openai_agents`. |
| `--team-size` | `4` (`single`: 1) | `2`, `4`, `8` or `10`. |
| `--communication` | `freeform` | `freeform`, `semi_structured` or `structured_soft`. |
| `--model` | required | `qwen` (`Qwen/Qwen3.5-9B`), `llama` (`meta-llama/Llama-3.1-8B-Instruct`) or a full model ID. |
| `--seed` | required | Optimizer seed `0`, `1` or `2`. |
| `--out` | required | Job folder. |
| `--budget` | `600` | Usable rollouts; a smaller value makes a non-conformant smoke run. |
| `--phase` | `all` | `optimize`, `validate`, `test` or `all` (whatever is missing). |
| `--allow-any-cell` | off | Run a cell outside the experiment grid. |
| `--task-endpoints` | `$TASK_ENDPOINTS`, else `$VLLM_BASE_URL` | Comma-separated task endpoints. |
| `--evaluation-cache DIR` | `<out>/evaluations` | Shared folder of content-addressed evaluations. |
| `--quiet` | off | No progress lines; errors are still logged. |
| `--log-level` | `$LOG_LEVEL`, else `INFO` | Console level. |

A registry key sets the framework, team size or communication format itself; an explicit flag that contradicts it is an error. The keys a dataset accepts:

| Key | Datasets | Runner |
| --- | --- | --- |
| `single`, `independent`, `sequential`, `centralized`, `decentralized` | all nine | LangGraph |
| `sequential_crewai`, `centralized_autogen`, `decentralized_openai_agents` | all nine | CrewAI, AutoGen, OpenAI Agents SDK |
| `<topology>_r<r>` | `hotpotqa`, `lcb`, `bfcl`, `apibank`, `toolhop` | [team size](../mas/team-sizes.md) r ∈ {2, 4, 8, 10} |
| `<topology>_communications_<format>` | `hotpotqa`, `lcb`, `bfcl`, `apibank`, `toolhop` | [communication format](../mas/communication-protocols.md) |

Here `<topology>` is `independent`, `sequential`, `centralized` or `decentralized`. The job exits with status 2 on a job, evaluation or installation error. Stdout carries only the one-line JSON summary.

## Aggregate

`python -m optimizers.protocol.aggregate ROOT [ROOT ...]` pairs the seeds of every cell found under the given job folders or roots.

| Option | Default | Meaning |
| --- | --- | --- |
| `ROOT` | required | Job folders, or roots searched for `result.json`. |
| `--out` | none | Write the JSON summary here. |
| `--bootstrap` | `10000` | Bootstrap replicates (at least 100). |
| `--family` | `task` | Holm family: `task` (dataset × model), `method`, or `none` (one family). |
| `--include-nonconformant` | off | Include smoke-budget, off-grid and other non-conformant jobs. |
| `--log-level` | `$LOG_LEVEL`, else `INFO` | Console level. |

It prints a tab-separated table to stdout and exits with status 1 when it finds no job result. [Read Run Outputs](../evaluation/outputs.md#aggregate-summary) explains the summary.

## Prompt generation

`python configs/generate_role_prompts.py` writes the seed prompts in `configs/prompts/`; [Add a Dataset](../extending/add-dataset.md#4-seed-prompts) explains how.

| Option | Default | Meaning |
| --- | --- | --- |
| `--base-url` | `$PROMPT_GEN_BASE_URL`, else `http://localhost:8200/v1` | Generator endpoint. |
| `--api-key` | `$PROMPT_GEN_API_KEY`, else `EMPTY` | API key. |
| `--model` | `$PROMPT_GEN_MODEL`, else `Qwen/Qwen3.5-122B-A10B-FP8` | Generator model. |
| `--temperature` | `0.0` | Sampling temperature. |
| `--seed` | `42` | Request seed. |
| `--force` | off | Overwrite existing prompt files. |
| `--only` | none | `<topology>`, `<topology>/<dataset>` or `<topology>/<dataset>/<role>`; repeatable. |

## Tests

`python -m tests.golden.record` re-records golden cells after an intended behavior change:

| Option | Default | Meaning |
| --- | --- | --- |
| `--only GLOB` | none | Cell IDs to record; repeatable. The cells are merged into the existing golden files. |
| `--force` | off | Overwrite existing golden cells. |
| `--workers` | `$GOLDEN_WORKERS`, else half the CPUs (2 to 32) | Parallel worker processes. |
| `--timeout` | `$GOLDEN_CELL_TIMEOUT`, else `900` | Seconds per cell. |
| `--list` | off | Print the cell IDs and exit. |
| `--keep-tmp` | off | Keep the worker temp folders, for debugging. |
| `--no-verify` | off | Skip the reproduction pass: faster, but a load-induced flake could be recorded. |

Replay the suite with `python -m pytest -p no:cacheprovider tests/golden -q`; the [golden tests README](https://github.com/juyangbai/MAS-PromptBench/blob/main/tests/golden/README.md) covers its prerequisites.
