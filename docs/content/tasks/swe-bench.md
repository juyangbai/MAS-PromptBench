# SWE-bench Verified

SWE-bench Verified pairs real GitHub issues with the tests that the maintainers' fix made pass. Agents edit a checkout of the repository, and the runner counts an instance as resolved only if the hidden tests pass with their change applied.
{ .lede }

<div class="facts" markdown>
<div><span>Domain</span>Coding</div>
<div><span>Metric</span>Resolved rate</div>
<div><span>Eval IDs</span>30</div>
<div><span>Train / val</span>146 / 50</div>
<div><span>Data</span>princeton-nlp/SWE-bench_Verified</div>
</div>

The CLI and folder name for this task is `swe`.

## The task

For each instance the runner clones `https://github.com/<repo>` into `<workdir-root>/<instance_id>` and checks out the instance's base commit. The agents get the issue text, the maintainers' hints when the dataset has them, and file tools scoped to that checkout:

| Topologies | Tools |
| --- | --- |
| Single, Independent | `file_read`, `file_write`, `list_dir`, `search_repo`, `shell_exec` |
| Sequential, Centralized, Decentralized | `file_read`, `str_replace`, `list_dir`, `search_repo`, `shell_exec` |

Individual roles may get only some of these tools. The issue text is cut to 16,000 characters and the hints to 4,000 (`SWE_PROBLEM_CHAR_BUDGET`, `SWE_HINTS_CHAR_BUDGET`). The checkout has no test dependencies installed, and the prompt tells agents not to run the repository's tests. Independent replicas and Decentralized peers each work in their own clone.

`shell_exec` runs inside the instance's SWE-bench Singularity image, the same image used for evaluation: networking off, a clean environment, only the checkout mounted (its `.git` read-only), and bounded output, time, memory and open files. `SWE_SHELL_SANDBOX=0` runs commands on the host instead; use it only for debugging without Singularity.

The submitted patch is `git diff HEAD` of the checkout after the agents finish. The output contract also asks the answering role to end with a fenced `diff` block, but scoring uses the checkout's diff, not that text. Independent and Decentralized teams submit their agents' most common non-empty patch, and only that patch is evaluated.

## How it is scored

SWE-bench grading applies the dataset's `test_patch` (the hidden tests) and the model patch, then runs the instance's `FAIL_TO_PASS` and `PASS_TO_PASS` tests with pytest. An instance is **resolved** when every test in both lists passes; `XFAIL` counts as a pass. The reported metric is the fraction of instances resolved.

The `--eval` flag picks where the tests run:

| Mode | What it does |
| --- | --- |
| `singularity` | Pulls `docker://swebench/sweb.eval.x86_64.<instance-tag>:latest` with `singularity pull`, caches it as `<instance_id>.sif` under `SWE_SIF_DIR` (default `~/containers/swe`), and runs both patches and pytest inside it. This matches the official Docker environment. |
| `local` | Applies the test patch in the checkout and runs pytest in your own environment. Fast, but not equivalent to the leaderboard. Single topology only. |
| `none` | Skips evaluation and only collects patches. |

Use `singularity` for reported numbers. Image pulls time out after 15 minutes and test runs after 30.

Every run writes to `--out-dir`: `predictions.jsonl` (`instance_id`, `model_patch`, `model_name_or_path`, the official harness format), `results.jsonl` (per-instance `f2p_rate`, `p2p_rate` and `resolved`), `patches/<instance_id>.diff`, `traces/<instance_id>.txt` and, with `singularity`, the test logs in `eval_logs/`. Both JSONL files are emptied when the batch starts, and each clone is deleted after its instance unless you pass `--keep-workdirs`. The end-of-batch report goes to stderr; the single-agent runner also logs a `swebench.harness.run_evaluation` command for the official Docker harness.

## Data

The runner loads `princeton-nlp/SWE-bench_Verified` from Hugging Face. You also need `git` and network access to GitHub for the clones, and Singularity on your `PATH` for the agents' shell and for `singularity` mode. A full set of instance images takes tens of gigabytes; [the benchmarks README](https://github.com/juyangbai/MAS-PromptBench/blob/main/benchmarks/README.md) lists the disk needs.

The 30 eval IDs in [`benchmarks/swe/swe_eval_ids.json`](https://github.com/juyangbai/MAS-PromptBench/blob/main/benchmarks/swe/swe_eval_ids.json) are a balanced sample: 15 instances labelled "<15 min fix" and 15 labelled "15 min - 1 hour". They are not a slice of the split, so pass them with `--only` rather than `--limit 30`.

## Run it

Run every command from the repository root. There is no smoke demo; with no arguments a runner solves the first 2 instances, and `--batch` changes nothing.

Score the eval IDs by passing them to `--only`:

```bash title="Load the eval IDs"
MANIFEST=benchmarks/swe/swe_eval_ids.json
IDS=$(python -c "import json,sys; print(*json.load(open(sys.argv[1]))['ids'])" $MANIFEST)
```

Then run a topology:

=== "Single"

    ```bash
    python -m topologies.single.swe.langgraph_swe --only $IDS --eval singularity \
      --out-dir results/topologies_baseline/single_swe
    ```

=== "Sequential (LangGraph)"

    ```bash
    python -m topologies.sequential.langgraph.swe.langgraph_swe --only $IDS --eval singularity \
      --out-dir results/topologies_baseline/sequential_langgraph_swe
    ```

SWE-bench also has [communication-protocol](../mas/communication-protocols.md) runners under `communications/` and [team-size](../mas/team-sizes.md) runners under `teamsizes/`.

## Optimize it

Every optimizer runs through the same protocol command; change `--method` to switch. For example, GEPA on the single agent:

```bash title="GEPA on Single · SWE-bench Verified"
python -m optimizers.protocol.run --method gepa --dataset swe --topology single \
  --model qwen --seed 0 --out runs/gepa/swe/single/qwen/0
```

Combinations outside the experiment grid need `--allow-any-cell`; see [Optimize a task](index.md#optimize-a-task).

!!! warning "A structural score during optimization"
    The run protocol scores SWE-bench with a structural check only: a rollout scores 1 when the agents produce a unified diff with a header and at least one changed line. The patch is not applied and no tests run. See [Evaluation Protocol](../evaluation/protocol.md).

## Flags

Beyond the common flags (`--batch`, `--limit`, `--offset`, `--only`, `--out-dir`, `--out`):

| Flag | Default | Effect |
| --- | --- | --- |
| `--eval MODE` | `local` (Single), `singularity` (others) | Evaluation backend. Multi-agent runners accept only `singularity` and `none`. |
| `--skip-eval` | off | Alias for `--eval none` (LangGraph Centralized and Decentralized runners). |
| `--workdir-root DIR` | a `swe_work*` folder in your home directory, different per topology | Where repositories are cloned. |
| `--keep-workdirs` | off | Keep each clone after the instance finishes. |
| `--subset SPLIT` | `test` | Hugging Face split to load. |

`--limit` defaults to 2. Without `--out-dir`, output goes to a `results/swe_bench*` folder that differs per topology.

## Related

- [LiveCodeBench](livecodebench.md) and [APPS](apps.md), the other coding tasks.
- [Workflow Topologies](../mas/topologies.md) for what each runner variant does.
- [Evaluation Protocol](../evaluation/protocol.md) for how the eval IDs and splits are used.
