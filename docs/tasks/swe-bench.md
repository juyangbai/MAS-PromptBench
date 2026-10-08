# SWE-bench Verified

SWE-bench Verified pairs real GitHub issues with the tests that the maintainers' fix made pass. Agents edit a checkout of the repository, and the runner counts an instance as resolved only if the hidden tests pass with their change applied.
{ .lede }

<div class="facts" markdown>
<div><span>Domain</span>Coding</div>
<div><span>Metric</span>Resolved rate</div>
<div><span>Launcher limit</span>30</div>
<div><span>Eval IDs</span>30</div>
<div><span>Data</span>princeton-nlp/SWE-bench_Verified</div>
</div>

The CLI and folder name for this task is `swe`.

## The task

For each instance the runner clones `https://github.com/<repo>` into `<workdir-root>/<instance_id>` and checks out the instance's base commit. The agents get the issue text, the maintainers' hints when the dataset has them, and file tools scoped to that checkout:

| Topologies | Tools |
| --- | --- |
| Single, Independent | `file_read`, `file_write`, `list_dir`, `search_repo`, `shell_exec` |
| Sequential, Centralized, Decentralized | `file_read`, `str_replace`, `list_dir`, `search_repo`, `shell_exec` |

Individual roles may get only some of these tools. The issue text is cut to 16,000 characters and the hints to 4,000 (`SWE_PROBLEM_CHAR_BUDGET`, `SWE_HINTS_CHAR_BUDGET`). The checkout has no test dependencies installed, and the prompt tells agents not to run the repository's tests.

The submitted patch is `git diff HEAD` of the checkout after the agents finish. The output contract also asks the final role to end with a fenced `diff` block, but scoring uses the checkout's diff, not that text.

## How it is scored

SWE-bench grading applies the dataset's `test_patch` (the hidden tests) and the model patch, then runs the instance's `FAIL_TO_PASS` and `PASS_TO_PASS` tests with pytest. An instance is **resolved** when every test in both lists passes; `XFAIL` counts as a pass. The reported metric is the fraction of instances resolved.

The `--eval` flag picks where the tests run:

| Mode | What it does |
| --- | --- |
| `singularity` | Pulls `docker://swebench/sweb.eval.x86_64.<tag>:latest` with `singularity pull`, caches it as `<instance_id>.sif` under `SWE_SIF_DIR` (default `~/containers/swe`), and runs both patches and pytest inside it. This matches the official Docker environment. |
| `local` | Applies the test patch in the checkout and runs pytest in your own environment. Fast, but not numerically equivalent to the leaderboard. Single topology only. |
| `none` | Skips evaluation and only collects patches. |

Use `singularity` for reported numbers. Image pulls time out after 15 minutes and test runs after 30.

Every run writes `predictions.jsonl` (`instance_id`, `model_patch`, `model_name_or_path`, the official harness format) and `results.jsonl` (per-instance `f2p_rate`, `p2p_rate` and `resolved`) to `--out-dir`, emptying both at the start of the batch. The runner prints the resolved count at the end; the single-agent runner also prints a `swebench.harness.run_evaluation` command for the official Docker harness.

## Data

The runner loads `princeton-nlp/SWE-bench_Verified` from Hugging Face. You also need `git` and network access to GitHub for the clones, and Singularity on your `PATH` for `singularity` mode.

The 30 frozen IDs in [`benchmarks/swe/swe_eval_ids.json`](https://github.com/juyangbai/MAS-PromptBench/blob/main/benchmarks/swe/swe_eval_ids.json) are a balanced sample: 15 instances labelled "<15 min fix" and 15 labelled "15 min - 1 hour". They are not a slice of the split, so pass them with `--only` rather than `--limit 30`.

## Run it

Run every command from the repository root. There is no smoke demo and no `--batch` flag (passing it is an error); with no arguments a runner solves the first 2 instances.

Score the frozen set by repeating `--only` once per ID. First turn the manifest into flags:

```bash title="Load the frozen IDs"
MANIFEST=benchmarks/swe/swe_eval_ids.json
IDS=$(python -c "import json,sys; print(*json.load(open(sys.argv[1]))['ids'])" $MANIFEST)
ONLY=$(printf -- '--only %s ' $IDS)
```

Then run a topology:

=== "Single"

    ```bash
    python -m topologies.single.swe.langgraph_swe $ONLY --eval singularity \
      --out-dir results/topologies_baseline/single_swe
    ```

=== "Sequential (LangGraph)"

    ```bash
    python -m topologies.sequential.langgraph.swe.langgraph_swe $ONLY --eval singularity \
      --out-dir results/topologies_baseline/sequential_langgraph_swe
    ```

To optimize the single-agent prompt with [GEPA](../optimizers/gepa.md):

```bash title="GEPA on SWE-bench Verified"
cd optimizers/gepa
python -m real_runner_gepa.pilots.run_gepa_dataset --dataset swe --topology single \
  --train-size 25 --val-size 25 --max-full-evals 5 --out results/gepa/single_swe
```

!!! warning "GEPA uses a structural score for SWE"
    During optimization, GEPA calls the runners with evaluation off and scores 1 when the agents produce a non-trivial unified diff, not when tests pass. The agents work in a copy of an existing checkout found under `SWE_WORK_ROOT` (for example, a folder kept with `--keep-workdirs`); if none is found, they get a tiny placeholder repository. Set `REQUIRE_REAL_SWE_WORKDIR=1` to fail instead.

SWE-bench also has [communication-protocol](../mas/communication-protocols.md) runners under `communications/`.

## Flags

| Flag | Default | Effect |
| --- | --- | --- |
| `--eval MODE` | `local` (Single), `singularity` (others) | Evaluation backend. Multi-agent runners accept only `singularity` and `none`. |
| `--workdir-root DIR` | a `swe_work*` folder in your home directory, different per topology | Where repositories are cloned. |
| `--keep-workdirs` | off | Keep each clone after the instance finishes. |
| `--out-dir DIR` | a `results/swe_bench*` folder, different per topology | Where output files go. |
| `--subset SPLIT` | `test` | Hugging Face split to load. |
| `--limit N` | `2` | Number of instances; ignored when `--only` is given. |
| `--offset K` | `0` | Skip the first K instances. |
| `--only ID` | none | Run this instance; repeat the flag for more. |

## Related

- [LiveCodeBench](livecodebench.md) and [APPS](apps.md), the other coding tasks.
- [Workflow Topologies](../mas/topologies.md) for what each runner variant does.
- [Evaluation Protocol](../evaluation/protocol.md) for how the frozen IDs are used.
