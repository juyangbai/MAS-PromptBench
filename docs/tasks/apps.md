# APPS

APPS is a set of Python programming problems, from introductory exercises to competition tasks. Agents write a solution, and the runner counts it as solved only if it passes every test it runs.
{ .lede }

<div class="facts" markdown>
<div><span>Domain</span>Coding</div>
<div><span>Metric</span>Strict accuracy (pass@1)</div>
<div><span>Launcher limit</span>50</div>
<div><span>Eval IDs</span>50</div>
<div><span>Data</span>codeparrot/apps (test)</div>
</div>

## The task

The runner frames each problem the way the APPS paper and common harnesses do: `QUESTION:`, the problem text, a format directive, then `ANSWER:`. The directive depends on the problem:

- **Standard input**: `Use Standard Input format.` The program reads stdin and prints the answer.
- **Call-based**: the starter code is shown, followed by `Use Call-Based format.` The solution defines the named function.

Agents can call `python_exec(code, stdin)`, which runs a snippet in a fresh Python subprocess with a 10-second timeout and returns stdout, stderr and the exit code.

The final agent must end with one fenced `python` code block holding the submitted solution. The runner takes the last fenced block that parses as Python, preferring blocks labelled `python` or `py` over bare fences.

## How it is scored

The runner runs the extracted code against the problem's tests, with the APPS reference timeout of 4 seconds per test. Each test passes if the output matches through a cascade of checks:

- **Standard input**: stripped output equal; else equal line by line; else all numeric tokens close (`numpy.allclose`, relative 1e-5, absolute 1e-6).
- **Call-based**: return value equal; else numerically close; else equal as sets; else equal as sets of floats rounded to 3 places. These tests run in a subprocess with a reliability guard and a memory cap (`APPS_CALL_BASED_MEMORY_BYTES`, 4 GB by default).

An instance scores 1 only if every test passes (APPS strict accuracy, reported as pass@1); otherwise 0. The batch prints the score over all instances, over instances with extracted code, and per difficulty tier.

!!! warning "Generated code runs on your machine"
    Model-written code runs as plain subprocesses on the host, both in `python_exec` and in scoring, with timeouts but no container.

## Data

The runner loads the Hugging Face dataset `codeparrot/apps`, split `test`. Each row's ID is its `problem_id` as a string, and its tests come from the `input_output` field. Rows without usable tests are skipped.

APPS problems can have hundreds of tests, so the runner keeps the first 20 per problem by default. Change this with `--max-tests-per-row`; `-1` keeps every test. The cap changes the score, so keep it fixed when you compare runs.

The 50 frozen IDs are problem IDs `0` to `49`, listed in [`benchmarks/apps/apps_eval_ids.json`](https://github.com/juyangbai/MAS-PromptBench/blob/main/benchmarks/apps/apps_eval_ids.json).

## Run it

Run every command from the repository root. With no arguments, the single-agent runner runs two built-in problems, one standard-input and one call-based, and prints the extracted code and test results:

```bash title="Smoke demo"
python -m topologies.single.apps.langgraph_apps
```

A batch needs `--batch`. Without `--out`, it only prints scores.

=== "Single"

    ```bash
    python -m topologies.single.apps.langgraph_apps --batch --limit 50 \
      --out results/topologies_baseline/single_apps/predictions.jsonl
    ```

=== "Centralized (AutoGen)"

    ```bash
    python -m topologies.centralized.autogen.apps.autogen_apps --batch --limit 50 \
      --out results/topologies_baseline/centralized_autogen_apps/predictions.jsonl
    ```

To score exactly the frozen set, pass its IDs to `--only`:

```bash title="Score the frozen eval IDs"
MANIFEST=benchmarks/apps/apps_eval_ids.json
IDS=$(python -c "import json,sys; print(*json.load(open(sys.argv[1]))['ids'])" $MANIFEST)
python -m topologies.single.apps.langgraph_apps --batch --only $IDS \
  --out results/topologies_baseline/single_apps/predictions.jsonl
```

To optimize the single-agent prompt with [GEPA](../optimizers/gepa.md):

```bash title="GEPA on APPS"
cd optimizers/gepa
python -m real_runner_gepa.pilots.run_gepa_dataset --dataset apps --topology single \
  --train-size 25 --val-size 25 --max-full-evals 5 --out results/gepa/single_apps
```

## Flags

Beyond the common flags (`--batch`, `--limit`, `--offset`, `--only`, `--out`), all eight APPS runners accept:

| Flag | Default | Effect |
| --- | --- | --- |
| `--difficulty {introductory,interview,competition}` | all | Keep one difficulty tier. |
| `--max-tests-per-row N` | `20` | Tests run per problem; `-1` runs all. |

## Related

- [LiveCodeBench](livecodebench.md) and [SWE-bench Verified](swe-bench.md), the other coding tasks.
- [Workflow Topologies](../mas/topologies.md) for what each runner variant does.
- [Evaluation Protocol](../evaluation/protocol.md) for how the frozen IDs are used.
