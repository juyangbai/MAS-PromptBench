# LiveCodeBench

LiveCodeBench collects programming problems from contest sites. Agents write a Python solution, and the runner counts it as solved only if it passes every hidden test.
{ .lede }

<div class="facts" markdown>
<div><span>Domain</span>Coding</div>
<div><span>Metric</span>pass@1</div>
<div><span>Launcher limit</span>50</div>
<div><span>Eval IDs</span>50</div>
<div><span>Data</span>livecodebench/code_generation_lite</div>
</div>

The CLI and folder name for this task is `lcb`.

## The task

Each instance is a problem statement in one of two modes, chosen the same way as LiveCodeBench's own prompt templates:

- **Stdin mode** (no starter code): the program reads input from stdin and writes the answer to stdout.
- **Functional mode** (LeetCode-style starter code): the prompt includes the starter code, and the solution fills in the method.

Public test cases are not shown, matching LiveCodeBench. Agents can call `python_exec(code, stdin)`, which runs a snippet in a fresh Python subprocess with a 10-second timeout and returns stdout, stderr and the exit code.

The final agent must end with one fenced `python` code block holding the submitted solution. The runner takes the last fenced block that parses as Python, preferring blocks labelled `python` or `py` over bare fences and skipping blocks that start with `{` or `[`.

## How it is scored

The runner runs the extracted code against every hidden test of the problem, with a 6-second timeout per test:

- **Stdin tests** run the program with the test input. The test passes if the program exits with code 0 and its output matches, either exactly or line by line with numeric tokens compared as decimals.
- **Functional tests** call `Solution().<fn_name>` (or a top-level `<fn_name>`) in a subprocess that applies LiveCodeBench's `reliability_guard` and a memory cap, then compare the return value with the expected one. The cap is `LCB_FUNCTIONAL_MEMORY_BYTES`, 4 GB by default; set it to `0` to disable it.

An instance scores 1 only if all tests pass (the pass@1 convention); otherwise 0. The batch prints pass@1 over all instances, over instances with extracted code, and per difficulty tier.

!!! warning "Generated code runs on your machine"
    Both `python_exec` and the scorer run model-written code as plain subprocesses on the host, with timeouts but no container. Run batches on a machine you are comfortable exposing to untrusted code.

## Data

The runner loads the Hugging Face dataset `livecodebench/code_generation_lite`, split `test`. Hidden tests come from each row's `private_test_cases` field, which is stored compressed; the runner decodes it and skips rows without usable tests. Row IDs are LiveCodeBench `question_id` values, such as `1873_A`.

The 50 frozen IDs are in [`benchmarks/lcb/lcb_eval_ids.json`](https://github.com/juyangbai/MAS-PromptBench/blob/main/benchmarks/lcb/lcb_eval_ids.json).

## Run it

Run every command from the repository root. With no arguments, the single-agent runner runs two built-in problems, one in stdin mode and one in functional mode, and prints the extracted code and test results for each:

```bash title="Smoke demo"
python -m topologies.single.lcb.langgraph_lcb
```

A batch needs `--batch`. Without `--out`, it only prints scores.

=== "Single"

    ```bash
    python -m topologies.single.lcb.langgraph_lcb --batch --limit 50 \
      --out results/topologies_baseline/single_lcb/predictions.jsonl
    ```

=== "Sequential (LangGraph)"

    ```bash
    python -m topologies.sequential.langgraph.lcb.langgraph_lcb --batch --limit 50 \
      --out results/topologies_baseline/sequential_langgraph_lcb/predictions.jsonl
    ```

To score exactly the frozen set, pass its IDs to `--only`:

```bash title="Score the frozen eval IDs"
MANIFEST=benchmarks/lcb/lcb_eval_ids.json
IDS=$(python -c "import json,sys; print(*json.load(open(sys.argv[1]))['ids'])" $MANIFEST)
python -m topologies.single.lcb.langgraph_lcb --batch --only $IDS \
  --out results/topologies_baseline/single_lcb/predictions.jsonl
```

Each output line records the predicted code, `pass` and `total` test counts, `pass_rate`, `em` (the pass@1 score), difficulty and platform.

To optimize the single-agent prompt with [GEPA](../optimizers/gepa.md):

```bash title="GEPA on LiveCodeBench"
cd optimizers/gepa
python -m real_runner_gepa.pilots.run_gepa_dataset --dataset lcb --topology single \
  --train-size 25 --val-size 25 --max-full-evals 5 --out results/gepa/single_lcb
```

LiveCodeBench also has [communication-protocol](../mas/communication-protocols.md) and [team-size](../mas/team-sizes.md) runners, and GEPA can target them with topology names such as `centralized_r4` or `independent_communications_freeform`.

## Flags

Beyond the common flags (`--batch`, `--limit`, `--offset`, `--only`, `--out`):

| Flag | Runners | Effect |
| --- | --- | --- |
| `--difficulty {easy,medium,hard}` | all eight | Keep one difficulty tier. |
| `--platform {codeforces,leetcode,atcoder}` | LangGraph Sequential, Centralized and Decentralized | Keep one source platform. |

## Related

- [APPS](apps.md) and [SWE-bench Verified](swe-bench.md), the other coding tasks.
- [Workflow Topologies](../mas/topologies.md) for what each runner variant does.
- [Evaluation Protocol](../evaluation/protocol.md) for how the frozen IDs are used.
