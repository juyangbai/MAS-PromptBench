# MATH

MATH is a set of competition math problems with a single final answer. MAS-PromptBench uses its hardest precalculus slice and scores the last boxed answer with Hendrycks' equivalence check.
{ .lede }

<div class="facts" markdown>
<div><span>Domain</span>Reasoning</div>
<div><span>Metric</span>Accuracy (is_equiv)</div>
<div><span>Launcher limit</span>100</div>
<div><span>Eval IDs</span>100</div>
<div><span>Data</span>qwedsacf/competition_math</div>
</div>

## The task

Each instance is one problem statement, passed to the agents as is. Agents can call `calculator`, which evaluates a numeric Python expression with the usual math functions. The Decentralized (OpenAI SDK) runner also gives its agents `solve_equation` for one-variable equations.

The final agent's output contract asks for a `\boxed{...}` line at the start, at most 12 short reasoning lines, and the same `\boxed{...}` line at the end. The scorer reads only the last boxed expression. The single-agent runner adds a format note with examples such as `\boxed{42}` and `\boxed{\frac{1}{2}}`.

## How it is scored

1. **Extract** the content of the last `\boxed{...}` in the final message, counting braces so nested expressions like `\boxed{\frac{1}{2}}` come out whole. No boxed answer means the instance is wrong.
2. **Compare** it to the gold answer with `is_equiv`, a verbatim port of Hendrycks' `math_equivalence.py`. Both strings are normalized and then compared for equality.

Normalization removes spaces, `\left` and `\right`, degree marks, `\$`, `\%` and trailing units; rewrites `tfrac` and `dfrac` as `frac`; repairs shorthand like `\frac12` and `\sqrt3`; turns `0.5` into `\frac{1}{2}` and simple `a/b` into `\frac{a}{b}`; and drops a short left-hand side such as `x =`.

The batch reports `EM`, the fraction of instances judged equivalent, over all instances (a missing answer counts as wrong) and over those with an extracted answer.

## Data

The runner loads the Hugging Face dataset `qwedsacf/competition_math` (split `train`) and keeps rows whose subject is `Precalculus` and level is `Level 5`. That dataset has no answer field, so the gold answer is the last `\boxed{...}` in the reference solution; rows without one are skipped. Each row's ID is `math_` plus the first 10 hex characters of the MD5 of the problem text, so IDs match across topologies.

The 100 frozen IDs are in [`benchmarks/math/math_eval_ids.json`](https://github.com/juyangbai/MAS-PromptBench/blob/main/benchmarks/math/math_eval_ids.json). No setup is needed beyond Hugging Face access.

!!! note "MATH-500 in the logs"
    Most MATH runners still say "MATH-500" in their `--help` text and log lines. All eight load the Precalculus, Level 5 slice described above.

## Run it

Run every command from the repository root. With no arguments, a runner solves \( 7!/5! \) (expected `42`) and prints the extracted answer, its score and the message trace:

```bash title="Smoke demo"
python -m topologies.single.math.langgraph_math
```

A batch needs `--batch`. Without `--out`, it only prints scores.

=== "Single"

    ```bash
    python -m topologies.single.math.langgraph_math --batch --limit 100 \
      --out results/topologies_baseline/single_math/predictions.jsonl
    ```

=== "Independent"

    ```bash
    python -m topologies.independent.math.langgraph_math --batch --limit 100 \
      --out results/topologies_baseline/independent_math/predictions.jsonl
    ```

The Independent runner runs `INDEPENDENT_N_AGENTS` replicas (default 4) and takes a majority vote, grouping answers that `is_equiv` treats as equal.

To score exactly the frozen set, pass its IDs to `--only`:

```bash title="Score the frozen eval IDs"
MANIFEST=benchmarks/math/math_eval_ids.json
IDS=$(python -c "import json,sys; print(*json.load(open(sys.argv[1]))['ids'])" $MANIFEST)
python -m topologies.single.math.langgraph_math --batch --only $IDS \
  --out results/topologies_baseline/single_math/predictions.jsonl
```

To optimize the single-agent prompt with [GEPA](../optimizers/gepa.md):

```bash title="GEPA on MATH"
cd optimizers/gepa
python -m real_runner_gepa.pilots.run_gepa_dataset --dataset math --topology single \
  --train-size 25 --val-size 25 --max-full-evals 5 --out results/gepa/single_math
```

## Flags

All eight MATH runners take the common flags and nothing else: `--batch`, `--limit N`, `--offset K`, `--only ID ...` (space-separated) and `--out PATH`. See [Command-Line Flags](../reference/cli.md).

## Related

- [GPQA-Diamond](gpqa.md) and [HotpotQA](hotpotqa.md), the other reasoning tasks.
- [Independent](../mas/independent.md) for the majority-vote ensemble.
- [Evaluation Protocol](../evaluation/protocol.md) for how the frozen IDs are used.
