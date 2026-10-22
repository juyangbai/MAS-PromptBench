# MATH

MATH is a set of competition math problems with a single final answer. MAS-PromptBench uses its hardest precalculus slice and scores the last boxed answer with Hendrycks' equivalence check.
{ .lede }

<div class="facts" markdown>
<div><span>Domain</span>Reasoning</div>
<div><span>Metric</span>Accuracy (is_equiv)</div>
<div><span>Eval IDs</span>100</div>
<div><span>Train / val</span>150 / 50</div>
<div><span>Data</span>qwedsacf/competition_math</div>
</div>

## The task

Each instance is one problem statement, passed to the agents as is. Agents can call `calculator`, which evaluates a numeric Python expression with the usual math functions.

The answering agent's output contract asks for a `\boxed{...}` line at the start, at most 12 short reasoning lines, and the same `\boxed{...}` line at the end. The scorer reads only the last boxed expression. The Single and Independent runners also add a format note with examples such as `\boxed{42}` and `\boxed{\frac{1}{2}}`.

## How it is scored

1. **Extract** the content of the last `\boxed{...}` in the final message, counting braces so nested expressions like `\boxed{\frac{1}{2}}` come out whole. No boxed answer means the instance is wrong.
2. **Compare** it to the gold answer with `is_equiv`, a port of Hendrycks' `math_equivalence.py`. Both strings are normalized and then compared for equality.

Normalization removes spaces, `\left` and `\right`, degree marks, `\$`, `\%` and trailing units; rewrites `tfrac` and `dfrac` as `frac`; repairs shorthand like `\frac12` and `\sqrt3`; turns `0.5` into `\frac{1}{2}` and simple `a/b` into `\frac{a}{b}`; and drops a short left-hand side such as `x =`.

The batch reports `EM`, the fraction of instances judged equivalent, over all instances (a missing answer counts as wrong) and over those with an extracted answer. Independent and Decentralized teams group their agents' answers into buckets that `is_equiv` treats as equal and submit the largest bucket, ties going to the lowest agent.

## Data

The runner loads the Hugging Face dataset `qwedsacf/competition_math` (split `train`) and keeps the 312 rows whose subject is `Precalculus` and level is `Level 5`. That dataset has no answer field, so the gold answer is the last `\boxed{...}` in the reference solution; rows without one are skipped. Each row's ID is `math_` plus the first 10 hex characters of the MD5 of the problem text, so IDs match across topologies.

The 100 eval IDs are in [`benchmarks/math/math_eval_ids.json`](https://github.com/juyangbai/MAS-PromptBench/blob/main/benchmarks/math/math_eval_ids.json); they are the first 100 problems of the slice, so `--limit 100` scores exactly that set. No setup is needed beyond Hugging Face access.

## Run it

Run every command from the repository root. With no arguments, a runner solves \( 7!/5! \) (expected `42`) and prints the extracted answer, its score and the message trace:

```bash title="Smoke demo"
python -m topologies.single.math.langgraph_math
```

A batch needs `--batch`:

=== "Single"

    ```bash
    python -m topologies.single.math.langgraph_math --batch --limit 100 \
      --out-dir results/topologies_baseline/single_math
    ```

=== "Independent"

    ```bash
    python -m topologies.independent.math.langgraph_math --batch --limit 100 \
      --out-dir results/topologies_baseline/independent_math
    ```

The Independent runner runs `INDEPENDENT_N_AGENTS` replicas (default 4) and submits the largest equivalence bucket.

## Optimize it

Every optimizer runs through the same protocol command; change `--method` to switch. For example, MAPRO on the Independent team:

```bash title="MAPRO on Independent · MATH"
python -m optimizers.protocol.run --method mapro --dataset math --topology independent \
  --model qwen --seed 0 --out runs/mapro/math/independent/qwen/0
```

Combinations outside the experiment grid need `--allow-any-cell`; see [Optimize a task](index.md#optimize-a-task).

## Flags

All eight MATH runners take the common flags and nothing else: `--batch`, `--limit N`, `--offset K`, `--only ID ...`, `--out-dir DIR` and `--out PATH`. See [Command-Line Flags](../reference/cli.md).

## Related

- [GPQA-Diamond](gpqa.md) and [HotpotQA](hotpotqa.md), the other reasoning tasks.
- [Independent](../mas/independent.md) for the majority-vote ensemble.
- [Evaluation Protocol](../evaluation/protocol.md) for how the eval IDs and splits are used.
