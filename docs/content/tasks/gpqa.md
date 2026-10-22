# GPQA-Diamond

GPQA-Diamond is a set of graduate-level science questions with four answer options. Agents must commit to one letter, and the runner scores it by exact letter match.
{ .lede }

<div class="facts" markdown>
<div><span>Domain</span>Reasoning</div>
<div><span>Metric</span>Accuracy (letter match)</div>
<div><span>Eval IDs</span>100</div>
<div><span>Train / val</span>48 / 50</div>
<div><span>Data</span>Idavidrein/gpqa (gated)</div>
</div>

## The task

Each instance is one multiple-choice question. The runner shows the agents the question followed by four lines, `A)` to `D)`. Agents can use one tool, `calculator`, which evaluates a numeric Python expression with the usual math functions (`sqrt`, `log`, `exp`, trigonometry, `pi`, `e`).

The answering agent must end with exactly one line of the form `Answer: A` (or B, C, D). This is the GPQA output contract from [`core/output_contracts.py`](https://github.com/juyangbai/MAS-PromptBench/blob/main/core/output_contracts.py). The runner adds it to the answering role's system prompt when it loads the prompt file, so it stays in place whatever the editable role prompt says.

## How it is scored

The runner extracts a letter from the final message. It first strips markdown emphasis, so `**Answer:** B` reads as `Answer: B`. It then tries three patterns in order:

1. `Answer: X` or `Final answer: X`
2. `option X` or `choice X`
3. a bare letter on its own line

The first pattern that matches wins, and within it the last match counts, since models often revise a letter mid-reasoning. An instance is correct when that letter equals the gold letter.

The batch summary reports `accuracy` (correct over all instances, so a missing letter counts as wrong) and `extracted_acc` (correct over instances where a letter was found). Independent and Decentralized teams submit the most common letter of their agents, ties going to the lowest agent.

## Data

The runner loads the `gpqa_diamond` config of the Hugging Face dataset `Idavidrein/gpqa` (split `train`, 198 questions). The dataset is gated: accept its terms on Hugging Face and log in with a Hugging Face token before the first batch.

The raw rows store the correct answer and three incorrect answers in separate fields. To keep the correct letter from always being A, the runner shuffles the four options per row with a seeded generator (`--shuffle-seed`, default `0`). The same seed gives the same option order in every topology. Rows with a missing option are skipped.

GPQA has no ID field, so the runner builds one from the question text: `gpqa_` plus the first 10 hex characters of its MD5 hash. The 100 eval IDs are in [`benchmarks/gpqa/gpqa_eval_ids.json`](https://github.com/juyangbai/MAS-PromptBench/blob/main/benchmarks/gpqa/gpqa_eval_ids.json); they are the first 100 questions, so `--limit 100` scores exactly that set. Only 98 questions remain for the optimizers, hence the smaller train split.

## Run it

Run every command from the repository root. With no arguments, a runner answers one built-in physics question (expected answer A) and prints the extracted letter and the full message trace:

```bash title="Smoke demo"
python -m topologies.single.gpqa.langgraph_gpqa
```

A batch needs `--batch`:

=== "Single"

    ```bash
    python -m topologies.single.gpqa.langgraph_gpqa --batch --limit 100 \
      --out-dir results/topologies_baseline/single_gpqa
    ```

=== "Decentralized (LangGraph)"

    ```bash
    python -m topologies.decentralized.langgraph.gpqa.langgraph_gpqa --batch --limit 100 \
      --out-dir results/topologies_baseline/decentralized_langgraph_gpqa
    ```

Each line of `predictions.jsonl` holds the question, the shuffled choices, `correct_letter`, `predicted_letter`, `correct`, the raw final message, latency and token counts.

## Optimize it

Every optimizer runs through the same protocol command; change `--method` to switch. For example, MIPRO on the single agent:

```bash title="MIPRO on Single · GPQA"
python -m optimizers.protocol.run --method mipro --dataset gpqa --topology single \
  --model qwen --seed 0 --out runs/mipro/gpqa/single/qwen/0
```

Combinations outside the experiment grid need `--allow-any-cell`; see [Optimize a task](index.md#optimize-a-task). [Run an Optimizer](../optimizers/running.md) covers endpoints, budgets and outputs.

## Flags

All eight GPQA runners take the common flags (`--batch`, `--limit`, `--offset`, `--only`, `--out-dir`, `--out`) and one more:

| Flag | Default | Effect |
| --- | --- | --- |
| `--shuffle-seed SEED` | `0` | Seed for the per-row option shuffle. |

## Related

- [HotpotQA](hotpotqa.md) and [MATH](math.md), the other reasoning tasks.
- [Workflow Topologies](../mas/topologies.md) for what each runner variant does.
- [Evaluation Protocol](../evaluation/protocol.md) for how the eval IDs and splits are used.
