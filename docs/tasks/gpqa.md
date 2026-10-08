# GPQA-Diamond

GPQA-Diamond is a set of graduate-level science questions with four answer options. Agents must commit to one letter, and the runner scores it by exact letter match.
{ .lede }

<div class="facts" markdown>
<div><span>Domain</span>Reasoning</div>
<div><span>Metric</span>Accuracy (letter match)</div>
<div><span>Launcher limit</span>100</div>
<div><span>Eval IDs</span>100</div>
<div><span>Data</span>Idavidrein/gpqa (gated)</div>
</div>

## The task

Each instance is one multiple-choice question. The runner shows the agents the question followed by four lines, `A)` to `D)`. Agents can use one tool, `calculator`, which evaluates a numeric Python expression with the usual math functions (`sqrt`, `log`, `exp`, trigonometry, `pi`, `e`).

The final agent must end with exactly one line of the form `Answer: A` (or B, C, D). This is the GPQA output contract from [`topologies/output_contracts.py`](https://github.com/juyangbai/MAS-PromptBench/blob/main/topologies/output_contracts.py). The runner adds it to the final role's system prompt when it loads the prompt file, so it stays in place whatever the editable role prompt says.

## How it is scored

The runner extracts a letter from the final message. It first strips markdown emphasis, so `**Answer:** B` reads as `Answer: B`. It then tries three patterns in order:

1. `Answer: X` or `Final answer: X`
2. `option X` or `choice X`
3. a bare letter on its own line

The first pattern that matches wins, and within it the last match counts, since models often revise a letter mid-reasoning. An instance is correct when that letter equals the gold letter.

The batch summary reports `accuracy` (correct over all instances, so a missing letter counts as wrong) and `extracted_acc` (correct over instances where a letter was found).

## Data

The runner loads the `gpqa_diamond` config of the Hugging Face dataset `Idavidrein/gpqa` (split `train`). The dataset is gated: accept its terms on Hugging Face and log in with a Hugging Face token before the first batch.

The raw rows store the correct answer and three incorrect answers in separate fields. To keep the correct letter from always being A, the runner shuffles the four options per row with a seeded generator (`--shuffle-seed`, default `0`). The same seed gives the same option order in every topology. Rows with a missing option are skipped.

GPQA has no ID field, so the runner builds one from the question text: `gpqa_` plus the first 10 hex characters of its MD5 hash. The 100 frozen IDs are in [`benchmarks/gpqa/gpqa_eval_ids.json`](https://github.com/juyangbai/MAS-PromptBench/blob/main/benchmarks/gpqa/gpqa_eval_ids.json).

## Run it

Run every command from the repository root. With no arguments, a runner answers one built-in physics question (expected answer A) and prints the extracted letter and the full message trace:

```bash title="Smoke demo"
python -m topologies.single.gpqa.langgraph_gpqa
```

A batch needs `--batch`. Without `--out`, it only prints scores.

=== "Single"

    ```bash
    python -m topologies.single.gpqa.langgraph_gpqa --batch --limit 100 \
      --out results/topologies_baseline/single_gpqa/predictions.jsonl
    ```

=== "Decentralized (LangGraph)"

    ```bash
    python -m topologies.decentralized.langgraph.gpqa.langgraph_gpqa --batch --limit 100 \
      --out results/topologies_baseline/decentralized_langgraph_gpqa/predictions.jsonl
    ```

To score exactly the frozen set, pass its IDs to `--only`:

```bash title="Score the frozen eval IDs"
MANIFEST=benchmarks/gpqa/gpqa_eval_ids.json
IDS=$(python -c "import json,sys; print(*json.load(open(sys.argv[1]))['ids'])" $MANIFEST)
python -m topologies.single.gpqa.langgraph_gpqa --batch --only $IDS \
  --out results/topologies_baseline/single_gpqa/predictions.jsonl
```

Each line of the output file holds the question, the shuffled choices, `correct_letter`, `predicted_letter`, `correct`, the raw final message, latency and token counts.

To optimize the single-agent prompt with [GEPA](../optimizers/gepa.md):

```bash title="GEPA on GPQA"
cd optimizers/gepa
python -m real_runner_gepa.pilots.run_gepa_dataset --dataset gpqa --topology single \
  --train-size 25 --val-size 25 --max-full-evals 5 --out results/gepa/single_gpqa
```

## Flags

All eight GPQA runners share the same interface:

| Flag | Default | Effect |
| --- | --- | --- |
| `--batch` | off | Run the dataset instead of the smoke demo. |
| `--limit N` | all rows | Stop after N rows. |
| `--offset K` | `0` | Skip the first K rows. |
| `--only ID ...` | none | Keep only these row IDs (space-separated). |
| `--shuffle-seed S` | `0` | Seed for the per-row option shuffle. |
| `--out PATH` | none | Write one JSON line per instance. |

## Related

- [HotpotQA](hotpotqa.md) and [MATH](math.md), the other reasoning tasks.
- [Workflow Topologies](../mas/topologies.md) for what each runner variant does.
- [Evaluation Protocol](../evaluation/protocol.md) for how the frozen IDs are used.
