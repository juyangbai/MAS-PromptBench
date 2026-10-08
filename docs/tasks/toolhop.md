# ToolHop

ToolHop asks multi-hop questions that need a chain of tool calls, where each tool's output feeds the next call. The tools are Python functions shipped with the dataset, so the runner executes dataset code and needs your explicit opt-in.
{ .lede }

<div class="facts" markdown>
<div><span>Domain</span>Tool calling</div>
<div><span>Metric</span>Answer accuracy</div>
<div><span>Launcher limit</span>100</div>
<div><span>Eval IDs</span>100</div>
<div><span>Data</span>bytedance-research/ToolHop</div>
</div>

## The task

Each row has a question, a gold answer, OpenAI-style tool schemas and the Python source of those tools. The agents run a standard tool-calling loop against your OpenAI-compatible endpoint: the model calls tools, the runner executes them locally and returns the results, and the loop repeats until the model answers without a tool call or reaches `TOOLHOP_MAX_TURNS` turns (default 9). Tool results longer than `TOOLHOP_TOOL_RESULT_CHAR_BUDGET` characters (default 6,000) are cut.

The user prompt fixes the answer format: dates as `YYYY-MM-DD`, names as `Firstname Lastname`, numbers as digits with no leading zeros. The output contract asks the final role to end with one short answer wrapped as `<answer>...</answer>`. If the final role stops without one, the runner makes one more short call that asks the model for its final answer in that format.

## How it is scored

The runner takes the text after the last `<answer>` tag in the final message, up to `</answer>`. If the message has no tag, it uses the whole message. Then:

1. If the gold answer parses as a Python literal (a number, a list), the instance is correct when the answer parses to an equal literal.
2. Otherwise, it is correct when the gold answer, lowercased, appears inside the answer, lowercased with commas removed. A trailing `.0` is dropped on both sides.
3. In either case, it also counts as correct when the gold answer appears in the last tool result before the final message.

Each line of `results.jsonl` has `correct` and `predicted_answer`; accuracy is the fraction of `correct` lines. The runner does not print a summary score.

## Data and setup

The runner downloads `data/ToolHop.json` from the Hugging Face dataset `bytedance-research/ToolHop`. Row IDs are the dataset's integer IDs; the 100 frozen IDs are `0` to `99`, listed in [`benchmarks/toolhop/toolhop_eval_ids.json`](https://github.com/juyangbai/MAS-PromptBench/blob/main/benchmarks/toolhop/toolhop_eval_ids.json).

Before it runs a tool, the runner reduces its source to function definitions, replaces the builtins with a restricted set, and allows imports only from a fixed list of modules (such as `math`, `datetime`, `re`, `json`, `numpy` and `sympy`). It still executes code from the dataset, so it refuses unless you opt in:

```bash title="Allow dataset tool execution"
export TOOLHOP_ALLOW_DATASET_EXEC=1
```

Without it, the runner raises an error on every instance and scores each one as wrong. `scripts/run_topologies.sh` sets this variable to `1` unless you set it yourself.

## Run it

Run every command from the repository root. To check the download and the dataset's structure without calling the model or executing any tool, use `--smoke-dataset`. It checks the first `--limit` rows (5 by default):

```bash title="Validate the dataset only"
python -m topologies.single.toolhop.langgraph_toolhop --smoke-dataset --limit 100
```

Runs are always batches; `--batch` is accepted and ignored. With no arguments, a runner solves the first 5 rows.

=== "Single"

    ```bash
    python -m topologies.single.toolhop.langgraph_toolhop --limit 100 \
      --out-dir results/topologies_baseline/single_toolhop
    ```

=== "Decentralized (OpenAI SDK)"

    ```bash
    python -m topologies.decentralized.openai.toolhop.openai_toolhop --limit 100 \
      --out-dir results/topologies_baseline/decentralized_openai_toolhop
    ```

To score exactly the frozen set, pass each ID as its own `--only` flag. `--only` overrides `--limit`:

```bash title="Score the frozen eval IDs"
MANIFEST=benchmarks/toolhop/toolhop_eval_ids.json
IDS=$(python -c "import json,sys; print(*json.load(open(sys.argv[1]))['ids'])" $MANIFEST)
python -m topologies.single.toolhop.langgraph_toolhop $(printf -- '--only %s ' $IDS) \
  --out-dir results/topologies_baseline/single_toolhop
```

The runner appends to `predictions.jsonl` and `results.jsonl` in `--out-dir` and writes one trace per instance under `traces/`. Use a fresh `--out-dir` for each run. Without `--out-dir`, output goes to `results/toolhop/<style>/`, for example `results/toolhop/single_langgraph/`.

To optimize the single-agent prompt with [GEPA](../optimizers/gepa.md):

```bash title="GEPA on ToolHop"
cd optimizers/gepa
python -m real_runner_gepa.pilots.run_gepa_dataset --dataset toolhop --topology single \
  --train-size 25 --val-size 25 --max-full-evals 5 --out results/gepa/single_toolhop
```

For ToolHop, GEPA always keeps the 100 frozen IDs out of its training and validation splits. ToolHop also has [communication-protocol](../mas/communication-protocols.md) and [team-size](../mas/team-sizes.md) variants that GEPA can target, such as `--topology decentralized_r8`.

## Flags

| Flag | Default | Effect |
| --- | --- | --- |
| `--limit N` | `5` | Number of rows; ignored when `--only` is given. |
| `--offset K` | `0` | Skip the first K rows. |
| `--only ID` | none | Keep this row ID; repeat the flag for more. |
| `--out-dir DIR` | `results/toolhop/<style>/` | Where output files go. |
| `--smoke-dataset` | off | Validate the dataset and exit; no model calls, no tool execution. |
| `--batch` | off | Accepted for uniformity; has no effect. |

## Related

- [BFCL](bfcl.md) and [API-Bank](api-bank.md), the other tool-calling tasks.
- [Workflow Topologies](../mas/topologies.md) for what each runner variant does.
- [Environment Variables](../reference/environment.md) for the other `TOOLHOP_*` settings.
