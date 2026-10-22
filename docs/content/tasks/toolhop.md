# ToolHop

ToolHop asks multi-hop questions that need a chain of tool calls, where each tool's output feeds the next call. The tools are Python functions shipped with the dataset, so the runner executes dataset code and needs your explicit opt-in.
{ .lede }

<div class="facts" markdown>
<div><span>Domain</span>Tool calling</div>
<div><span>Metric</span>Answer accuracy</div>
<div><span>Eval IDs</span>100</div>
<div><span>Train / val</span>150 / 50</div>
<div><span>Data</span>bytedance-research/ToolHop</div>
</div>

## The task

Each row has a question, a gold answer, OpenAI-style tool schemas and the Python source of those tools. An agent runs a standard tool-calling loop against your OpenAI-compatible endpoint: the model calls tools, the runner executes them locally and returns the results, and the loop repeats until the model answers without a tool call or reaches `TOOLHOP_MAX_TURNS` turns (default 9). Tool results longer than `TOOLHOP_TOOL_RESULT_CHAR_BUDGET` characters (default 6,000) are cut.

The user prompt fixes the answer format: dates as `YYYY-MM-DD`, names as `Firstname Lastname`, numbers as digits with no leading zeros. The output contract asks the answering role to end with one short answer wrapped as `<answer>...</answer>`. If an answering role stops without one, the runner makes one more short call that asks the model for its final answer in that format.

## How it is scored

The runner takes the text after the last `<answer>` tag in the final message, up to `</answer>`. If the message has no tag, it uses the whole message. Then:

1. If the gold answer parses as a Python literal (a number, a list), the instance is correct when the answer parses to an equal literal.
2. Otherwise, it is correct when the gold answer, lowercased, appears inside the answer, lowercased with commas removed. A trailing `.0` is dropped on both sides.
3. In either case, it also counts as correct when the gold answer appears in the last tool result before the final message.

Each line of `results.jsonl` has `correct` and `predicted_answer`; accuracy is the fraction of `correct` lines. Independent and Decentralized teams submit their agents' most common answer, ties going to the earliest agent.

## Data and setup

The runner downloads `data/ToolHop.json` from the Hugging Face dataset `bytedance-research/ToolHop`. Row IDs are the dataset's integer IDs; the 100 eval IDs are `0` to `99`, listed in [`benchmarks/toolhop/toolhop_eval_ids.json`](https://github.com/juyangbai/MAS-PromptBench/blob/main/benchmarks/toolhop/toolhop_eval_ids.json). They are the first 100 rows, so `--limit 100` scores exactly that set.

Before it runs a tool, the runner reduces its source to function definitions, replaces the builtins with a restricted set, and allows imports only from a fixed list of modules (such as `math`, `datetime`, `re`, `json`, `numpy` and `sympy`). It still executes code from the dataset, so it refuses unless you opt in:

```bash title="Allow dataset tool execution"
export TOOLHOP_ALLOW_DATASET_EXEC=1
```

Without it, every instance fails with an error that names the variable and counts as wrong. The sweep scripts set it to `1` unless you set it yourself.

## Run it

Run every command from the repository root. To check the download and the dataset's structure without calling the model or executing any tool, use `--smoke-dataset`. It checks the first `--limit` rows (5 by default):

```bash title="Validate the dataset only"
python -m topologies.single.toolhop.langgraph_toolhop --smoke-dataset --limit 100
```

There is no smoke demo; with no arguments a runner solves the first 5 rows, and `--batch` changes nothing.

=== "Single"

    ```bash
    python -m topologies.single.toolhop.langgraph_toolhop --limit 100 \
      --out-dir results/topologies_baseline/single_toolhop
    ```

=== "Decentralized (Agents SDK)"

    ```bash
    python -m topologies.decentralized.openai_agents.toolhop.openai_agents_toolhop --limit 100 \
      --out-dir results/topologies_baseline/decentralized_openai_agents_toolhop
    ```

The runner writes `predictions.jsonl` and `results.jsonl` to `--out-dir`, emptying both first, and one trace per instance under `traces/`. Without `--out-dir`, output goes to `results/toolhop/<style>/`, for example `results/toolhop/single_langgraph/`. ToolHop also has [communication-protocol](../mas/communication-protocols.md) and [team-size](../mas/team-sizes.md) runners.

## Optimize it

Every optimizer runs through the same protocol command; change `--method` to switch. For example, GEPA on the Centralized team:

```bash title="GEPA on Centralized · ToolHop"
export TOOLHOP_ALLOW_DATASET_EXEC=1
python -m optimizers.protocol.run --method gepa --dataset toolhop --topology centralized \
  --model qwen --seed 0 --out runs/gepa/toolhop/centralized/qwen/0
```

Combinations outside the experiment grid need `--allow-any-cell`; see [Optimize a task](index.md#optimize-a-task).

## Flags

Beyond the common flags (`--batch`, `--limit`, `--offset`, `--only`, `--out-dir`, `--out`):

| Flag | Default | Effect |
| --- | --- | --- |
| `--smoke-dataset` | off | Validate the dataset and exit; no model calls, no tool execution. |

`--limit` defaults to 5.

## Related

- [BFCL](bfcl.md) and [API-Bank](api-bank.md), the other tool-calling tasks.
- [Workflow Topologies](../mas/topologies.md) for what each runner variant does.
- [Environment Variables](../reference/environment.md) for the other `TOOLHOP_*` settings.
