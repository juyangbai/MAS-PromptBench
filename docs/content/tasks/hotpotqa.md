# HotpotQA

HotpotQA asks multi-hop questions whose answer needs facts from more than one Wikipedia article. Agents search live Wikipedia, then give a short answer scored with the official exact-match and F1 metrics.
{ .lede }

<div class="facts" markdown>
<div><span>Domain</span>Reasoning</div>
<div><span>Metric</span>Exact match and F1</div>
<div><span>Eval IDs</span>100</div>
<div><span>Train / val</span>150 / 50</div>
<div><span>Data</span>hotpot_qa (distractor)</div>
</div>

## The task

Each instance is one question, such as "Were Scott Derrickson and Ed Wood of the same nationality?". The agents start from the question alone. They do not get the dataset's context paragraphs; they retrieve evidence themselves with two tools:

| Tool | What it returns |
| --- | --- |
| `wikipedia_search(query)` | Titles and two-sentence summaries of the top 3 matching articles. |
| `wikipedia_page(title)` | The article text for an exact title, cut to 4,000 characters. |

The tools use the `wikipedia` Python client, so runs need network access to Wikipedia.

The answering agent must end with one line `Answer: <short-form>`. The Single, Independent and Decentralized runners also append a format note to the prompt: `yes` or `no` for yes/no questions, a bare year for "when" questions, a full name for "who" questions, a place name for "where" questions, and no explanation on the answer line.

## How it is scored

The runner takes the last `Answer: X` match in the final message (case-insensitive, markdown bold allowed). If there is none, it falls back to the last non-empty line.

Both metrics come from the official `hotpot_evaluate_v1.py`, kept as published:

- **Normalization**: lowercase, remove punctuation, remove the articles a, an and the, collapse whitespace.
- **Exact match (EM)**: 1 if the normalized prediction equals the normalized gold answer, else 0.
- **F1**: token-level F1 between the normalized strings. For yes, no and noanswer there is no partial credit: a mismatch scores 0.

The batch summary averages EM and F1 over all instances, counting a missing answer as 0, and also reports both over extracted answers only. Each record has `em`, `f1`, `precision`, `recall`, the question `type` (comparison or bridge) and `level`. Independent and Decentralized teams submit the most common normalized answer of their agents. The optimizers score HotpotQA by exact match only.

## Data

The runner loads the Hugging Face dataset `hotpot_qa`, config `distractor`, split `validation`. It keeps only the ID, question, answer, type and level of each row; the distractor paragraphs are never shown to agents. Rows use HotpotQA's own string IDs.

The 100 eval IDs are in [`benchmarks/hotpotqa/hotpotqa_eval_ids.json`](https://github.com/juyangbai/MAS-PromptBench/blob/main/benchmarks/hotpotqa/hotpotqa_eval_ids.json); they are the first 100 rows, so `--limit 100` scores exactly that set. The `wikipedia` client is already in `environment.yml`; nothing else needs installing.

## Run it

Run every command from the repository root. With no arguments, a runner answers the Derrickson and Wood question (expected `yes`) and prints the extracted answer, EM, F1 and the message trace:

```bash title="Smoke demo"
python -m topologies.single.hotpotqa.langgraph_hotpotqa
```

A batch needs `--batch`:

=== "Single"

    ```bash
    python -m topologies.single.hotpotqa.langgraph_hotpotqa --batch --limit 100 \
      --out-dir results/topologies_baseline/single_hotpotqa
    ```

=== "Centralized (LangGraph)"

    ```bash
    python -m topologies.centralized.langgraph.hotpotqa.langgraph_hotpotqa \
      --batch --limit 100 \
      --out-dir results/topologies_baseline/centralized_langgraph_hotpotqa
    ```

HotpotQA also has [communication-protocol](../mas/communication-protocols.md) runners under `communications/` and [team-size](../mas/team-sizes.md) runners under `teamsizes/`.

## Optimize it

HotpotQA is one of three tasks, with LiveCodeBench and BFCL, whose experiment grid has all eight optimizers on every multi-agent LangGraph team, plus the protocol and team-size variants for GEPA, MIPRO, MAPRO and MASPO. Swap `--method` for any key. For example, HiveMind on the Centralized team:

```bash title="HiveMind on Centralized · HotpotQA"
python -m optimizers.protocol.run --method hivemind --dataset hotpotqa --topology centralized \
  --model qwen --seed 0 --out runs/hivemind/hotpotqa/centralized/qwen/0
```

Add `--team-size 8` or `--communication structured_soft` to target a variant. See [Run an Optimizer](../optimizers/running.md).

## Flags

All eight HotpotQA runners take the common flags and nothing else: `--batch`, `--limit N`, `--offset K`, `--only ID ...`, `--out-dir DIR` and `--out PATH`. See [Command-Line Flags](../reference/cli.md).

## Related

- [GPQA-Diamond](gpqa.md) and [MATH](math.md), the other reasoning tasks.
- [Workflow Topologies](../mas/topologies.md) for what each runner variant does.
- [Evaluation Protocol](../evaluation/protocol.md) for how the eval IDs and splits are used.
