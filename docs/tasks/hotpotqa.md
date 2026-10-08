# HotpotQA

HotpotQA asks multi-hop questions whose answer needs facts from more than one Wikipedia article. Agents search live Wikipedia, then give a short answer scored with the official exact-match and F1 metrics.
{ .lede }

<div class="facts" markdown>
<div><span>Domain</span>Reasoning</div>
<div><span>Metric</span>Exact match and F1</div>
<div><span>Launcher limit</span>100</div>
<div><span>Eval IDs</span>100</div>
<div><span>Data</span>hotpot_qa (distractor)</div>
</div>

## The task

Each instance is one question, such as "Were Scott Derrickson and Ed Wood of the same nationality?". The agents start from the question alone. They do not get the dataset's context paragraphs; they retrieve evidence themselves with two tools:

| Tool | What it returns |
| --- | --- |
| `wikipedia_search(query)` | Titles and two-sentence summaries of the top 3 matching articles. |
| `wikipedia_page(title)` | The article text for an exact title, cut to 4,000 characters. |

All eight HotpotQA runners use these tools through the `wikipedia` Python client, so runs need network access to Wikipedia.

The final agent must end with one line `Answer: <short-form>`. The single-agent runner also appends a format note to its prompt: `yes` or `no` for yes/no questions, a bare year for "when" questions, a full name for "who" questions, a place name for "where" questions, and no explanation on the answer line.

## How it is scored

The runner takes the last `Answer: X` match in the final message (case-insensitive, markdown bold allowed). If there is none, it falls back to the last non-empty line.

Both metrics come from the official `hotpot_evaluate_v1.py`, reimplemented in the runner:

- **Normalization**: lowercase, remove punctuation, remove the articles a, an and the, collapse whitespace.
- **Exact match (EM)**: 1 if the normalized prediction equals the normalized gold answer, else 0.
- **F1**: token-level F1 between the normalized strings. For yes, no and noanswer there is no partial credit: a mismatch scores 0.

The batch summary averages EM and F1 over all instances, counting a missing answer as 0, and also reports both over extracted answers only. Each output line has `em`, `f1`, `precision`, `recall`, the question `type` (comparison or bridge) and `level`. When [GEPA](../optimizers/gepa.md) optimizes a HotpotQA prompt, its metric is exact match.

## Data

The runner loads the Hugging Face dataset `hotpot_qa`, config `distractor`, split `validation`. It keeps only the ID, question, answer, type and level of each row; the distractor paragraphs are never shown to agents. Rows use HotpotQA's own string IDs.

The 100 frozen IDs are in [`benchmarks/hotpotqa/hotpotqa_eval_ids.json`](https://github.com/juyangbai/MAS-PromptBench/blob/main/benchmarks/hotpotqa/hotpotqa_eval_ids.json). The `wikipedia` client is already in `environment.yml`; nothing else needs installing.

## Run it

Run every command from the repository root. With no arguments, a runner answers the Derrickson and Wood question (expected `yes`) and prints the extracted answer, EM, F1 and the message trace:

```bash title="Smoke demo"
python -m topologies.single.hotpotqa.langgraph_hotpotqa
```

A batch needs `--batch`. Without `--out`, it only prints scores.

=== "Single"

    ```bash
    python -m topologies.single.hotpotqa.langgraph_hotpotqa --batch --limit 100 \
      --out results/topologies_baseline/single_hotpotqa/predictions.jsonl
    ```

=== "Centralized (LangGraph)"

    ```bash
    python -m topologies.centralized.langgraph.hotpotqa.langgraph_hotpotqa \
      --batch --limit 100 \
      --out results/topologies_baseline/centralized_langgraph_hotpotqa/predictions.jsonl
    ```

To score exactly the frozen set, pass its IDs to `--only`:

```bash title="Score the frozen eval IDs"
MANIFEST=benchmarks/hotpotqa/hotpotqa_eval_ids.json
IDS=$(python -c "import json,sys; print(*json.load(open(sys.argv[1]))['ids'])" $MANIFEST)
python -m topologies.single.hotpotqa.langgraph_hotpotqa --batch --only $IDS \
  --out results/topologies_baseline/single_hotpotqa/predictions.jsonl
```

To optimize the single-agent prompt with GEPA:

```bash title="GEPA on HotpotQA"
cd optimizers/gepa
python -m real_runner_gepa.pilots.run_gepa_dataset --dataset hotpotqa --topology single \
  --train-size 25 --val-size 25 --max-full-evals 5 --out results/gepa/single_hotpotqa
```

HotpotQA is one of the tasks with extra runner families: [communication-protocol](../mas/communication-protocols.md) runners under `communications/` and [team-size](../mas/team-sizes.md) runners under `teamsizes/`. GEPA can target those variants too, for example `--topology centralized_r4` or `--topology sequential_communications_structured_soft`.

## Flags

All eight HotpotQA runners take the common flags and nothing else: `--batch`, `--limit N`, `--offset K`, `--only ID ...` (space-separated HotpotQA IDs) and `--out PATH`. See [Command-Line Flags](../reference/cli.md) for the full list.

## Related

- [GPQA-Diamond](gpqa.md) and [MATH](math.md), the other reasoning tasks.
- [Workflow Topologies](../mas/topologies.md) for what each runner variant does.
- [Evaluation Protocol](../evaluation/protocol.md) for how the frozen IDs are used.
