# BFCL

The Berkeley Function Calling Leaderboard (BFCL) gives the agents one or more function schemas and a user request. They must emit the right call or calls, and the official BFCL AST checker decides whether each call matches.
{ .lede }

<div class="facts" markdown>
<div><span>Domain</span>Tool calling</div>
<div><span>Metric</span>AST match (valid rate)</div>
<div><span>Eval IDs</span>100</div>
<div><span>Train / val</span>150 / 50</div>
<div><span>Data</span>gorilla-llm/Berkeley-Function-Calling-Leaderboard</div>
</div>

## The task

MAS-PromptBench uses the four single-turn BFCL v3 categories that the AST checker can score. A run covers one category, chosen with `--category`:

| Category | What the request needs |
| --- | --- |
| `simple` | One call to the one given function. |
| `multiple` | One call, picking the right function from several. |
| `parallel` | Several calls to one function. |
| `parallel_multiple` | Several calls across several functions. |

How the agents answer depends on the runner:

- **Single and Independent** use native tool calling. Each schema becomes a LangChain tool whose body returns an empty string, and an agent's answer is the tool calls of its first model message that has any. Later turns are ignored.
- **Sequential, Centralized and Decentralized** follow the output contract: the answering role ends with one fenced `json` block holding a non-empty list of calls in canonical form, `[{"function_name": {"arg": value}}]`. The runner parses the last fenced block that is a list of objects.

## How it is scored

The runner passes the canonical calls, with the schemas, the gold answers and the category, to `ast_checker` from the `bfcl-eval` package (Python language). The checker verifies function names and required parameters, and that each argument's type and value appear among the possible answers. For parallel categories, call order does not matter.

Each line of `results.jsonl` carries `valid` (true when the checker accepts the output) and, on failure, the checker's `error_type`. The runner prints `valid N/M` at the end; the metric is the valid rate. Independent and Decentralized teams submit their agents' most common canonical call list, and only that list is checked.

The runner registers your `MODEL_ID` in BFCL's model table before scoring, so dotted function names such as `math.factorial` are checked as written.

## Data

Each category is two files in the Hugging Face dataset `gorilla-llm/Berkeley-Function-Calling-Leaderboard`: `BFCL_v3_<category>.json` for the requests and `possible_answer/BFCL_v3_<category>.json` for the gold answers. The runner downloads them with `hf_hub_download`; no other setup is needed.

The 100 eval IDs in [`benchmarks/bfcl/bfcl_eval_ids.json`](https://github.com/juyangbai/MAS-PromptBench/blob/main/benchmarks/bfcl/bfcl_eval_ids.json) are a seed-0 stratified sample: 40 `simple`, 20 `multiple`, 20 `parallel` and 20 `parallel_multiple`. They are not a slice of any category, so a plain `--limit` run does not reproduce them.

## Run it

Run every command from the repository root. There is no smoke demo; with no arguments a runner scores the first 5 `simple` instances, and `--batch` changes nothing.

=== "Single"

    ```bash
    python -m topologies.single.bfcl.langgraph_bfcl --category simple --limit 100 \
      --out-dir results/topologies_baseline/single_bfcl/simple
    ```

=== "Sequential (CrewAI)"

    ```bash
    python -m topologies.sequential.crewai.bfcl.crewai_bfcl --category simple --limit 100 \
      --out-dir results/topologies_baseline/sequential_crewai_bfcl/simple
    ```

To score the eval IDs, run each category with all 100 IDs. `--only` keeps only the IDs found in the chosen category's file, so each pass picks up its own share:

```bash title="Score the eval IDs"
MANIFEST=benchmarks/bfcl/bfcl_eval_ids.json
IDS=$(python -c "import json,sys; print(*json.load(open(sys.argv[1]))['ids'])" $MANIFEST)
for cat in simple multiple parallel parallel_multiple; do
  python -m topologies.single.bfcl.langgraph_bfcl --category $cat --only $IDS \
    --out-dir results/topologies_baseline/single_bfcl/$cat
done
```

!!! warning "One output folder per category"
    A BFCL run writes `predictions.jsonl` and `results.jsonl` to `--out-dir`, emptying both first, plus one trace per instance in `traces/<id>.txt`. Give each category its own folder, as above, or the next category overwrites the last. `scripts/run_topologies.sh` does the same.

## Optimize it

BFCL is one of the three tasks whose experiment grid has all eight optimizers on every multi-agent LangGraph team. Swap `--method` for any key. For example, MASPOB on the Decentralized debate:

```bash title="MASPOB on Decentralized · BFCL"
python -m optimizers.protocol.run --method maspob --dataset bfcl --topology decentralized \
  --model qwen --seed 0 --out runs/maspob/bfcl/decentralized/qwen/0
```

The protocol draws its train and validation rows from all four categories and scores them with the same AST checker. For the CrewAI pipeline, pass `--topology sequential --framework crewai`. See [Run an Optimizer](../optimizers/running.md).

## Flags

Beyond the common flags (`--batch`, `--limit`, `--offset`, `--only`, `--out-dir`, `--out`):

| Flag | Default | Effect |
| --- | --- | --- |
| `--category` | `simple` | One of `simple`, `multiple`, `parallel`, `parallel_multiple`. |

`--limit` defaults to 5. Without `--out-dir`, output goes to a `results/bfcl*` folder that differs per topology.

## Related

- [ToolHop](toolhop.md) and [API-Bank](api-bank.md), the other tool-calling tasks.
- [Sequential](../mas/sequential.md) for the CrewAI pipeline.
- [Evaluation Protocol](../evaluation/protocol.md) for how the eval IDs and splits are used.
