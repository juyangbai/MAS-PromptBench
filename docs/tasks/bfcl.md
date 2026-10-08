# BFCL

The Berkeley Function Calling Leaderboard (BFCL) gives the agents one or more function schemas and a user request. They must emit the right call or calls, and the official BFCL AST checker decides whether each call matches.
{ .lede }

<div class="facts" markdown>
<div><span>Domain</span>Tool calling</div>
<div><span>Metric</span>AST match (valid rate)</div>
<div><span>Launcher limit</span>100</div>
<div><span>Eval IDs</span>25</div>
<div><span>Data</span>gorilla-llm/Berkeley-Function-Calling-Leaderboard</div>
</div>

## The task

MAS-PromptBench uses the four single-turn BFCL categories that the AST checker can score. A run covers one category, chosen with `--category`:

| Category | What the request needs |
| --- | --- |
| `simple` | One call to the one given function. |
| `multiple` | One call, picking the right function from several. |
| `parallel` | Several calls to one function. |
| `parallel_multiple` | Several calls across several functions. |

How the agents answer depends on the runner:

- **Single and Independent** use native tool calling. Each schema becomes a LangChain tool whose body does nothing, and an agent's answer is its first model message that contains tool calls. Later turns are ignored.
- **Sequential, Centralized and Decentralized** follow the output contract: the final role ends with one fenced `json` block holding a non-empty list of calls in canonical form, `[{"function_name": {"arg": value}}]`. The runner parses the last fenced block that is a list of objects.

## How it is scored

The runner passes the canonical calls, with the schemas, the gold answers and the category, to `ast_checker` from the `bfcl-eval` package (Python language). The checker verifies function names and required parameters, and that each argument's type and value appear among the possible answers. For parallel categories, call order does not matter.

Each line of `results.jsonl` carries `valid` (true when the checker accepts the output) and, on failure, the checker's `error_type`. The runner prints `valid N/M` at the end; the metric is the valid rate.

The runner registers your `MODEL_ID` in BFCL's model table before scoring, so dotted function names such as `math.factorial` are checked as written.

## Data

Each category is two files in the Hugging Face dataset `gorilla-llm/Berkeley-Function-Calling-Leaderboard`: `BFCL_v3_<category>.json` for the requests and `possible_answer/BFCL_v3_<category>.json` for the gold answers. The runner downloads them with `hf_hub_download`; no other setup is needed.

The 25 frozen IDs in [`benchmarks/bfcl/bfcl_eval_ids.json`](https://github.com/juyangbai/MAS-PromptBench/blob/main/benchmarks/bfcl/bfcl_eval_ids.json) mix all four categories: `simple_0` to `simple_9`, and `multiple_`, `parallel_` and `parallel_multiple_` 0 to 4. A single run covers one category, so a plain `--limit` run, such as the launcher's `--limit 100` on the default `simple` category, does not reproduce this set.

## Run it

Run every command from the repository root. There is no smoke demo and no `--batch` flag; every run is a batch, and passing `--batch` is an error. With no arguments, a runner scores the first 5 `simple` instances.

=== "Single"

    ```bash
    python -m topologies.single.bfcl.langgraph_bfcl --category simple --limit 100 \
      --out-dir results/topologies_baseline/single_bfcl
    ```

=== "Sequential (CrewAI)"

    ```bash
    python -m topologies.sequential.crewai.bfcl.crewai_bfcl --category simple --limit 100 \
      --out-dir results/topologies_baseline/sequential_crewai_bfcl
    ```

To score the frozen set, run each category with all 25 IDs as repeated `--only` flags. `--only` keeps only the IDs found in the chosen category's file, so each pass picks up its own share:

```bash title="Score the frozen eval IDs"
MANIFEST=benchmarks/bfcl/bfcl_eval_ids.json
IDS=$(python -c "import json,sys; print(*json.load(open(sys.argv[1]))['ids'])" $MANIFEST)
ONLY=$(printf -- '--only %s ' $IDS)
for cat in simple multiple parallel parallel_multiple; do
  python -m topologies.single.bfcl.langgraph_bfcl --category $cat $ONLY \
    --out-dir results/topologies_baseline/single_bfcl
done
```

!!! warning "Output files are appended"
    BFCL runners append to `predictions.jsonl` and `results.jsonl` in `--out-dir`, and write one trace per instance to `traces/<id>.txt`. The loop above relies on this to collect all four categories in one place. Use a fresh `--out-dir` for each new run, or the old lines stay in the file.

To optimize the single-agent prompt with [GEPA](../optimizers/gepa.md):

```bash title="GEPA on BFCL"
cd optimizers/gepa
python -m real_runner_gepa.pilots.run_gepa_dataset --dataset bfcl --topology single \
  --train-size 25 --val-size 25 --max-full-evals 5 --out results/gepa/single_bfcl
```

To optimize the CrewAI pipeline instead, use `--topology sequential_crewai`.

## Flags

| Flag | Default | Effect |
| --- | --- | --- |
| `--category` | `simple` | One of `simple`, `multiple`, `parallel`, `parallel_multiple`. |
| `--limit N` | `5` | Number of instances; ignored when `--only` is given. |
| `--offset K` | `0` | Skip the first K instances. |
| `--only ID` | none | Keep this instance ID; repeat the flag for more. |
| `--out-dir DIR` | a `results/bfcl*` folder, different per topology | Where output files go. |

## Related

- [ToolHop](toolhop.md) and [API-Bank](api-bank.md), the other tool-calling tasks.
- [Sequential](../mas/sequential.md) for the CrewAI pipeline behind the best result.
- [Evaluation Protocol](../evaluation/protocol.md) for how the frozen IDs are used.
