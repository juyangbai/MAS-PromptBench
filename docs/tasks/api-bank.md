# API-Bank

API-Bank shows the agents a dialogue between a user and an assistant that calls APIs, and asks for the next API call. The runner executes that call with API-Bank's own API code and checks the result against the recorded one.
{ .lede }

<div class="facts" markdown>
<div><span>Domain</span>Tool calling</div>
<div><span>Metric</span>API-call accuracy</div>
<div><span>Launcher limit</span>100</div>
<div><span>Eval IDs</span>100</div>
<div><span>Data</span>Vendored in benchmarks/apibank</div>
</div>

The CLI and folder name for this task is `apibank`.

## The task

Each instance is a dialogue cut just before an API call. The prompt holds the dialogue so far, with the inputs and outputs of earlier API calls, plus API descriptions that depend on the level:

| Level | API descriptions given | What makes it harder |
| --- | --- | --- |
| 1 | The APIs the dialogue needs. | Pick the API and fill its arguments. |
| 2 | Only `ToolSearcher`, plus any earlier search results in the dialogue. | Search for the right API first. |
| 3 | Only `ToolSearcher`, as in Level 2. | Several calls may be needed; predict only the next. |

Agents do not execute APIs while solving. Each agent makes one chat completion and must answer with exactly one call in brackets with keyword arguments, in the form `[ApiName(arg='value')]`. That is the API-Bank output contract. The runner takes the last bracketed call that parses.

## How it is scored

1. **Parse** the call. A missing call, positional arguments or bad syntax scores 0.
2. **Match the name.** The API name must equal the gold API name.
3. **Execute and compare.** The runner loads the API's class from the vendored API-Bank source (Level 3 uses its `lv3_apis` set), replays the earlier API calls in the dialogue, runs the predicted call, and compares its result with the gold result using that API's own `check_api_call_correctness`.

`ToolSearcher` calls use a separate scorer, set by `--toolsearcher-scorer`:

| Scorer | Default for | How it decides |
| --- | --- | --- |
| `official` | levels all, 2, 3 | Looks up the keywords in recorded ToolSearcher outputs, else picks the closest API by sentence-embedding similarity; correct if the output equals the gold output. |
| `keyword` | level 1 | Normalized keywords equal the gold keywords. |
| `upstream` | none | Runs API-Bank's original ToolSearcher class. |

The `official` scorer needs the `sentence_transformers` package; it loads `sentence-transformers/paraphrase-MiniLM-L3-v2` on CPU by default.

Each line of `results.jsonl` has `correct`, the failing `stage` and `error`, and the predicted call. Accuracy is the fraction of `correct` lines.

## Data

The API-Bank source from `AlibabaResearch/DAMO-ConvAI` ships in the repository at [`benchmarks/apibank/apibank_upstream/`](https://github.com/juyangbai/MAS-PromptBench/tree/main/benchmarks/apibank/apibank_upstream), under its own license, so nothing needs downloading. Set `APIBANK_ROOT` to use another checkout.

With the default level, `all`, the runner reads the frozen manifest [`benchmarks/apibank/apibank_eval_ids.json`](https://github.com/juyangbai/MAS-PromptBench/blob/main/benchmarks/apibank/apibank_eval_ids.json) directly: 100 IDs, 33 from Level 1, 33 from Level 2 and 34 from Level 3. So `--limit 100` scores exactly the frozen set.

For a single level, the runner builds a task list on the fly: the first tasks whose gold call replays correctly, skipping those that need `SearchEngine` or `Translate`, up to 100 for Levels 1 and 2 or 245 for Level 3.

## Run it

Run every command from the repository root. To check the data and the gold-call replay without calling the model, use `--summary`. It covers the first `--limit` instances:

```bash title="Dataset summary, no model calls"
python -m topologies.single.apibank.langgraph_apibank --summary --limit 100
```

Runs are always batches; `--batch` is accepted and ignored. With no arguments, a runner solves the first 2 instances.

=== "Single"

    ```bash
    python -m topologies.single.apibank.langgraph_apibank --limit 100 \
      --out-dir results/topologies_baseline/single_apibank
    ```

=== "Centralized (LangGraph)"

    ```bash
    python -m topologies.centralized.langgraph.apibank.langgraph_apibank --limit 100 \
      --out-dir results/topologies_baseline/centralized_langgraph_apibank
    ```

The runner appends to `predictions.jsonl` and `results.jsonl` in `--out-dir` and writes one JSON trace per instance under `traces/`. Use a fresh `--out-dir` for each run. Without `--out-dir`, output goes to `results/apibank/<style>/`.

To optimize the single-agent prompt with [GEPA](../optimizers/gepa.md):

```bash title="GEPA on API-Bank"
cd optimizers/gepa
python -m real_runner_gepa.pilots.run_gepa_dataset --dataset apibank --topology single \
  --train-size 25 --val-size 25 --max-full-evals 5 --out results/gepa/single_apibank
```

For API-Bank, GEPA always keeps the 100 frozen IDs out of its training and validation splits. API-Bank also has [communication-protocol](../mas/communication-protocols.md) and [team-size](../mas/team-sizes.md) variants that GEPA can target, such as `--topology sequential_r2`.

## Flags

| Flag | Default | Effect |
| --- | --- | --- |
| `--level` | `all` (or `APIBANK_LEVEL`) | `all`, `1`, `2` or `3`; aliases such as `l1` and `level-2` also work. |
| `--summary` | off | Print a dataset summary with gold-call replay checks, then exit. |
| `--curated-path FILE` | none | Read task IDs from another manifest (a JSON file with an `ids` list). |
| `--toolsearcher-scorer` | by level | `official`, `upstream` or `keyword`. |
| `--limit N` | `2` | Number of instances. Also applies with `--only`. |
| `--only ID` | none | Keep this task ID; repeat the flag for more. |
| `--out-dir DIR` | `results/apibank/<style>/` | Where output files go. |

## Related

- [BFCL](bfcl.md) and [ToolHop](toolhop.md), the other tool-calling tasks.
- [Workflow Topologies](../mas/topologies.md) for what each runner variant does.
- [Evaluation Protocol](../evaluation/protocol.md) for how the frozen IDs are used.
