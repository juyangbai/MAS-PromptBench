# Single

One agent solves the task end to end in a reason-act loop with its tools. It is the control condition: every multi-agent result is read against it.
{ .lede }

--8<-- "diagrams/single.svg"

<div class="facts" markdown>
<div><span>Agents</span>1</div>
<div><span>Rounds</span>ReAct loop</div>
<div><span>Frameworks</span>LangGraph</div>
<div><span>Aggregation</span>None</div>
</div>

## How it works

1. The runner reads the seed prompt `configs/prompts/single/<dataset>/solver.txt` and wraps it with the dataset's protected output contract from `topologies/output_contracts.py` (for HotpotQA: end with one line `Answer: <short-form>`).
2. It builds one agent with LangGraph's prebuilt `create_react_agent`, bound to the dataset's tools: `wikipedia_search` and `wikipedia_page` for HotpotQA, `calculator` for GPQA and MATH, `python_exec` for LiveCodeBench and APPS, `file_read`, `file_write`, `list_dir`, `search_repo` and `shell_exec` for SWE-bench, and tools built from the function schemas for BFCL.
3. The agent alternates between reasoning and tool calls. Each tool result is appended to its history. The loop ends when the model replies without a tool call, or when LangGraph's recursion limit is reached (25 steps; 100 for SWE-bench).
4. The runner parses the final message with the dataset's extractor and scores it.

There is no hand-off, no second agent and no vote. Whatever the one prompt makes the model do is the result.

## Agent role and seed prompt

Every dataset uses a single role, `solver`:

| Dataset folder | Seed prompt |
| --- | --- |
| `gpqa`, `hotpotqa`, `math`, `lcb`, `apps`, `bfcl`, `swe`, `toolhop`, `apibank` | `configs/prompts/single/<dataset>/solver.txt` |

The role descriptions the prompts were generated from are under `single:` in [`configs/prompts/roles.yaml`](https://github.com/juyangbai/MAS-PromptBench/blob/main/configs/prompts/roles.yaml). The optimizer reads these files and never overwrites them; it writes its compiled prompt to its own output folder.

## Implementation

- **Reference scaffold:** [`topologies/single/langgraph_base.py`](https://github.com/juyangbai/MAS-PromptBench/blob/main/topologies/single/langgraph_base.py) shows the pattern in 54 lines: `create_react_agent` with toy tools (`add`, `multiply`, a mock `web_search`). It is a demo, not a benchmark runner.
- **Dataset runners:** `topologies/single/<dataset>/langgraph_<dataset>.py`, one per dataset. Each wires real tools, the dataset loader, the scorer and the batch loop around the same ReAct agent.
- **ToolHop and API-Bank:** these two runners don't use LangGraph objects. They call the endpoint through the `openai` client in a plain tool loop (ToolHop allows 9 turns by default, `TOOLHOP_MAX_TURNS`). Every other topology's ToolHop and API-Bank runner carries a copy of the same core.

## Run it

Run from the repository root with a model endpoint configured ([Connect a Model](../getting-started/connect-a-model.md)).

```bash title="Smoke demo (built-in example, no dataset download)"
python -m topologies.single.hotpotqa.langgraph_hotpotqa
```

```bash title="Batch run on 100 HotpotQA questions"
python -m topologies.single.hotpotqa.langgraph_hotpotqa --batch --limit 100 \
  --out results/topologies_baseline/single_hotpotqa/predictions.jsonl
```

Other datasets follow the same module path with a different folder and file name, for example `topologies.single.math.langgraph_math`. BFCL, SWE-bench, ToolHop and API-Bank take `--out-dir` instead of `--out`, and BFCL and SWE-bench reject `--batch`. See [Command-Line Flags](../reference/cli.md) for each family.

## Optimize it

The optimizer topology name is `single`. It has one prompt to tune, `solver`.

```bash title="GEPA on Single / MATH"
cd optimizers/gepa
python -m real_runner_gepa.pilots.run_gepa_dataset --dataset math --topology single \
  --train-size 25 --val-size 25 --max-full-evals 5 --out results/gepa/single_math
```

See [Run an Optimizer](../optimizers/running.md) for endpoints and outputs.
