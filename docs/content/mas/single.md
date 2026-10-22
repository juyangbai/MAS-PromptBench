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

1. The runner reads the seed prompt `configs/prompts/single/<dataset>/solver.txt` and wraps it with the dataset's protected output contract from `core/output_contracts.py` (for HotpotQA: end with one line `Answer: <short-form>`). HotpotQA and MATH also append a short format note.
2. It builds one agent with LangGraph's prebuilt `create_react_agent`, bound to the dataset's tools: `wikipedia_search` and `wikipedia_page` for HotpotQA, `calculator` for GPQA and MATH, `python_exec` for LiveCodeBench and APPS, `file_read`, `file_write`, `list_dir`, `search_repo` and `shell_exec` for SWE-bench, and tools built from the function schemas for BFCL.
3. The agent alternates between reasoning and tool calls. Each tool result is appended to its history. The loop ends when the model replies without a tool call, or when LangGraph's recursion limit is reached (25 steps; 100 for SWE-bench).
4. The runner parses the final message with the dataset's extractor and scores it.

There is no hand-off, no second agent and no vote. Whatever the one prompt makes the model do is the result.

## Agent role and seed prompt

Every dataset uses a single role, `solver`:

| Dataset folder | Seed prompt |
| --- | --- |
| `gpqa`, `hotpotqa`, `math`, `lcb`, `apps`, `bfcl`, `swe`, `toolhop`, `apibank` | `configs/prompts/single/<dataset>/solver.txt` |

The role descriptions the prompts were generated from are under `single:` in [`configs/prompts/roles.yaml`](https://github.com/juyangbai/MAS-PromptBench/blob/main/configs/prompts/roles.yaml). The optimizers read these files and never overwrite them; a job keeps its prompts under its own `--out` folder.

## Implementation

- **Reference scaffold:** [`topologies/single/langgraph_base.py`](https://github.com/juyangbai/MAS-PromptBench/blob/main/topologies/single/langgraph_base.py) shows the pattern in 54 lines: `create_react_agent` with toy tools (`add`, `multiply`, a mock `web_search`). It is a demo, not a benchmark runner.
- **Dataset runners:** `topologies/single/<dataset>/langgraph_<dataset>.py`, one per dataset. Each wires the real tools to the same ReAct agent; the loader, scorer and records come from the dataset's task module in `core/tasks/`.
- **ToolHop and API-Bank:** these two runners don't use LangGraph objects. ToolHop runs a plain tool loop on the `openai` client (9 turns by default, `TOOLHOP_MAX_TURNS`); API-Bank makes one model call. Both live in `core/tasks/toolhop.py` and `core/tasks/apibank.py`, shared by every topology's runner.

## Run it

Run from the repository root with a model endpoint configured ([Connect a Model](../getting-started/connect-a-model.md)).

```bash title="Smoke demo (built-in example, no dataset download)"
python -m topologies.single.hotpotqa.langgraph_hotpotqa
```

```bash title="Batch run on 100 HotpotQA questions"
python -m topologies.single.hotpotqa.langgraph_hotpotqa --batch --limit 100 \
  --out-dir results/topologies_baseline/single_hotpotqa
```

Other datasets follow the same module path with a different folder and file name, for example `topologies.single.math.langgraph_math`. BFCL, SWE-bench, ToolHop and API-Bank have no demo and always run a batch. See the [task pages](../tasks/index.md) for each dataset's options.

## Optimize it

The protocol topology is `single`, with one prompt to tune, `solver`. In the experiment grid, the Single topology is optimized by GEPA and MIPRO; the other six methods optimize multi-agent teams.

```bash title="GEPA on Single · MATH"
python -m optimizers.protocol.run --method gepa --dataset math --topology single \
  --model qwen --seed 0 --out runs/gepa/math/single/qwen/0
```

See [Run an Optimizer](../optimizers/running.md) for endpoints and outputs.
