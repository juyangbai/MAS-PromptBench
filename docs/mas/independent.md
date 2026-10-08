# Independent

Several copies of one agent answer the same input in parallel and never see each other's work. A fixed rule then picks one answer. It measures what an ensemble adds without any communication.
{ .lede }

--8<-- "diagrams/independent.svg"

<div class="facts" markdown>
<div><span>Agents</span>4 replicas</div>
<div><span>Rounds</span>1</div>
<div><span>Frameworks</span>LangGraph</div>
<div><span>Aggregation</span>Vote or best-of-N</div>
</div>

## How it works

1. The runner reads one seed prompt for the dataset and gives it, unchanged, to every replica.
2. A LangGraph `StateGraph` fans out from `START`: a conditional edge returns one `Send` per replica (`agent_0` to `agent_3`). Each node runs a ReAct agent built with `create_react_agent` and the dataset's tools.
3. Replicas differ only by sampling seed. Replica `i` uses seed `i` at temperature 0.2, so the four runs diverge without any change to the prompt.
4. All nodes edge to `END`. Their answers are merged into one list by an `operator.add` reducer. No replica reads another's output at any point.
5. The runner aggregates the list with the dataset's rule:

| Datasets | Rule |
| --- | --- |
| GPQA, HotpotQA | Majority vote over normalized answers; ties go to the lowest replica index. |
| MATH | Majority over buckets of equivalent `\boxed{}` answers. |
| BFCL | Majority over canonical function-call forms. |
| LiveCodeBench, APPS | Run each program on the problem's tests; return the first that passes all, else the highest pass rate. |
| SWE-bench | Evaluate each patch; return the first resolved one, else the best fail-to-pass × pass-to-pass rate. |
| ToolHop, API-Bank | Majority over normalized final answers or canonical API calls. |

Change the number of replicas with `INDEPENDENT_N_AGENTS` (default `4`). The ToolHop and API-Bank runners read `TOOLHOP_INDEPENDENT_N_AGENTS` or `APIBANK_INDEPENDENT_N_AGENTS` first. For the 2, 8 and 10 replica variants used in the paper, see [Team Sizes](team-sizes.md).

## Agent role and seed prompt

One role per dataset, shared by all replicas:

| Role | Datasets | Seed prompt |
| --- | --- | --- |
| `solver` | `gpqa`, `hotpotqa`, `math`, `toolhop`, `apibank` | `configs/prompts/independent/<dataset>/solver.txt` |
| `coder` | `lcb`, `apps` | `configs/prompts/independent/<dataset>/coder.txt` |
| `caller` | `bfcl` | `configs/prompts/independent/bfcl/caller.txt` |
| `patcher` | `swe` | `configs/prompts/independent/swe/patcher.txt` |

Because every replica reads the same file, optimizing this topology means tuning one prompt that all four agents then use.

## Implementation

- **Reference scaffold:** [`topologies/independent/langgraph_base.py`](https://github.com/juyangbai/MAS-PromptBench/blob/main/topologies/independent/langgraph_base.py) shows the `Send` fan-out and fan-in with four agents. Its demo agents have different personas; the benchmark runners use one shared prompt instead.
- **Dataset runners:** `topologies/independent/<dataset>/langgraph_<dataset>.py`. The GPQA, HotpotQA and MATH runners cap each row at 120 s of wall-clock time and the LiveCodeBench and APPS runners at 180 s, so one stuck replica can't stall a batch.
- **ToolHop and API-Bank:** no LangGraph graph. The runner calls the shared tool loop once per seed in plain Python and votes over the results.

## Run it

```bash title="Smoke demo"
python -m topologies.independent.hotpotqa.langgraph_hotpotqa
```

```bash title="Batch run with the default 4 replicas"
python -m topologies.independent.hotpotqa.langgraph_hotpotqa --batch --limit 100 \
  --out results/topologies_baseline/independent_hotpotqa/predictions.jsonl
```

```bash title="ToolHop with 8 replicas"
export TOOLHOP_ALLOW_DATASET_EXEC=1
INDEPENDENT_N_AGENTS=8 python -m topologies.independent.toolhop.langgraph_toolhop \
  --limit 100 --out-dir results/topologies_baseline/independent_toolhop_n8
```

Flag families differ by dataset; see [Command-Line Flags](../reference/cli.md).

## Optimize it

| Optimizer topology name | What it runs |
| --- | --- |
| `independent` | the runners on this page |
| `independent_r2`, `independent_r4`, `independent_r8`, `independent_r10` | [team-size](team-sizes.md) variants (HotpotQA, LiveCodeBench, ToolHop, API-Bank) |
| `independent_communications_<format>` | [communication-protocol](communication-protocols.md) variants (same four datasets) |

The optimizer pilot sets the replica count itself with `--n-agents`, which defaults to `2`. Pass `--n-agents 4` to optimize the same team the baseline runs:

```bash title="GEPA on Independent / MATH with 4 replicas"
cd optimizers/gepa
python -m real_runner_gepa.pilots.run_gepa_dataset --dataset math --topology independent \
  --n-agents 4 --train-size 25 --val-size 25 --max-full-evals 5 \
  --out results/gepa/independent_math
```
