# Independent

Several copies of one agent answer the same input in parallel and never see each other's work. A majority vote then picks one answer. It measures what an ensemble adds without any communication.
{ .lede }

--8<-- "diagrams/independent.svg"

<div class="facts" markdown>
<div><span>Agents</span>4 replicas</div>
<div><span>Rounds</span>1</div>
<div><span>Frameworks</span>LangGraph</div>
<div><span>Aggregation</span>Majority vote</div>
</div>

## How it works

1. The runner reads one seed prompt for the dataset and gives it, unchanged, to every replica.
2. A LangGraph `StateGraph` fans out from `START`: a conditional edge returns one `Send` per replica (`agent_0` to `agent_3`). Each node runs a ReAct agent built with `create_react_agent` and the dataset's tools.
3. Replica `i` sends its requests with seed `i`; otherwise the replicas are identical.
4. All nodes edge to `END`. Their answers are merged into one list by an `operator.add` reducer. No replica reads another's output at any point.
5. The runner submits the majority answer, without looking at the gold answer or the tests:

| Datasets | Vote over |
| --- | --- |
| GPQA | extracted letters |
| HotpotQA | normalized short-form answers |
| MATH | buckets of `\boxed{}` answers that `is_equiv` treats as equal |
| BFCL | canonical function-call lists |
| LiveCodeBench, APPS | programs, compared with whitespace normalized; only the winner is tested |
| SWE-bench | non-empty patches, compared with whitespace normalized; only the winner is evaluated |
| ToolHop, API-Bank | final answers or API calls |

The largest bucket wins, ties go to the bucket with the lowest replica, and replicas without an answer abstain. The shared rule is in [`core/voting.py`](https://github.com/juyangbai/MAS-PromptBench/blob/main/core/voting.py).

The replica count comes from the team spec (4) and `INDEPENDENT_N_AGENTS` overrides it. The ToolHop and API-Bank runners read `TOOLHOP_INDEPENDENT_N_AGENTS` or `APIBANK_INDEPENDENT_N_AGENTS` first. For the 2, 8 and 10 replica variants, see [Team Sizes](team-sizes.md).

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
- **Dataset runners:** `topologies/independent/<dataset>/langgraph_<dataset>.py`. The GPQA, HotpotQA and MATH runners cap each row at 120 s of wall-clock time and the LiveCodeBench and APPS runners at 180 s, so one stuck replica can't stall a batch. On SWE-bench, each replica edits its own clone of the repository.
- **ToolHop and API-Bank:** no LangGraph graph. The runner calls the shared tool loop (ToolHop) or model call (API-Bank) once per seed in plain Python and votes over the results.

## Run it

```bash title="Smoke demo"
python -m topologies.independent.hotpotqa.langgraph_hotpotqa
```

```bash title="Batch run with the default 4 replicas"
python -m topologies.independent.hotpotqa.langgraph_hotpotqa --batch --limit 100 \
  --out-dir results/topologies_baseline/independent_hotpotqa
```

```bash title="ToolHop with 8 replicas"
export TOOLHOP_ALLOW_DATASET_EXEC=1
INDEPENDENT_N_AGENTS=8 python -m topologies.independent.toolhop.langgraph_toolhop \
  --limit 100 --out-dir results/topologies_baseline/independent_toolhop_n8
```

See the [task pages](../tasks/index.md) for each dataset's options.

## Optimize it

The protocol topology is `independent`; add `--team-size` or `--communication` for the [team-size](team-sizes.md) and [communication-protocol](communication-protocols.md) variants. The optimizer tunes the one shared prompt. Any of the eight methods runs it; for example, MAMUT-GEPA on HotpotQA:

```bash title="MAMUT-GEPA on Independent · HotpotQA"
python -m optimizers.protocol.run --method mamut_gepa --dataset hotpotqa --topology independent \
  --model qwen --seed 0 --out runs/mamut_gepa/hotpotqa/independent/qwen/0
```

See [Run an Optimizer](../optimizers/running.md).
