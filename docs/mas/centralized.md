# Centralized

One manager plans the work, delegates each step to a specialist worker and writes the final answer. All control flows through the manager; workers never address each other.
{ .lede }

--8<-- "diagrams/centralized.svg"

<div class="facts" markdown>
<div><span>Agents</span>1 manager + 3 workers</div>
<div><span>Rounds</span>Until TERMINATE or turn cap</div>
<div><span>Frameworks</span>LangGraph, AutoGen</div>
<div><span>Aggregation</span>Manager answers</div>
</div>

## How it works

1. The manager receives the task. Its seed prompt is extended with an instruction to end its final message with the word `TERMINATE`.
2. The manager picks one worker and gives it an instruction.
3. The worker runs, using the dataset's tools where it has them, and its reply goes back to the manager.
4. Control returns to the manager after every worker turn. The manager may delegate again, call a task tool itself, or finish.
5. The loop stops when the manager writes `TERMINATE` or the turn cap is reached.
6. The runner extracts and scores the answer from the manager's final message.

The cap is the same in both frameworks (LangGraph `MAX_TURNS`, AutoGen `MaxMessageTermination`): 16 for GPQA, 18 for MATH and HotpotQA, 24 for BFCL, 26 for LiveCodeBench and APPS, 30 for SWE-bench.

Workers see the shared conversation, not only the manager's latest instruction. The role descriptions in `roles.yaml` describe stricter isolation; the comment there explains that AutoGen's `SelectorGroupChat` shares one transcript, and the LangGraph runners also pass the full message state to each worker.

## Roles and seed prompts

Each dataset has a `manager` and three workers:

| Dataset | Workers |
| --- | --- |
| `gpqa` | `analyzer_worker`, `solver_worker`, `verifier_worker` |
| `math` | `decomposer_worker`, `computation_worker`, `verifier_worker` |
| `hotpotqa` | `retriever_worker`, `reasoner_worker`, `writer_worker` |
| `lcb`, `apps` | `analyzer_worker`, `coder_worker`, `tester_worker` |
| `bfcl` | `inspector_worker`, `caller_worker`, `validator_worker` |
| `swe` | `navigator_worker`, `patcher_worker`, `tester_worker` |
| `toolhop` | `planner_worker`, `caller_worker`, `validator_worker` |
| `apibank` | `inspector_worker`, `caller_worker`, `validator_worker` |

Seed prompts live at `configs/prompts/centralized/<dataset>/manager.txt` and `configs/prompts/centralized/<dataset>/<worker>.txt`. Each folder also holds `manager_r8.txt`, `manager_r10.txt` and seven more workers, used only by the larger [team sizes](team-sizes.md). Role descriptions are under `centralized:` in [`configs/prompts/roles.yaml`](https://github.com/juyangbai/MAS-PromptBench/blob/main/configs/prompts/roles.yaml).

## Implementations

### LangGraph

`topologies/centralized/langgraph/<dataset>/langgraph_<dataset>.py` builds a `StateGraph` with a `manager` node, a `manager_tools` node (`ToolNode`) and one node per worker. The manager is bound to one `delegate_to_<worker>` tool per worker and, on every dataset except BFCL, to the task tools as well. When the manager calls a delegate tool, a conditional edge routes to that worker; each worker is a `create_react_agent` node with a plain edge back to `manager`. A router ends the graph on `TERMINATE` or at `MAX_TURNS`.

### AutoGen

`topologies/centralized/autogen/<dataset>/autogen_<dataset>.py` creates one `AssistantAgent` per role with the seed prompt as its `system_message`, and puts them in a `SelectorGroupChat`. A `selector_func` returns the manager whenever the last speaker was a worker; after a manager turn, AutoGen's model-based selector picks who speaks next. Termination is `TextMentionTermination("TERMINATE") | MaxMessageTermination(N)`.

[`topologies/centralized/autogen/autogen_base.py`](https://github.com/juyangbai/MAS-PromptBench/blob/main/topologies/centralized/autogen/autogen_base.py) is the reference demo of this pattern: a `PlanningAgent` with `Researcher`, `Analyst` and `Writer` workers.

!!! note "ToolHop and API-Bank"
    These runners use no framework objects, and both variants run identical code. Each worker runs once on the task, then the manager reads all three reports and writes the answer. There is no delegation loop.

## Run it

=== "LangGraph"

    ```bash
    python -m topologies.centralized.langgraph.hotpotqa.langgraph_hotpotqa --batch --limit 100 \
      --out results/topologies_baseline/centralized_langgraph_hotpotqa/predictions.jsonl
    ```

=== "AutoGen"

    ```bash
    python -m topologies.centralized.autogen.hotpotqa.autogen_hotpotqa --batch --limit 100 \
      --out results/topologies_baseline/centralized_autogen_hotpotqa/predictions.jsonl
    ```

Drop `--batch` and the other flags to run the built-in smoke demo. BFCL, SWE-bench, ToolHop and API-Bank use `--out-dir`; see [Command-Line Flags](../reference/cli.md).

## Optimize it

| Optimizer topology name | What it runs |
| --- | --- |
| `centralized` | the LangGraph team |
| `centralized_autogen` | the AutoGen team |
| `centralized_r2`, `centralized_r4`, `centralized_r8`, `centralized_r10` | team-size variants (HotpotQA, LiveCodeBench, ToolHop, API-Bank) |
| `centralized_communications_<format>` | [communication-protocol](communication-protocols.md) variants (same four datasets) |

The optimizer tunes the manager prompt and all three worker prompts.

```bash title="GEPA on Centralized (AutoGen) / HotpotQA"
cd optimizers/gepa
python -m real_runner_gepa.pilots.run_gepa_dataset --dataset hotpotqa \
  --topology centralized_autogen --train-size 25 --val-size 25 --max-full-evals 5 \
  --out results/gepa/centralized_autogen_hotpotqa
```
