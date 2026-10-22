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

1. The manager receives the task. Its seed prompt is extended with an instruction to end its final message with the word `TERMINATE` and a note naming its workers.
2. The manager picks one worker and gives it an instruction.
3. The worker runs, using the dataset's tools where it has them, and its reply goes back to the manager.
4. Control returns to the manager after every worker turn. The manager may delegate again, call a task tool itself, or finish.
5. The loop stops when the manager writes `TERMINATE` or the turn cap is reached.
6. The runner extracts and scores the answer from the manager's final message.

The cap comes from the team spec and is the same in both frameworks (LangGraph `MAX_TURNS`, AutoGen `MaxMessageTermination`): 16 for GPQA, 18 for MATH and HotpotQA, 24 for BFCL, 26 for LiveCodeBench and APPS, 30 for SWE-bench.

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

Seed prompts live at `configs/prompts/centralized/<dataset>/manager.txt` and `configs/prompts/centralized/<dataset>/<worker>.txt`. Each folder also holds `manager_r8.txt`, `manager_r10.txt` and six more workers, used only by the larger [team sizes](team-sizes.md). The workers, their tools, the manager's tools and the turn caps come from the team spec in `configs/teams/<dataset>.yaml`; ToolHop and API-Bank fix their roles in code. Role descriptions are under `centralized:` in [`configs/prompts/roles.yaml`](https://github.com/juyangbai/MAS-PromptBench/blob/main/configs/prompts/roles.yaml).

## Implementations

### LangGraph

`topologies/centralized/langgraph/<dataset>/langgraph_<dataset>.py` builds a `StateGraph` with a `manager` node, a `manager_tools` node (`ToolNode`) and one node per worker. The manager is bound to one `delegate_to_<worker>` tool per worker and, on every dataset except BFCL, to the task tools as well. When the manager calls a delegate tool, a conditional edge routes to that worker; each worker is a `create_react_agent` node with a plain edge back to `manager`. A router ends the graph on `TERMINATE` or at `MAX_TURNS`.

### AutoGen

`topologies/centralized/autogen/<dataset>/autogen_<dataset>.py` creates one `AssistantAgent` per role with the seed prompt as its `system_message`, and puts them in a `SelectorGroupChat`. A `selector_func` returns the manager whenever the last speaker was a worker; after a manager turn, AutoGen's model-based selector picks who speaks next. Termination is `TextMentionTermination("TERMINATE") | MaxMessageTermination(N)`.

[`topologies/centralized/autogen/autogen_base.py`](https://github.com/juyangbai/MAS-PromptBench/blob/main/topologies/centralized/autogen/autogen_base.py) is the reference demo of this pattern: a `PlanningAgent` with `Researcher`, `Analyst` and `Writer` workers.

!!! note "ToolHop and API-Bank"
    These runners use no framework objects, and the AutoGen runner is the LangGraph runner's code under the `centralized_autogen` label. Each worker runs once on the task, then the manager reads all three reports and writes the answer. There is no delegation loop.

## Run it

=== "LangGraph"

    ```bash
    python -m topologies.centralized.langgraph.hotpotqa.langgraph_hotpotqa --batch --limit 100 \
      --out-dir results/topologies_baseline/centralized_langgraph_hotpotqa
    ```

=== "AutoGen"

    ```bash
    python -m topologies.centralized.autogen.hotpotqa.autogen_hotpotqa --batch --limit 100 \
      --out-dir results/topologies_baseline/centralized_autogen_hotpotqa
    ```

Drop `--batch` and the other flags to run the built-in smoke demo. See the [task pages](../tasks/index.md) for each dataset's options.

## Optimize it

| Protocol flags | What it runs |
| --- | --- |
| `--topology centralized` | the LangGraph team |
| `--topology centralized --framework autogen` | the AutoGen team (`centralized_autogen`) |
| `--topology centralized --team-size N` | the manager with N − 1 workers (`centralized_r<N>`) |
| `--topology centralized --communication FORMAT` | a [communication-protocol](communication-protocols.md) variant (`centralized_communications_<format>`) |

The optimizer tunes the manager prompt and every worker prompt. Any of the eight methods runs it; for example, TAVO on LiveCodeBench:

```bash title="TAVO on Centralized · LiveCodeBench"
python -m optimizers.protocol.run --method tavo --dataset lcb --topology centralized \
  --model qwen --seed 0 --out runs/tavo/lcb/centralized/qwen/0
```

See [Run an Optimizer](../optimizers/running.md).
