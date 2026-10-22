# Sequential

Specialist agents run one after another in a fixed pipeline. Each stage reads the task plus everything the earlier stages wrote, and the last stage produces the answer.
{ .lede }

--8<-- "diagrams/sequential.svg"

<div class="facts" markdown>
<div><span>Agents</span>4 stages</div>
<div><span>Rounds</span>1 pass</div>
<div><span>Frameworks</span>LangGraph, CrewAI</div>
<div><span>Aggregation</span>Last stage answers</div>
</div>

## How it works

1. The runner loads one seed prompt per stage. Only the final stage's prompt carries the dataset's protected output contract.
2. Stage 1 receives the task.
3. Each later stage receives the task plus the output of every earlier stage, appended in order as `--- PRIOR STAGE: <role> ---` blocks (LangGraph) or as task context (CrewAI).
4. Stages may use the dataset's tools; HotpotQA's retriever searches Wikipedia, LiveCodeBench's tester runs code with `python_exec`.
5. The pipeline has no back-edges. A later stage can correct an earlier one only by what it writes forward, never by sending work back.
6. The runner parses the final stage's output and scores it.

## Stages and seed prompts

The four default stages per dataset, in order:

| Dataset | Stage 1 | Stage 2 | Stage 3 | Stage 4 |
| --- | --- | --- | --- | --- |
| `gpqa` | `analyzer` | `solver` | `critic` | `verifier` |
| `math` | `decomposer` | `computer` | `checker` | `verifier` |
| `hotpotqa` | `planner` | `retriever` | `reasoner` | `writer` |
| `lcb`, `apps` | `analyzer` | `coder` | `tester` | `debugger` |
| `bfcl` | `analyzer` | `inspector` | `caller` | `verifier` |
| `swe` | `investigator` | `planner` | `patcher` | `tester` |
| `toolhop` | `planner` | `caller` | `checker` | `verifier` |
| `apibank` | `dialogue_reader` | `schema_mapper` | `argument_planner` | `verifier` |

Each role's seed prompt is `configs/prompts/sequential/<dataset>/<role>.txt`, and both frameworks read the same files. The LangGraph stages, their tools and each stage's task message come from the team spec in `configs/teams/<dataset>.yaml`; ToolHop and API-Bank fix their four stages in code. Every dataset folder holds ten role files: the four above plus six specialists that only the 8- and 10-stage [team-size](team-sizes.md) variants use. For HotpotQA those are `query_decomposer`, `searcher`, `entity_disambiguator`, `evidence_filter`, `citation_compiler` and `answer_simplifier`. All role descriptions are under `sequential:` in [`configs/prompts/roles.yaml`](https://github.com/juyangbai/MAS-PromptBench/blob/main/configs/prompts/roles.yaml).

## Implementations

### LangGraph

`topologies/sequential/langgraph/<dataset>/langgraph_<dataset>.py` builds a `StateGraph` with one node per stage and plain edges `START → stage 1 → … → stage 4 → END`. A stage with tools is a `create_react_agent`; a stage without tools is a single chat call. Stage outputs collect in a `by_stage` dict in the graph state, which the next node formats into its prompt. The [communication-protocol](communication-protocols.md) and [team-size](team-sizes.md) runners run this implementation with a preset format or team.

### CrewAI

`topologies/sequential/crewai/<dataset>/crewai_<dataset>.py` builds one CrewAI `Agent` per stage, with the seed prompt as the agent's `backstory`, and one `Task` per stage whose `context` lists every earlier task. A `Crew` runs them with `Process.sequential`. The model is reached through CrewAI's `LLM` class at `openai/<MODEL_ID>` on `VLLM_BASE_URL`, with the same decoding settings as every runner.

[`topologies/sequential/crewai/crewai_base/`](https://github.com/juyangbai/MAS-PromptBench/tree/main/topologies/sequential/crewai/crewai_base) is a CrewAI project scaffold (`crew.py`, `main.py`, `config/agents.yaml`, `config/tasks.yaml`) with a researcher, analyst, writer and editor. It is a reference demo, not a benchmark runner. There is no separate LangGraph base file for this topology.

!!! note "ToolHop and API-Bank"
    These runners use no framework objects. They run the four roles in a plain-Python loop, passing each stage the trimmed reports of the earlier stages. The CrewAI runner is the LangGraph runner's code under the `sequential_crewai` label.

## Run it

=== "LangGraph"

    ```bash
    python -m topologies.sequential.langgraph.bfcl.langgraph_bfcl --category simple --limit 100 \
      --out-dir results/topologies_baseline/sequential_langgraph_bfcl/simple
    ```

=== "CrewAI"

    ```bash
    python -m topologies.sequential.crewai.bfcl.crewai_bfcl --category simple --limit 100 \
      --out-dir results/topologies_baseline/sequential_crewai_bfcl/simple
    ```

BFCL runners always run a batch and score one category at a time. For HotpotQA, MATH, GPQA, LiveCodeBench and APPS, add `--batch`, or the runner plays its demo:

```bash title="HotpotQA, LangGraph pipeline"
python -m topologies.sequential.langgraph.hotpotqa.langgraph_hotpotqa --batch --limit 100 \
  --out-dir results/topologies_baseline/sequential_langgraph_hotpotqa
```

## Optimize it

| Protocol flags | What it runs |
| --- | --- |
| `--topology sequential` | the LangGraph pipeline |
| `--topology sequential --framework crewai` | the CrewAI pipeline (`sequential_crewai`) |
| `--topology sequential --team-size N` | the N-stage [team-size](team-sizes.md) variant (`sequential_r<N>`) |
| `--topology sequential --communication FORMAT` | a [communication-protocol](communication-protocols.md) variant (`sequential_communications_<format>`) |

The optimizer tunes every stage prompt. Any of the eight methods runs it; for example, GEPA on the CrewAI pipeline for BFCL:

```bash title="GEPA on Sequential (CrewAI) · BFCL"
python -m optimizers.protocol.run --method gepa --dataset bfcl \
  --topology sequential --framework crewai \
  --model qwen --seed 0 --out runs/gepa/bfcl/sequential_crewai/qwen/0
```

See [Run an Optimizer](../optimizers/running.md).
