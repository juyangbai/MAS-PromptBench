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

Each role's seed prompt is `configs/prompts/sequential/<dataset>/<role>.txt`. Both frameworks read the same files. Every dataset folder holds ten role files: the four above plus six specialists that only the 8- and 10-stage [team-size](team-sizes.md) variants use. For HotpotQA those are `query_decomposer`, `searcher`, `entity_disambiguator`, `evidence_filter`, `citation_compiler` and `answer_simplifier`. All role descriptions are under `sequential:` in [`configs/prompts/roles.yaml`](https://github.com/juyangbai/MAS-PromptBench/blob/main/configs/prompts/roles.yaml).

## Implementations

### LangGraph

`topologies/sequential/langgraph/<dataset>/langgraph_<dataset>.py` builds a `StateGraph` with one node per stage and plain edges `START → stage 1 → … → stage 4 → END`. A stage with tools is a `create_react_agent`; a stage without tools is a single chat call. Stage outputs collect in a `by_stage` dict in the graph state, which the next node formats into its prompt. The [communication-protocol](communication-protocols.md) runners wrap this implementation, and the team-size runners mirror it.

### CrewAI

`topologies/sequential/crewai/<dataset>/crewai_<dataset>.py` builds one CrewAI `Agent` per stage, with the seed prompt as the agent's `backstory`, and one `Task` per stage whose `context` lists every earlier task. A `Crew` runs them with `Process.sequential`. The model is reached through CrewAI's `LLM` class at `openai/<MODEL_ID>` on `VLLM_BASE_URL`.

[`topologies/sequential/crewai/crewai_base/`](https://github.com/juyangbai/MAS-PromptBench/tree/main/topologies/sequential/crewai/crewai_base) is a CrewAI project scaffold (`crew.py`, `main.py`, `config/agents.yaml`, `config/tasks.yaml`) with a researcher, analyst, writer and editor. It is a reference demo, not a benchmark runner. There is no separate LangGraph base file for this topology.

!!! note "ToolHop and API-Bank"
    These runners use no framework objects. Both variants run the same plain-Python loop over the four roles, passing each stage the trimmed reports of the earlier stages. The two files differ only in their `STYLE` label.

## Run it

=== "LangGraph"

    ```bash
    python -m topologies.sequential.langgraph.bfcl.langgraph_bfcl --limit 100 \
      --out-dir results/topologies_baseline/sequential_langgraph_bfcl
    ```

=== "CrewAI"

    ```bash
    python -m topologies.sequential.crewai.bfcl.crewai_bfcl --limit 100 \
      --out-dir results/topologies_baseline/sequential_crewai_bfcl
    ```

BFCL runners always run a batch, take `--out-dir`, and reject `--batch`. They score the `simple` category unless you pass `--category`. For HotpotQA, MATH, GPQA, LiveCodeBench and APPS use `--batch` and `--out <file.jsonl>` instead:

```bash title="HotpotQA, LangGraph pipeline"
python -m topologies.sequential.langgraph.hotpotqa.langgraph_hotpotqa --batch --limit 100 \
  --out results/topologies_baseline/sequential_langgraph_hotpotqa/predictions.jsonl
```

## Optimize it

| Optimizer topology name | What it runs |
| --- | --- |
| `sequential` | the LangGraph pipeline |
| `sequential_crewai` | the CrewAI pipeline |
| `sequential_r2`, `sequential_r4`, `sequential_r8`, `sequential_r10` | team-size variants (HotpotQA, LiveCodeBench, ToolHop, API-Bank) |
| `sequential_communications_<format>` | communication-protocol variants (same four datasets) |

The optimizer tunes all four stage prompts. For example, GEPA on `sequential_crewai` / `bfcl`:

```bash
cd optimizers/gepa
python -m real_runner_gepa.pilots.run_gepa_dataset --dataset bfcl --topology sequential_crewai \
  --train-size 25 --val-size 25 --max-full-evals 5 --out results/gepa/sequential_crewai_bfcl
```
