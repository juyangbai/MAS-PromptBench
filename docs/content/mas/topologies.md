# Workflow Topologies

MAS-PromptBench runs every task under five ways of wiring agents together, from one agent to a peer debate. This page defines what a multi-agent system is in the benchmark, compares the five topologies and shows which frameworks implement each.
{ .lede }

## How a multi-agent system is modeled

A multi-agent system (MAS) in the benchmark has three parts:

- **Agents** \( a_1, \dots, a_n \). Each agent \( a_i = (M, s_i) \) pairs a frozen LLM \( M \) with a learnable system prompt \( s_i \).
- **A coordination workflow** \( G \): who acts, in what order, and whose output each agent reads. This is the topology.
- **A communication protocol** \( P \): the shape of the messages agents pass to each other.

Prompt optimization changes only the prompts \( s_1, \dots, s_n \). The model, the workflow, the protocol and the number of agents stay fixed. In the repository:

| Part | Where it lives |
| --- | --- |
| \( M \) | One OpenAI-compatible endpoint, `VLLM_BASE_URL` + `MODEL_ID` (default `Qwen/Qwen3.5-9B`) |
| \( s_i \) | Seed prompts in `configs/prompts/<topology>/<dataset>/<role>.txt` |
| \( G \) | The runners in [`topologies/`](https://github.com/juyangbai/MAS-PromptBench/tree/main/topologies), one per topology, framework and dataset |
| \( P \) | Free text by default; the formats in [Communication Protocols](communication-protocols.md) |
| \( n \) | The team specs in [`configs/teams/`](https://github.com/juyangbai/MAS-PromptBench/tree/main/configs/teams), 4 agents by default; varied in [Team Sizes](team-sizes.md) |

The seed prompts were written by an LLM from the role catalog in [`configs/prompts/roles.yaml`](https://github.com/juyangbai/MAS-PromptBench/blob/main/configs/prompts/roles.yaml), the domain and tool lists next to it, and the template `configs/prompts/meta_prompt.txt`. Optimizers read them and never overwrite them. A team spec, `configs/teams/<dataset>.yaml`, names each topology's roles, their tools and order, and the turn caps, for every team size; ToolHop and API-Bank fix their roles in code instead.

## The five topologies

| Topology | Shape | Who talks to whom | Frameworks | Default size | Protocol `--topology` |
| --- | --- | --- | --- | --- | --- |
| [Single](single.md) | One ReAct loop | The agent and its tools only | LangGraph | 1 agent | `single` |
| [Independent](independent.md) | Parallel fan-out, fan-in | Nobody; a majority vote picks one answer | LangGraph | 4 replicas, 1 round | `independent` |
| [Sequential](sequential.md) | Linear pipeline | Each stage reads all earlier stages | LangGraph, CrewAI | 4 stages, 1 pass | `sequential`, `sequential_crewai` |
| [Centralized](centralized.md) | Hub and spoke | Manager and each worker; workers never address each other | LangGraph, AutoGen | 1 manager + 3 workers, until `TERMINATE` or a turn cap | `centralized`, `centralized_autogen` |
| [Decentralized](decentralized.md) | Peer debate | Every peer reads every other peer's previous answer | LangGraph, OpenAI Agents SDK | 4 peers, 2 rounds | `decentralized`, `decentralized_openai_agents` |

The last column is what `python -m optimizers.protocol.run --topology` takes; `--topology sequential --framework crewai` is the same as `sequential_crewai`. Team-size and protocol variants of the four multi-agent topologies take `--team-size` and `--communication`, or keys such as `centralized_r8` and `sequential_communications_structured_soft`. See [Run an Optimizer](../optimizers/running.md).

<div class="cards" markdown>

- [Single](single.md)
  One agent in a reason-act loop; the control condition.
- [Independent](independent.md)
  Parallel replicas with no communication, aggregated by a vote.
- [Sequential](sequential.md)
  A fixed pipeline of specialist stages, each reading the ones before it.
- [Centralized](centralized.md)
  A manager that delegates to workers and writes the answer.
- [Decentralized](decentralized.md)
  Peers that debate over rounds and vote at the end.

</div>

## Framework by topology

Every topology runs on every one of the nine datasets: 72 runner modules in all. Single and Independent have one implementation each; Sequential, Centralized and Decentralized have two, so you can compare frameworks on the same task.

| Topology | LangGraph | CrewAI | AutoGen | OpenAI Agents SDK |
| --- | --- | --- | --- | --- |
| Single | `single/<ds>/langgraph_<ds>.py` | | | |
| Independent | `independent/<ds>/langgraph_<ds>.py` | | | |
| Sequential | `sequential/langgraph/<ds>/langgraph_<ds>.py` | `sequential/crewai/<ds>/crewai_<ds>.py` | | |
| Centralized | `centralized/langgraph/<ds>/langgraph_<ds>.py` | | `centralized/autogen/<ds>/autogen_<ds>.py` | |
| Decentralized | `decentralized/langgraph/<ds>/langgraph_<ds>.py` | | | `decentralized/openai_agents/<ds>/openai_agents_<ds>.py` |

Paths are under `topologies/`. `<ds>` is the dataset folder: `gpqa`, `hotpotqa`, `math`, `lcb`, `apps`, `swe`, `bfcl`, `toolhop` or `apibank`. The runners are thin: the command line, batch loop, model clients, team specs, prompts and one task module per dataset live in the shared `core/` package. See [Repository Map](../reference/repository.md).

!!! note "ToolHop and API-Bank"
    The ToolHop and API-Bank runners call the endpoint through the `openai` client and build their topology in plain Python. Their CrewAI and AutoGen variants run the LangGraph runner's code under another `STYLE` label; the Agents SDK variant runs the SDK debate engine.

## Run a topology

Run every command from the repository root, using the module form:

```bash title="Run one topology on 100 HotpotQA questions"
python -m topologies.centralized.autogen.hotpotqa.autogen_hotpotqa --batch --limit 100 \
  --out-dir results/topologies_baseline/centralized_autogen_hotpotqa
```

Every runner takes the same command line; the [task pages](../tasks/index.md) list each dataset's extra options and how to select its eval IDs. To sweep all eight topology variants over all nine datasets on the eval IDs, run `scripts/run_topologies.sh`. It reads `VLLM_BASE_URL`, `MODEL_ID`, `DATASETS` and `OUT_ROOT` (default `results/topologies_baseline`) and runs each cell as its own process.
