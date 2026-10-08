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
| \( n \) | Fixed per topology; varied in [Team Sizes](team-sizes.md) |

The seed prompts were written by an LLM from the role catalog in [`configs/prompts/roles.yaml`](https://github.com/juyangbai/MAS-PromptBench/blob/main/configs/prompts/roles.yaml), the domain and tool lists next to it, and the template `configs/prompts/meta_prompt.txt`. Optimizers read them and never overwrite them.

## The five topologies

| Topology | Shape | Who talks to whom | Frameworks | Default size | Optimizer names |
| --- | --- | --- | --- | --- | --- |
| [Single](single.md) | One ReAct loop | The agent and its tools only | LangGraph | 1 agent | `single` |
| [Independent](independent.md) | Parallel fan-out, fan-in | Nobody; a fixed rule picks one answer | LangGraph | 4 replicas, 1 round | `independent` |
| [Sequential](sequential.md) | Linear pipeline | Each stage reads all earlier stages | LangGraph, CrewAI | 4 stages, 1 pass | `sequential`, `sequential_crewai` |
| [Centralized](centralized.md) | Hub and spoke | Manager and each worker; workers never address each other | LangGraph, AutoGen | 1 manager + 3 workers, until `TERMINATE` or a turn cap | `centralized`, `centralized_autogen` |
| [Decentralized](decentralized.md) | Peer debate | Every peer reads every other peer's previous answer | LangGraph, OpenAI SDK | 4 peers, 2 rounds | `decentralized`, `decentralized_openai` |

For HotpotQA, LiveCodeBench, ToolHop and API-Bank, the four multi-agent topologies also have optimizer names for team-size variants (`<topology>_r<N>`) and protocol variants (`<topology>_communications_<format>`). List every name for a dataset with `real_runner_gepa.registry.topologies(dataset)`.

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

Every topology runs on every one of the nine datasets: 72 runner scripts in all. Single and Independent have one implementation each; the three multi-agent topologies have two, so you can compare frameworks on the same task.

| Topology | LangGraph | CrewAI | AutoGen | OpenAI SDK |
| --- | --- | --- | --- | --- |
| Single | `single/<ds>/langgraph_<ds>.py` | | | |
| Independent | `independent/<ds>/langgraph_<ds>.py` | | | |
| Sequential | `sequential/langgraph/<ds>/langgraph_<ds>.py` | `sequential/crewai/<ds>/crewai_<ds>.py` | | |
| Centralized | `centralized/langgraph/<ds>/langgraph_<ds>.py` | | `centralized/autogen/<ds>/autogen_<ds>.py` | |
| Decentralized | `decentralized/langgraph/<ds>/langgraph_<ds>.py` | | | `decentralized/openai/<ds>/openai_<ds>.py` |

Paths are under `topologies/`. `<ds>` is the dataset folder: `gpqa`, `hotpotqa`, `math`, `lcb`, `apps`, `swe`, `bfcl`, `toolhop` or `apibank`.

!!! note "ToolHop and API-Bank don't use the frameworks"
    The ToolHop and API-Bank runners are self-contained. Each calls the endpoint through the `openai` client and builds its topology in plain Python, so the two framework variants of a topology run identical code under a different `STYLE` label.

## Run a topology

Run every command from the repository root, using the module form:

```bash title="Run one topology on 100 HotpotQA questions"
python -m topologies.centralized.autogen.hotpotqa.autogen_hotpotqa --batch --limit 100 \
  --out results/topologies_baseline/centralized_autogen_hotpotqa/predictions.jsonl
```

!!! tip "Use `python -m` from the repository root"
    The runners import the `topologies` package, which the module form puts on Python's import path. To run a file by its path instead, set `PYTHONPATH=.` first.

To sweep all eight topology variants over all nine datasets, use `scripts/run_topologies.sh` (it passes `--batch` and `--out` to the BFCL and SWE-bench runners, which accept neither, so run those cells by hand). Each topology page shows the commands for its frameworks; [Command-Line Flags](../reference/cli.md) lists the flags per dataset family.
