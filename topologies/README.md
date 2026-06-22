# Topologies

This folder holds reference implementations of **5 multi-agent topologies**, evaluated across **9 benchmark datasets** — the core of MAS-PromptBench. Each `(topology, dataset)` pair is **runnable**, with a no-arg smoke demo and a batch-evaluation mode that writes predictions, per-instance results, and traces to `results/<dataset>/`.

Each pair is also an **optimizer target**: prompt-optimization methods such as GEPA and MIPRO mutate the per-role prompts in `configs/prompts/` and re-run it to measure improvement (see [Prompt optimization](#prompt-optimization)).

## Overview

| Topology | Shape | Inter-agent communication | Frameworks |
|---|---|---|---|
| `single` | self-loop (1 agent) | — | LangGraph |
| `independent` | parallel fan-out | none (ensemble) | LangGraph |
| `sequential` | linear pipeline | stage → stage | LangGraph, CrewAI |
| `centralized` | hub-and-spoke | via manager only | LangGraph, AutoGen |
| `decentralized` | peer debate | all-to-all, per round | LangGraph, OpenAI SDK |

### Directory layout

```
topologies/
├── telemetry.py            # token / round-count extraction
├── output_contracts.py     # per-dataset final-answer contracts
├── code_extract.py         # robust code-block extraction
│
├── single/                  # LangGraph only
│   └── <dataset>/<framework>_<dataset>.py
├── independent/             # LangGraph only
├── sequential/{langgraph,crewai}/
├── centralized/{langgraph,autogen}/
└── decentralized/{langgraph,openai}/
```

Path pattern: `topologies/<topology>/[<framework>/]<dataset>/<framework>_<dataset>.py`. `single` and `independent` are LangGraph-only; the three multi-agent topologies ship two framework implementations each for comparison.

---

## Topologies

Each topology has a reference base implementation (`<topology>/.../<framework>_base.py`) plus one runner per dataset.

### `single` — one agent, ReAct loop

One LLM in a reason→act self-loop; terminates when it replies with no tool call. Baseline control.

<div align="left">

<pre>
                 ┌──────┐
                 │ user │
                 └───┬──┘
                     v
           ┌───────────────────┐
    ┌─────>│        LLM        │<─────┐
    │      └─────────┬─────────┘      │
    │        has tool_calls?          │
    │           ┌────┴────┐           │
    │          YES       NO           │
    │           v         v           │
    │      ┌─────────┐ ┌─────┐        │
    │      │  tools  │ │ END │        │
    │      └────┬────┘ └─────┘        │
    └───────────┴─────────────────────┘
              tool results appended
</pre>

</div>

**Implementation:** `single/langgraph_base.py` — `create_react_agent`.

### `independent` — parallel agents, no communication

N agents answer the same input concurrently; outputs are aggregated (ensemble). Single round, no iteration.

<div align="left">

<pre>
             ┌────────┐
             │  task  │
             └────┬───┘
                  │ fan-out
    ┌────────┬────┴───┬────────┐
    v        v        v        v
  ┌────┐   ┌────┐   ┌────┐   ┌────┐
  │ A1 │   │ A2 │   │ A3 │   │ A4 │   (each agent independent;
  └─┬──┘   └─┬──┘   └─┬──┘   └─┬──┘    no edges between them)
    └────────┴────┬───┴────────┘
                  v fan-in (aggregate)
             ┌─────────┐
             │ answers │
             └─────────┘
</pre>

</div>

**Implementation:** `independent/langgraph_base.py` — `Send` fan-out / fan-in.

### `sequential` — linear pipeline

N agents in a chain; each agent's output becomes the next agent's context. Low coordination overhead, no cross-stage error correction.

<div align="left">

<pre>
  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
  │   A1     │──>│   A2     │──>│   A3     │──>│   A4     │──> output
  │(research)│   │(analyze) │   │ (write)  │   │  (edit)  │
  └──────────┘   └──────────┘   └──────────┘   └──────────┘
</pre>

</div>

**Implementations:** `sequential/langgraph/` (4-stage `StateGraph`), `sequential/crewai/` (`Process.sequential` scaffold).

### `centralized` — hub-and-spoke

One **manager** coordinates N **workers**; all delegation flows through the manager and workers never talk directly. Strong control; the manager is the bottleneck.

<div align="left">

<pre>
              ┌────────────┐
              │  Manager   │<────────────┐
              │ (planner)  │             │
              └─────┬──────┘             │
           ┌────────┼────────┐    workers report back
           v        v        v    to the manager,
       ┌──────┐ ┌──────┐ ┌──────┐ never to each other
       │  W1  │ │  W2  │ │  W3  │────────┘
       └──────┘ └──────┘ └──────┘
</pre>

</div>

**Implementations:** `centralized/langgraph/`, `centralized/autogen/` (`SelectorGroupChat`).

### `decentralized` — peer debate

N peers debate over R rounds (default 4×2); from round 1 each peer sees every other peer's previous answer (complete graph per round).

<div align="left">

<pre>
  round 0 (independent):
  ┌─────┐              ┌─────┐              ┌─────┐
  │ A1  │              │ A2  │              │ A3  │
  └──┬──┘              └──┬──┘              └──┬──┘
     v                    v                    v
  ┌─────┐              ┌─────┐              ┌─────┐
  │ a01 │              │ a02 │              │ a03 │
  └─────┘              └─────┘              └─────┘

  round 1 (each peer sees every other peer's round-0 answer):
  ┌─────┐              ┌─────┐              ┌─────┐
  │ A1  │ <─(a02,a03)  │ A2  │ <─(a01,a03)  │ A3  │ <─(a01,a02)
  └──┬──┘              └──┬──┘              └──┬──┘
     v                    v                    v
  ┌─────┐              ┌─────┐              ┌─────┐
  │ a11 │              │ a12 │              │ a13 │
  └─────┘              └─────┘              └─────┘

  ... continue until R rounds complete
</pre>

</div>

**Implementations:** `decentralized/langgraph/`, `decentralized/openai/` (`debate_base.py`). After Du et al. 2023 ([arXiv:2305.14325](https://arxiv.org/abs/2305.14325)).

---

## Usage

### Point at an endpoint

```bash
export VLLM_BASE_URL=http://localhost:8000/v1   # any OpenAI-compatible server
export MODEL_ID=Qwen/Qwen3.5-9B
```

### Run a baseline

```bash
# smoke demo (built-in example, no dataset download)
python topologies/single/hotpotqa/langgraph_hotpotqa.py

# real batch on a slice
python topologies/single/hotpotqa/langgraph_hotpotqa.py --batch --limit 100

# dataset summary only (no model call)
python topologies/single/apibank/langgraph_apibank.py --summary
```

### CLI flags

Common: `--batch`, `--limit N`, `--offset K`, `--only ID...`, `--out PATH`. Dataset-specific examples: `--level` (apibank), `--eval singularity` (swe). Run any pair with `--help` for its exact interface. Outputs land in `results/<dataset>/`.

---

## Prompt optimization

Every runner loads each agent's prompt from `configs/prompts/<topology>/<dataset>/<role>.txt`. An optimizer improves a pair by mutating those role prompts and re-scoring:

| Method | Workspace | Approach |
|---|---|---|
| **GEPA** | `optimizers/gepa/` | reflective prompt evolution |
| **MIPRO** | `optimizers/mipro/` | instruction + demo optimization |

Both drive these runners through a thin real-runner bridge and report before/after scores. See each workspace's README for the run interface.
