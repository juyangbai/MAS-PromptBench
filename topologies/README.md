# Topologies

This folder holds reference implementations of **5 multi-agent topologies**, evaluated across **9 benchmark datasets** — the core of MAS-PromptBench. Each `(topology, dataset)` pair is **runnable**: `--batch` evaluates a slice and writes predictions and per-instance records under `results/`; without it, the GPQA, HotpotQA, MATH, LiveCodeBench and APPS runners run a canned demo and the other four (BFCL, SWE-bench, API-Bank, ToolHop) a small batch.

Each pair is also an **optimizer target**: the prompt optimizers mutate the per-role prompts in `configs/prompts/` and re-run it to measure improvement (see [Prompt optimization](#prompt-optimization)).

## Overview

| Topology | Shape | Inter-agent communication | Frameworks |
|---|---|---|---|
| `single` | self-loop (1 agent) | — | LangGraph |
| `independent` | parallel fan-out | none (ensemble) | LangGraph |
| `sequential` | linear pipeline | stage → stage | LangGraph, CrewAI |
| `centralized` | hub-and-spoke | via manager only | LangGraph, AutoGen |
| `decentralized` | peer debate | all-to-all, per round | LangGraph, OpenAI Agents SDK |

### Directory layout

```
topologies/
├── single/                  # LangGraph only
│   └── <dataset>/<framework>_<dataset>.py
├── independent/             # LangGraph only
├── sequential/{langgraph,crewai}/
├── centralized/{langgraph,autogen}/
└── decentralized/{langgraph,openai_agents}/
```

Path pattern: `topologies/<topology>/[<framework>/]<dataset>/<framework>_<dataset>.py`. Shared runner code (CLI, batch loop, team specs, one task module per dataset) lives in [`core/`](../core/); see the [repository map](../docs/content/reference/repository.md) and [Add a Dataset](../docs/content/extending/add-dataset.md).

---

## Topologies

Each topology has one runner per dataset and framework; the `*_base` modules below implement the pattern.

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

N agents answer the same input concurrently; a gold-free majority vote over their answers ([`core/voting.py`](../core/voting.py)) picks the submission. Single round, no iteration.

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

**Implementations:** `sequential/langgraph/` (4-stage `StateGraph`), `sequential/crewai/` (`Process.sequential`; base in `crewai_base/`).

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

**Implementations:** `centralized/langgraph/`, `centralized/autogen/` (`SelectorGroupChat`; base in `autogen_base.py`).

### `decentralized` — peer debate

N peers debate over R rounds (default 4×2); from round 1 each peer sees every other peer's previous answer (complete graph per round). The submission is the majority vote over the final-round answers.

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

**Implementations:** `decentralized/langgraph/`, `decentralized/openai_agents/` (`agents_sdk_base.py`; needs an [isolated install](#install-the-openai-agents-sdk)). After Du et al. 2023 ([arXiv:2305.14325](https://arxiv.org/abs/2305.14325)).

---

## Usage

### Point at an endpoint

```bash
export VLLM_BASE_URL=http://localhost:8000/v1   # any OpenAI-compatible server
export MODEL_ID=Qwen/Qwen3.5-9B
export TOOLHOP_ALLOW_DATASET_EXEC=1             # required for ToolHop pairs
```

Decoding is greedy with up to 32,768 output tokens per call ([decoding](../docs/content/reference/environment.md#task-model-and-decoding)), so serve a context window larger than that. Per-dataset setup, including the SWE-bench Singularity sandbox, is in [benchmarks/README.md](../benchmarks/README.md#environment-requirements).

### Run a baseline

```bash
# canned demo (built-in example, no dataset download)
python -m topologies.single.hotpotqa.langgraph_hotpotqa

# real batch on a slice
python -m topologies.single.hotpotqa.langgraph_hotpotqa --batch --limit 100

# dataset summary only (no model call)
python -m topologies.single.apibank.langgraph_apibank --summary
```

### CLI flags

Common: `--batch`, `--limit N`, `--offset K`, `--only ID ...`, `--out-dir DIR`, `--out PATH`. Dataset-specific examples: `--category` (bfcl), `--level` and `--summary` (apibank), `--eval` (swe). All are listed in [Command-Line Flags](../docs/content/reference/cli.md#runner-options); run any pair with `--help` for its exact interface. Progress goes to stderr ([`LOG_LEVEL`](../docs/content/reference/environment.md#logging)), the end-of-batch report to stdout (SWE-bench: stderr).

### Install the OpenAI Agents SDK

openai-agents needs `openai>=3`, which conflicts with the main environment, so install it in isolation from [`requirements-openai-agents.txt`](../requirements-openai-agents.txt):

```bash
pip install --target vendor/openai_agents -r requirements-openai-agents.txt
```

The runners put it first on `PYTHONPATH` themselves and exit with status 1 if the SDK is unusable ([`OPENAI_AGENTS_PATH`](../docs/content/reference/environment.md#openai-agents-sdk)).

---

## Prompt optimization

Every runner loads each agent's prompt from `configs/prompts/<topology>/<dataset>/<role>.txt`. The eight [prompt optimizers](../optimizers/README.md) — GEPA, MIPRO, MAPRO, MASPO, HiveMind, MAMUT-GEPA, MASPOB and TAVO — improve a pair by mutating those role prompts and re-scoring the real runner through the [bridge](../optimizers/bridge/README.md).
