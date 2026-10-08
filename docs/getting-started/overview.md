# Overview

MAS-PromptBench measures when optimizing system prompts improves a multi-agent LLM system, and by how much. This page explains the model of a multi-agent system the benchmark uses, the four factors it varies, and the one number every run reports.
{ .lede }

## The question

Prompt optimizers such as GEPA reliably improve a single LLM agent. A multi-agent system (MAS) is harder: each agent's prompt can improve locally, yet the system can still get worse once agents hand work to each other. MAS-PromptBench runs the same optimizer across many controlled MAS configurations so you can see where gains transfer and where they break.

## A multi-agent system, formally

The benchmark models a MAS as a tuple \( \mathcal{M} = (\mathcal{A}, G, P) \):

- \( \mathcal{A} \) is the set of agents. Each agent pairs a frozen LLM with a learnable **system prompt** for its role (planner, solver, manager, debater, …).
- \( G \) is the **workflow topology**: who sends work to whom.
- \( P \) is the **communication protocol**: the format of the messages agents exchange.

Model weights never change. Optimization edits only the joint set of role prompts \( \pi \). The seed prompts \( \pi^0 \) live in `configs/prompts/<topology>/<dataset>/<role>.txt`.

## The prompt-optimization gain

For a configuration \( (\mathcal{T}, G, n, P) \) (task, topology, team size, protocol), the benchmark reports the gain of the optimized prompts \( \pi^\star \) over the seed prompts \( \pi^0 \):

\[
\Delta(\mathcal{T}, G, n, P) = \mathbb{E}_{(x,y)\sim\mathcal{T}}\big[\,\mu(\mathcal{M}(x;\pi^\star), y) - \mu(\mathcal{M}(x;\pi^0), y)\,\big]
\]

\( \mu \) is the task's own scorer (exact match, pass@1, AST match, …). Each optimizer run writes this as `delta` in its `meta.json`, as a fraction; the paper reports it in percentage points. A positive \( \Delta \) means the optimized prompts helped; a negative one means they hurt.

## The four factors

The benchmark varies one factor at a time and holds the others at their defaults.

<div class="cards factors" markdown>

- [Task](../tasks/index.md)
  Nine datasets in three domains: reasoning, coding and tool-calling.
- [Workflow topology](../mas/topologies.md)
  Single, Independent, Sequential, Centralized and Decentralized, on four frameworks.
- [Communication protocol](../mas/communication-protocols.md)
  Freeform, Semi-structured or Structured messages between agents.
- [Team size](../mas/team-sizes.md)
  Teams of 2, 4, 8 and 10 agents.

</div>

Two [optimizers](../optimizers/index.md) run over every factor: **GEPA** (reflective prompt evolution) and **MIPRO** (instruction and few-shot example search). Both score candidate prompts by running the real topology runners, so the prompts they return run unchanged in the benchmark.

## Vocabulary

Cell
:   One configuration, such as *BFCL · Sequential (CrewAI)*. Each cell is evaluated twice, with seed and optimized prompts, on the same examples.

Runner
:   The Python module that runs one cell, for example `topologies.sequential.crewai.bfcl.crewai_bfcl`. Every runner has a batch mode; most also have a smoke demo.

Role and seed prompt
:   A position in the topology (stage, worker, manager, peer) and the system-prompt file it starts from.

Eval IDs
:   The frozen example IDs per dataset in `benchmarks/<dataset>/<dataset>_eval_ids.json`. The optimizers keep them out of their train and validation splits. See [Evaluation Protocol](../evaluation/protocol.md).

## Next steps

<div class="cards" markdown>

- [Installation](installation.md)
  Clone with submodules and create the conda environment.
- [Quick Start](quick-start.md)
  Run a baseline, optimize it and read \( \Delta \) in a few commands.

</div>
