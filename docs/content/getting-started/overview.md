# Overview

MAS-PromptBench measures when optimizing system prompts improves a multi-agent LLM system, and by how much. This page explains the model of a multi-agent system the benchmark uses, the factors it varies, and the one number every optimizer job reports.
{ .lede }

## The question

Most prompt optimizers were designed for a single LLM agent. A multi-agent system (MAS) is harder: each agent's prompt can improve locally, yet the system can still get worse once agents hand work to each other. MAS-PromptBench runs eight optimizers across many controlled MAS configurations so you can see where gains transfer and where they break.

## A multi-agent system, formally

The benchmark models a MAS as a tuple \( \mathcal{M} = (\mathcal{A}, G, P) \):

- \( \mathcal{A} \) is the set of agents. Each agent pairs a frozen LLM with a learnable **system prompt** for its role (planner, solver, manager, debater, …).
- \( G \) is the **workflow topology**: who sends work to whom.
- \( P \) is the **communication protocol**: the format of the messages agents exchange.

Model weights never change. Optimization edits only the joint set of role prompts \( \pi \). The seed prompts \( \pi^0 \) live in `configs/prompts/<topology>/<dataset>/<role>.txt`, and no optimizer writes to them.

## The prompt-optimization gain

For a configuration \( (\mathcal{T}, G, n, P) \) (task, topology, team size, protocol), the benchmark reports the gain of the deployed prompts \( \hat{\pi} \) over the seed prompts \( \pi^0 \) on the task's held-out test split:

\[
\Delta(\mathcal{T}, G, n, P) = \mathbb{E}_{(x,y)\sim\mathcal{T}_{\text{test}}}\big[\,\mu(\mathcal{M}(x;\hat{\pi}), y) - \mu(\mathcal{M}(x;\pi^0), y)\,\big]
\]

\( \mu \) is the task's scorer (exact match, pass@1, AST match, …), 0 or 1 per example. \( \hat{\pi} \) is the optimizer's best prompt set only if it scores strictly higher than the seeds on the validation split; otherwise the seeds stay deployed. Each job writes \( \Delta \) as `delta_pp`, in percentage points, in its `result.json`. A positive \( \Delta \) means the deployed prompts helped; a negative one means they hurt.

## The factors

The benchmark varies one factor at a time and holds the others at their defaults.

<div class="cards factors" markdown>

- [Task](../tasks/index.md)
  Nine datasets in three domains: reasoning, coding and tool calling.
- [Workflow topology](../mas/topologies.md)
  Single, Independent, Sequential, Centralized and Decentralized, on four frameworks.
- [Communication protocol](../mas/communication-protocols.md)
  Freeform, Semi-structured or Structured messages between agents.
- [Team size](../mas/team-sizes.md)
  Teams of 2, 4, 8 and 10 agents.

</div>

Every agent runs on the task model `Qwen/Qwen3.5-9B`; `meta-llama/Llama-3.1-8B-Instruct` repeats a subset of cells. Eight [optimizers](../optimizers/index.md) run over these factors: **GEPA**, **MIPRO**, **MAPRO**, **MASPO**, **HiveMind**, **MAMUT-GEPA**, **MASPOB** and **TAVO**. All eight follow one run protocol and score candidate prompts by running the real topology runners, so the prompts they return run unchanged in the benchmark.

## Vocabulary

Cell
:   One runtime configuration: task, topology, framework, communication format, team size and task model, such as *BFCL · Sequential (CrewAI) · freeform · 4 agents · Qwen3.5-9B*.

Job
:   One optimizer on one cell with one optimizer seed (0, 1 or 2). It optimizes, selects on validation and scores seed and deployed prompts on the same test examples.

Runner
:   The Python module that runs one cell, for example `topologies.sequential.crewai.bfcl.crewai_bfcl`. Every runner has a batch mode; five datasets also have a smoke demo.

Role and seed prompt
:   A position in the topology (stage, worker, manager, peer) and the system-prompt file it starts from.

Splits and eval IDs
:   Fixed `train`, `validation` and `test` IDs per dataset in `benchmarks/<dataset>/<dataset>_splits.json`. `test` is the evaluation set, also listed in `<dataset>_eval_ids.json`; `train` and `validation` never overlap it. See [Evaluation Protocol](../evaluation/protocol.md).

## Next steps

<div class="cards" markdown>

- [Installation](installation.md)
  Create the conda environment and the isolated Agents SDK install.
- [Quick Start](quick-start.md)
  Run a baseline, optimize it and read \( \Delta \) in a few commands.

</div>
