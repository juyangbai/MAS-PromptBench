---
title: Home
hide:
  - toc
---

<div class="home-hero" markdown>

<span class="eyebrow">Tutorial and reference</span>

# MAS-PromptBench Docs { .brand-title .mpb-hero }

When does prompt optimization improve a multi-agent LLM system? These docs show you how to run the benchmark, optimize system prompts with GEPA or MIPRO, and read the gain they produce.
{ .hero-lede }

<div class="hero-actions" markdown>
[Get started](getting-started/installation.md){ .btn .btn-primary }
</div>

<div class="hero-meta">
<span><b>9</b> tasks</span><span><b>5</b> topologies</span><span><b>4</b> frameworks</span><span><b>3</b> protocols</span><span><b>4</b> team sizes</span><span><b>2</b> optimizers</span>
</div>

</div>

## Start here

<div class="cards" markdown>

- [Overview](getting-started/overview.md)
  The model of a multi-agent system, the four factors and the gain \( \Delta \).
- [Installation](getting-started/installation.md)
  Clone with submodules and create the conda environment.
- [Connect a Model](getting-started/connect-a-model.md)
  Use a hosted API or serve Qwen locally with vLLM.
- [Quick Start](getting-started/quick-start.md)
  Baseline, optimize and read the result in five steps.

</div>

## The benchmark

Each factor is varied on its own while the others stay at their defaults.

<div class="cards factors" markdown>

- [Tasks](tasks/index.md)
  GPQA-Diamond, HotpotQA, MATH, LiveCodeBench, APPS, SWE-bench Verified, BFCL, ToolHop and API-Bank.
- [Workflow Topologies](mas/topologies.md)
  Single, Independent, Sequential, Centralized and Decentralized.
- [Communication Protocols](mas/communication-protocols.md)
  Freeform, Semi-structured and Structured messages.
- [Team Sizes](mas/team-sizes.md)
  The same topology with 2, 4, 8 and 10 agents.
- [Optimizers](optimizers/index.md)
  GEPA and MIPRO, run over the real topology runners.

</div>

## Extend and look up

<div class="cards" markdown>

- [Add a Dataset](extending/add-dataset.md)
  Runners, scorer, output contract, eval IDs, prompts and optimizer adapter.
- [Add a Topology](extending/add-topology.md)
  A new coordination structure or a framework variant of an existing one.
- [Command-Line Flags](reference/cli.md)
  Every runner and optimizer flag, with defaults.
- [Read the Results](evaluation/results.md)
  The files an optimizer run writes and how to turn them into numbers.

</div>
