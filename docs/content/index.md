---
title: Home
hide:
  - toc
---

<div class="home-hero" markdown>

<span class="eyebrow">Tutorial and reference</span>

# MAS-PromptBench Docs { .brand-title .mpb-hero }

When does prompt optimization improve a multi-agent LLM system? These docs show you how to run the benchmark, optimize its system prompts with eight optimizers under one run protocol, and measure the gain they produce.
{ .hero-lede }

<div class="hero-actions" markdown>
[Get started](getting-started/installation.md){ .btn .btn-primary }
[Ask DeepWiki](https://deepwiki.com/juyangbai/MAS-PromptBench){ .btn .btn-deepwiki }
</div>

<div class="hero-meta">
<span><b>9</b> tasks</span><span><b>5</b> topologies</span><span><b>4</b> frameworks</span><span><b>3</b> protocols</span><span><b>4</b> team sizes</span><span><b>8</b> optimizers</span>
</div>

</div>

## Start here

<div class="cards" markdown>

- [Overview](getting-started/overview.md)
  The model of a multi-agent system, the factors it varies and the gain \( \Delta \).
- [Installation](getting-started/installation.md)
  Create the conda environment and the isolated Agents SDK install.
- [Connect a Model](getting-started/connect-a-model.md)
  Serve Qwen3.5-9B with vLLM and point the runners and optimizers at it.
- [Quick Start](getting-started/quick-start.md)
  Run a baseline, optimize its prompts and read the result.

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
  GEPA, MIPRO, MAPRO, MASPO, HiveMind, MAMUT-GEPA, MASPOB and TAVO, run over the real topology runners.

</div>

## Extend and look up

<div class="cards" markdown>

- [Add a Dataset](extending/add-dataset.md)
  Task module, team spec, runners, prompts, splits and optimizer adapter.
- [Add a Topology](extending/add-topology.md)
  A new coordination structure or a framework variant of an existing one.
- [Command-Line Flags](reference/cli.md)
  Every runner and run-protocol flag, with defaults.
- [Environment Variables](reference/environment.md)
  Every setting the runners and optimizers read, with defaults.

</div>
