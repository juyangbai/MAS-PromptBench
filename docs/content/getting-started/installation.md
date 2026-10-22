# Installation

Create one conda environment for the benchmark, then install the OpenAI Agents SDK into its own folder. It takes a few minutes and needs no GPU unless you serve models locally.
{ .lede }

<div class="facts" markdown>
<div><span>Python</span>3.11</div>
<div><span>Environment</span>conda</div>
<div><span>Frameworks</span>Pinned upstream commits</div>
<div><span>GPU</span>Optional</div>
</div>

## Clone the repository

The code is on [GitHub](https://github.com/juyangbai/MAS-PromptBench).

```bash title="Clone"
git clone https://github.com/juyangbai/MAS-PromptBench.git
cd MAS-PromptBench
```

The agent frameworks come with the environment in the next step; nothing else needs cloning.

## Create the environment

```bash title="Create and activate"
conda env create -f environment.yml
conda activate mas-promptbench
```

The environment holds:

- **Python 3.11** with NumPy and pandas.
- **The agent frameworks**: LangGraph, CrewAI and AutoGen, each installed from the upstream commit pinned in [`environment.yml`](https://github.com/juyangbai/MAS-PromptBench/blob/main/environment.yml), plus LangChain's OpenAI client and the OpenAI SDK (`openai<3`).
- **Benchmark tooling**: Hugging Face `datasets` for loading tasks, the `wikipedia` client used by HotpotQA retrieval, the official BFCL AST checker (`bfcl-eval`) and the SWE-bench harness (`swebench`).
- **Prompt optimization**: DSPy and `gepa`, plus `torch`, `torch_geometric` and `sentence-transformers` (CPU is enough).
- **Local serving**: vLLM and a CUDA 12.8 toolchain, used only if you serve models yourself.

## Install the OpenAI Agents SDK

The Decentralized topology has a variant on the OpenAI Agents SDK. The SDK needs `openai>=3`, which conflicts with the main environment, so install it into its own folder from the pinned list in [`requirements-openai-agents.txt`](https://github.com/juyangbai/MAS-PromptBench/blob/main/requirements-openai-agents.txt):

```bash title="Isolated SDK install"
pip install --target vendor/openai_agents -r requirements-openai-agents.txt
```

You never put this folder on the path yourself. The Agents SDK runners, and `python -m optimizers.protocol.run` for an Agents SDK cell, restart themselves with it first on `PYTHONPATH`. If the SDK is missing or unusable, an Agents SDK runner exits with status 1 before the first row and prints the reason and the fix. To use an install elsewhere, set `OPENAI_AGENTS_PATH`.

Skip this step if you don't run Agents SDK cells.

## Check the install

Run one smoke demo from the repository root. It uses a built-in example, so it doesn't download a dataset, but it does call a model, so [connect a model](connect-a-model.md) first:

```bash title="Smoke test"
python -m topologies.single.hotpotqa.langgraph_hotpotqa
```

The demo asks one built-in question, "Were Scott Derrickson and Ed Wood of the same nationality?", whose expected answer is `yes`. A correct run prints lines like these:

```text title="Expected output"
=== Extracted answer: 'yes'  (expected: 'yes') ===
=== EM: 1.00   F1: 1.00   P: 1.00   R: 1.00 ===
=== Full message trace ===
```

followed by every message the agent exchanged, including any Wikipedia searches.

!!! warning "Run runners as modules"
    Start every runner with `python -m` from the repository root. The runners import the shared `core` package, so `python topologies/single/hotpotqa/langgraph_hotpotqa.py` stops with `ModuleNotFoundError: No module named 'core'`. If you prefer file paths, prefix the command with `PYTHONPATH=.`.

## Next steps

<div class="cards" markdown>

- [Connect a Model](connect-a-model.md)
  Serve the task model and point every agent at it.
- [Quick Start](quick-start.md)
  Run a baseline, optimize its prompts and read the result.

</div>
