# Installation

Clone the repository with its agent-framework submodules and create one conda environment. It takes a few minutes and needs no GPU unless you plan to serve models locally.
{ .lede }

<div class="facts" markdown>
<div><span>Python</span>3.11</div>
<div><span>Environment</span>conda</div>
<div><span>Frameworks</span>4 submodules</div>
<div><span>GPU</span>Optional</div>
</div>

## Clone the repository

The four agent frameworks live in `frameworks/` as git submodules, so clone them along with the code:

```bash title="Clone with submodules"
git clone https://github.com/juyangbai/MAS-PromptBench.git
cd MAS-PromptBench
git submodule update --init --recursive
```

| Submodule | Used by |
| --- | --- |
| `frameworks/langgraph` | every topology (the reference implementation) |
| `frameworks/crewAI` | the CrewAI variant of [Sequential](../mas/sequential.md) |
| `frameworks/autogen` | the AutoGen variant of [Centralized](../mas/centralized.md) |
| `frameworks/llm_multiagent_debate` | reference for the [Decentralized](../mas/decentralized.md) debate |

!!! warning "Don't skip the submodules"
    `environment.yml` installs LangGraph, CrewAI and AutoGen in editable mode from `frameworks/`. If those folders are empty, the environment build fails.

## Create the environment

```bash title="Create and activate"
conda env create -f environment.yml
conda activate mas-promptbench
```

The environment holds:

- **Python 3.11** with NumPy and pandas.
- **The agent frameworks**: LangGraph, CrewAI and AutoGen (editable installs), plus the OpenAI SDK.
- **Benchmark tooling**: Hugging Face `datasets` for loading tasks, the `wikipedia` client used by HotpotQA retrieval, the official BFCL AST checker (`bfcl-eval`) and the SWE-bench harness (`swebench`).
- **Prompt optimization**: DSPy, which runs GEPA and MIPRO.
- **Local serving**: vLLM and a CUDA 12.8 toolchain, used only if you serve models yourself.

!!! warning "Upgrade DSPy before running GEPA"
    `environment.yml` pins `dspy>=2.6,<3`, but the GEPA optimizer (`dspy.teleprompt.GEPA`) ships with DSPy 3. Upgrade inside the environment before you run GEPA:

    ```bash
    pip install -U "dspy[optuna]>=3"
    ```

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
    Start runners with `python -m` from the repository root. Most runner files import the `topologies` package before they add the repository root to the import path, so `python topologies/single/hotpotqa/langgraph_hotpotqa.py` stops with `ModuleNotFoundError: No module named 'topologies'`. If you prefer file paths, prefix the command with `PYTHONPATH=.`.

## Next steps

<div class="cards" markdown>

- [Connect a Model](connect-a-model.md)
  Point every agent at a hosted API or a local vLLM server.
- [Quick Start](quick-start.md)
  Run a baseline, optimize its prompts and read the result.

</div>
