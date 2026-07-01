

<div align="center">
  <img src="assets/MAS-PromptBench_lockup.svg" alt="MAS-PromptBench" width="560">
</div>

<h2 align="center">
  A Benchmark of Prompt Optimization for Multi-Agent LLM Systems
</h2>

<p align="center">
  <a href="https://juyangbai.github.io/MAS-PromptBench/" style="margin: 0 10px;">🌐 Website</a> |
  <a href="https://arxiv.org/abs/2606.23664" style="margin: 0 10px;">📖 Paper</a> |
  <a href="https://github.com/juyangbai/MAS-PromptBench" style="margin: 0 10px;">💻 GitHub</a>
</p>

<details open>
<summary><b>Contents</b></summary>

1. [News and Updates](#-news-and-updates)
2. [Introduction](#-introduction)
3. [Code Structure](#-code-structure)
4. [Quickstart](#-quickstart)
5. [Referenced Resources](#-referenced-resources)
6. [Contributing](#-contributing)

</details>

---

## 🔔 News and Updates

- **[2026-06-22]** — Initial public release of the code!

---

## 📖 Introduction

<div align="center">
  <img src="assets/MAS-PromptBench_overview.png" alt="MAS-PromptBench overview" width="820">
</div>

A reproducible benchmark for studying when prompt optimization improves multi-agent LLM systems across optimizers, tasks, topologies, communication formats, and team sizes.

- **Optimizer** — GEPA and MIPRO prompt optimizers, run on the real multi-agent runners.
- **Task dataset** — 9 reasoning, coding, and tool-use benchmarks, each scored with its official / community-standard scorer.
- **Workflow Topology** — `single`, `independent`, `sequential`, `centralized`, and `decentralized`, implemented across LangGraph, CrewAI, AutoGen, and the OpenAI SDK.
- **Communication format** — three inter-agent message formats (`freeform`, `semi_structured`, `structured_soft`).
- **Team size** — the number of agents per team, `r ∈ {2, 4, 8, 10}`.

---

## 🌳 Code Structure


| Path                                 | Contents                                                                  |
| ------------------------------------ | ------------------------------------------------------------------------- |
| [`benchmarks/`](benchmarks/)         | per-dataset evaluation-ID manifests + the API-Bank source                 |
| [`communications/`](communications/) | inter-agent message-format studies                                        |
| [`configs/`](configs/)               | seed per-role prompts (`configs/prompts/<topology>/<dataset>/<role>.txt`) |
| [`frameworks/`](frameworks/)         | LangGraph, CrewAI, AutoGen, and debate submodules (editable installs)     |
| [`models/`](models/)                 | node-agnostic vLLM serve scripts (Qwen3.5-9B / 122B)                      |
| [`optimizers/`](optimizers/)         | GEPA and MIPRO prompt optimizers over the real runners                    |
| [`scripts/`](scripts/)               | helper and launcher scripts                                               |
| [`teamsizes/`](teamsizes/)           | team-size sweeps (number of agents per team)                              |
| [`topologies/`](topologies/)         | the core benchmark — 5 topologies × 9 datasets, one runnable pair each    |


---

## 🚀 Quickstart

Go from a fresh clone to scored results in four steps — install, connect a model, run a baseline, then optimize its prompts.

### 1. Install

Clone with the framework submodules and create the conda environment:

```bash
git clone https://github.com/juyangbai/MAS-PromptBench.git
cd MAS-PromptBench
git submodule update --init --recursive

conda env create -f environment.yml      # Python 3.11 + vLLM + benchmark deps
conda activate mas-promptbench
```

### 2. Connect a model

Every agent talks to the same **OpenAI-compatible** chat endpoint, configured via `VLLM_BASE_URL` and `MODEL_ID` (plus `OPENAI_API_KEY` when the provider requires one). Pick one path:

*Option A — Use a model API (no GPU).* Manage keys in a `.env` file — uncomment one provider block, fill in your key, and load it:

```bash
# edit .env — pick a provider, set OPENAI_API_KEY
set -a && source .env && set +a
```

`.env` ships ready-to-use blocks for **OpenAI**, **Anthropic**, **Google Gemini**, **Mistral AI**, **DeepSeek**, and **local vLLM** — no local GPU required.

*Option B — Local serving (vLLM, on your own GPUs).* Serve a Qwen model from [`models/`](models/), then point runs at it:

```bash
bash models/serve_qwen3_5_9b.sh
export VLLM_BASE_URL=http://localhost:8000/v1
export MODEL_ID=Qwen/Qwen3.5-9B
# OPENAI_API_KEY not needed (the local endpoint accepts any key)
```

| Script | Model | Serving | GPUs needed |
|---|---|---|---|
| [`serve_qwen3_5_9b.sh`](models/serve_qwen3_5_9b.sh) | `Qwen/Qwen3.5-9B` | one replica per GPU | **≥ 1** CUDA GPU |
| [`serve_qwen3_5_122b.sh`](models/serve_qwen3_5_122b.sh) | `Qwen/Qwen3.5-122B-A10B-FP8` | tensor-parallel (TP=4) | **4** FP8-capable GPUs (Hopper / Blackwell) |

### 3. Run a baseline

Every `(topology, dataset)` pair ships a no-arg smoke demo and a `--batch` mode that writes predictions, per-instance results, and traces to `results/<dataset>/`:

```bash
# smoke demo (built-in example, no dataset download)
python topologies/single/hotpotqa/langgraph_hotpotqa.py

# real batch on a slice
python topologies/single/hotpotqa/langgraph_hotpotqa.py --batch --limit 100
```

See [topologies/README.md](topologies/README.md) for the full run interface, per-dataset setup, and scoring details.

### 4. Optimize prompts

[`optimizers/`](optimizers/) holds two optimizers — **GEPA** (reflective prompt evolution) and **MIPRO** (instruction + few-shot example search) — that improve the seed prompts by running the **real** topology runners and re-scoring, so gains transfer directly back to the benchmark:

```bash
# from optimizers/gepa/
python -m real_runner_gepa.pilots.run_gepa_dataset \
  --dataset math --topology single --train-size 25 --val-size 25 \
  --max-full-evals 5 --out results/gepa/single_math
```

Train/val splits draw from the frozen eval-ID manifests, so optimization never trains on reported eval instances. See [optimizers/README.md](optimizers/README.md).

---

## 🔗 Referenced Resources

The agent frameworks under `frameworks/` retain their own upstream licenses:

- **[LangGraph](https://github.com/langchain-ai/langgraph)** — stateful graph-based agent orchestration
- **[CrewAI](https://github.com/crewAIInc/crewAI)** — role-based multi-agent framework
- **[AutoGen](https://github.com/microsoft/autogen)** — conversational multi-agent framework
- **[LLM Multi-Agent Debate](https://github.com/composable-models/llm_multiagent_debate)** — multi-agent debate reference implementation

MAS-PromptBench evaluates on nine existing benchmarks. Please cite and comply with the license of each original dataset when reporting results:

- **[GPQA](https://github.com/idavidrein/gpqa)** — graduate-level science multiple-choice QA
- **[HotpotQA](https://hotpotqa.github.io/)** — multi-hop open-domain QA
- **[MATH](https://github.com/hendrycks/math)** — competition mathematics
- **[LiveCodeBench](https://livecodebench.github.io/)** — contamination-free code generation
- **[APPS](https://github.com/hendrycks/apps)** — programming problems
- **[Berkeley Function Calling Leaderboard (BFCL)](https://gorilla.cs.berkeley.edu/leaderboard.html)** — function / tool calling
- **[SWE-bench Verified](https://www.swebench.com/)** — real-world GitHub issue resolution
- **[API-Bank](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/api-bank)** — tool-augmented API calling
- **[ToolHop](https://huggingface.co/datasets/bytedance-research/ToolHop)** — multi-hop tool use

MAS-PromptBench also builds on:

- **[vLLM](https://github.com/vllm-project/vllm)** — high-throughput, OpenAI-compatible LLM serving (the `models/` endpoints)
- **[DSPy](https://github.com/stanfordnlp/dspy)** — the prompt-optimization backend behind GEPA and MIPRO

---

## 🤝 Contributing

Contributions are very welcome — a new topology, dataset, or optimizer, a framework integration, a bug fix, or even a typo. Every bit helps!

A few tips to make it smooth:

- For anything substantial, open an issue first so we can talk through the approach together.
- Match the existing pair / runner conventions — [topologies/README.md](topologies/README.md) and the per-optimizer `ADDING_PAIR.md` guides are great starting points.
- Give the relevant smoke demo a quick run (plus a small `--batch --limit` slice) before opening your PR.

Then send over a pull request — and thank you for helping make MAS-PromptBench better! 🙌

---

## 📚 Citation

If you use MAS-PromptBench in your research, please cite it:

```bibtex
@article{bai2026mas,
  title={MAS-PromptBench: When Does Prompt Optimization Improve Multi-Agent LLM Systems?},
  author={Bai, Juyang and Shi, Laixi},
  journal={arXiv preprint arXiv:2606.23664},
  year={2026}
}
```

---

## ⚖️ License

MAS-PromptBench is released under the [MIT License](LICENSE). The framework libraries under `frameworks/` and the API-Bank source under `benchmarks/apibank/apibank_upstream/` retain their respective upstream licenses — comply with each when redistributing.