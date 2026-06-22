# Benchmarks

This folder defines the nine evaluation datasets that MAS-PromptBench runs its
multi-agent topologies against. For each dataset,
`<dataset>/<dataset>_eval_ids.json` lists the exact instance IDs used for the
reported scores (`{dataset, sample, n, source, ids[]}`); the runners load them
through their `--only` filter, so every topology, team-size, and
communication-format run is scored on the same instances. Dataset content is
fetched from Hugging Face at load time — the sole exception is API-Bank, whose
source is vendored in-repo at
[`apibank/apibank_upstream/`](apibank/apibank_upstream) (from
`AlibabaResearch/DAMO-ConvAI`, under its bundled `LICENSE`) and rebuilt into its
curated manifest automatically, so no setup is required.

The sections below cover the benchmark matrix and each dataset's environment
requirements.

## Benchmark Matrix

**5 topologies** — each a distinct coordination pattern, with one or two framework implementations:

| Topology | Shape | Inter-agent communication | Frameworks |
|---|---|---|---|
| `single` | self-loop (1 agent) | — | LangGraph |
| `independent` | parallel fan-out | none (ensemble) | LangGraph |
| `sequential` | linear pipeline | stage → stage | LangGraph, CrewAI |
| `centralized` | hub-and-spoke | via manager only | LangGraph, AutoGen |
| `decentralized` | peer debate | all-to-all, per round | LangGraph, OpenAI SDK |

**9 datasets** — each scored by its official / community-standard scorer:

| Dataset | Task | Scoring | Eval set |
|---|---|---|---:|
| `gpqa` | Graduate-level science MCQ | letter-match accuracy | 100 |
| `hotpotqa` | Multi-hop open-domain QA | official EM + token F1 | 100 |
| `math` | Competition math | Hendrycks `is_equiv` (`\boxed{}`) | 100 |
| `lcb` | LiveCodeBench coding | pass@1 (stdin + functional) | 50 |
| `apps` | APPS coding | pass@1 (stdin + call-based) | 50 |
| `bfcl` | Function calling | official `bfcl_eval` AST scorer | 25 |
| `swe` | SWE-bench Verified patching | `FAIL_TO_PASS` + `PASS_TO_PASS` | 30 |
| `apibank` | API-Bank API calls (L1–L3) | official API-Bank harness | 100 |
| `toolhop` | ToolHop multi-hop tool use | exact / normalized answer match | 100 |

The frozen evaluation IDs for each dataset ship under `benchmarks/<dataset>/<dataset>_eval_ids.json` — **655 instances total** — so every run scores the same set.

## Environment Requirements

| Dataset | Sandbox | Python deps to add | External service / repo | Auth | Disk |
|---|---|---|---|---|---|
| **GPQA** | — | — | HF (gated) | HF token + terms | <10 MB |
| **MATH** | — | — | HF (public) | — | <10 MB |
| **HotpotQA** | — | — (`wikipedia` already in env.yml) | HF + Wikipedia REST | — | <2 GB |
| **BFCL** | — | — | HF `gorilla-llm/Berkeley-Function-Calling-Leaderboard` | — | <50 MB |
| **LCB** | Tier 1 | — | HF `livecodebench/code_generation_lite` | — | ~100 MB data + 150 MB SIF |
| **APPS** | Tier 1 | — (`numpy` already in env) | HF `codeparrot/apps` | — | ~300 MB data + shared SIF |
| **SWE-bench** | Tier 2 | `swebench>=4.0,<5` | HF + Docker Hub (`docker://swebench/sweb.eval.x86_64.*`) | — | ~67 GB (optimized) to ~189 GB full |
| **API-Bank curated** | — | — | ships in-repo (`apibank_upstream/`) | — | 4.6 MB in-repo |
| **ToolHop** | — | — | HF `bytedance-research/ToolHop` | — | <50 MB |

The **Sandbox tiers** refers to two execution tiers:

- **Tier 1** (`python311.sif`) — model-generated Python is executed; single shared
  SIF covers LCB and APPS.
- **Tier 2** (per-instance SIFs) — the repository's own code is installed and its
  test suite is run; used only by SWE-bench. Images are pulled from
  `docker://swebench/sweb.eval.x86_64.<instance_id>`.
