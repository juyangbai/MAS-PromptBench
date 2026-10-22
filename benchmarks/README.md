# Benchmarks

This folder defines the nine evaluation datasets that MAS-PromptBench runs its
multi-agent topologies against. Per dataset, `<dataset>/<dataset>_eval_ids.json`
lists the exact instance IDs of the reported scores
(`{dataset, sample, n, source, ids[]}`) and `<dataset>/<dataset>_splits.json`
the fixed [optimization splits](#optimization-splits). Dataset content is
fetched from Hugging Face at load time, except API-Bank: its source is vendored
at [`apibank/apibank_upstream/`](apibank/apibank_upstream) (from
`AlibabaResearch/DAMO-ConvAI`, under its bundled `LICENSE`) and rebuilt into its
curated manifest automatically, so no setup is required.

## Benchmark Matrix

**5 topologies** — `single`, `independent`, `sequential`, `centralized` and `decentralized`, described in [topologies/](../topologies/README.md#overview).

**9 datasets** — each scored by its task module (`core/tasks/<dataset>.py`):

| Dataset | Task | Scoring | Train | Val | Test (eval) |
|---|---|---|---:|---:|---:|
| `gpqa` | GPQA-Diamond science MCQ | extracted option letter = gold letter | 48 | 50 | 100 |
| `hotpotqa` | Multi-hop open-domain QA | official EM and token F1 | 150 | 50 | 100 |
| `math` | Competition math (Precalculus, Level 5) | Hendrycks `is_equiv` of the last `\boxed{}` | 150 | 50 | 100 |
| `lcb` | LiveCodeBench coding | pass@1 on all private tests (stdin and functional) | 150 | 50 | 50 |
| `apps` | APPS coding (interview) | pass@1 (strict accuracy) on the first 20 tests (`--max-tests-per-row`) | 150 | 50 | 50 |
| `bfcl` | Function calling | `bfcl_eval` AST checker | 150 | 50 | 100 |
| `swe` | SWE-bench Verified patching | resolved: every `FAIL_TO_PASS` and `PASS_TO_PASS` test passes in the instance image (`--eval singularity`) | 146 | 50 | 30 |
| `apibank` | API-Bank API calls (L1–L3) | the call, replayed through API-Bank's API classes, passes the API's own check | 150 | 50 | 100 |
| `toolhop` | ToolHop multi-hop tool use | ToolHop's matcher: equal Python literals, else the gold answer in the prediction (case-insensitive) or the last tool result | 150 | 50 | 100 |

- **730 evaluation IDs** in total. The sweep scripts select them with `--only` (BFCL, SWE-bench) or `--limit N` (whose first N instances are exactly these IDs), so every topology, team-size and format run scores the same instances.
- **BFCL**'s IDs are a seed-0 stratified sample (40 `simple`, 20 each of `multiple`, `parallel`, `parallel_multiple`), run one AST category at a time.
- **The run protocol** scores LiveCodeBench and APPS on the first 3 tests only and SWE-bench with a structural patch check (no tests run); see its [Scoring](../optimizers/protocol/README.md#scoring).

## Optimization Splits

**3 fixed splits** per dataset (sizes above) for the prompt optimizers: `test` is the evaluation set, in the same order; `train` and `validation` are drawn with `split_seed` 0 from the rest, so optimization never sees a reported instance. GPQA-Diamond has only 198 questions, hence 48 for train. API-Bank's splits come from the 445-task curated pool in `apibank/apibank_pool_ids.json`.

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
  `docker://swebench/sweb.eval.x86_64.<instance_id>`. The agents' `shell_exec`
  tool runs sandboxed in the same image ([`core/swe_sandbox.py`](../core/swe_sandbox.py):
  no network, `.git` read-only, bounded resources; `SWE_SHELL_SANDBOX=0` runs it
  on the host for debugging, see [settings](../docs/content/reference/environment.md#dataset-settings)).

ToolHop runners execute dataset-provided tool code and require `TOOLHOP_ALLOW_DATASET_EXEC=1` ([settings](../docs/content/reference/environment.md#dataset-settings)).
