# Environment Variables

Every environment variable the benchmark reads, grouped by area, with its default in the code. Defaults are the code's unless a script sets its own; variables read only inside the agent frameworks are left out.
{ .lede }

Most runs need only `VLLM_BASE_URL` and `MODEL_ID` for the runners, or `TASK_ENDPOINTS` and `REFLECTION_MODEL_BASE_URL` for the optimizers. Command-line options are on [Command-Line Flags](cli.md).

## Logging

| Variable / option | Default | Meaning |
| --- | --- | --- |
| `LOG_LEVEL` | `INFO` | Console level of the repository's loggers, for every command line (`core/logs.py`). |
| `--log-level` | `$LOG_LEVEL` | Overrides it for `optimizers.protocol.run` and `optimizers.protocol.aggregate`. |

Progress goes to stderr. Stdout carries only results: reports, JSON summaries, demos and the aggregate table.

## Task model and decoding

The runners read these through `core/settings.py` at call time, so a change applies to the next client a runner builds.

| Variable | Default | Meaning |
| --- | --- | --- |
| `VLLM_BASE_URL` | `http://localhost:8000/v1` | OpenAI-compatible endpoint of every agent. |
| `MODEL_ID` | `Qwen/Qwen3.5-9B` | Served model name sent with each request. |
| `OPENAI_API_KEY` | `EMPTY` | API key; a local vLLM server accepts any value. |
| `TASK_MODEL_TEMPERATURE` | `0.0` | Sampling temperature. |
| `TASK_MODEL_TOP_P` | `0.9` | Nucleus mass. |
| `TASK_MODEL_MAX_TOKENS` | `32768` | Output-token cap per model call. |
| `REQUEST_SEED` | `0` | Per-request sampling seed. |

Fixed in code: the vLLM `extra_body` with `repetition_penalty` 1.05 and `chat_template_kwargs.enable_thinking` false. The [sweep launchers](#sweep-launchers) and the [run protocol](#run-protocol) set some of these variables themselves.

## Team shape

For runners and adapters without a team spec (`configs/teams/<dataset>.yaml`) or a team size from the protocol:

| Variable | Default | Read by |
| --- | --- | --- |
| `INDEPENDENT_N_AGENTS` | `4` | API-Bank and ToolHop runners; bridge adapters when no team size is passed |
| `DECENTRALIZED_N_AGENTS` | `4` | same |
| `DECENTRALIZED_N_ROUNDS` | `2` | same |
| `TOOLHOP_<VAR>`, `APIBANK_<VAR>` | the unprefixed value | ToolHop and API-Bank, before the three above |
| `N_AGENTS`, `N_ROUNDS` | `4`, `2` | bridge communication and team-size adapters, after the above |

## Dataset settings

Read by `core/tasks/<dataset>.py` unless noted.

| Variable | Default | Meaning |
| --- | --- | --- |
| `TOOLHOP_ALLOW_DATASET_EXEC` | unset | Must be `1` to execute the dataset-provided tool source. The sweep launchers and the run protocol set it. |
| `TOOLHOP_MAX_TURNS` | `9` | Tool-loop turns per agent. |
| `TOOLHOP_TOOL_RESULT_CHAR_BUDGET` | `6000` | Characters of a tool result kept in the conversation. |
| `APIBANK_LEVEL` | `all` | Level slice; `--level` overrides it. |
| `APIBANK_ROOT` | `benchmarks/apibank/apibank_upstream/api-bank` | API-Bank source checkout. |
| `APIBANK_CURATED_PATH` | `benchmarks/apibank/apibank_eval_ids.json` (levels 1 to 3: per-level manifests) | Curated manifest; `--curated-path` sets it, and the bridge reads it too. |
| `APIBANK_CURATED_LIMIT` | unset | Cap when a manifest has to be rebuilt. |
| `APIBANK_LEVEL3_JSON` | upstream `level-3.json` | Level-3 data file. |
| `APIBANK_TOOLSEARCHER_SCORER` | `official` (level 1: `keyword`) | ToolSearcher scorer; `--toolsearcher-scorer` sets it. |
| `APIBANK_TOOLSEARCHER_MODEL` | `sentence-transformers/paraphrase-MiniLM-L3-v2` | ToolSearcher embedding model. |
| `APIBANK_TOOLSEARCHER_DEVICE` | `cpu` | Its device. |
| `APIBANK_TOOLSEARCHER_{SKLEARN,GOOGLETRANS,NLTK,BM25}_SHIM` | `1` | `0` disables the stub of that optional upstream dependency. |
| `APIBANK_REQUEST_TIMEOUT` | `60` | Seconds per model request. |
| `APIBANK_OPENAI_MAX_RETRIES` | `0` | Client retries. |
| `SWE_SIF_DIR` | `~/containers/swe` | Per-instance Singularity images, pulled on first use. |
| `SWE_SHELL_SANDBOX` | `1` | `0` runs the agents' shell commands on the host, for debugging only (`core/swe_sandbox.py`). |
| `SWE_EVAL_LOG_DIR` | `<out-dir>/eval_logs` | Pytest logs of the image evaluation (`eval_<instance id>.log`). |
| `SWE_REPO_DIR` | `.` | Repository of the tools when no checkout is bound. |
| `SWE_PROBLEM_CHAR_BUDGET`, `SWE_HINTS_CHAR_BUDGET` | `16000`, `4000` | Issue and hints characters in the brief. |
| `SWE_WORKDIR_ROOT` | `~/swe_work_independent` (team sizes: `..._r<r>`) | Clone root of the replicas (`topologies/independent/swe`). |
| `LCB_FUNCTIONAL_MEMORY_BYTES`, `APPS_CALL_BASED_MEMORY_BYTES` | 4 GiB | Memory cap of a guarded test call; `0` means none. |
| `{LCB,APPS,HOTPOTQA}_INDEPENDENT_RECURSION_LIMIT` | team spec | Independent runners: LangGraph recursion limit per replica. |
| `{LCB,APPS}_TOOL_TIMEOUT_S`, `{LCB,APPS}_TOOL_OUTPUT_CHAR_BUDGET` | `10`, `4000` | Independent runners: `python_exec` timeout and output budget. |
| `APPS_TEST_TIMEOUT_S` | `4` | Independent APPS runner: per-test timeout. |

### OpenAI Agents SDK

`OPENAI_AGENTS_PATH` (default `vendor/openai_agents`) is the isolated install of `openai-agents`, which needs `openai>=3`:

```bash title="Install the SDK"
pip install --target vendor/openai_agents -r requirements-openai-agents.txt
```

It must be first on `PYTHONPATH` for the whole process. The `decentralized/openai_agents` runners and `optimizers.protocol.run` restart themselves with it prepended when it is not. Without a usable SDK the runners exit with status 1 before the first row, and a protocol job with status 2.

### Wall-clock caps

A row that hits its cap is recorded with its error. Team-size and communication variants run the same code.

| Runners | Cap | Set by | Mechanism |
| --- | --- | --- | --- |
| `independent/{lcb,apps}` | 180 s | `LCB_INDEPENDENT_ROW_TIMEOUT_S`, `APPS_INDEPENDENT_ROW_TIMEOUT_S` | `asyncio.wait_for` |
| `independent/hotpotqa` | 120 s | `HOTPOTQA_INDEPENDENT_ROW_TIMEOUT_S` | `asyncio.wait_for` |
| `independent/{math,gpqa}` | 120 s | hardcoded `PER_ROW_TIMEOUT_S` | `asyncio.wait_for` |
| `centralized/autogen/{hotpotqa,gpqa,math}` | 120 s | hardcoded `PER_ROW_TIMEOUT_S` | `asyncio.wait_for` |
| `decentralized/langgraph/{gpqa,math}` | 120 s | hardcoded `PER_ROW_TIMEOUT_S` | SIGALRM (`row_timeout` in `core/runtime.py`) |

The SIGALRM cap rarely ends a row: the alarm usually fires during a model request, which the OpenAI client catches and retries, so it only stops a row between requests. It stays unchanged so existing results remain reproducible.

## Run protocol

Read by `optimizers.protocol.run` (`optimizers/protocol/settings.py`) when the job needs the value. The command-line options are under [Run protocol](cli.md#run-protocol).

| Variable | Default | Meaning |
| --- | --- | --- |
| `TASK_ENDPOINTS` | unset | Comma-separated task endpoints (`--task-endpoints` sets it); fallback `VLLM_BASE_URL`. |
| `MODEL_ID`, `TASK_MODEL` | from `--model` | Task model. |
| `REFLECTION_MODEL_ID` | `Qwen/Qwen3.5-122B-A10B-FP8` | Reflection model; any other value makes the job non-conformant. |
| `REFLECTION_MODEL_BASE_URL` | `http://localhost:8200/v1` | Reflection endpoint. |
| `OPENAI_API_KEY` | `EMPTY` | Key of both endpoints. |
| `REFLECTION_CONTEXT_LIMIT` | served `max_model_len`, else 65,536 | Reflection window used by context fitting. |
| `TOKENIZER_CACHE_DIRS` | unset | Colon-separated extra tokenizer caches. |
| `DSPY_CACHEDIR` | `<out>/optimization/dspy_cache` | The job's DSPy disk cache. |

Per rollout the protocol sets, then restores, `MODEL_ID`, `REQUEST_SEED`, `TASK_MODEL_TEMPERATURE`, `TASK_MODEL_TOP_P` and `TASK_MODEL_MAX_TOKENS` (optimization 0.2 / 0.9 / 32,768; evaluation 0.0 / 0.9 / 32,768), `TASK_TEMP_OPTIMIZE`, `TASK_TEMP_EVAL` and, for ToolHop, `TOOLHOP_ALLOW_DATASET_EXEC=1`.

## Optimizer bridge

Read by `optimizers/bridge`; the index is [`env.py`](https://github.com/juyangbai/MAS-PromptBench/blob/main/optimizers/bridge/env.py).

| Variable | Default | Meaning |
| --- | --- | --- |
| `TASK_MODEL` | `Qwen/Qwen3.5-9B` | Adapters' task model when `MODEL_ID` is unset. |
| `REFL_MODEL` | `Qwen/Qwen3.5-122B-A10B-FP8` | Reflection-model name (`lm.REFL_MODEL`); protocol jobs use `REFLECTION_MODEL_ID`. |
| `TASK_ENDPOINTS` | `http://localhost:8000/v1` | Task endpoints, round robin; each runner's `VLLM_BASE_URL` is repointed to them. |
| `REFLECTION_COMPACT_DATASETS` | `lcb` | Datasets whose role traces are compacted for reflection; `*` means all. |
| `MIPRO_REFLECTION_COMPACT_DATASETS` | `REFLECTION_COMPACT_DATASETS` | The same for MIPRO's program view, read first. |
| `TASK_TEMP_OPTIMIZE`, `TASK_TEMP_EVAL` | `0.2`, `0` | Rollout temperature outside and inside `lm.eval_mode`, mirrored into `TASK_MODEL_TEMPERATURE`. |
| `REAL_RUNNER_FAIL_ON_ADAPTER_ERROR` | `1` | `0` scores an adapter exception as a failed row instead of raising. |
| `REAL_RUNNER_KEEP_MESSAGES` | `0` | Keep full message lists in single and independent BFCL and GPQA outputs. |
| `REAL_RUNNER_RECURSION_LIMIT`, `BFCL_{SINGLE,INDEPENDENT,CENTRALIZED_GRAPH,CENTRALIZED_WORKER}_RECURSION_LIMIT` | `100` | LangGraph limits of the BFCL adapters. |
| `GPQA_SINGLE_RECURSION_LIMIT`, `GPQA_INDEPENDENT_RECURSION_LIMIT` | `25`, `15` | LangGraph limits of the GPQA adapters. |
| `BFCL_CATEGORY` | all four with fixed splits, else `simple` | BFCL subset of `load_all()`. |
| `SWE_WORK_ROOT` | `~/swe_work` | Repository clones of the SWE adapters. |
| `ALLOW_TOY_SWE_WORKDIR` | `0` | Offline tests only: a toy repository when no clone exists. |

## Method settings

Each method's knobs are one frozen dataclass, listed on [Optimizers](../optimizers/index.md#method-settings). Only MAPRO and TAVO also read environment variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `MAPRO_LISTWISE` | `0` | `1` selects the paper's listwise scoring, which the protocol refuses. |
| `MAPRO_PAPER_ANCHOR` | `0` | `1` selects the latest-selection anchor, which the protocol refuses. |
| `MAPRO_INIT_MAX_TOKENS` | `1024` | Rewrite cap of the pool initialization. |
| `MAPRO_MUTATE_MAX_TOKENS` | `512` | Rewrite cap of a mutation. |
| `TAVO_CREDIT` | `1` | `0`, `false`, `no` or `off` turns off trajectory credit (an ablation). |

## Sweep launchers

`scripts/run_topologies.sh`, `run_teamsizes.sh` and `run_communications.sh` run one process per cell. Each exports `VLLM_BASE_URL` (`http://localhost:8000/v1`), `MODEL_ID` (`Qwen/Qwen3.5-9B`) and `TOOLHOP_ALLOW_DATASET_EXEC=1` unless they are set, and reads:

| Variable | `run_topologies.sh` | `run_teamsizes.sh` | `run_communications.sh` |
| --- | --- | --- | --- |
| `DATASETS` | all 9 | all 9 | `hotpotqa lcb bfcl toolhop apibank swe` |
| `TOPOLOGIES` | not read (the 8 runner folders in `TOPOS`) | `independent sequential centralized decentralized` | same |
| `RVALUES` | not read | `2 4 8 10` | not read |
| `FORMATS` | not read | not read | `freeform semi_structured structured_soft` |
| `OUT_ROOT` | `results/topologies_baseline` | `results/teamsizes` | not read (runner defaults) |

- BFCL cells run their evaluation IDs one AST category at a time. SWE-bench cells run their 30 IDs, with `--eval singularity` in `run_topologies.sh` and `run_teamsizes.sh`. Each script sets the other datasets' `--limit`.
- `run_topologies.sh` puts `vendor/openai_agents`, when present, first on `PYTHONPATH` for the Agents SDK cells, so they need no restart.

## Model serving

`models/serve_qwen3_5_9b.sh` and `serve_llama3_1_8b.sh` start one vLLM replica per GPU on consecutive ports; `serve_qwen3_5_122b.sh` starts one tensor-parallel instance.

| Variable | 9B | Llama 8B | 122B |
| --- | --- | --- | --- |
| `MODEL_ID` | `Qwen/Qwen3.5-9B` | `meta-llama/Llama-3.1-8B-Instruct` | `Qwen/Qwen3.5-122B-A10B-FP8` |
| port | `VLLM_BASE_PORT` (or `BASE_PORT`) 8000 | `VLLM_BASE_PORT` (or `BASE_PORT`) 8100 | `VLLM_PORT` 8200 |
| GPUs | `VLLM_GPU_LIST` (or `GPU_LIST`): every visible GPU; `NUM_REPLICAS`: their count | same | `TENSOR_PARALLEL_SIZE` 4 over `CUDA_VISIBLE_DEVICES` (default the first 4) |
| `MAX_MODEL_LEN` | 131072 | 131072 | 262144 |
| `GPU_MEMORY_UTIL` | 0.90 | 0.90 | 0.95 |
| `KV_CACHE_DTYPE` | `auto` | `auto` | `fp8` |
| other | none | `CHAT_TEMPLATE` (`models/tool_chat_template_llama3.1_json_multi.jinja`), `MAX_NUM_SEQS` 256, `MAX_NUM_BATCHED_TOKENS` 32768 | none |

All three also read `VLLM_HOST` (`0.0.0.0`), `CONDA_ENV` (`mas-promptbench`, activated unless it is already `CONDA_DEFAULT_ENV`), `HF_HOME` (`$HOME/models`, also the default of `MODEL_PATH`, vLLM's `--download-dir`, and of `TRANSFORMERS_CACHE`) and `LOG_DIR` (`results/vllm_<model>`). They append the conda environment's CUDA driver stubs to `LIBRARY_PATH` for flashinfer's JIT build.

- `$HOME/models` is a flat download folder, not the `~/.cache/huggingface/hub` layout. To serve from an existing Hugging Face cache, set `HF_HOME` to it and `MODEL_PATH` and `TRANSFORMERS_CACHE` to `<HF_HOME>/hub`.
- The Llama model is gated: set `HF_TOKEN` for the download, or serve a cached copy with `HF_HUB_OFFLINE=1`.

!!! tip "Serve in a clean shell"
    The serve scripts and the runners share `MODEL_ID`. If your shell exports `MODEL_ID=Qwen/Qwen3.5-9B` for the runners, `serve_qwen3_5_122b.sh` serves the 9B model. Start it in a clean shell or pass `MODEL_ID=Qwen/Qwen3.5-122B-A10B-FP8`.

## Prompt generation

`configs/generate_role_prompts.py` reads `PROMPT_GEN_BASE_URL` (`http://localhost:8200/v1`), `PROMPT_GEN_API_KEY` (`EMPTY`) and `PROMPT_GEN_MODEL` (`Qwen/Qwen3.5-122B-A10B-FP8`). The options `--base-url`, `--api-key` and `--model` override them; the other options are on [Command-Line Flags](cli.md#prompt-generation).

## Tests

| Variable | Default | Meaning |
| --- | --- | --- |
| `GOLDEN_WORKERS` | half the CPUs (2 to 32) | Parallel golden cell workers. |
| `GOLDEN_EXEC_WORKERS` | `6` | Concurrent APPS cells. |
| `GOLDEN_CELL_TIMEOUT` | `900` | Seconds per cell. |
| `GOLDEN_RERUN_MAX` | `25` | Differing cells are re-run, to rule out machine load, only when at most this many differ. |
| `GOLDEN_DIFF_LINES` | `400` | Diff lines shown per failing cell. |
| `GOLDEN_REAL_HOME` | `~` | Home whose Hugging Face cache the workers read. |
| `HF_HOME`, `HF_DATASETS_CACHE` | `~/.cache/huggingface` | Local Hugging Face cache. |
| `MASPOB_TEST_SITE_PACKAGES` | unset | Extra site-packages with `torch_geometric` for the MASPOB tests, which skip without it. |

The suite runs offline with `HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 TOOLHOP_ALLOW_DATASET_EXEC=1`; see the [golden tests README](https://github.com/juyangbai/MAS-PromptBench/blob/main/tests/golden/README.md).
