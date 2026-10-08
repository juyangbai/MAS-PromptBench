# Environment Variables

Every environment variable the benchmark reads, grouped by area, with its default in the code and the component that reads it. Variables used only inside the vendored frameworks are left out.
{ .lede }

Most runs need only the three model-connection variables. The rest tune one dataset, an optimizer or a serving script.

## Model connection

| Variable | Default | Read by | Meaning |
| --- | --- | --- | --- |
| `VLLM_BASE_URL` | `http://localhost:8001/v1` or `http://localhost:8000/v1`, depending on the runner | every topology and team-size runner; communication runners inherit it from the runner they wrap | OpenAI-compatible base URL for every agent. The defaults differ between runners, so always set it. The sweep scripts default it to `http://localhost:8000/v1`. |
| `MODEL_ID` | `Qwen/Qwen3.5-9B` | every runner; optimizer adapters; `models/*.sh` | Model name sent with each request. Optimizer adapters set the runner's model from `MODEL_ID`, falling back to `TASK_MODEL`. The serve scripts read it as the model to serve. |
| `OPENAI_API_KEY` | `EMPTY` | every runner, optimizer LM and adapter | API key for the endpoint. Most `*_base.py` demo files assert that it is set. |

The optimizers do not send task calls to `VLLM_BASE_URL`; they use the endpoint lists under [Optimizers](#optimizers).

### Seed-prompt generator

`configs/generate_role_prompts.py` reads only these, so a runtime `VLLM_BASE_URL` or `MODEL_ID` never changes which model writes the prompts.

| Variable | Default | Same as flag | Meaning |
| --- | --- | --- | --- |
| `PROMPT_GEN_BASE_URL` | `http://localhost:8000/v1` | `--base-url` | Endpoint serving the generator model. |
| `PROMPT_GEN_MODEL` | `Qwen/Qwen3.5-122B-A10B-FP8` | `--model` | Generator model. |
| `PROMPT_GEN_API_KEY` | `EMPTY` | `--api-key` | API key. |

## Team shape

| Variable | Default | Read by | Meaning |
| --- | --- | --- | --- |
| `INDEPENDENT_N_AGENTS` | `4` | independent topology runners | Parallel replicas. |
| `DECENTRALIZED_N_AGENTS` | `4` | decentralized topology runners | Debating peers. |
| `DECENTRALIZED_N_ROUNDS` | `2` | decentralized topology and team-size runners | Debate rounds. |

Independent and decentralized team-size runners read `INDEPENDENT_N_AGENTS` and `DECENTRALIZED_N_AGENTS` too, with the team size `r` as the default (the ToolHop and API-Bank team-size runners take `r` from the file only). Leave them unset there, or the run no longer matches its file name. ToolHop and API-Bank topology runners check a dataset-prefixed variable first (below).

## Datasets

### ToolHop

| Variable | Default | Read by | Meaning |
| --- | --- | --- | --- |
| `TOOLHOP_ALLOW_DATASET_EXEC` | unset | ToolHop runners | Must be `1`; otherwise solving stops with an error, because ToolHop executes dataset-provided Python tools. The sweep scripts set it to `1`. |
| `TOOLHOP_MAX_TOKENS` | `512` | ToolHop runners | Max tokens per model call. |
| `TOOLHOP_FINAL_MAX_TOKENS` | `64` | ToolHop runners | Max tokens for the forced final answer once the tool budget is spent. |
| `TOOLHOP_MAX_TURNS` | `9` | ToolHop runners | Tool-calling turns before the final answer is forced. |
| `TOOLHOP_TOOL_RESULT_CHAR_BUDGET` | `6000` | ToolHop runners | Characters kept from each tool result. |
| `TOOLHOP_INDEPENDENT_N_AGENTS` | `INDEPENDENT_N_AGENTS`, else `4` | ToolHop runners | Replicas for independent ToolHop. |
| `TOOLHOP_DECENTRALIZED_N_AGENTS`, `TOOLHOP_DECENTRALIZED_N_ROUNDS` | `DECENTRALIZED_N_AGENTS` / `_N_ROUNDS`, else `4` / `2` | ToolHop runners | Peers and rounds for decentralized ToolHop. |

### API-Bank

| Variable | Default | Read by | Meaning |
| --- | --- | --- | --- |
| `APIBANK_ROOT` | `benchmarks/apibank/apibank_upstream/api-bank`, then `.cache/apibank_smoke_damo/api-bank` | API-Bank runners | API-Bank source checkout (must contain `apis/` and `init_database/`). |
| `APIBANK_LEVEL` | `all` | API-Bank runners | Level slice when `--level` is not given: `all`, `1`, `2` or `3`. |
| `APIBANK_CURATED_PATH` | per level: `benchmarks/apibank/apibank_eval_ids.json` for `all`, `benchmarks/apibank/apibank_level<N>_curated.json` otherwise | API-Bank runners and team-size runners | Curated task manifest. Set by `--curated-path`. If the file does not exist, the manifest is built from the API-Bank source. |
| `APIBANK_CURATED_LIMIT` | unset | API-Bank runners | Row cap when a curated manifest has to be rebuilt. |
| `APIBANK_LEVEL3_JSON` | unset | API-Bank runners | Path to a local `level-3.json` when the vendored copy is missing. |
| `APIBANK_MAX_TOKENS` | `1024` | API-Bank runners | Max tokens per model call. |
| `APIBANK_REQUEST_TIMEOUT` | `60` | API-Bank runners | Seconds per request. |
| `APIBANK_OPENAI_MAX_RETRIES` | `0` | API-Bank runners | OpenAI SDK retries per request. |
| `APIBANK_TOOLSEARCHER_SCORER` | `official` for levels `all`, 2, 3; `keyword` for level 1 | API-Bank runners and team-size runners | ToolSearcher scorer: `official`, `upstream` or `keyword`. Set by `--toolsearcher-scorer`. |
| `APIBANK_TOOLSEARCHER_DEVICE` | `cpu` | API-Bank runners | Device for the ToolSearcher sentence-embedding model. |
| `APIBANK_TOOLSEARCHER_SKLEARN_SHIM`, `_GOOGLETRANS_SHIM`, `_NLTK_SHIM`, `_BM25_SHIM` | `1` | API-Bank runners | Install a stub for that package when it is missing; `0` disables. |
| `APIBANK_INDEPENDENT_N_AGENTS` | `INDEPENDENT_N_AGENTS`, else `4` | API-Bank runners | Replicas for independent API-Bank. |
| `APIBANK_DECENTRALIZED_N_AGENTS`, `APIBANK_DECENTRALIZED_N_ROUNDS` | `DECENTRALIZED_N_AGENTS` / `_N_ROUNDS`, else `4` / `2` | API-Bank runners | Peers and rounds for decentralized API-Bank. |

### SWE-bench Verified

| Variable | Default | Read by | Meaning |
| --- | --- | --- | --- |
| `SWE_SIF_DIR` | `~/containers/swe` | SWE runners | Cache of per-instance Singularity images pulled from `docker://swebench/sweb.eval.x86_64.*`. |
| `SWE_EVAL_LOG_DIR` | `/tmp` | SWE runners | Where raw pytest output is saved as `eval_<instance>.log`. |
| `SWE_PROBLEM_CHAR_BUDGET` | `16000` | SWE runners | Characters of the issue text kept in the prompt. |
| `SWE_HINTS_CHAR_BUDGET` | `4000` | SWE runners | Characters of the hints kept in the prompt. |
| `SWE_REPO_DIR` | `.` | single, sequential (both frameworks) and centralized LangGraph SWE runners | Initial repository folder for the file tools; a batch sets a per-instance clone. |
| `SWE_WORKDIR_ROOT` | `~/swe_work_independent` | independent SWE runners | Root for per-replica clones; `--workdir-root` replaces it for a batch. |

### LiveCodeBench, APPS, HotpotQA

| Variable | Default | Read by | Meaning |
| --- | --- | --- | --- |
| `LCB_FUNCTIONAL_MEMORY_BYTES` | 4 GiB | LCB runners | Memory cap per functional test; `0` disables. |
| `APPS_CALL_BASED_MEMORY_BYTES` | 4 GiB | APPS runners | Memory cap per call-based test. |
| `APPS_TEST_TIMEOUT_S` | `4` | independent APPS runners | Seconds per test. |
| `LCB_TOOL_TIMEOUT_S`, `APPS_TOOL_TIMEOUT_S` | `10` | independent LCB and APPS runners | Seconds per `python_exec` tool call. |
| `LCB_TOOL_OUTPUT_CHAR_BUDGET`, `APPS_TOOL_OUTPUT_CHAR_BUDGET` | `4000` | independent LCB and APPS runners | Characters kept from tool output. |
| `LCB_INDEPENDENT_RECURSION_LIMIT`, `APPS_INDEPENDENT_RECURSION_LIMIT` | `20` | independent LCB and APPS runners | LangGraph recursion limit per replica. |
| `LCB_INDEPENDENT_ROW_TIMEOUT_S`, `APPS_INDEPENDENT_ROW_TIMEOUT_S` | `180` | independent LCB and APPS runners | Seconds per row. |
| `HOTPOTQA_INDEPENDENT_RECURSION_LIMIT` | `20` (`35` in the r=2 team-size runner) | independent HotpotQA runners | LangGraph recursion limit per replica. |
| `HOTPOTQA_INDEPENDENT_ROW_TIMEOUT_S` | `120` | independent HotpotQA topology runner | Seconds per row. |
| `GPQA_INDEPENDENT_RECURSION_LIMIT` | `35` | independent GPQA r=2 team-size runner | LangGraph recursion limit per replica. |

The independent team-size runners for LCB and APPS read the same `*_INDEPENDENT_*` and `*_TOOL_*` variables.

## Optimizers

GEPA reads `GEPA_*`; MIPRO reads `MIPRO_*` first and falls back to the GEPA name. Both pilots set `DSPY_CACHEDIR` and `DSP_CACHEDIR` to `cache/<topology>_<dataset>` inside their workspace; you do not set those.

| Variable | Default | Read by | Meaning |
| --- | --- | --- | --- |
| `GEPA_TASK_ENDPOINTS` | `localhost:10000`, `localhost:15000`–`15007`, `20000`–`20007`, `30000`–`30007` and `9000`–`9003` (all `http://localhost:…/v1`) | GEPA; MIPRO fallback | Comma-separated task endpoints. The DSPy task LM and the runner clients rotate through them. |
| `MIPRO_TASK_ENDPOINTS` | `http://localhost:15000/v1` to `15007` | MIPRO | Comma-separated task endpoints. |
| `GEPA_REFL_ENDPOINT` | `http://localhost:8000/v1` | GEPA; MIPRO fallback | Reflection-model endpoint. |
| `MIPRO_REFL_ENDPOINT` | `http://localhost:15000/v1` | MIPRO | Proposal-model endpoint. |
| `TASK_MODEL` | `Qwen/Qwen3.5-9B` | both | Task model name. |
| `REFL_MODEL` | `Qwen/Qwen3.5-122B-A10B-FP8` | both | Reflection or proposal model name. |
| `MIPRO_TASK_MODEL`, `MIPRO_REFL_MODEL` | `TASK_MODEL`, `REFL_MODEL` | MIPRO | MIPRO-only overrides. |
| `GEPA_EXCLUDE_REAL_EVAL_IDS` | on when unset | GEPA; MIPRO fallback | Keep the IDs in `benchmarks/<dataset>/<dataset>_eval_ids.json` out of train and validation. `0`, `false`, `no` or `off` disables it. API-Bank and ToolHop always exclude them. |
| `MIPRO_EXCLUDE_REAL_EVAL_IDS` | `GEPA_EXCLUDE_REAL_EVAL_IDS`, on when both unset | MIPRO | Same, for MIPRO. |
| `GEPA_USE_SIGNAL_25_SPLITS` | on when unset | GEPA API-Bank and ToolHop loaders | For a 25/25 split with seed 0 and offset 0, use `benchmarks/<dataset>/<dataset>_gepa25_signal_split.json` if it exists. None ship, so the normal split is used. |
| `GEPA_REFLECTION_COMPACT_DATASETS` | `lcb` | GEPA; MIPRO fallback | Comma-separated datasets (or `*`) whose traces are shortened before reflection. Scoring still uses the full data. |
| `MIPRO_REFLECTION_COMPACT_DATASETS`, `MIPRO_DATA_SUMMARY_COMPACT_DATASETS` | GEPA value, then `lcb` | MIPRO | Same, for MIPRO's traces and its dataset summary. |
| `GEPA_EARLY_STOP_PATIENCE` | `8` | GEPA `early_stop.py` | Used only when no patience is passed; the pilot always passes `--early-stop-patience`, so it has no effect there. |
| `REAL_RUNNER_FAIL_ON_ADAPTER_ERROR` | `1` | both | Abort on an adapter error that is not an endpoint or timeout error. `0` scores that row 0 instead. |
| `REAL_RUNNER_TASK_MAX_TOKENS` | `4096` | both, shared module adapters | Max tokens for runner models built by the adapters. |
| `MATH_TASK_MAX_TOKENS` | `4096` | both, MATH adapters | Same, for MATH. |
| `REAL_RUNNER_RECURSION_LIMIT` | unset | both, BFCL adapters | Fallback LangGraph recursion limit. |
| `GPQA_SINGLE_RECURSION_LIMIT`, `GPQA_INDEPENDENT_RECURSION_LIMIT` | `25`, `15` | both, GPQA adapters | LangGraph recursion limits. |
| `REAL_RUNNER_KEEP_MESSAGES` | `0` | both, single and independent BFCL and GPQA adapters | `1` keeps full message lists in the output. |
| `LCB_COMPILED_PROMPT_CHAR_LIMIT`, `LCB_COMPILED_PROMPT_HEAD_CHARS` | `24000`, `6000` | both pilots | Cap on compiled LCB prompts, and how much of the head is kept when truncating. |
| `BFCL_CATEGORY` | `simple` | both, BFCL loader | BFCL subset to optimize on. |
| `SWE_GEPA_EVAL_SAMPLE`, `SWE_SAMPLE` | `balanced_30` | both | Name of the protected SWE-bench sample recorded in `meta.json`. MIPRO checks `SWE_MIPRO_EVAL_SAMPLE` first. |
| `SWE_WORK_ROOT` | unset | both, SWE adapters | Folder searched for existing SWE clones. |
| `REQUIRE_REAL_SWE_WORKDIR` | `0` | both, SWE adapters | `1` fails when no clone is found; otherwise a tiny temporary git repository stands in. |
| `N_AGENTS`, `N_ROUNDS` | `4`, `2` | both, team-size and protocol adapters | Fallback team shape when the pilot passes none. |

!!! warning "`run_gepa.sh` leaves the task endpoints at their defaults"
    `optimizers/mipro/run_mipro.sh` exports `MIPRO_TASK_ENDPOINTS=http://localhost:8000/v1`, but `optimizers/gepa/run_gepa.sh` sets only `GEPA_REFL_ENDPOINT`. Export `GEPA_TASK_ENDPOINTS` before a GEPA run, or task calls go to the built-in list above.

## Sweeps

The baseline sweeps in `scripts/` read these (see [Command-Line Flags](cli.md#sweep-scripts) for what each script passes):

| Variable | Default | Read by |
| --- | --- | --- |
| `VLLM_BASE_URL` | `http://localhost:8000/v1` | all three |
| `MODEL_ID` | `Qwen/Qwen3.5-9B` | all three |
| `TOOLHOP_ALLOW_DATASET_EXEC` | `1` | all three |
| `DATASETS` | all nine datasets; `hotpotqa lcb toolhop apibank swe` for protocols | all three |
| `TOPOLOGIES` | `independent sequential centralized decentralized` | `run_communications.sh`, `run_teamsizes.sh` |
| `FORMATS` | `freeform semi_structured structured_soft` | `run_communications.sh` |
| `RVALUES` | `2 4 8 10` | `run_teamsizes.sh` |
| `OUT_ROOT` | `results/topologies_baseline` | `run_topologies.sh` |

The optimizer launchers change into their workspace, so a relative `OUT_ROOT` lands under `optimizers/gepa/` or `optimizers/mipro/`.

| Variable | `run_gepa.sh` | `run_mipro.sh` |
| --- | --- | --- |
| `GEPA_REFL_ENDPOINT` | `http://localhost:8000/v1` | not set |
| `MIPRO_REFL_ENDPOINT`, `MIPRO_TASK_ENDPOINTS` | not set | `http://localhost:8000/v1` each |
| `DATASETS` | `bfcl gpqa hotpotqa math apps lcb swe apibank toolhop` | same |
| `TOPOLOGIES` | `single independent sequential sequential_crewai centralized centralized_autogen decentralized decentralized_openai` | same |
| `TRAIN_SIZE`, `VAL_SIZE` | `25`, `25` | `25`, `25` |
| `NUM_THREADS` | `4` | `4` |
| `OUT_ROOT` | `results/gepa` | `results/mipro` |
| `MAX_FULL_EVALS`, `REFLECTION_MINIBATCH_SIZE` | `5`, `3` | — |
| `N_AGENTS`, `N_ROUNDS` | `4`, `2` | — |
| `COMPONENT_SELECTOR`, `EARLY_STOP_PATIENCE` | `round_robin`, `3` | — |
| `NUM_CANDIDATES`, `NUM_TRIALS` | — | `3`, `3` |

## Local serving

The scripts in `models/` read these. Raw `HOST` and `PORT` are avoided because conda's compiler activation sets `HOST`.

| Variable | `serve_qwen3_5_9b.sh` | `serve_qwen3_5_122b.sh` | Meaning |
| --- | --- | --- | --- |
| `MODEL_ID` | `Qwen/Qwen3.5-9B` | `Qwen/Qwen3.5-122B-A10B-FP8` | Model to serve. |
| `CONDA_ENV` | `mas-promptbench` | `mas-promptbench` | Environment activated if not already active. |
| `HF_HOME` | `$HOME/models` | `$HOME/models` | Model cache; `MODEL_PATH` and `TRANSFORMERS_CACHE` default to it. |
| `MODEL_PATH` | `$HF_HOME` | `$HF_HOME` | vLLM `--download-dir`. |
| `VLLM_HOST` | `0.0.0.0` | `0.0.0.0` | Bind address. |
| `VLLM_BASE_PORT` (then `BASE_PORT`) | `8000` | — | First port; replica `i` listens on base + `i`. |
| `VLLM_PORT` | — | `8000` | Server port. |
| `VLLM_GPU_LIST` (then `GPU_LIST`) | every visible GPU | — | Comma-separated GPU indices, one replica each. |
| `NUM_REPLICAS` | GPUs in the list | — | Replicas to start. |
| `TENSOR_PARALLEL_SIZE` | — | `4` | Tensor-parallel degree. |
| `CUDA_VISIBLE_DEVICES` | set per replica | `0` to `TENSOR_PARALLEL_SIZE - 1` | GPUs used. |
| `MAX_MODEL_LEN` | `131072` | `262144` | Context length. |
| `GPU_MEMORY_UTIL` | `0.90` | `0.95` | vLLM GPU memory fraction. |
| `KV_CACHE_DTYPE` | `auto` | `fp8` | KV-cache dtype; `fp8` needs Hopper or Blackwell GPUs. |
| `LOG_DIR` | `results/vllm_qwen3_5_9b` | `results/vllm_qwen3_5_122b` | Log folder, relative to the repository root. |

Both scripts also append `$CONDA_PREFIX/targets/x86_64-linux/lib/stubs` to `LIBRARY_PATH` for flashinfer's JIT build.

!!! tip "Unset `MODEL_ID` before serving the 122B model"
    The serve scripts and the runners share `MODEL_ID`. If your shell still exports `MODEL_ID=Qwen/Qwen3.5-9B` for runners, `serve_qwen3_5_122b.sh` serves the 9B model. Start it in a clean shell or pass `MODEL_ID=Qwen/Qwen3.5-122B-A10B-FP8`.
