# Repository Map

Where everything lives in the MAS-PromptBench repository, what each folder is for, and which file to edit for the most common changes.
{ .lede }

## Folder tree

```text
MAS-PromptBench/
├── README.md
├── environment.yml                  # conda env "mas-promptbench" (Python 3.11)
├── assets/                          # logos and the overview figure
├── benchmarks/
│   ├── <dataset>/<dataset>_eval_ids.json   # frozen eval IDs, one per dataset
│   └── apibank/apibank_upstream/           # vendored API-Bank source
├── communications/
│   ├── communication_formats.py     # format contracts, prompt injection, parsing, CLI
│   ├── output_contracts.py          # re-exports topologies/output_contracts.py
│   └── <topology>/<dataset>/<dataset>_<format>.py
├── configs/
│   ├── generate_role_prompts.py     # writes the seed prompts with an LLM
│   └── prompts/
│       ├── meta_prompt.txt  domains.yaml  roles.yaml  tools.yaml
│       └── <topology>/<dataset>/<role>.txt
├── frameworks/                      # submodules: langgraph, crewAI, autogen,
│                                    #   llm_multiagent_debate
├── models/
│   ├── serve_qwen3_5_9b.sh
│   └── serve_qwen3_5_122b.sh
├── optimizers/
│   ├── gepa/
│   │   ├── run_gepa.sh
│   │   ├── real_runner_gepa/        # adapters/ datasets/ pilots/ registry.py ...
│   │   ├── docs/ADDING_PAIR.md
│   │   ├── templates/               # adapter_template.py, dataset_template.py
│   │   └── scripts/smoke_new_adapters.py
│   └── mipro/
│       ├── run_mipro.sh
│       ├── real_runner_mipro/
│       └── docs/ADDING_PAIR.md
├── scripts/
│   ├── run_topologies.sh
│   ├── run_communications.sh
│   └── run_teamsizes.sh
├── teamsizes/
│   ├── output_contracts.py  apibank_common.py  toolhop_common.py
│   ├── independent/langgraph_base.py
│   └── <topology>/<dataset>/<dataset>_r{2,4,8,10}.py
└── topologies/
    ├── output_contracts.py  telemetry.py  code_extract.py
    ├── single/                      # LangGraph only
    ├── independent/                 # LangGraph only
    ├── sequential/{langgraph,crewai}/
    ├── centralized/{langgraph,autogen}/
    └── decentralized/{langgraph,openai}/
```

Runs write to `results/` (and the optimizer workspaces to their own `results/` and `cache/`). These folders are created on first use and are gitignored.

## Folders

| Folder | Purpose | Key files |
| --- | --- | --- |
| `topologies/` | The core benchmark: 72 runners, one per topology variant and dataset, at `<topology>/[<framework>/]<dataset>/<framework>_<dataset>.py`. Each runner holds its own loader, scorer, batch loop and CLI. | `output_contracts.py` (final-answer contracts), `telemetry.py` (token and call counts per framework), `code_extract.py` (code-block extraction), base demos such as `single/langgraph_base.py` |
| `communications/` | 60 runners that rerun the LangGraph variants of four topologies under the three message formats, for hotpotqa, lcb, toolhop, apibank and swe. | `communication_formats.py` (`install_proxy`, `cli_main`) |
| `teamsizes/` | 144 runners that sweep the team size r over 2, 4, 8 and 10 for four topologies and all nine datasets. | `<topology>/<dataset>/<dataset>_r<N>.py`, `toolhop_common.py`, `apibank_common.py` |
| `configs/` | Seed system prompts, one per topology, dataset and role, plus the spec and script that generated them. | `generate_role_prompts.py`, `prompts/roles.yaml`, `prompts/tools.yaml`, `prompts/domains.yaml`, `prompts/meta_prompt.txt` |
| `benchmarks/` | Frozen eval-ID manifests (655 IDs over nine datasets) and the vendored API-Bank source. | `<dataset>/<dataset>_eval_ids.json`, `apibank/apibank_upstream/` |
| `optimizers/` | The GEPA and MIPRO workspaces, each with a bridge to the real runners, a per-pair pilot and a sweep launcher. | `real_runner_*/registry.py`, `real_runner_*/protocol.py`, `real_runner_*/pilots/run_*_dataset.py`, `run_*.sh`, `docs/ADDING_PAIR.md` |
| `scripts/` | Baseline sweep launchers, one process per cell. | `run_topologies.sh`, `run_communications.sh`, `run_teamsizes.sh` |
| `models/` | vLLM serve scripts for the task model and the reflection model. | `serve_qwen3_5_9b.sh`, `serve_qwen3_5_122b.sh` |
| `frameworks/` | Agent frameworks as git submodules. `environment.yml` installs LangGraph, CrewAI and AutoGen from here in editable mode; the debate repository is a reference for the decentralized topology. | `langgraph`, `crewAI`, `autogen`, `llm_multiagent_debate` |
| `assets/` | Logos and figures for the README and website. | `MAS-PromptBench_overview.png` |

## Where do I change...

| To change | Edit | Notes |
| --- | --- | --- |
| An agent's prompt | `configs/prompts/<topology>/<dataset>/<role>.txt` | The CrewAI runners read `sequential/`, the AutoGen runners `centralized/`, and the OpenAI-SDK debate runners `decentralized/` (MATH alone uses `decentralized_openai/`). To regenerate, edit `roles.yaml` or `tools.yaml` and run `configs/generate_role_prompts.py --force --only <topology>/<dataset>`. The optimizers read these files and never write them. |
| The required final-answer format | `DATASET_CONTRACTS` in `topologies/output_contracts.py` | Keep `teamsizes/output_contracts.py` and the copies in `optimizers/*/real_runner_*/output_contracts.py` in step. Some runners add a dataset-specific `_OUTPUT_FORMAT_NUDGE`. |
| A scorer | the scoring section of every runner for that dataset | Each runner carries its own copy, including the team-size runners. Optimizer metrics live in `optimizers/*/real_runner_*/datasets/<dataset>.py`; the MATH metric imports `exact_match_score` from `topologies/single/math/langgraph_math.py`. |
| The model endpoint for runs | `VLLM_BASE_URL`, `MODEL_ID`, `OPENAI_API_KEY` | See [Environment Variables](environment.md#model-connection). |
| The endpoints for optimization | `GEPA_TASK_ENDPOINTS`, `GEPA_REFL_ENDPOINT`, `MIPRO_TASK_ENDPOINTS`, `MIPRO_REFL_ENDPOINT` | Defaults live in `optimizers/*/real_runner_*/lm.py`. |
| Sampling settings | the client factory in each runner (`build_agent`, `_build_llm` or `_build_client`); `task_sampling()` and `reflection_sampling()` in each optimizer's `lm.py` | |
| A baseline sweep grid | `DATASETS`, `TOPOLOGIES`, `FORMATS`, `RVALUES` for `scripts/run_*.sh`; the `TOPOS` string inside `run_topologies.sh` | Row limits are the `LIMIT` table inside each script. See [Command-Line Flags](cli.md#sweep-scripts). |
| An optimizer sweep grid or budget | `DATASETS`, `TOPOLOGIES` and the budget variables for `optimizers/gepa/run_gepa.sh` and `optimizers/mipro/run_mipro.sh` | Or call the pilot directly with flags. |
| The eval IDs | `benchmarks/<dataset>/<dataset>_eval_ids.json` | Optimizers exclude these IDs from their splits. Runners score them when you pass the IDs with `--only`; the API-Bank runners read the file directly. |
| Which pairs an optimizer can target | `DATASET_ADAPTERS` in `optimizers/*/real_runner_*/registry.py` | See [Add a Topology](../extending/add-topology.md). |
| Team size outside the team-size sweep | `INDEPENDENT_N_AGENTS`, `DECENTRALIZED_N_AGENTS`, `DECENTRALIZED_N_ROUNDS` | Or the pilot's `--n-agents` and `--n-rounds`. |
| Where results go | `--out` or `--out-dir` on a runner, `--out` on a pilot, `OUT_ROOT` on a launcher | Defaults are listed in [Command-Line Flags](cli.md#default-output-folders). |
