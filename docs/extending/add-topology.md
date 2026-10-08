# Add a Topology

A topology is a way of wiring agents together, implemented once per dataset. This page covers adding a new topology, or a new framework implementation of an existing one, from the runners to the optimizer adapters.
{ .lede }

## Choose what you are adding

| Adding | Runner path | Seed prompts | Optimizer topology name |
| --- | --- | --- | --- |
| A framework variant of an existing topology | `topologies/sequential/<fw>/<ds>/<fw>_<ds>.py` | reuse `configs/prompts/sequential/<ds>/` | `sequential_<fw>` |
| A new topology | `topologies/<topo>/[<fw>/]<ds>/<fw>_<ds>.py` | new `configs/prompts/<topo>/<ds>/` | `<topo>` |

The existing variants follow the first row: the CrewAI runners read the same prompts as the LangGraph sequential runners, and the optimizers call them `sequential_crewai`. The same holds for `centralized_autogen` and for `decentralized_openai`, except that its MATH runner reads `configs/prompts/decentralized_openai/math/`. See [Workflow Topologies](../mas/topologies.md) for the five shapes that exist today.

## 1. Write a base implementation

A base file is a small framework demo of the coordination pattern, kept next to the runners:

| Base | Pattern |
| --- | --- |
| `topologies/single/langgraph_base.py` | LangGraph `create_react_agent` |
| `topologies/independent/langgraph_base.py` | LangGraph `Send` fan-out and fan-in |
| `topologies/sequential/crewai/crewai_base/` | CrewAI `Process.sequential` crew with `config/agents.yaml` and `config/tasks.yaml` |
| `topologies/centralized/autogen/autogen_base.py` | AutoGen `SelectorGroupChat` |
| `topologies/decentralized/openai/debate_base.py` | Multi-round debate after Du et al. 2023, on the OpenAI SDK |

The Python bases are standalone demos that call `gpt-4o-mini` and assert that `OPENAI_API_KEY` is set; the CrewAI base is a four-agent research crew. No runner imports a base file, and the LangGraph variants of sequential, centralized and decentralized have none. Write one if it helps you settle the agent wiring.

## 2. Write one runner per dataset

Add nine runners, one per dataset folder. Copy each from an existing runner for the same dataset and replace only the agent wiring, so the loader, scorer and CLI (see [Command-Line Flags](../reference/cli.md)) stay identical across topologies. Each runner exposes these module-level names:

| Name | Purpose |
| --- | --- |
| `VLLM_BASE_URL`, `MODEL_ID` | endpoint and model, read from the environment |
| `_load_prompt(role)` or `SYSTEM_PROMPT` | role prompt from `configs/prompts/`, wrapped by `append_output_contract_from_path(text, __file__, role)` |
| `_build_llm()` or `_build_client()` | model client factory |
| `N_AGENTS`, `N_ROUNDS` | team shape, when it is configurable |
| `solve(...)` | runs one instance, returns a dict with `answer` and `raw` |
| `load_instances(...)`, `run_batch(...)` | dataset slice and batch scoring |

The optimizer adapters patch these names at run time (`patched_module` in `optimizers/gepa/real_runner_gepa/adapters/module_common.py`). They replace `_load_prompt`, `SYSTEM_PROMPT`, `_build_llm`, `_build_client`, `VLLM_BASE_URL`, `MODEL_ID`, `N_AGENTS`, `N_ROUNDS` and `_RECURSION_LIMIT` when present, and append `_OUTPUT_FORMAT_NUDGE` and `_OUTPUT_FORMAT_APPENDIX` to prompts. Keep the names and the adapters work without edits to your runner.

For token and call counts, pass your framework's output through the matching extractor in `topologies/telemetry.py` (`langchain_telemetry`, `crewai_telemetry`, `autogen_telemetry`, `openai_sdk_telemetry` and others), then `normalize`.

## 3. Define the prompt roles

A framework variant reuses the existing prompt folder, so skip to step 4. For a new topology:

1. Add a block to `configs/prompts/roles.yaml` with a `description` of the topology and, under `benchmarks`, one entry per dataset listing `<role>: "<job description>"`.
2. Add `<topo>:` to `configs/prompts/tools.yaml` with a tool list per dataset.
3. Run `python configs/generate_role_prompts.py --only <topo>` to write `configs/prompts/<topo>/<ds>/<role>.txt`. [Add a Dataset](add-dataset.md#4-generate-the-seed-prompts) describes how the generator works.
4. Teach `output_contract()` in `topologies/output_contracts.py` which roles produce the final answer. It only recognizes the topology names `single`, `independent`, `sequential`, `sequential_crewai`, `centralized`, `centralized_autogen`, `decentralized` and `decentralized_openai`; any other name gets no contract. Make the same change in `teamsizes/output_contracts.py` and in both optimizer copies.

`append_output_contract_from_path` takes the topology from the folder right after `topologies/`, so a framework variant under an existing topology folder inherits its contract rules.

## 4. Register the optimizer adapters

Both optimizers document this in `ADDING_PAIR.md` ([GEPA](https://github.com/juyangbai/MAS-PromptBench/blob/main/optimizers/gepa/docs/ADDING_PAIR.md), [MIPRO](https://github.com/juyangbai/MAS-PromptBench/blob/main/optimizers/mipro/docs/ADDING_PAIR.md)). A pair is `dataset + topology + execution framework`, and adding one takes:

1. A dataset loader in `real_runner_*/datasets/`. All nine datasets have one.
2. Adapters in `real_runner_*/adapters/` implementing `RealRunnerAdapter`: `roles()`, `get_prompt(role)`, `set_prompt(role, text)`, `reset()`, `run_example(example)` and `format_role_trace(role, output)`. `run_example` returns a dict with `model_output` (what the metric reads) and optional `winner` and `buckets`.
3. An entry in `DATASET_ADAPTERS` in `real_runner_*/registry.py`. Values are import strings, so framework dependencies load only for the requested pair.
4. A pilot. The generic `run_gepa_dataset.py` and `run_mipro_dataset.py` handle any registered pair.
5. A smoke command and expected artifacts, recorded in `ADDING_PAIR.md`.

For a framework variant, subclass the existing adapter for each dataset and change three attributes. This is how the CrewAI MATH adapter is defined:

```python title="optimizers/gepa/real_runner_gepa/adapters/module_math.py"
class SequentialCrewAIMATHAdapter(SequentialMATHAdapter):
    topology = "sequential_crewai"
    framework = "crewai"
    module_name = "topologies.sequential.crewai.math.crewai_math"
```

```python title="optimizers/gepa/real_runner_gepa/registry.py"
"sequential_crewai": "real_runner_gepa.adapters.module_math:SequentialCrewAIMATHAdapter",
```

For a new topology, also set `prompt_topology` (the prompt folder name) and `roles_` (the role files to load). Two places key on names:

- The `framework` attribute picks the client the adapter injects: `crewai` gets a CrewAI `LLM`, `openai` an OpenAI client, and anything else a LangChain model for `_build_llm` or an AutoGen client for `_build_client`. A new framework needs its own branch in `patched_module` and in the per-dataset copies such as `_patched_module` in `module_math.py`.
- `build_program` in each pilot passes `--n-agents` only to `independent`, `decentralized` and `decentralized_openai`, and `--n-rounds` only to the decentralized ones. Add your topology there if its team size is configurable.

Repeat everything under `optimizers/mipro/real_runner_mipro/`. MIPRO has no `templates/` folder; its guide suggests copying `adapters/single_bfcl.py`. MIPRO renders its chosen instructions and few-shot demos into the role prompts through `set_prompt()` before each `run_example()`, so the adapter needs no MIPRO-specific code.

### Smoke-test the pair

Before a long run, follow the `ADDING_PAIR.md` checklist: `python3 -m py_compile` the new files, instantiate the adapter and call `describe_runtime()` or run `run_example()` on one example, then run a minimal optimization and check the artifacts under `--out`.

=== "GEPA"

    ```bash
    cd optimizers/gepa
    python -m real_runner_gepa.pilots.run_gepa_dataset --dataset math \
      --topology sequential_<fw> --train-size 1 --val-size 1 --max-full-evals 1 \
      --out results/gepa/sequential_<fw>_math_smoke
    ```

    Expect `meta.json`, `baseline_val.jsonl`, `compiled_val.jsonl`, `compiled/<role>.txt` and `compiled_raw/<role>.txt`.

=== "MIPRO"

    ```bash
    cd optimizers/mipro
    python -m real_runner_mipro.pilots.run_mipro_dataset --dataset math \
      --topology sequential_<fw> --train-size 1 --val-size 1 --num-candidates 1 \
      --num-trials 1 --out results/mipro/sequential_<fw>_math_smoke
    ```

    Also expect `status.json`, `optimized_val.jsonl` and `compiled_demos/`.

!!! warning "Skip the GEPA adapter smoke script"
    The GEPA README points to `optimizers/gepa/scripts/smoke_new_adapters.py`, but that script imports a `travel` dataset module that is not in the repository, so it fails on import. Use the one-row run above instead.

## 5. Add it to the launchers

`scripts/run_topologies.sh` lists topology variants as `path:framework` pairs and skips any module that does not exist:

```bash title="scripts/run_topologies.sh"
TOPOS="single:langgraph independent:langgraph \
sequential/langgraph:langgraph sequential/crewai:crewai \
centralized/langgraph:langgraph centralized/autogen:autogen \
decentralized/langgraph:langgraph decentralized/openai:openai"
```

Append your variant, for example `sequential/<fw>:<fw>`. For optimizer sweeps, add the topology name to the `TOPOLOGIES` default in `optimizers/gepa/run_gepa.sh` and `optimizers/mipro/run_mipro.sh`, or pass `TOPOLOGIES=...` when you launch.

## Checklist

1. Optional base file under `topologies/<topo>/[<fw>/]`.
2. Nine runners exposing the shared names, each reusing its dataset's loader, scorer and CLI.
3. New topology only: YAML entries, generated prompts, and final roles in all four `output_contracts.py` copies.
4. Adapters and `DATASET_ADAPTERS` entries for GEPA and MIPRO, plus `build_program` and `patched_module` changes if needed.
5. A smoke demo and a `--batch --limit` slice per runner, and a one-row GEPA and MIPRO run.
6. `TOPOS` in `scripts/run_topologies.sh` and `TOPOLOGIES` in the optimizer launchers.
