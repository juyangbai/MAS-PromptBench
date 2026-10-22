# Add a Topology

A topology is a way of wiring agents together, implemented once per dataset as a thin runner on top of `core/`. This page covers adding a new topology, or a new framework implementation of an existing one, from the runners to the bridge adapters the optimizers use.
{ .lede }

## Choose what you are adding

| Adding | Runner path | Seed prompts | Bridge key |
| --- | --- | --- | --- |
| A framework variant of an existing topology | `topologies/<topology>/<fw>/<ds>/<fw>_<ds>.py` | reuse `configs/prompts/<topology>/<ds>/` | `<topology>_<fw>` |
| A new topology | `topologies/<topo>/[<fw>/]<ds>/<fw>_<ds>.py` | new `configs/prompts/<topo>/<ds>/` | `<topo>` |

The existing variants follow the first row: the CrewAI, AutoGen and OpenAI Agents SDK runners read the prompts of `sequential/`, `centralized/` and `decentralized/`, and the bridge calls them `sequential_crewai`, `centralized_autogen` and `decentralized_openai_agents`. [Workflow Topologies](../mas/topologies.md) describes the five shapes that exist today.

## 1. Write one runner per dataset

Add nine runners, one per dataset folder. Each wires its framework and takes everything else from `core`: the task module `core/tasks/<ds>.py` (loader, prompt text, tools, scorer, records), `core.prompts.role_prompt` for the seed prompt plus its output contract, `core.llm` for the model clients, `core.teams` for the team, and `core.cli.main` for the [shared command line](../reference/cli.md#runner-options). [Add a Dataset](add-dataset.md#3-topology-runners) shows the module shape, and the runners of an existing topology are the closest template.

Keep the module-level names the adapters patch: `_load_prompt` or `SYSTEM_PROMPT`, `_build_llm` or `_build_client`, `VLLM_BASE_URL`, `MODEL_ID`, `N_AGENTS` and `N_ROUNDS` where the team shape is configurable, `_RECURSION_LIMIT`, `_OUTPUT_FORMAT_NUDGE`, and `solve`, `run_batch` and `main`. Read them at call time so a patched value takes effect.

For token and call counts, pass the framework's output through the matching extractor in `core/telemetry.py` (`langchain_telemetry`, `langchain_ensemble_telemetry`, `crewai_telemetry`, `autogen_telemetry`, or `openai_sdk_accumulate` per call), then `normalize`.

Pattern demos sit next to the runners. No runner imports them:

| File | Pattern |
| --- | --- |
| `topologies/single/langgraph_base.py` | LangGraph `create_react_agent` |
| `topologies/independent/langgraph_base.py` | LangGraph `Send` fan-out and fan-in |
| `topologies/sequential/crewai/crewai_base/` | CrewAI `Process.sequential` crew |
| `topologies/centralized/autogen/autogen_base.py` | AutoGen `SelectorGroupChat` |

`topologies/decentralized/openai_agents/agents_sdk_base.py` is different: it is the debate engine the Agents SDK runners share, and it loads the [isolated SDK install](../reference/environment.md#openai-agents-sdk).

## 2. Define the team and the prompts

A framework variant reuses its topology's team spec and prompts, so skip to step 3. For a new topology:

1. **Team.** Add a `<topo>:` entry to each `configs/teams/<ds>.yaml` and a builder for it in `_FIELDS` in `core/teams.py`, which turns the YAML into a `TeamSpec` for r ∈ {2, 4, 8, 10}. The [team-spec README](https://github.com/juyangbai/MAS-PromptBench/blob/main/configs/teams/README.md) documents the existing schemas.
2. **Roles.** Add a block to `configs/prompts/roles.yaml` with a `description` of the topology and, under `benchmarks`, one entry per dataset listing `<role>: "<job description>"`. Add `<topo>:` to `configs/prompts/tools.yaml` with a tool list per dataset.
3. **Prompts.** Run `python configs/generate_role_prompts.py --only <topo>` to write `configs/prompts/<topo>/<ds>/<role>.txt`; [Add a Dataset](add-dataset.md#4-seed-prompts) describes the generator.
4. **Contract.** Teach `output_contract()` in `core/output_contracts.py` which roles produce the final answer. It recognizes only `single`, `independent`, `sequential`, `sequential_crewai`, `centralized`, `centralized_autogen`, `decentralized` and `decentralized_openai_agents`; any other name gets no contract. Make the same change in `optimizers/bridge/output_contracts.py`.

## 3. Add it to the launchers

`scripts/run_topologies.sh` lists runner folders as `path:framework` pairs in `TOPOS` and skips any module that does not exist:

```bash title="scripts/run_topologies.sh"
TOPOS="single:langgraph independent:langgraph \
sequential/langgraph:langgraph sequential/crewai:crewai \
centralized/langgraph:langgraph centralized/autogen:autogen \
decentralized/langgraph:langgraph decentralized/openai_agents:openai_agents"
```

Append your variant, for example `sequential/<fw>:<fw>`.

## 4. Bridge adapters

The optimizers reach a runner through an adapter in `optimizers/bridge/adapters/` that implements `RealRunnerAdapter` (`adapter_protocol.py`): `roles()`, `get_prompt(role)`, `set_prompt(role, text)`, `reset()`, `run_example(example)` and `format_role_trace(role, output)`. `run_example` returns a dict with `model_output` (what the metric reads) and, optionally, `winner` and `buckets`. Keep state per instance; MIPRO's demos arrive through `set_prompt()`, so an adapter needs no optimizer-specific code.

For a framework variant, subclass the existing adapter of each dataset and change three attributes. This is how the CrewAI MATH adapter is defined:

```python title="optimizers/bridge/adapters/module_math.py"
class SequentialCrewAIMATHAdapter(SequentialMATHAdapter):
    topology = "sequential_crewai"
    framework = "crewai"
    module_name = "topologies.sequential.crewai.math.crewai_math"
```

```python title="optimizers/bridge/registry.py"
"sequential_crewai": "optimizers.bridge.adapters.module_math:SequentialCrewAIMATHAdapter",
```

For a new topology, also set `prompt_topology` (the prompt folder) and `roles_` (the role files to load). Two places key on names:

- The `framework` attribute picks the client the adapter injects into `_build_llm` and `_build_client` (`patched_module` in `adapters/module_common.py`). A new framework needs its own branch there.
- `default_adapter_kwargs` in `optimizers/protocol/runner.py` passes the team size as `n_agents` to `independent` and `decentralized` adapters and `n_rounds=2` to `decentralized` ones. Add your topology if its team shape is configurable.

Registry values are import strings, so a framework is imported only when its pair is requested. The [bridge README](https://github.com/juyangbai/MAS-PromptBench/blob/main/optimizers/bridge/README.md#adding-a-pair) has the full checklist and the [adapter template](https://github.com/juyangbai/MAS-PromptBench/blob/main/optimizers/bridge/templates/adapter_template.py).

## 5. Run it through the protocol

`optimizers/protocol/cells.py` maps a cell to a bridge key: `registry_key()` and `parse_registry_key()` know the five topologies (`TOPOLOGIES` in `config.py`) and the three framework keys in `FRAMEWORK_KEYS`. Teach them your name; a framework also goes into the `--framework` choices in `run.py`. The experiment grid (`build_grid`) and its frozen sizes (`EXPECTED_COUNTS`) describe the published experiments, so a new pair runs with `--allow-any-cell` and is reported as non-conformant.

```bash title="Smoke-test the pair"
python -m optimizers.protocol.run --method gepa --dataset math --topology sequential_<fw> \
  --model qwen --seed 0 --budget 56 --phase optimize --allow-any-cell \
  --out runs/smoke/math_sequential_<fw>
```

Expect `job.json`, `optimization.json` and `optimization/optimizer_result.json` under `--out`. The golden inventory is derived from the file tree and the bridge registry, so new runners and pairs show up in `test_inventory`; record them with `python -m tests.golden.record --only '<glob>'`.

## Checklist

1. Nine runners on `core`, keeping the module-level names the adapters patch.
2. New topology only: team specs and their `core/teams.py` builder, YAML entries and generated prompts, final roles in both `output_contracts.py` copies.
3. `TOPOS` in `scripts/run_topologies.sh`.
4. Adapters and `DATASET_ADAPTERS` entries, plus `patched_module` and `default_adapter_kwargs` changes if needed.
5. The protocol's cell mapping, a smoke demo and a `--batch --limit` slice per runner, a smoke job and golden cells.
