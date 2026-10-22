# Add a Dataset

A run is one dataset solved by one topology runner with one team and one communication format. Each piece is written once and parameters choose the run, so a new dataset is a task module, a team spec, a set of thin runners, seed prompts, its evaluation IDs and splits, and a bridge loader for the optimizers.
{ .lede }

The steps use `<ds>` for the dataset's short name. Use the same name everywhere: several components find the dataset by that string.

| Piece | Lives in | Provides |
| --- | --- | --- |
| Task module | `core/tasks/<ds>.py` | Loading, prompt text, tools, answer extraction, scoring, records. |
| Seed prompts | `configs/prompts/<topology>/<ds>/<role>.txt` | Each agent's seed system prompt, generated from YAML. |
| Output contract | `core/output_contracts.py` | The protected final-output instruction of the answering roles. |
| Team | `configs/teams/<ds>.yaml`, read by `core/teams.py` (`TeamSpec`) | The agents at team size r ∈ {2, 4, 8, 10}. |
| Topology runners | `topologies/<topology>/[<framework>/]<ds>/<framework>_<ds>.py` | The framework wiring: graph, crew, group chat or debate. |
| Team sizes | `teamsizes/<topology>/<ds>/<ds>_r<r>.py`, via `core/variant.py` | The LangGraph runner with another r. |
| Communication formats | `core/communication.py` (`CommPolicy`), `communications/<topology>/<ds>/<ds>_<format>.py` | How agents report to each other, and the parse metrics. |
| Evaluation IDs and splits | `benchmarks/<ds>/<ds>_eval_ids.json`, `<ds>_splits.json` | The reported instances and the optimizers' fixed splits. |
| Bridge | `optimizers/bridge/datasets/<ds>.py`, `adapters/`, `registry.py` | The optimizers' loader, metric and adapters. |

All three runner families run the same runner module:

```bash title="Entry points"
python -m topologies.<topology>.[<framework>.]<ds>.<framework>_<ds>   # team r=4, freeform reports
python -m teamsizes.<topology>.<ds>.<ds>_r8                           # the runner with TEAM_SIZE=8
python -m communications.<topology>.<ds>.<ds>_structured_soft         # with COMMUNICATION_FORMAT preset
```

## 1. Task module

`core/tasks/<ds>.py` holds what every topology shares, with no framework imports at module level.

- **Constants.** `DATASET`, the source constants and `SOURCE` (the "loading ..." line).
- **Loader.** `load_instances(limit=None, offset=0, only=None, ...)`: `only` selects IDs, then `offset` and `limit` apply. Dataset options are explicit parameters, declared by `add_arguments(parser)`.
- **Prompt text.** The user message (`format_prompt`) and prompt nudges, as constants.
- **Tools.** Plain functions plus a factory taking the docstring (`make_x(doc)`). A tool's docstring is its description as the model sees it, so every wording is a constant, indentation included; runners add their framework's wrapper.
- **Scoring.** Answer extraction and the reference scorer, as published.
- **Records.** `record(inst, pred, **fields)`, `summarize(records)`, progress lines and `run_batch(instances, row, *, out_path, verbose, label)` over `core.batch.run_batch`.
- **Reuse** the shared [core modules](../reference/repository.md#core) instead of copying them.
- **Tests** go to `core/tasks/tests/test_<ds>.py`; tests of shared modules go to `tests/core/`.

## 2. Team spec

`configs/teams/<ds>.yaml` has one entry per multi-agent topology, following the [schema](https://github.com/juyangbai/MAS-PromptBench/blob/main/configs/teams/README.md): a replicated role (`independent`, `decentralized`), the r stages (`sequential`), or a manager and its r - 1 workers (`centralized`). `recursion_limit` is the same at every r and is omitted when the agents make plain model calls. All text is sent to the model verbatim.

## 3. Topology runners

Write one module per (topology, framework), eight in all. Each has this shape:

```python title="topologies/centralized/langgraph/<ds>/langgraph_<ds>.py (sketch)"
TOPOLOGY = "centralized"
TEAM_SIZE = globals().get("TEAM_SIZE")  # preset by the team-size variants
TEAM = teams.spec(TOPOLOGY, task.DATASET, TEAM_SIZE)

VLLM_BASE_URL = settings.base_url()     # repointed per rollout by the optimizers
MODEL_ID = settings.model_id()

def _load_prompt(role: str) -> str:
    return prompts.role_prompt(TOPOLOGY, task.DATASET, role)

def _build_llm():
    return chat_openai(model=MODEL_ID, base_url=VLLM_BASE_URL)

def solve(...) -> dict: ...                     # one instance
def run_batch(instances, out_path=None, verbose=True) -> dict: ...
def main(argv=None) -> int:
    return cli.main(argv, description=..., load_instances=task.load_instances, run_batch=run_batch,
                    demo=_canned_demo, source=task.SOURCE, add_arguments=task.add_arguments)
```

[`topologies/single/math/langgraph_math.py`](https://github.com/juyangbai/MAS-PromptBench/blob/main/topologies/single/math/langgraph_math.py) is a complete single-agent example. The `core/llm.py` docstring lists the client builders for every runner style.

!!! warning "Module attributes are an interface"
    Read them at call time, keep their names, and keep when they are called (how many clients a row builds, when a prompt is loaded). The optimizer adapters patch or read `_load_prompt`, `SYSTEM_PROMPT`, `_build_llm`, `_build_client`, `VLLM_BASE_URL`, `MODEL_ID`, `N_AGENTS`, `N_ROUNDS`, `_RECURSION_LIMIT` and `_OUTPUT_FORMAT_NUDGE`. HiveMind patches `DELEGATION_TOOLS`, `DELEGATION_NAMES`, `MANAGER_TOOLS`, `_manager_tool_node` and `_MANAGER_TERMINATE_NUDGE` of the centralized LangGraph runners. The golden harness reads `solve`, `run_one`, `run_batch`, `load_instances` and the scorers.

The command line comes from `core.cli.main`, which adds the [shared runner options](../reference/cli.md#runner-options) and the dataset's `add_arguments`:

- A dataset may also pass `default_limit`, an `epilog`, an info mode (`cli.InfoMode("--summary", dataset_summary)`: a JSON report instead of a run) and `configure`, applied to the parsed options first.
- The loader, `run_batch`, `configure` and the info report receive the options they declare as parameters.
- Diagnostics go through `logging.getLogger(__name__)`; stdout is for results only.

## 4. Seed prompts

Seed prompts are written by an LLM. For every `(topology, dataset, role)` in `roles.yaml`, [`configs/generate_role_prompts.py`](https://github.com/juyangbai/MAS-PromptBench/blob/main/configs/generate_role_prompts.py) fills `meta_prompt.txt` with the dataset line from `domains.yaml`, the topology and role descriptions from `roles.yaml` and the tool list from `tools.yaml`. It sends the result as one user message, strips any `<think>` block and writes the reply to `configs/prompts/<topology>/<ds>/<role>.txt`.

Add the dataset to the three YAML files in `configs/prompts/`:

- `domains.yaml`: one line, `<ds>: "..."`. The generator only visits datasets listed here.
- `roles.yaml`: under `topologies.<topology>.benchmarks` for each of the five topologies, `<ds>:` with `<role>: "<job description>"`. Role names must match what the runners and the team spec load.
- `tools.yaml`: under each topology, `<ds>:` with the tool list. A missing entry becomes `None.`, which tells the prompt to use no tools.

Then generate; existing files are skipped unless you pass `--force`:

```bash title="Write the new prompt files"
python configs/generate_role_prompts.py --only single/<ds> --only independent/<ds> \
  --only sequential/<ds> --only centralized/<ds> --only decentralized/<ds>
```

The generator defaults to `Qwen/Qwen3.5-122B-A10B-FP8` at `http://localhost:8200/v1`, temperature 0 and seed 42 ([options](../reference/cli.md#prompt-generation)). The CrewAI, AutoGen and Agents SDK runners read the prompts of their topology folder.

## 5. Output contract

Add one sentence to `DATASET_CONTRACTS` in `core/output_contracts.py` that names the final artifact your scorer reads, for example "End with exactly one final line: Answer: <short-form>." Then list the roles that emit it:

- `single` and `independent`: roles in `PRIMARY_FINAL_ROLES` (`solver`, `caller`, `coder`, `patcher`, `planner`);
- `decentralized`: `debater`;
- `sequential`: `SEQUENTIAL_FINAL_ROLES["<ds>"]`;
- `centralized`: `CENTRALIZED_FINAL_ROLES_BY_DATASET["<ds>"]`.

The bridge adapters attach their own copy at execution time, so make the same entries in `optimizers/bridge/output_contracts.py`, whose contracts also carry the `PROTECTED FINAL OUTPUT CONTRACT:` header. [Output contracts](../evaluation/protocol.md#output-contracts) explains how they are applied.

## 6. Team sizes

A team-size module runs the topology runner's source in its own namespace with a preset parameter (`core.variant`), so it owns a full set of hooks:

```python title="teamsizes/centralized/<ds>/<ds>_r8.py"
"""Centralized <DATASET> with team size r=8: a manager and 7 workers."""

from core import variant

variant.load(globals(), "topologies.centralized.langgraph.<ds>.langgraph_<ds>", TEAM_SIZE=8)
```

- The runner derives everything size-dependent from `TEAM`: agents, delegate tools, `max_turns` and the default output folder.
- API-Bank and ToolHop are the exception: their team sizes vote over r replicas of one role (`teamsizes/apibank_common.py`, `teamsizes/toolhop_common.py`, loaded with `TOPOLOGY` and `TEAM_SIZE`), and their CrewAI and AutoGen runners are the LangGraph runners under another `STYLE`.

## 7. Communication formats

A multi-agent runner selects its format once, at load time, and never rebinds it, so runners loaded with different formats never share one:

```python title="Format selection in a runner"
COMMUNICATION_FORMAT = globals().get("COMMUNICATION_FORMAT", "freeform")  # preset by communications/
COMMUNICATION = CommPolicy(COMMUNICATION_FORMAT, task.DATASET, TOPOLOGY)

def _load_prompt(role: str) -> str:
    return COMMUNICATION.system_prompt(prompts.role_prompt(TOPOLOGY, task.DATASET, role))
```

- `COMMUNICATION.system_prompt` appends the format's contract wherever a system prompt is completed: `_load_prompt`, a module-level `SYSTEM_PROMPT`, API-Bank's four-argument `_load_prompt` and ToolHop's `_system_prompt`.
- `COMMUNICATION.handoff(role, text, ...)` renders one agent's output before another receives it.
- API-Bank and ToolHop runners default to `None`, not `"freeform"`: outside a communications run their agents read each other's outputs as plain text.
- Each `communications/<topology>/<ds>/<ds>_<format>.py` calls `communication_formats.install(globals(), topology=..., dataset=..., fmt=...)`. The dataset's `Pair` subclass in `communications/communication_formats.py` loads the LangGraph runner with the format preset (`core.variant.module`) and defines the pair's `solve` (handoffs recorded, reports scored), `run_one`, `run_batch` and `main`.
- The bridge adapters load the runner the same way: `import_isolated_real_module(module_name, COMMUNICATION_FORMAT=fmt)`.

## 8. Evaluation IDs and splits

Ship two files in `benchmarks/<ds>/`, with IDs equal to the `id` values of `load_instances`:

- `<ds>_eval_ids.json`: `dataset`, `sample`, `n`, `source` and `ids`, the reported instances.
- `<ds>_splits.json`: fixed `train`, `validation` and `test` ID lists. `test` is the evaluation set in the same order; `train` and `validation` are drawn from the remaining pool and never overlap it.

[Evaluation Protocol](../evaluation/protocol.md#fixed-splits) describes both files.

## 9. Optimizer bridge

1. Add `optimizers/bridge/datasets/<ds>.py` from [`templates/dataset_template.py`](https://github.com/juyangbai/MAS-PromptBench/blob/main/optimizers/bridge/templates/dataset_template.py): `load_all()` returns `dspy.Example`s with an `id` and a `task_instance` input, and `metric(example, prediction, ...)` returns `dspy.Prediction(score=..., feedback=...)`. The protocol imports the module by the dataset name.
2. Add adapters under `optimizers/bridge/adapters/`, usually a `module_<ds>.py` whose classes subclass `ModuleAdapterBase` (`module_common.py`) as `module_lcb.py` does, and register them in `DATASET_ADAPTERS` in `optimizers/bridge/registry.py`. [Add a Topology](add-topology.md#4-bridge-adapters) covers the adapter contract.
3. A new dataset is outside the experiment grid, so protocol jobs on it need `--allow-any-cell` and are non-conformant. Run the adapter's `run_example()` on one example, then the optimization phase on a [smoke budget](../optimizers/running.md#smoke-runs), and check that `job.json`, `optimization.json` and `optimization/optimizer_result.json` exist under `--out`:

```bash title="Smoke-test the bridge"
python -m optimizers.protocol.run --method gepa --dataset <ds> --topology single --model qwen \
  --seed 0 --budget 56 --phase optimize --allow-any-cell --out runs/smoke/<ds>_single
```

## 10. Golden tests

Add the dataset to `tests/golden/cells.py` and record its cells once with `python -m tests.golden.record --only '<glob>'`. From then on, change behavior only on purpose:

```bash title="Check, then re-record an intended change"
GOLDEN_WORKERS=16 HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 \
  TOOLHOP_ALLOW_DATASET_EXEC=1 python -m pytest -q tests/golden tests/core core/tasks/tests optimizers
python -m tests.golden.record --only '<cells whose change is intended>' --force  # after reading the diff
ruff check <files> && ruff format --check <files>
```

## 11. Launchers

The sweep scripts list their datasets and row limits in code: add `<ds>` to the `DATASETS` default and the `LIMIT` table of `scripts/run_topologies.sh`, and of `run_teamsizes.sh` and `run_communications.sh` if you add those variants. A dataset missing from `LIMIT` in `run_topologies.sh` runs 50 rows.

## Checklist

1. `core/tasks/<ds>.py` with its tests, and `configs/teams/<ds>.yaml`.
2. Eight topology runners, plus team-size and communication modules if you add those variants.
3. YAML entries and generated prompts under `configs/prompts/<topology>/<ds>/`.
4. Contract and final roles in `core/output_contracts.py` and `optimizers/bridge/output_contracts.py`.
5. `<ds>_eval_ids.json` and `<ds>_splits.json` in `benchmarks/<ds>/`.
6. The bridge loader, adapters and registry entry, and a smoke job.
7. Golden cells, and the launchers' `DATASETS` and `LIMIT`.
