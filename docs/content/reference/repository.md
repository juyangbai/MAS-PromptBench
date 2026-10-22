# Repository Map

Where everything lives in the MAS-PromptBench repository, how the layers depend on each other, how one task and one optimizer job flow through the code, and which file to edit for common changes.
{ .lede }

## Folder tree

```text
MAS-PromptBench/
├── environment.yml                  # conda env "mas-promptbench" (Python 3.11), pinned frameworks
├── requirements-openai-agents.txt   # the isolated OpenAI Agents SDK install
├── assets/                          # logo and overview images
├── benchmarks/
│   ├── <dataset>/<dataset>_eval_ids.json   # frozen evaluation IDs
│   ├── <dataset>/<dataset>_splits.json     # fixed train / validation / test IDs
│   └── apibank/apibank_upstream/           # vendored API-Bank source
├── configs/
│   ├── generate_role_prompts.py     # writes the seed prompts with an LLM
│   ├── prompts/<topology>/<dataset>/<role>.txt
│   └── teams/<dataset>.yaml         # team specs per topology and team size
├── core/                            # code shared by every runner
│   └── tasks/<dataset>.py           # one task module per dataset
├── topologies/<topology>/[<framework>/]<dataset>/<framework>_<dataset>.py
├── teamsizes/<topology>/<dataset>/<dataset>_r<r>.py
├── communications/<topology>/<dataset>/<dataset>_<format>.py
├── optimizers/
│   ├── <method>/                    # one package per method, with integration.py
│   ├── protocol/                    # the run protocol; methods/ holds the registry
│   └── bridge/                      # real-runner adapters, dataset loaders, templates/
├── models/                          # vLLM serve scripts
├── scripts/                         # baseline sweep launchers
├── tests/                           # core/ unit tests, golden/ behavior snapshots
└── docs/                            # this site: mkdocs.yml, content/, theme/
```

The eight method packages are `gepa`, `mipro`, `mapro`, `maspo`, `hivemind`, `mamut_gepa`, `maspob` and `tavo`. Runs write to `results/` (runners) and `runs/` (the examples on these pages use it for protocol jobs); both are gitignored, and so is `vendor/`, where the OpenAI Agents SDK is installed.

## Layers

Each layer imports only the layers above it. The protocol loads the methods by name from its registry, `optimizers/protocol/methods/__init__.py`.

| Layer | Path | Contents |
| --- | --- | --- |
| Benchmarks | `benchmarks/<dataset>/` | Evaluation IDs, fixed splits, the API-Bank source. |
| Configuration | `configs/` | Seed role prompts and team specs. |
| Core | `core/` | Code shared by every runner; one task module per dataset in `core/tasks/`. |
| Runners | `topologies/`, `teamsizes/`, `communications/` | One module per (topology, framework, dataset), plus the team-size and communication variants. |
| Optimizer bridge | `optimizers/bridge/` | Prompt-mutable runner adapters, dataset loaders and metrics, endpoints, DSPy programs. |
| Run protocol | `optimizers/protocol/` | One job = (method, cell, optimizer seed): optimize, validate, test; aggregation; the method registry and the `identity` method. |
| Methods | `optimizers/<method>/` | Eight optimizers, one package each, registered class in `integration.py`. |

## Core

| Module | Contents |
| --- | --- |
| `core/tasks/<dataset>.py` | Loading, prompt text, tools, answer extraction, reference scorer, records, batch summary. |
| `settings.py`, `llm.py` | Endpoint and decoding, read at call time; chat-client builders for every framework. |
| `prompts.py`, `output_contracts.py` | A role's seed prompt plus its protected final-output contract. |
| `teams.py`, `variant.py` | Team specs per (topology, dataset, r); running a runner's source with a preset `TEAM_SIZE`, `TOPOLOGY`, `STYLE` or `COMMUNICATION_FORMAT`. |
| `communication.py` | The three inter-agent formats (`CommPolicy`): contract, handoff rendering, parse metrics. |
| `cli.py`, `batch.py`, `logs.py` | Shared command line; batch loop (`attempt`, `run_batch`); console logging. |
| `telemetry.py`, `thinking.py`, `voting.py` | Token and call counts; `<think>` removal; gold-free majority vote of ensembles. |
| `calculator.py`, `code_tasks.py`, `bfcl_calls.py` | Calculator tool; code extraction, `python_exec`, guarded calls and code-task records; BFCL call parsing. |
| `swe_sandbox.py`, `agent_runs.py`, `runtime.py` | SWE-bench shell sandbox; API-Bank and ToolHop runners whose agents call the endpoint directly; replay-safe LangChain histories and `row_timeout`. |
| `paths.py` | Repository folders: the root, `configs/`, `configs/prompts/`, `benchmarks/` and `results/`. |

## Runners

- **Topologies.** `topologies/<topology>/[<framework>/]<dataset>/<framework>_<dataset>.py`. `single` and `independent` are LangGraph only; `sequential` adds CrewAI, `centralized` AutoGen and `decentralized` the OpenAI Agents SDK. A runner wires its framework and takes everything else from `core`.
- **Team sizes.** `teamsizes/<topology>/<dataset>/<dataset>_r<r>.py` runs the LangGraph runner's source with `TEAM_SIZE=r` (`core.variant.load`). API-Bank and ToolHop vote over r replicas instead (`teamsizes/apibank_common.py`, `toolhop_common.py`).
- **Communication formats.** `communications/<topology>/<dataset>/<dataset>_<format>.py` loads the LangGraph runner as its own module with `COMMUNICATION_FORMAT` preset (`core.variant.module`).

## One task through a runner

```bash
python -m topologies.centralized.langgraph.hotpotqa.langgraph_hotpotqa --batch --limit 5
```

1. **Import.** The module builds its team from `configs/teams/hotpotqa.yaml` (r = 4 unless `TEAM_SIZE` is preset), its tools (one `delegate_to_<worker>` per worker) and its `CommPolicy` (default `freeform`).
2. **Command line.** `core.cli.main` parses the options, configures logging and loads the instances (`core.tasks.hotpotqa.load_instances`).
3. **Batch.** `core.batch.run_batch` empties the output files and calls the runner's `row` once per instance.
4. **Solve.** `solve(question)` runs inside `core.batch.attempt`, which times it and turns an exception into the record's `error`. Each system prompt is the role's seed prompt, its output contract and the format's contract. Every model call goes through `core.llm.chat_openai` with the decoding of `core.settings`.
5. **Score.** The reference scorer (here official EM and token F1) scores the extracted answer; `task.record` adds the telemetry.
6. **Output.** Records are appended to the predictions JSONL as they complete. Progress goes to stderr, the end-of-batch report to stdout.

## One optimizer job

```bash
python -m optimizers.protocol.run --method gepa --dataset hotpotqa --topology centralized \
  --model qwen --seed 0 --out runs/x
```

1. **Job setup.** Check the cell against the grid (`cells.py`), load the fixed splits (test rows stay locked until selection), and build the adapter, the metric scorer and the seed bundle (each role's shipped prompt). `job.json` seals the job's identity.
2. **Optimize.** The method gets a `ProtocolRunner`, a `BudgetLedger` of 600 usable rollouts and the train and validation rows; it returns its incumbent bundle (`optimization.json`).
3. **One rollout.** Install the bundle's prompts, set `MODEL_ID`, `REQUEST_SEED` and the phase decoding, and call `adapter.run_example`, which runs the runner's solve code in a private copy of its module (`import_isolated_real_module`). The dataset metric scores the prediction. Infrastructure failures are retried up to twice and never charged.
4. **Final validation.** Uncharged and greedy, with paired request seeds. The incumbent is deployed only if its validation mean is strictly higher than the seed bundle's; `selection.json` locks the decision.
5. **Test.** Seed and deployed bundles on the test split with the same seeds (`test.json`, `result.json` and a one-line JSON summary on stdout).
6. **Aggregate.** `python -m optimizers.protocol.aggregate runs/` pairs the three optimizer seeds of each cell.

The protocol scores with the bridge's dataset metrics, which are cheaper than the runners' scorers for LiveCodeBench, APPS and SWE-bench; see [Scorers](../evaluation/protocol.md#scorers).

## Where do I change...

| To change | Edit | Notes |
| --- | --- | --- |
| An agent's prompt | `configs/prompts/<topology>/<dataset>/<role>.txt` | The CrewAI runners read `sequential/`, the AutoGen runners `centralized/` and the Agents SDK runners `decentralized/`. To regenerate, see [Prompt generation](cli.md#prompt-generation). The optimizers never write these files. |
| A team | `configs/teams/<dataset>.yaml` | API-Bank and ToolHop fix their roles in code. |
| The required final-answer format | `core/output_contracts.py` | The bridge adapters carry their own copy, `optimizers/bridge/output_contracts.py`. |
| A scorer | `core/tasks/<dataset>.py` | The protocol's metric is `metric` in `optimizers/bridge/datasets/<dataset>.py`. |
| The model endpoint of runs | `VLLM_BASE_URL`, `MODEL_ID`, `OPENAI_API_KEY` | See [Task model and decoding](environment.md#task-model-and-decoding). |
| The endpoints of optimization | `TASK_ENDPOINTS`, `REFLECTION_MODEL_BASE_URL` | See [Run protocol](environment.md#run-protocol). |
| The protocol's budget, decoding or splits | `optimizers/protocol/config.py` | Hashed into every job's identity (`protocol_hash`), so a change makes jobs incomparable with earlier ones. |
| A method's settings | its settings dataclass | Listed under [Method settings](../optimizers/index.md#method-settings). |
| The experiment grid | `optimizers/protocol/cells.py` | Jobs outside it need `--allow-any-cell` and are non-conformant. |
| A baseline sweep grid | `DATASETS`, `TOPOLOGIES`, `FORMATS`, `RVALUES` for `scripts/run_*.sh`; `TOPOS` inside `run_topologies.sh` | Row limits are the `LIMIT` table in each script. See [Sweep launchers](environment.md#sweep-launchers). |
| The evaluation IDs or splits | `benchmarks/<dataset>/<dataset>_eval_ids.json`, `<dataset>_splits.json` | The `test` split is the evaluation set. |
| Which pairs an optimizer can target | `DATASET_ADAPTERS` in `optimizers/bridge/registry.py` | See [Add a Topology](../extending/add-topology.md). |
| Where results go | `--out` or `--out-dir` on a runner, `--out` on a job, `OUT_ROOT` on a launcher | See [Command-Line Flags](cli.md). |
