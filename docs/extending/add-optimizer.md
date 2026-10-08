# Add an Optimizer

An optimizer plugs into MAS-PromptBench by rewriting the per-role prompts of a real topology runner and scoring each candidate by running that runner. This page describes what GEPA and MIPRO do so that a third optimizer produces results you can compare with theirs.
{ .lede }

!!! note "A guide, not a plugin API"
    The repository has no optimizer registry or shared base class. GEPA and MIPRO are self-contained workspaces that share one design by copy, kept separate on purpose because their dataset and LM internals have diverged (`optimizers/README.md`). Everything below is derived from those two workspaces.

## What the two optimizers share

| Piece | GEPA | MIPRO |
| --- | --- | --- |
| Bridge package | `real_runner_gepa/` | `real_runner_mipro/` |
| Adapter protocol | `protocol.py` | `protocol.py` (identical) |
| Pair registry | `registry.py` | `registry.py` (same pairs) |
| DSPy program | `programs.py` | `mipro_programs.py` (builds on `programs.py`) |
| Dataset loaders and metrics | `datasets/<ds>.py` | `datasets/<ds>.py` |
| Endpoints and sampling | `lm.py` (`GEPA_*` variables) | `lm.py` (`MIPRO_*`, falling back to `GEPA_*`) |
| Per-pair entry point | `pilots/run_gepa_dataset.py` | `pilots/run_mipro_dataset.py` |
| Sweep launcher | `run_gepa.sh` | `run_mipro.sh` |

The quickest route for a new optimizer is a third workspace, `optimizers/<name>/`, that copies this layout and replaces only the optimization call. Run its pilot from the workspace folder with `python -m`, as GEPA and MIPRO do; the pilots put the workspace and the repository root on `sys.path` themselves.

## 1. Drive a pair through the adapter

Every `(dataset, topology)` pair is reached through an adapter that implements the `RealRunnerAdapter` protocol:

```python title="optimizers/gepa/real_runner_gepa/protocol.py (signatures)"
class RealRunnerAdapter(Protocol):
    topology: str
    dataset: str

    def roles(self) -> list[str]: ...
    def get_prompt(self, role: str) -> str: ...
    def set_prompt(self, role: str, text: str) -> None: ...
    def reset(self) -> None: ...
    def run_example(self, example: Any) -> Any: ...
    def format_role_trace(self, role: str, output: Any) -> str: ...
```

`validate_adapter` requires the first five methods. `registry.get_adapter_class(dataset, topology)` returns an adapter class and `registry.topologies(dataset)` lists the valid names. Adapters read seed prompts from `configs/prompts/<topology>/<dataset>/<role>.txt` and never write back. The shared module adapters patch the runner's prompt loader, model client and team size, call its `solve()`, and return a dict with `model_output`, `winner`, `buckets` and the raw `runner_output`. `format_role_trace` gives short per-role text for reflection.

To an optimizer, then, a pair is a set of named text parameters (the roles) plus a black-box `run_example` that runs the whole multi-agent system once.

## 2. Expose the prompts as parameters

Both existing optimizers are DSPy teleprompters, so they wrap the adapter in a `dspy.Module` with one predictor per role:

- **GEPA** uses `AdapterBackedProgram`. Each role becomes a `RealRunnerRolePredict` whose signature instructions are the role prompt, so GEPA finds and edits them through `named_predictors()`. `forward(task_instance)` copies every predictor's instructions into the adapter with `set_prompt`, runs the adapter once, converts the output into a `dspy.Prediction` (`answer`, `tool_calls`, `winner`, `vote_summary`, `agent_trace`) and appends one trace entry per role.
- **MIPRO** uses `MIPROAdapterBackedProgram`. Its `MIPRORolePredict` renders the selected instruction and few-shot demos into one prompt before calling `set_prompt`, and `role_artifacts()` returns each role's final prompt and demos for saving.

A DSPy-based optimizer can reuse `AdapterBackedProgram` unchanged. An optimizer outside DSPy can call `set_prompt` and `run_example` directly, but the dataset metrics expect a prediction with an `answer` field, which `prediction_from_adapter_output` in `programs.py` builds. Calling the program and scoring its output is the simpler path.

## 3. Score candidates the same way

Use the dataset modules and the same split, metric and acceptance rule as the existing pilots:

1. **Load and split.** `rows = dataset_mod.load_all()`, then `dataset_mod.train_val_split(rows, train_size, val_size, seed, offset)`.
2. **Keep eval IDs out.** The split functions drop every ID in `benchmarks/<dataset>/<dataset>_eval_ids.json` unless `GEPA_EXCLUDE_REAL_EVAL_IDS` (or `MIPRO_EXCLUDE_REAL_EVAL_IDS`) is set to `0`, `false`, `no` or `off`; API-Bank and ToolHop always drop them. Reuse these functions rather than writing your own, and record what was excluded in `meta.json`.
3. **Score.** `dataset_mod.metric(example, prediction)` returns an object whose `.score` is in [0, 1] and whose `.feedback` text GEPA uses for reflection. A split's score is the mean of its per-row scores.
4. **Baseline and compiled.** Evaluate the seed prompts on the validation split (baseline), optimize on the train split, then evaluate the optimized prompts on the same validation split (compiled). `delta = compiled_score - baseline_score`.
5. **Accept or keep the seeds.** If any compiled validation record contains an infrastructure-error marker (connection errors, CUDA out-of-memory, context-length errors, Python tracebacks; see `HARD_INFRA_MARKERS` in the pilots), the compiled prompts are rejected. ToolHop uses a stricter rule that also rejects ties without row-level gains and prompts that fail its anti-memorization checks. Otherwise the compiled prompts are accepted when `compiled_score + 1e-9 >= baseline_score`. The records of the selected side are written to `optimized_val.jsonl`.

Before the compiled evaluation, both pilots cap LCB prompts at `LCB_COMPILED_PROMPT_CHAR_LIMIT` characters (default 24000) and strip concrete examples from ToolHop prompts. Copy those steps if your optimizer can produce long or example-heavy prompts.

## 4. Write the same artifacts

Write everything under the `--out` folder in the layout the existing pilots use:

```text
<out>/
├── status.json               # current phase; "complete" or "failed" at the end
├── meta.json                 # settings, splits, scores, acceptance decision
├── baseline_val.jsonl        # per-row validation records, seed prompts
├── compiled_val.jsonl        # per-row validation records, optimized prompts
├── optimized_val.jsonl       # whichever of the two was selected
├── compiled/<role>.txt       # deployable prompts
├── compiled_raw/<role>.txt   # raw optimizer output
└── <name>_state/             # optimizer logs (gepa_state/, mipro_state/)
```

Each validation record holds `id`, `score`, `latency_s`, `answer`, `winner`, `vote_summary` and `agent_trace`. MIPRO also writes `compiled_demos/<role>.json`.

The repository ships no analysis script that reads `meta.json`; the fields below are what both pilots write, so keeping them makes your runs comparable. [Read the Results](../evaluation/results.md) explains how to use them.

| Group | Fields |
| --- | --- |
| Identity | `cell`, `mode`, `dataset`, `topology` |
| Split | `train_size`, `val_size`, `offset`, `split_seed`, `actual_train_ids`, `actual_val_ids`, `split_manifest` |
| Eval protection | `exclude_real_eval_ids`, `excluded_real_eval_id_count`, `excluded_real_eval_ids`, `train_real_eval_overlap`, `val_real_eval_overlap`, `protected_eval_sample` |
| Prompts | `adapter_roles`, `compiled_prompt_files`, `compiled_prompt_quality`, `compiled_prompt_sanitizer` |
| Models | `task_model`, `reflection_model`, `task_endpoints`, `reflection_endpoint` |
| Settings | `n_agents`, `n_rounds`, `auto`, `num_threads`, `output_contract_version`, `dspy_version`, plus your optimizer's budget settings |
| Outcome | `baseline_score`, `compiled_score`, `delta`, `accept_compiled_prompt`, `selected_prompt_source`, `selection_reason`, `acceptance_diagnostics`, `baseline_failure_summary`, `compiled_failure_summary`, `baseline_records`, `compiled_records` |

For team-size and protocol pairs the pilots add `team_size`, `base_topology` and `teamsizes_module`, or `communications_format` and `communications_module`.

## 5. Use the same models and a launcher

Copy `lm.py` and give its variables your optimizer's prefix. It builds a round-robin task LM over a comma-separated endpoint list and a separate reflection or proposal LM. Task sampling is fixed at temperature 0.2, top-p 0.9, seed 0, 1024 max tokens and thinking disabled; keep it so your baselines run under the same conditions as GEPA's and MIPRO's.

Finally, add `run_<name>.sh`: loop over `DATASETS` and `TOPOLOGIES`, start one pilot process per pair with `--out "$OUT_ROOT/${topo}_${dataset}"`, and continue past failures, as `run_gepa.sh` does. See [Run an Optimizer](../optimizers/running.md) for how the existing launchers are used.

## Checklist

1. A workspace `optimizers/<name>/` with its own bridge package, pilot and launcher.
2. The adapter protocol and registry reused unchanged, so every existing pair is a target.
3. Prompts exposed per role, synced into the adapter with `set_prompt` before each run.
4. Train and validation splits from the dataset modules, with eval IDs excluded and recorded.
5. Baseline and compiled scores on the same validation split, and the acceptance rule above.
6. `status.json`, `meta.json`, the three validation JSONL files, `compiled/` and `compiled_raw/` under `--out`; seed prompts in `configs/` left untouched.
7. A one-row run (`--train-size 1 --val-size 1`) before any sweep.
