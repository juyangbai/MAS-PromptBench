# Optimizers

MAS-PromptBench optimizes the system prompts of a multi-agent system while the model weights stay frozen. GEPA and MIPRO both score every candidate prompt set by running the real topology runner, so the prompts they return run unchanged in the benchmark.
{ .lede }

## What gets optimized

A multi-agent system \( M \) is a topology plus a set of roles: a planner, a solver, a manager, a debater and so on. Each role has a system prompt. The seed prompts live in `configs/prompts/<topology>/<dataset>/<role>.txt`, one file per role. The optimizers treat the joint set \( \pi = (\pi_1, \dots, \pi_K) \) of role prompts as the only variable. The agent model, the topology wiring, the tools and the scorer stay fixed.

The quantity of interest is the prompt-optimization gain:

\[
\Delta = \mathbb{E}_{(x, y) \sim \mathcal{D}}\left[\mu\big(M(x; \pi^\ast), y\big) - \mu\big(M(x; \pi^0), y\big)\right]
\]

Here \( \pi^0 \) is the seed prompt set, \( \pi^\ast \) the optimized set, \( \mu \) the task scorer and \( (x, y) \) a task instance with its reference answer. A run's `meta.json` reports the sample estimate as `delta = compiled_score - baseline_score`, both means over the same validation rows. Scores are fractions in [0, 1]; multiply by 100 for percentage points.

Neither optimizer writes to `configs/`. Compiled prompts go to the run's `--out` folder.

## The real-runner bridge

Each optimizer has its own bridge package: [`real_runner_gepa/`](https://github.com/juyangbai/MAS-PromptBench/tree/main/optimizers/gepa/real_runner_gepa) and [`real_runner_mipro/`](https://github.com/juyangbai/MAS-PromptBench/tree/main/optimizers/mipro/real_runner_mipro). They follow the same three layers.

1. **Adapter** (`adapters/`). One class per `(topology, dataset)` pair implements the `RealRunnerAdapter` protocol in `protocol.py`: `roles()`, `get_prompt()`, `set_prompt()`, `reset()`, `run_example()` and `format_role_trace()`. It loads the seed prompts, imports the real runner module (for example `topologies.single.hotpotqa.langgraph_hotpotqa`) and, for each example, patches the module's prompt loader, model client, endpoint, model name and agent or round counts before calling the runner's own solve function.
2. **Program** (`programs.py`, plus `mipro_programs.py` for MIPRO). A DSPy module registers one predictor per mutable role. The predictor's instruction text is the role prompt, so the optimizer finds and edits the roles through `named_predictors()`.
3. **Forward pass.** `forward()` copies the current instructions into the adapter, runs the real runner once, and records one trace per role for the optimizer to read.

`registry.py` maps each `(dataset, topology)` name to its adapter class. The bridge also re-attaches two things the optimizer cannot edit: the protected final-output contract for the role that emits the answer, and a few format nudges (code-first for APPS and LCB coders, patch-first for SWE). See [Evaluation Protocol](../evaluation/protocol.md#output-contracts).

!!! note "Scoring during optimization"
    The runner code is the real one, but the metric each optimizer maximizes lives in `datasets/<dataset>.py`. For LCB and APPS it runs only the first three tests, and for SWE-bench it checks that the patch is a non-trivial unified diff. [Evaluation Protocol](../evaluation/protocol.md#scorers) lists every metric.

## GEPA and MIPRO compared

| | GEPA | MIPRO |
| --- | --- | --- |
| Method | Reflective prompt evolution (DSPy `GEPA`) | Instruction and few-shot demo search (DSPy `MIPROv2`) |
| What changes | Role instruction text | Role instruction text plus selected demos, rendered into the prompt |
| Proposal signal | Reflection model reads each role's trace and the metric's feedback text | Proposal model sees the program, a data summary, tips and bootstrapped demos |
| Search | Mutates one role or all roles per step, tracks candidates on the validation split | Proposes candidates, then a Bayesian search over instruction and demo combinations |
| Budget flags | `--max-full-evals` or `--auto` | `--num-candidates` and `--num-trials`, or `--auto` |
| Early stopping | `--early-stop-patience` | None |
| Extra output | `gepa_state/` | `mipro_state/`, `compiled_demos/` |
| Entry point | `real_runner_gepa.pilots.run_gepa_dataset` | `real_runner_mipro.pilots.run_mipro_dataset` |
| Sweep launcher | `optimizers/gepa/run_gepa.sh` | `optimizers/mipro/run_mipro.sh` |

The paper's headline results use GEPA.

## Train/val split and acceptance

Each run draws two disjoint splits from the dataset's optimization pool:

- **Train** (`--train-size`): rows the optimizer learns from. GEPA reflects on minibatches of them; MIPRO bootstraps demos and writes its data summary from them.
- **Validation** (`--val-size`): rows the optimizer uses to compare candidates, and the rows the pilot uses for the before/after measurement.

The frozen eval IDs in `benchmarks/<dataset>/<dataset>_eval_ids.json` are excluded from the pool unless you turn exclusion off. The pilot defaults are 1 train and 1 validation row, a smoke-test size; the sweeps use 25 and 25. [Evaluation Protocol](../evaluation/protocol.md#trainval-splits-for-optimization) covers the split rules.

After optimization the pilot scores the seed prompts and the compiled prompts on the same validation rows and applies one rule: keep the compiled prompts when `compiled_score + 1e-9 >= baseline_score`, so a tie keeps the compiled prompts. Two cases override it. If any compiled validation record shows an infrastructure error, the baseline prompts are kept. ToolHop uses a stricter rule that also rejects ties without row-level gains and prompts that memorize examples. The decision lands in `meta.json` as `accept_compiled_prompt`, `selected_prompt_source` and `selection_reason`. See [Read the Results](../evaluation/results.md#acceptance-and-selection_reason).

## Next steps

<div class="cards" markdown>

- [GEPA](gepa.md)
  The reflective loop, every flag and its default, and a worked command.
- [MIPRO](mipro.md)
  Instruction proposal, demo search, every flag and its default.
- [Run an Optimizer](running.md)
  Endpoints, topology names, sweeps and compute tips, end to end.

</div>
