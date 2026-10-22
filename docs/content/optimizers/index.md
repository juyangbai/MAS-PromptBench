# Optimizers

MAS-PromptBench optimizes the system prompts of a multi-agent system while the model weights stay frozen. Eight methods share one run protocol and one bridge to the real topology runners, so every method spends the same budget, is selected by the same rule and is scored on the same held-out rows.
{ .lede }

<div class="facts" markdown>
<div><span>Methods</span>8</div>
<div><span>Protocol</span>mas-promptbench</div>
<div><span>Budget</span>600 rollouts</div>
<div><span>Optimizer seeds</span>0, 1, 2</div>
</div>

## What gets optimized

A multi-agent system \( M \) is a topology plus a set of roles: a planner, a solver, a manager, a debater and so on. Each role has a system prompt, and the seed prompts live in `configs/prompts/<topology>/<dataset>/<role>.txt`. A method treats the joint set \( \pi = (\pi_1, \dots, \pi_K) \) of role prompts as the only variable; the model, the topology wiring, the tools and the scorer stay fixed. MIPRO's few-shot demos are rendered into the role prompts, so they are part of \( \pi \) too.

The quantity of interest is the prompt-optimization gain on the held-out test split:

\[
\Delta = \mathbb{E}_{(x, y) \sim \mathcal{D}_{\text{test}}}\left[\mu\big(M(x; \pi^{\text{dep}}), y\big) - \mu\big(M(x; \pi^0), y\big)\right]
\]

Here \( \pi^0 \) is the seed prompt set, \( \pi^{\text{dep}} \) the deployed set, \( \mu \) the task metric and \( (x, y) \) a test instance with its reference answer. The deployed set is the method's incumbent only if it beats the seeds on validation; otherwise it is \( \pi^0 \) and the gain is zero. A job's `result.json` reports the estimate as `delta_pp`, in percentage points, and [aggregation](running.md#aggregate-the-seeds) averages it over the three optimizer seeds.

No method writes to `configs/`. Optimized prompts stay in the job's `--out` folder.

## The eight methods

| Method | `--method` | Approach | Code | Grid cells |
| --- | --- | --- | --- | ---: |
| GEPA | `gepa` | Reflective prompt evolution (DSPy `GEPA`) | [`gepa/`](https://github.com/juyangbai/MAS-PromptBench/tree/main/optimizers/gepa) | 147 |
| MIPRO | `mipro` | Instruction and few-shot demo search (DSPy `MIPROv2`) | [`mipro/`](https://github.com/juyangbai/MAS-PromptBench/tree/main/optimizers/mipro) | 147 |
| MAPRO | `mapro` | Per-role prompt pools, max-product belief propagation, blame-driven mutation | [`mapro/`](https://github.com/juyangbai/MAS-PromptBench/tree/main/optimizers/mapro) | 135 |
| MASPO | `maspo` | Role-wise evolutionary beam search with pairwise judging | [`maspo/`](https://github.com/juyangbai/MAS-PromptBench/tree/main/optimizers/maspo) | 135 |
| HiveMind | `hivemind` | Coalition (Shapley) credit, lesson-based refinement of the lowest-credit role | [`hivemind/`](https://github.com/juyangbai/MAS-PromptBench/tree/main/optimizers/hivemind) | 12 |
| MAMUT-GEPA | `mamut_gepa` | One joint GEPA search over all role prompts (`gepa` engine) | [`mamut_gepa/`](https://github.com/juyangbai/MAS-PromptBench/tree/main/optimizers/mamut_gepa) | 12 |
| MASPOB | `maspob` | Prompt-variant bandit with a GATv2 surrogate (LinUCB) | [`maspob/`](https://github.com/juyangbai/MAS-PromptBench/tree/main/optimizers/maspob) | 12 |
| TAVO | `tavo` | Trajectory credit assignment and a shared verbalized-policy overlay | [`tavo/`](https://github.com/juyangbai/MAS-PromptBench/tree/main/optimizers/tavo) | 12 |

A grid cell is one (dataset, topology, framework, communication format, team size, task model) configuration of the experiment grid in `optimizers/protocol/cells.py`, run once per optimizer seed:

- **GEPA and MIPRO** cover every dataset on all five topologies, in each topology's native framework and in LangGraph, plus the communication formats and team sizes on HotpotQA, LiveCodeBench and BFCL, and those three datasets with `meta-llama/Llama-3.1-8B-Instruct`.
- **MAPRO and MASPO** cover the same cells except `single`.
- **HiveMind, MAMUT-GEPA, MASPOB and TAVO** cover HotpotQA, LiveCodeBench and BFCL on the four multi-agent LangGraph topologies.

The built-in `identity` method returns the seed prompts without a rollout; it runs on any grid configuration and gives the seed baseline of a cell. Any method runs on any registered pair with `--allow-any-cell`, as a non-conformant job.

## One protocol for every method

Every method runs as a job of the protocol `mas-promptbench`, `python -m optimizers.protocol.run`:

1. **Optimize.** The method gets the fixed, ordered train and validation rows, the frozen seed bundle and a ledger of 600 usable full-system rollouts, sampled at temperature 0.2, top-p 0.9 and up to 32,768 tokens. It returns its incumbent bundle.
2. **Final validation.** Uncharged and greedy: the seed and the incumbent run on the full validation split with paired per-item request seeds. The incumbent is deployed only if it is strictly better; a tie or a regression keeps the seeds. The decision is sealed in `selection.json`.
3. **Test.** After the lock, the seed and deployed bundles run on the held-out test split with the same paired seeds.

The methods differ only in step 1. Their reflection or proposal calls go to `Qwen/Qwen3.5-122B-A10B-FP8` with thinking on and up to 48,000 output tokens, at temperature and top-p 1.0 unless the method sets them. [Evaluation Protocol](../evaluation/protocol.md) covers budgets, seeds and scorers in detail.

## The real-runner bridge

Every rollout runs through [`optimizers/bridge/`](https://github.com/juyangbai/MAS-PromptBench/tree/main/optimizers/bridge), which wraps the topology runners instead of re-implementing them, so an optimized prompt runs unchanged in the benchmark.

1. **Adapter** (`adapters/`). One class per (dataset, registry key) pair exposes the per-role prompts (`roles()`, `get_prompt()`, `set_prompt()`) and `run_example()`, one execution of the real runner. It patches the runner module's prompt loader, model client, endpoint and team size in a private copy of the module.
2. **Registry** (`registry.py`). Maps each dataset and key, such as `sequential_crewai` or `centralized_r8`, to its adapter class.
3. **Programs** (`programs.py`, `mipro_programs.py`). Register one DSPy predictor per role, so GEPA and MIPRO edit the role instructions through `named_predictors()`. The other methods pass their candidates to the protocol's runner, directly or through its `RunnerSession`, which runs the same adapter.
4. **Metric** (`datasets/<dataset>.py`). `load_all()` and `metric()`, which returns a 0 or 1 score and feedback text for reflection.

The bridge also re-attaches what a method cannot edit: the protected final-output contract of the answering roles and a few format nudges. See [Output contracts](../evaluation/protocol.md#output-contracts).

!!! note "Cheaper checks for code and patches"
    The protocol scores LiveCodeBench and APPS on the first three tests only, and SWE-bench with a structural check of the diff, cheaper than the topology runners' scorers. [Scorers](../evaluation/protocol.md#scorers) lists every metric.

## Method settings

Each method's knobs are one frozen dataclass; the protocol fixes everything else. Only MAPRO and TAVO also read [environment variables](../reference/environment.md#method-settings).

| Method | Settings | Defaults |
| --- | --- | --- |
| GEPA | `gepa/integration.py: GEPAPolicy` | `max_full_evals=5`, minibatch 3, Pareto selection, round-robin components, merge (at most 5), plateau patience 3, seed 0 |
| MIPRO | `mipro/integration.py: MIPROPolicy` | 3 candidates, 3 trials, at most 4 bootstrapped and 0 labeled demos, no minibatch, seed 9, all proposer hints on |
| MAPRO | `mapro/regime.py: MAPROSettings` | 5 candidates per role, at most 8 rounds, patience 3, scoring batch 3, feedback 3, 12 threads |
| MASPO | `maspo/integration.py: MASPOSettings` | beam 2, 2 offspring, minibatch 10, depth at most 9, 3 rounds per turn, 12 threads, 4 judge calls in flight |
| HiveMind | `hivemind/regime.py: HiveMindSettings` | coalition batch 5, acceptance batch 5, at most 40 coalitions, manager every 3rd cycle, at most 6 lessons, reflection temperature 0.7, cycles until B |
| MAMUT-GEPA | `mamut_gepa/integration.py: MAMUTGEPASettings` | `max_metric_calls = min(600, B)`, Pareto, round-robin, merge, minibatch 3, reflection temperature 0.7 |
| MASPOB | `maspob/regime.py: MASPOBSettings` | 20 variants per role, generation temperature 0.5, MiniLM embeddings, seed 42 (+1000 per optimizer seed), minibatch 5, at most 5 warm-up pulls |
| TAVO | `tavo/settings.py: TAVOSettings` | train batch 6, at most 5 outer rounds, validation batch at least 3, adoption threshold 0.01, 2 attempts per round, patience 2, temperature 0.5 |

The [run protocol README](https://github.com/juyangbai/MAS-PromptBench/blob/main/optimizers/protocol/README.md#settings) keeps the full table. MASPOB also needs `torch`, `torch_geometric` and `sentence-transformers`, which `environment.yml` installs; a CPU is enough.

## Next steps

<div class="cards" markdown>

- [Run an Optimizer](running.md)
  Endpoints, cells, one job, smoke runs and aggregation over seeds.
- [Evaluation Protocol](../evaluation/protocol.md)
  Splits, budget, decoding, selection and scorers.
- [Read Run Outputs](../evaluation/outputs.md)
  Every file of a job folder and the aggregate summary.
- [Add an Optimizer](../extending/add-optimizer.md)
  The interface a ninth method implements.

</div>
