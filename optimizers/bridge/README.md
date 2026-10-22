# Real-Runner Bridge

**`optimizers.bridge`** plugs the **real** topology runners into the prompt optimizers: every method of the [run protocol](../protocol/README.md) runs its rollouts through it, so an optimized prompt is scored by the same runner the benchmark reports. Adapters hold role prompts per instance, so no optimizer edits `configs/` or `topologies/`.

Part of [optimizers/](../README.md).

## Overview

The registry covers all 9 datasets on every topology and framework of `topologies/`, plus the team-size and communication variants of HotpotQA, LCB, BFCL, API-Bank and ToolHop.

### Directory layout

```
bridge/
├── registry.py                     # pair → adapter class (import strings)
├── adapter_protocol.py             # RealRunnerAdapter: the adapter interface
├── adapters/                       # one prompt-mutable adapter per pair
├── datasets/                       # per-dataset loader and metric
│   └── split_utils.py                  # the fixed splits
├── lm.py                           # task endpoints and decoding (eval_mode)
├── output_contracts.py             # protected final-output contracts
├── programs.py                     # DSPy programs as GEPA sees them (GEPA_VIEW)
├── mipro_programs.py               # MIPRO's view (MIPRO_VIEW): demo rendering, MIPRORolePredict
├── env.py                          # environment variables
├── templates/                      # adapter and dataset skeletons
└── tests/
```

---

## How it works

1. An **adapter** owns one `(topology, dataset)` pair: its per-role prompts (`roles()` / `get_prompt()` / `set_prompt()`) and `run_example()`, one real-runner execution.
2. A **program** registers one DSPy predictor per mutable role, so a DSPy optimizer (GEPA, MIPRO) mutates the role instructions through `named_predictors()`.
3. `forward()` syncs the candidate prompts into the adapter, runs it and emits one trace per role; the dataset `metric` scores it.

**Keep `MIPRORolePredict` and `MIPROAdapterBackedProgram` verbatim**: MIPRO's program-aware proposer shows their source to the prompt model, and `optimizers/mipro/tests/test_mipro.py` pins its SHA-256.

---

## Settings

Environment variables are listed in [`env.py`](env.py) and described in [Environment Variables](../../docs/content/reference/environment.md#optimizer-bridge).

---

## Adding a pair

A **pair** is `dataset + topology + execution framework`, e.g. `bfcl/single`, `bfcl/sequential_crewai`, `bfcl/decentralized_openai_agents`. Team-size and communication variants register as `<topology>_r<r>` and `<topology>_communications_<format>`.

1. Add a dataset loader `datasets/<dataset>.py` ([contract](#dataset-loader-contract)).
2. Add one or more adapters under `adapters/` ([contract](#adapter-contract)).
3. Register the pair in `registry.py` ([registry](#registry)).
4. Smoke-test the wiring before a long run ([checklist](#smoke-test-checklist)).

### Dataset loader contract

Add the dataset's runners first ([Add a Dataset](../../docs/content/extending/add-dataset.md)). Skeleton: [`templates/dataset_template.py`](templates/dataset_template.py).

- `load_all()` returns `dspy.Example`s whose single input is the task instance:

  ```python
  dspy.Example(
      id="stable_id",
      task_instance={"id": "stable_id", ...},  # fields consumed by the adapter
      answer=...,                               # fields consumed by the metric
  ).with_inputs("task_instance")
  ```

- `metric(example, prediction, ...)` returns `dspy.Prediction(score=..., feedback=...)`.
- `benchmarks/<dataset>/<dataset>_splits.json` lists the `train`, `validation` and `test` ids the protocol takes from `load_all()`.

### Adapter contract

Implement `RealRunnerAdapter` (skeleton: [`templates/adapter_template.py`](templates/adapter_template.py)):

```python
class MyAdapter:
    topology = "my_topology"
    dataset = "my_dataset"

    def roles(self) -> list[str]: ...
    def get_prompt(self, role: str) -> str: ...
    def set_prompt(self, role: str, text: str) -> None: ...
    def reset(self) -> None: ...
    def run_example(self, example) -> dict: ...
    def format_role_trace(self, role: str, output) -> str: ...
```

- `run_example()` returns a dict with `model_output` (what the metric reads), optionally `winner` and `buckets` (vote/consensus summary), and role-specific trace fields.
- Keep state per instance. MIPRO's demos arrive through `set_prompt()`, so an adapter need not know which optimizer drives it.

### Registry

```python
DATASET_ADAPTERS = {
    "my_dataset": {"my_topology": "optimizers.bridge.adapters.my_topology:MyAdapter"},
}
```

Values are import strings, so a framework is imported only when its pair is requested. The key is what `--topology` takes.

### Smoke test checklist

1. Run the adapter's `run_example()` on one example.
2. Run the optimization phase on a (non-conformant) [smoke budget](../protocol/README.md#smoke-budgets):

   ```bash
   python -m optimizers.protocol.run --method gepa --dataset <dataset> --topology <key> --model qwen \
       --seed 0 --budget 56 --phase optimize --allow-any-cell --out runs/smoke/<dataset>_<key>
   ```

3. Check that `job.json`, `optimization.json` and `optimization/optimizer_result.json` exist under `--out`.
