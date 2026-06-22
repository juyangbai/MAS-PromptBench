# Adding A New Dataset-Topology Pair

This workspace treats each runnable optimization target as a **pair**:

```text
dataset + topology + execution framework
```

Examples:

- `bfcl/single`
- `bfcl/decentralized_openai`
- `bfcl/sequential_crewai`
- `bfcl/centralized_autogen`

## Required Pieces

1. Add a dataset loader under `real_runner_mipro/datasets/<dataset>.py`.
2. Add one or more adapters under `real_runner_mipro/adapters/`.
3. Register the adapter map in `real_runner_mipro/registry.py`.
4. Reuse the generic pilot `real_runner_mipro/pilots/run_mipro_dataset.py` (only extend it if the pair needs new handling).
5. Add a smoke command and expected artifact location to this document.

## Dataset Loader Contract

The loader should return `dspy.Example` objects with `.with_inputs(...)`.
For real-runner MIPRO, the recommended input is one `task_instance` dict:

```python
dspy.Example(
    id="stable_id",
    task_instance={
        "id": "stable_id",
        # dataset-specific fields consumed by the adapter
    },
    # metric fields
    answer=...,
).with_inputs("task_instance")
```

Also expose a scorer or metric helper that the pilot can call. Train/val splits
draw from `benchmarks/<dataset>/<dataset>_eval_ids.json`; set
`MIPRO_EXCLUDE_REAL_EVAL_IDS=1` to hold the reported eval IDs out of the
optimization pool.

## Adapter Contract

Adapters implement `RealRunnerAdapter` (`real_runner_mipro/protocol.py`):

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

Return shape from `run_example()` should usually include:

```python
{
    "model_output": ...,   # canonical output consumed by metric
    "winner": ...,         # optional
    "buckets": ...,        # optional vote/consensus summary
    # role-specific trace fields
}
```

MIPRO renders its selected instructions and few-shot demos into the role prompts
via `set_prompt()` before each `run_example()`, so the adapter does not need to
know it is being optimized.

## Registry

Add the adapter map to `real_runner_mipro/registry.py`:

```python
DATASET_ADAPTERS = {
    "bfcl": {...},
    "my_dataset": {
        "my_topology": "real_runner_mipro.adapters.my_topology:MyAdapter",
    },
}
```

Registry values are import strings on purpose. This keeps unrelated framework
dependencies from loading until that specific pair is requested. There is no
`templates/` directory — copy the closest existing module (e.g.
`real_runner_mipro/adapters/single_bfcl.py`) as the starting point.

## Smoke Test Checklist

Before a long MIPRO run:

1. `python3 -m py_compile` the new adapter and dataset loader.
2. Instantiate the adapter and run `run_example()` on one example end to end.
   (`real_runner_mipro/pilots/debug_trace.py` prints one DSPy trace — it defaults
   to `bfcl`, so adapt it when inspecting a different pair.)
3. Run a minimal optimization: `--train-size 1 --val-size 1 --num-candidates 1 --num-trials 1`.
4. Verify these artifacts exist under `--out`:
   - `status.json`
   - `meta.json`
   - `baseline_val.jsonl`
   - `compiled_val.jsonl` and `optimized_val.jsonl`
   - `compiled/<role>.txt`
   - `compiled_raw/<role>.txt`
   - `compiled_demos/` (selected few-shot demos)

## Prompt-Cleaning Rule

Write raw optimized prompts to `compiled_raw/`, deployable prompts to `compiled/`,
and the selected few-shot demos to `compiled_demos/`. Only strip proposal
preambles when the text looks like an optimizer output, not when a seed prompt
merely contains a fenced JSON example.
