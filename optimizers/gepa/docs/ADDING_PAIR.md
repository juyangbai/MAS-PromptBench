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

1. Add a dataset loader under `real_runner_gepa/datasets/<dataset>.py`.
2. Add one or more adapters under `real_runner_gepa/adapters/`.
3. Register the adapter map in `real_runner_gepa/registry.py`.
4. Add or update a pilot runner under `real_runner_gepa/pilots/`.
5. Add a smoke command and expected artifact location to this document.

## Dataset Loader Contract

The loader should return `dspy.Example` objects with `.with_inputs(...)`.
For real-runner GEPA, the recommended input is one `task_instance` dict:

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

Also expose a scorer or metric helper that the pilot can call.

## Adapter Contract

Adapters implement `RealRunnerAdapter`:

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

## Registry

Add the adapter map to `real_runner_gepa/registry.py`:

```python
DATASET_ADAPTERS = {
    "bfcl": {...},
    "my_dataset": {
        "my_topology": "real_runner_gepa.adapters.my_topology:MyAdapter",
    },
}
```

Registry values are import strings on purpose. This keeps unrelated framework
dependencies from loading until that specific pair is requested.

## Smoke Test Checklist

Before a long GEPA run:

1. `python3 -m py_compile` the new adapter and pilot.
2. Instantiate the adapter and call `describe_runtime()` if available.
3. Run `train=1`, `val=1`, `max_full_evals=1`.
4. Verify these artifacts exist:
   - `meta.json`
   - `baseline_val.jsonl`
   - `compiled_val.jsonl`
   - `compiled/<role>.txt`
   - `compiled_raw/<role>.txt`

## Prompt-Cleaning Rule

Write raw optimized prompts to `compiled_raw/`. Write deployable prompts to
`compiled/`. Only strip reflection preambles when the text looks like a
reflection output, not when a seed prompt merely contains a fenced JSON example.
