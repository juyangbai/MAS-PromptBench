# Add an Optimizer

A method is one package, `optimizers/<method>/`, whose `integration.py` exposes an optimizer class registered with the run protocol. The protocol hands it a budgeted runner and the train and validation rows, and does everything else: final validation, selection, test and artifacts. Your results are then comparable with the other eight methods by construction.
{ .lede }

## What the protocol owns

| The protocol | The method |
| --- | --- |
| Builds the cell's adapter, the seed bundle and the dataset metric. | Proposes candidate bundles. |
| Charges usable rollouts to a `BudgetLedger` of B = 600 and never overshoots it. | Decides which rows and bundles to run, within B. |
| Retries infrastructure failures with the same seed, uncharged. | Chooses its incumbent. |
| Runs the uncharged final validation and the strictly-better selection. | Nothing: it never sees the selection. |
| Runs the test split after the selection is locked. | Nothing: test rows never reach it. |
| Writes `job.json`, `optimization.json`, `selection.json`, `test.json` and `result.json`. | Writes optional scratch files under its `run_dir`. |

[Evaluation Protocol](../evaluation/protocol.md) defines each of these steps.

## 1. Create and register the package

Put the method in `optimizers/<method>/`, with its optimizer class in `integration.py` and its knobs in one frozen settings dataclass, as the existing methods do. Register the class in `METHODS` in [`optimizers/protocol/methods/__init__.py`](https://github.com/juyangbai/MAS-PromptBench/blob/main/optimizers/protocol/methods/__init__.py):

```python title="optimizers/protocol/methods/__init__.py"
METHODS: dict[str, str] = {
    "gepa": "optimizers.gepa.integration:GEPAOptimizer",
    # ...
    "my_method": "optimizers.my_method.integration:MyOptimizer",
}
```

The registry is lazy: nothing is imported until a job asks for the method, and the `--method` choices of `optimizers.protocol.run` come from it. [`identity.py`](https://github.com/juyangbai/MAS-PromptBench/blob/main/optimizers/protocol/methods/identity.py) is the smallest complete method: it returns the seed bundle without a rollout.

## 2. Implement the interface

```python title="optimizers/my_method/integration.py"
class MyOptimizer:
    def __init__(self, seed_bundle: PromptBundle, *, run_dir=None, reflection_lm=None): ...
    def optimize(self, cell: CellSpec, runner, budget: BudgetLedger,
                 training: list[dict], validation: list[dict]) -> OptimizerResult: ...
```

- **Constructor.** `methods.build_optimizer` passes the seed bundle as `seed_bundle` or `initial_bundle`, whichever the signature declares. If declared, it also passes `run_dir` (a scratch folder, `optimization/native/`), `reflection_lm` (`dspy_bridge.build_reflection_lm()`) and `task_lm` (`dspy_bridge.build_task_lm()`).
- **Bundles.** A `PromptBundle` holds `roles` (role name to prompt text), optional `demos` and `metadata`; its `digest` covers all three. A candidate must have exactly the seed bundle's roles.
- **Rows.** JSON-safe dicts with an `id`, a `task_instance` and the gold labels. Pass them to the runner unchanged.
- **Runner.** `ProtocolRunner.run(example, bundle, request_seed) -> RunRecord` and `run_batch(examples, bundle, request_seeds) -> BatchExecution`, plus `budget`, `cell`, `phase`, `seed_bundle` / `initial_bundle`, `required_roles`, `validate_bundle`, `artifact_store`, `artifact_directory` and `supports_concurrent = False`. A batch reserves budget before dispatch and is trimmed at B.
- **Records.** A `RunRecord` carries `score`, `status`, `final_output`, `messages`, `usage` and the metric's feedback text in `metadata["scorer_metadata"]["feedback"]`. Only `success` and `semantic_failure` records are usable and charged.

The protocol package has building blocks for the common cases: `seeding.request_seeds` for logical request seeds, `rollouts` for text methods, `session.RunnerSession` for a method with its own adapter-driven loop, `reflection.ReflectionClient` for the reflection model, and `methods/dspy_bridge.py` for DSPy programs. Rollouts are serialized per process; run jobs in parallel as processes.

## 3. Return a result

`optimize` returns a `schema.OptimizerResult` with `method`, `cell_id`, the seed and incumbent bundles, `budget_snapshot == budget.snapshot()` and a `schema.StopReason`. The protocol rejects a result whose identity or budget snapshot differs from the job, or whose incumbent the runtime cannot execute.

- **Layouts.** The result is published as `SEARCH_LAYOUT` (the incumbent as `selected_bundle`; GEPA, MIPRO, MASPOB and TAVO) or `JOURNAL_LAYOUT` (`incumbent_bundle`; `identity` and the `RunnerSession` methods, through `journal.LifecycleJournal`).
- **Errors.** All exceptions live in `optimizers/protocol/errors.py`. An infrastructure failure (`PreObservationInfrastructureFailure`, `NativeInfrastructureExhausted`, `OptimizerInfrastructureFailure`) keeps the seed bundle with the fallback reason `infrastructure_invalid_optimization`; any other exception fails the job.
- **Masked execution.** A bundle with `metadata["optimizer_control"]` runs through the hook registered by `runner.register_execution_hook("optimizer_control", hook)`, which provides `build_adapter(runtime, request, control)` and `verify(runtime, request, control, output)`; without a hook it is rejected.

## 4. Test it

1. **Unit tests** go to `optimizers/<method>/tests/`. Use fakes (see `optimizers/conftest.py`) so they need no endpoint:

    ```bash
    PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider optimizers/<method> -q
    ```

2. **Smoke job.** A new method is outside the experiment grid, so it needs `--allow-any-cell`. Methods charge rollouts before proposing anything, so pick a budget that reaches one proposal (see [smoke runs](../optimizers/running.md#smoke-runs)):

    ```bash
    python -m optimizers.protocol.run --method my_method --dataset hotpotqa \
      --topology centralized --model qwen --seed 0 --budget 60 --allow-any-cell \
      --out runs/smoke/my_method
    ```

3. **Golden cell.** Add a row to `METHOD_CELLS` in `tests/golden/methods.py`. These cells pin each method's end-to-end trace against a scripted fake server. Record it with `python -m tests.golden.record --only 'methods/my_method/*'`.

Jobs outside the grid are non-conformant, so `optimizers.protocol.aggregate` skips them unless you pass `--include-nonconformant`. The grid of published cells is `build_grid` in `optimizers/protocol/cells.py`.

## Checklist

1. `optimizers/<method>/integration.py` with the optimizer class and a frozen settings dataclass.
2. A `METHODS` entry in `optimizers/protocol/methods/__init__.py`.
3. `optimize` spends only the runner's ledger, uses train and validation rows only, and returns an `OptimizerResult` with the matching budget snapshot.
4. Seed prompts in `configs/` left untouched; scratch files under `run_dir`.
5. Unit tests with fakes, a smoke job and a golden cell.
