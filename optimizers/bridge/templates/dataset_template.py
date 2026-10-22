"""Template for a dataset module of the shared bridge (`optimizers/bridge/datasets/<dataset>.py`).

The run protocol takes the train/validation/test rows of `load_all()` listed in
`benchmarks/<dataset>/<dataset>_splits.json`, so every split id must be a
`load_all()` id.
"""

from __future__ import annotations

import dspy


def load_all() -> list[dspy.Example]:
    """Return examples with stable ids and `task_instance` input."""
    examples: list[dspy.Example] = []
    # TODO: populate examples.
    return examples


def metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
    """Score one prediction: `dspy.Prediction(score=0.0 or 1.0, feedback=...)`."""
    # TODO: extract the answer from `prediction` and compare it with `example`.
    return dspy.Prediction(score=0.0, feedback="TODO")
