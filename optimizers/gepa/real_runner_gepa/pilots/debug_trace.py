"""Print one DSPy trace for the real-runner program."""
from __future__ import annotations

import sys
from pathlib import Path

import dspy

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = WORKSPACE_ROOT.parent.parent
for path in (str(WORKSPACE_ROOT), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from real_runner_gepa.datasets.bfcl import load_all
from real_runner_gepa.programs import IndependentBFCLRealRunnerProgram


def main() -> int:
    row = load_all()[0]
    program = IndependentBFCLRealRunnerProgram(n_agents=1)
    trace = []
    with dspy.context(trace=trace):
        pred = program(task_instance=row.task_instance)
    print("prediction:", pred)
    print("trace_len:", len(trace))
    for idx, item in enumerate(trace):
        module, inputs, outputs = item
        print(f"trace[{idx}] module={type(module).__name__}")
        print("  signature:", getattr(module, "signature", None))
        print("  inputs:", inputs)
        print("  outputs:", outputs)
    print("named_predictors:", [(n, type(p).__name__) for n, p in program.named_predictors()])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

