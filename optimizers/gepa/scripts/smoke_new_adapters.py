"""Fast smoke test for newly registered real-runner adapters.

This imports every real topology module, patches its `solve` function with a
deterministic fake response, then verifies adapter execution, DSPy prediction
conversion, output-contract injection, and dataset metric extraction.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
WORKSPACE = ROOT / "optimizers/gepa"
sys.path[:0] = [str(WORKSPACE), str(ROOT)]

from real_runner_gepa.datasets import apps, lcb, swe, travel
from real_runner_gepa.output_contracts import output_contract
from real_runner_gepa.programs import RealRunnerProgram
from real_runner_gepa.registry import get_adapter_class, topologies


DATASETS = {
    "apps": (
        apps.metric,
        {
            "id": "apps_fake",
            "problem": "Read stdin and print 1.",
            "starter_code": "",
            "input_output": {"inputs": [""], "outputs": ["1"]},
        },
    ),
    "lcb": (
        lcb.metric,
        {
            "id": "lcb_fake",
            "problem": "Read stdin and print 1.",
            "starter_code": "",
            "tests": [{"input": "", "output": "1"}],
        },
    ),
    "swe": (
        swe.metric,
        {
            "id": "swe_fake",
            "instance_id": "repo__1",
            "repo": "fake/repo",
            "base_commit": "",
            "problem_statement": "Change buggy.py.",
            "hints_text": "",
            "FAIL_TO_PASS": [],
            "PASS_TO_PASS": [],
        },
    ),
    "travel": (
        travel.metric,
        {
            "id": "travel_fake",
            "query": "Plan a 1 day trip from A to B.",
            "query_data": {},
            "days": 1,
        },
    ),
}

CODE_RAW = "```python\nprint(1)\n```"
DIFF_RAW = """```diff
diff --git a/buggy.py b/buggy.py
--- a/buggy.py
+++ b/buggy.py
@@ -1,2 +1,2 @@
-def buggy():
-    return "replace me if needed"
+def buggy():
+    return "fixed"
```"""
PATCH = """diff --git a/buggy.py b/buggy.py
--- a/buggy.py
+++ b/buggy.py
@@ -1,2 +1,2 @@
-def buggy():
-    return "replace me if needed"
+def buggy():
+    return "fixed"
"""
PLAN = [{"days": 1, "current_city": "from A to B", "transportation": "Flight"}]
PLAN_RAW = '```json\n[{"days":1,"current_city":"from A to B","transportation":"Flight"}]\n```'


def fake_output(dataset: str, topology: str) -> dict:
    if dataset in {"apps", "lcb"}:
        out = {"code": "print(1)", "raw": CODE_RAW, "messages": [{"source": "manager", "content": CODE_RAW}]}
        if topology.startswith("sequential"):
            out["by_stage"] = {"debugger": CODE_RAW, "coder": CODE_RAW}
        if topology in {"independent", "decentralized", "decentralized_openai"}:
            out["winner"] = 0
            out["per_agent" if topology == "independent" else "per_peer"] = [
                {"agent_id": 0, "peer": 0, "code": "print(1)", "raw": CODE_RAW}
            ]
        return out
    if dataset == "swe":
        out = {"patch": PATCH, "raw": DIFF_RAW, "messages": [{"source": "manager", "content": DIFF_RAW}]}
        if topology.startswith("sequential"):
            out["by_stage"] = {"tester": DIFF_RAW, "patcher": DIFF_RAW}
        if topology in {"independent", "decentralized", "decentralized_openai"}:
            out["winner"] = 0
            out["per_agent" if topology == "independent" else "per_peer"] = [
                {"agent_id": 0, "peer": 0, "patch": PATCH, "raw": DIFF_RAW}
            ]
        return out
    out = {"plan": PLAN, "raw": PLAN_RAW, "messages": [{"source": "manager", "content": PLAN_RAW}]}
    if topology.startswith("sequential"):
        out["by_stage"] = {"finalizer": PLAN_RAW, "itinerary_drafter": PLAN_RAW}
    if topology in {"independent", "decentralized", "decentralized_openai"}:
        out["winner"] = 0
        out["per_agent" if topology == "independent" else "per_peer"] = [
            {"agent_id": 0, "peer": 0, "plan": PLAN, "raw": PLAN_RAW}
        ]
    return out


def as_example(instance: dict):
    class Example:
        pass

    ex = Example()
    for key, value in instance.items():
        setattr(ex, key, value)
    return ex


def main() -> int:
    failures = []
    for dataset, (metric, instance) in DATASETS.items():
        for topology in topologies(dataset):
            cls = get_adapter_class(dataset, topology)
            adapter = cls(n_agents=2, n_rounds=1) if topology in {"independent", "decentralized", "decentralized_openai"} else cls()
            if dataset == "travel":
                from real_runner_gepa.adapters.module_travel import ensure_travel_eval_stubs

                ensure_travel_eval_stubs()
            module = importlib.import_module(adapter.module_name)
            original = getattr(module, "solve")

            def fake_solve(*args, _dataset=dataset, _topology=topology, **kwargs):
                return fake_output(_dataset, _topology)

            setattr(module, "solve", fake_solve)
            try:
                kwargs = (
                    {"n_agents": 2, "n_rounds": 1}
                    if topology in {"decentralized", "decentralized_openai"}
                    else {"n_agents": 2}
                    if topology == "independent"
                    else {}
                )
                program = RealRunnerProgram(dataset, topology, **kwargs)
                pred = program(task_instance=instance)
                score = float(metric(as_example(instance), pred).score)
                contracted = [role for role in adapter.roles() if output_contract(dataset, adapter.prompt_topology, role)]
                print(f"{dataset}/{topology}: score={score} execution_contract_roles={contracted}")
                if score < 1.0 or not contracted:
                    failures.append((dataset, topology, score, contracted))
            except Exception as exc:
                print(f"{dataset}/{topology}: FAIL {type(exc).__name__}: {exc}")
                failures.append((dataset, topology, type(exc).__name__, str(exc)))
            finally:
                setattr(module, "solve", original)

    if failures:
        print(f"failures={failures}")
        return 1
    print("all adapter smokes passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
