"""Run ONE golden cell in this (fresh) process and write its snapshot as JSON.

    python -m tests.golden.worker --cell <cell-id> --out <file.json>

The parent (``tests.golden.harness``) prepares a hermetic environment (fixed
decoding env, offline HF, temp HOME/TMPDIR, proxies pointing nowhere) and runs
every cell in its own process so module-level state cannot leak between
cells. This process starts the fake LLM server, blocks other network access
and dispatches to the driver for the cell kind.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import inspect
import io
import json
import os
import pickle
import sys
import time
import traceback
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
sys.dont_write_bytecode = True

from tests.golden import cells as cellmod  # noqa: E402
from tests.golden import netguard, responder, scrub  # noqa: E402
from tests.golden.fake_server import FakeChatServer  # noqa: E402

SCRIPT: dict[str, Any] = {}
SERVER: FakeChatServer | None = None
TMP = Path(os.environ.get("GOLDEN_CELL_TMP") or "/nonexistent")
SESSION = Path(os.environ.get("GOLDEN_SESSION_DIR") or TMP)

TOY_PATCH = (
    "```diff\n"
    "diff --git a/buggy.py b/buggy.py\n"
    "--- a/buggy.py\n"
    "+++ b/buggy.py\n"
    "@@ -1,2 +1,2 @@\n"
    " def buggy():\n"
    "-    return 'replace me if needed'\n"
    "+    return 'fixed'\n"
    "```"
)
# Reference solution for the first LCB eval id (Codeforces 1873A); any other
# id gets a trivially wrong program.
LCB_SOLUTIONS = {
    "1873_A": (
        "t = int(input())\n"
        "for _ in range(t):\n"
        "    s = input().strip()\n"
        '    print("YES" if sum(a != b for a, b in zip(s, "abc")) <= 2 else "NO")'
    ),
}
WRONG_PROGRAM = "print(0)"


# ----------------------------------------------------------------------------- scripts
def _apps_reference_solution(problem_id: str) -> str:
    from datasets import load_dataset

    ds = load_dataset("codeparrot/apps", split="test", trust_remote_code=True)
    for row in ds:
        if str(row.get("problem_id")) == str(problem_id):
            try:
                solutions = json.loads(row.get("solutions") or "[]")
            except Exception:
                solutions = []
            return solutions[0].strip() if solutions else WRONG_PROGRAM
    return WRONG_PROGRAM


def script_for(dataset: str, inst: dict) -> dict:
    """Scripted answers derived from the gold label of the fixed instance."""
    fence = "```python\n{}\n```"
    if dataset == "gpqa":
        letter = inst.get("correct_letter") or inst.get("answer")
        return {"final": f"Answer: {letter}", "exercise_tools": ["calculator"]}
    if dataset == "hotpotqa":
        return {"final": f"Answer: {inst.get('answer')}"}
    if dataset == "math":
        return {"final": f"\\boxed{{{inst.get('answer')}}}", "exercise_tools": ["calculator"]}
    if dataset == "lcb":
        code = LCB_SOLUTIONS.get(str(inst.get("id")), WRONG_PROGRAM)
        return {"final": fence.format(code), "exercise_tools": ["python_exec"]}
    if dataset == "apps":
        return {"final": fence.format(_apps_reference_solution(str(inst.get("id")))), "exercise_tools": ["python_exec"]}
    if dataset == "bfcl":
        calls = responder.bfcl_domain_calls(inst.get("ground_truth") or [])
        return {"final": responder.bfcl_final(calls), "domain_calls": calls}
    if dataset == "apibank":
        return {"final": responder.apibank_final(inst.get("gold_api_call") or inst.get("answer"))}
    if dataset == "toolhop":
        return {"final": f"<answer>{inst.get('answer')}</answer>", "exercise_tools": "*"}
    if dataset == "swe":
        return {
            "final": TOY_PATCH,
            "exercise_tools": ["file_write", "str_replace"],
            "tool_args": {
                "file_write": {"path": "buggy.py", "content": "def buggy():\n    return 'fixed'\n"},
                "str_replace": {"path": "buggy.py", "old": "replace me if needed", "new": "fixed"},
            },
        }
    raise KeyError(dataset)


# ----------------------------------------------------------------------------- helpers
def bfcl_category(instance_id: str) -> str:
    return instance_id.rsplit("_", 1)[0]


def import_module(name: str):
    return importlib.import_module(name)


def jsonl_artifacts(out_dir: Path) -> dict:
    """Read result/prediction JSONL files a runner wrote (traces are skipped)."""
    found = {}
    for path in sorted(out_dir.rglob("*.jsonl")):
        rel = path.relative_to(out_dir).as_posix()
        rows = []
        for line in path.read_text().splitlines():
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    rows.append({"unparsed": line})
        found[rel] = rows
    diffs = {p.relative_to(out_dir).as_posix(): p.read_text(errors="replace") for p in sorted(out_dir.rglob("*.diff"))}
    others = sorted(
        p.relative_to(out_dir).as_posix()
        for p in out_dir.rglob("*")
        if p.is_file() and p.suffix not in (".jsonl", ".diff")
    )
    out = {"jsonl": found, "other_files": others}
    if diffs:
        out["diffs"] = diffs
    return out


def reduce_result(value: Any) -> Any:
    return scrub.to_data(value, long_limit=scrub.LONG_STRING_LIMIT)


SCORE_KEYS = (
    "accuracy",
    "em",
    "em_sum",
    "n_correct",
    "correct",
    "valid",
    "valid_rate",
    "pass_rate",
    "f1",
    "answer_correct",
    "resolved",
    "score",
    "n",
    "n_extracted",
    "extracted_acc",
    "extracted_em",
)


def pick_score(summary: Any, artifacts: dict | None = None) -> dict:
    out = {}
    if isinstance(summary, dict):
        out.update({k: summary[k] for k in SCORE_KEYS if k in summary and not isinstance(summary[k], (dict, list))})
        per = summary.get("per_instance")
        if isinstance(per, list) and per and isinstance(per[0], dict):
            out["instance"] = {
                k: per[0][k] for k in SCORE_KEYS if k in per[0] and not isinstance(per[0][k], (dict, list))
            }
    if artifacts:
        for name, rows in (artifacts.get("jsonl") or {}).items():
            if "result" in name and rows and isinstance(rows[0], dict):
                out["instance"] = {
                    k: rows[0][k] for k in SCORE_KEYS if k in rows[0] and not isinstance(rows[0][k], (dict, list))
                }
    return scrub.to_data(out)


def call_run_batch(module, dataset: str, instance_id: str, instances: list, out_dir: Path):
    params = inspect.signature(module.run_batch).parameters
    kwargs: dict[str, Any] = {}
    if "instances" in params:
        kwargs["instances"] = instances
    if "category" in params:
        kwargs["category"] = bfcl_category(instance_id)
    if "only" in params:
        kwargs["only"] = [instance_id]
    if "limit" in params and "instances" not in params:
        kwargs["limit"] = None
    if "out_dir" in params:
        kwargs["out_dir"] = out_dir
    if "out_path" in params:
        kwargs["out_path"] = out_dir / "results.jsonl"
    if "verbose" in params:
        kwargs["verbose"] = False
    if "workdir_root" in params:
        kwargs["workdir_root"] = TMP / "swe_work"
    if "eval_mode" in params:
        kwargs["eval_mode"] = "none"
    for name in ("style", "topology", "role"):
        if name in params and params[name].default is inspect.Parameter.empty:
            kwargs[name] = getattr(module, name.upper())
    return module.run_batch(**kwargs)


def load_runner_instance(module, dataset: str, instance_id: str) -> dict:
    if dataset == "bfcl":
        loaded = module.load_instances(category=bfcl_category(instance_id), only=[instance_id])
        if isinstance(loaded, tuple):
            rows, gts = loaded
            return {**rows[0], "ground_truth": (gts[0] or {}).get("ground_truth") or []}
        return dict(loaded[0])
    loader = getattr(module, "load_instances", None)
    if loader is None:  # centralized/autogen/swe has no loader of its own
        from datasets import load_dataset

        rows = [
            r for r in load_dataset("princeton-nlp/SWE-bench_Verified", split="test") if r["instance_id"] == instance_id
        ]
        return dict(rows[0])
    rows = loader(only=[instance_id])
    if not rows:
        raise RuntimeError(f"instance {instance_id!r} not found by {module.__name__}.load_instances")
    return dict(rows[0])


def install_sdk_fake() -> None:
    from tests.golden import sdk_fake

    sdk_fake.install(SERVER.record, lambda body: responder.respond(body, SCRIPT))


# ----------------------------------------------------------------------------- drivers
def drive_runner(cell: dict) -> dict:
    dataset = cell["dataset"]
    instance_id = cellmod.first_eval_id(dataset)
    if cell["framework"] == "openai_agents":
        install_sdk_fake()
    module = import_module(cell["module"])
    if dataset == "swe":
        return drive_swe(cell, module, module)
    instance = load_runner_instance(module, dataset, instance_id)
    SCRIPT.update(script_for(dataset, instance))
    out_dir = TMP / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = call_run_batch(module, dataset, instance_id, [instance], out_dir)
    artifacts = jsonl_artifacts(out_dir)
    return {
        "result": reduce_result(summary),
        "artifacts": reduce_result(artifacts),
        "score": pick_score(summary, artifacts),
    }


def drive_communication(cell: dict) -> dict:
    dataset = cell["dataset"]
    instance_id = cellmod.first_eval_id(dataset)
    module = import_module(cell["module"])
    base = sys.modules[module.BASE_MODULE]
    if dataset == "swe":
        return drive_swe(cell, module, base)
    if dataset == "bfcl":
        instances = module.load_instances(category=bfcl_category(instance_id), only=[instance_id])
    else:
        instances = module.load_instances(only=[instance_id])
    SCRIPT.update(script_for(dataset, dict(instances[0])))
    out_dir = TMP / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = module.run_batch(instances, out_path=out_dir / "results.jsonl", verbose=False)
    artifacts = jsonl_artifacts(out_dir)
    return {
        "result": reduce_result(summary),
        "artifacts": reduce_result(artifacts),
        "score": pick_score(summary, artifacts),
    }


SWE_SKIP_REASON = (
    "grading needs the per-instance SWE-bench Singularity image (and the clone needs GitHub): "
    "the run path executes with eval_mode='none' on a toy git repository standing in for the clone; "
    "the offline scoring helpers are snapshotted on canned reports instead"
)


def drive_swe(cell: dict, entry, base) -> dict:
    """SWE: real run path with a faked clone and eval_mode='none' + offline helpers."""
    from tests.golden import swe_fake

    swe_fake.install()
    instance_id = cellmod.first_eval_id("swe")
    instance = load_runner_instance(entry if hasattr(entry, "load_instances") else base, "swe", instance_id)
    SCRIPT.update(script_for("swe", instance))
    out_dir = TMP / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    if cell["kind"] == "communication":
        summary = entry.run_one(instance, workdir_root=TMP / "swe_work", out_dir=out_dir, eval_mode="none")
    else:
        summary = call_run_batch(entry, "swe", instance_id, [instance], out_dir)
    artifacts = jsonl_artifacts(out_dir)
    return {
        "result": reduce_result(summary),
        "artifacts": reduce_result(artifacts),
        "helpers": swe_helpers(base, instance),
        "skipped": {"grading": SWE_SKIP_REASON},
        "score": pick_score(summary, artifacts),
    }


SWE_REPORTS = {
    "resolved": {
        "fail_to_pass": {"success": ["t1"], "failure": []},
        "pass_to_pass": {"success": ["t2"], "failure": []},
        "f2p_rate": 1.0,
        "p2p_rate": 1.0,
    },
    "partial": {
        "fail_to_pass": {"success": [], "failure": ["t1"]},
        "pass_to_pass": {"success": ["t2"], "failure": []},
        "f2p_rate": 0.0,
        "p2p_rate": 1.0,
    },
    "empty": {},
}


def swe_helpers(base, instance: dict) -> dict:
    out: dict[str, Any] = {}
    for name in ("is_resolved", "exact_match_score"):
        fn = getattr(base, name, None)
        if callable(fn):
            out[name] = {label: _safe(lambda r=report, f=fn: f(r)) for label, report in SWE_REPORTS.items()}
    fn = getattr(base, "predictions_entry", None)
    if callable(fn):
        out["predictions_entry"] = _safe(lambda: fn(instance.get("instance_id"), "PATCH"))
    for name in ("extract_answer", "extract_patch", "strip_thinking"):
        fn = getattr(base, name, None)
        if callable(fn):
            out[name] = {
                label: _safe(lambda t=text, f=fn: f(t))
                for label, text in (("diff", TOY_PATCH), ("think", "<think>x</think>done"), ("empty", ""))
            }
    fn = getattr(base, "format_prompt", None)
    if callable(fn):
        params = inspect.signature(fn).parameters
        if "problem_statement" in params:
            out["format_prompt"] = _safe(
                lambda: fn(instance["problem_statement"], instance.get("instance_id"), instance.get("hints_text"))
            )
        else:
            out["format_prompt"] = _safe(lambda: fn(instance))
    for name in ("SYSTEM_PROMPT",):
        if isinstance(getattr(base, name, None), str):
            out[name] = getattr(base, name)
    tools = getattr(base, "TOOLS", None)
    if isinstance(tools, (list, tuple)):
        out["TOOLS"] = [_tool_schema(t) for t in tools]
    return scrub.to_data(out)


def _tool_schema(tool) -> Any:
    try:
        from langchain_core.utils.function_calling import convert_to_openai_tool

        return convert_to_openai_tool(tool)
    except Exception as exc:
        return {"name": getattr(tool, "name", repr(tool)), "error": f"{type(exc).__name__}: {exc}"}


def _safe(fn):
    try:
        return {"value": fn()}
    except Exception as exc:
        return {"raises": f"{type(exc).__name__}: {exc}"}


# ---------------------------------------------------------------- optimizer registry
BRIDGE = "optimizers.bridge"


def optimizer_setup(optimizer: str):
    """Package, ``lm`` and ``registry`` behind an optimizer's cells (the shared bridge for both)."""
    lm = importlib.import_module(f"{BRIDGE}.lm")
    registry = importlib.import_module(f"{BRIDGE}.registry")
    return BRIDGE, lm, registry


def adapter_label(adapter_class, optimizer: str) -> str:
    """Adapter class as recorded: bridge modules are labelled ``real_runner_<optimizer>.<submodule>``."""
    module = f"real_runner_{optimizer}" + adapter_class.__module__.removeprefix(BRIDGE)
    return f"{module}:{adapter_class.__name__}"


def adapter_kwargs(adapter_class, key: str) -> dict:
    """Constructor kwargs of the run protocol: team size from the key, else 4
    replicas/peers for independent/decentralized, 2 debate rounds."""
    import re

    topology = key.split("_", 1)[0]
    kwargs: dict[str, Any] = {}
    match = re.search(r"_r(\d+)$", key)
    if match:
        kwargs["n_agents"] = int(match.group(1))
    elif topology in ("independent", "decentralized"):
        kwargs["n_agents"] = 4
    if topology == "decentralized":
        kwargs["n_rounds"] = 2
    params = inspect.signature(adapter_class).parameters
    variadic = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    return {k: v for k, v in kwargs.items() if variadic or k in params}


def example_path(optimizer: str, dataset: str) -> Path:
    return SESSION / "examples" / f"{optimizer}_{dataset}.pkl"


def find_example(optimizer: str, dataset: str):
    package, _, _ = optimizer_setup(optimizer)
    module = importlib.import_module(f"{package}.datasets.{dataset}")
    target = cellmod.first_eval_id(dataset)
    rows = module.load_all()
    for row in rows:
        if str(getattr(row, "id", None)) == target:
            return row, rows
    raise RuntimeError(f"{optimizer}/{dataset}: first eval id {target!r} not in load_all()")


def load_example(optimizer: str, dataset: str):
    path = example_path(optimizer, dataset)
    if path.exists():
        with path.open("rb") as fh:
            return pickle.load(fh)
    example, _ = find_example(optimizer, dataset)
    return example


def example_instance(example) -> dict:
    data = example.toDict() if hasattr(example, "toDict") else dict(example)
    task = data.get("task_instance")
    task = task.toDict() if hasattr(task, "toDict") else dict(task or {})
    return {**task, **{k: v for k, v in data.items() if k != "task_instance"}}


SCORER_PRIVATE_FIELDS = {
    "bfcl": ("ground_truth",),
    "gpqa": ("correct_letter", "correct_answer", "incorrect_answers", "raw"),
}


def task_input(example, dataset: str) -> dict:
    """What the run protocol hands the MAS: the task instance minus scorer-private fields."""
    task = getattr(example, "task_instance", None)
    task = task.toDict() if hasattr(task, "toDict") else dict(task or {})
    for key in SCORER_PRIVATE_FIELDS.get(dataset, ()):
        task.pop(key, None)
    return task


def drive_dataset(cell: dict) -> dict:
    example, rows = find_example(cell["optimizer"], cell["dataset"])
    path = example_path(cell["optimizer"], cell["dataset"])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    with tmp.open("wb") as fh:
        pickle.dump(example, fh)
    tmp.replace(path)
    import hashlib

    ids = [str(getattr(row, "id", "")) for row in rows]
    data = example.toDict()
    return {
        "n_rows": len(rows),
        "ids_sha1": hashlib.sha1("\n".join(ids).encode()).hexdigest(),
        "first_ids": ids[:5],
        "inputs": sorted(example.inputs().keys()) if hasattr(example, "inputs") else None,
        "example": scrub.to_data(data, long_limit=2000),
        "task_input": scrub.to_data(task_input(example, cell["dataset"]), long_limit=2000),
    }


def drive_registry(cell: dict) -> dict:
    optimizer, dataset, key = cell["optimizer"], cell["dataset"], cell["key"]
    package, lm, registry = optimizer_setup(optimizer)
    if "openai_agents" in key:
        install_sdk_fake()
    example = load_example(optimizer, dataset)
    SCRIPT.update(script_for(dataset, example_instance(example)))
    adapter_class = registry.get_adapter_class(dataset, key)
    kwargs = adapter_kwargs(adapter_class, key)
    adapter = adapter_class(**kwargs)
    roles = list(adapter.roles())
    programs = importlib.import_module(f"{package}.programs")
    metric_module = importlib.import_module(f"{package}.datasets.{dataset}")
    if dataset == "swe":
        from tests.golden import swe_fake

        swe_fake.install()
        os.environ["SWE_WORK_ROOT"] = str(TMP / "swe_work")
    with lm.eval_mode(True):
        value = adapter.run_example(task_input(example, dataset))
        prediction = programs.prediction_from_adapter_output(adapter, roles[0], value)
        scored = metric_module.metric(example, prediction)
    return {
        "adapter": adapter_label(type(adapter), optimizer),
        "kwargs": kwargs,
        "roles": roles,
        "result": reduce_result(value),
        "prediction": reduce_result(prediction.toDict() if hasattr(prediction, "toDict") else prediction),
        "score": scrub.to_data(
            {"score": float(getattr(scored, "score", scored)), "feedback": getattr(scored, "feedback", None)}
        ),
    }


def drive_prompts(cell: dict) -> dict:
    optimizer, dataset = cell["optimizer"], cell["dataset"]
    _, _, registry = optimizer_setup(optimizer)
    out = {}
    for key in cell["keys"]:
        try:
            adapter_class = registry.get_adapter_class(dataset, key)
            kwargs = adapter_kwargs(adapter_class, key)
            adapter = adapter_class(**kwargs)
            roles = list(adapter.roles())
            entry = {
                "adapter": adapter_label(adapter_class, optimizer),
                "kwargs": kwargs,
                "roles": roles,
                "prompts": {role: adapter.get_prompt(role) for role in roles},
            }
            for attr in (
                "topology",
                "dataset",
                "framework",
                "prompt_topology",
                "module_name",
                "n_agents",
                "n_rounds",
                "team_size",
                "communication_format",
                "base_topology",
            ):
                if hasattr(adapter, attr):
                    value = getattr(adapter, attr)
                    if isinstance(value, (str, int, float, bool, type(None), list, tuple)):
                        entry[attr] = value
            out[key] = entry
        except Exception as exc:
            out[key] = {"error": f"{type(exc).__name__}: {exc}"}
    return {"keys": scrub.to_data(out)}


# ----------------------------------------------------------------------------- CLI
class _Captured(Exception):
    pass


def _parser_surface(parser) -> dict:
    actions = []
    for action in parser._actions:
        actions.append(
            {
                "option_strings": list(action.option_strings),
                "dest": action.dest,
                "action": type(action).__name__,
                "nargs": action.nargs,
                "const": action.const,
                "default": action.default if action.default is not argparse.SUPPRESS else "<SUPPRESS>",
                "type": getattr(action.type, "__name__", repr(action.type)) if action.type is not None else None,
                "choices": list(action.choices)
                if action.choices is not None and not isinstance(action.choices, dict)
                else (sorted(action.choices) if isinstance(action.choices, dict) else None),
                "required": action.required,
                "help": action.help,
                "metavar": action.metavar,
            }
        )
    return {"description": parser.description, "epilog": parser.epilog, "actions": actions}


def drive_cli(cell: dict) -> dict:
    import runpy

    captured: list = []
    original_parse_args = argparse.ArgumentParser.parse_args
    original_parse_known = argparse.ArgumentParser.parse_known_args

    def capture(self, *args, **kwargs):
        captured.append(self)
        raise _Captured()

    argparse.ArgumentParser.parse_args = capture
    argparse.ArgumentParser.parse_known_args = capture
    surfaces = {}
    try:
        for rel in cell["paths"]:
            captured.clear()
            path = REPO / rel
            saved_argv, saved_path0 = sys.argv, sys.path[0]
            sys.argv = [rel]
            sys.path[0] = str(path.parent)
            if str(REPO) not in sys.path:
                sys.path.insert(1, str(REPO))
            sink = io.StringIO()
            try:
                with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                    runpy.run_path(str(path), run_name="__main__")
                surfaces[rel] = {"error": "argparse was never invoked"}
            except _Captured:
                surfaces[rel] = _parser_surface(captured[-1])
            except SystemExit as exc:
                surfaces[rel] = {"error": f"SystemExit({exc.code}) before argparse"}
            except Exception as exc:
                surfaces[rel] = {"error": f"{type(exc).__name__}: {exc}"}
            finally:
                sys.argv = saved_argv
                sys.path[0] = saved_path0
    finally:
        argparse.ArgumentParser.parse_args = original_parse_args
        argparse.ArgumentParser.parse_known_args = original_parse_known
    return {"parsers": scrub.to_data(surfaces)}


# ----------------------------------------------------------------------------- scorers / static
def drive_scorer(cell: dict) -> dict:
    from tests.golden import canned

    return canned.score_dataset(cell["dataset"], optimizer_setup, load_example, example_instance, script_for)


def drive_comm_parser(cell: dict) -> dict:
    from tests.golden import canned

    return canned.communications()


def drive_decoding(cell: dict) -> dict:
    out: dict[str, Any] = {}
    for optimizer in cellmod.OPTIMIZERS:
        _, lm, _ = optimizer_setup(optimizer)
        entry = {"default_mode": lm.task_sampling()}
        with lm.eval_mode(True):
            entry["eval_mode"] = lm.task_sampling()
            entry["eval_env_temperature"] = os.environ.get("TASK_MODEL_TEMPERATURE")
        with lm.eval_mode(False):
            entry["optimize_mode"] = lm.task_sampling()
            entry["optimize_env_temperature"] = os.environ.get("TASK_MODEL_TEMPERATURE")
        entry["reflection"] = lm.reflection_sampling()
        entry["task_model"] = lm.TASK_MODEL
        entry["refl_model"] = lm.REFL_MODEL
        out[optimizer] = entry
        oc = importlib.import_module(f"{BRIDGE}.output_contracts")
        out[f"{optimizer}_output_contracts"] = _public_constants(oc)
    # Keys keep the original module names; topologies' contracts now live in core.
    sources = {
        "topologies.output_contracts": "core.output_contracts",
        "teamsizes.output_contracts": "teamsizes.output_contracts",
        "communications.output_contracts": "communications.output_contracts",
    }
    for name, module in sources.items():
        out[name] = _public_constants(importlib.import_module(module))
    # MIPRO demonstrations are rendered into the role prompts the runners execute.
    mipro_programs = importlib.import_module(f"{BRIDGE}.mipro_programs")
    demos = [
        {"task_instance": {"id": "demo-1", "question": "Which is larger, 2 or 3?"}, "answer": "Answer: 3"},
        {"question": "Capital of France?", "answer": "Answer: Paris", "augmented": True},
    ]
    out["mipro_render_instruction_with_demos"] = {
        "no_demos": _safe(lambda: mipro_programs.render_instruction_with_demos("You are a solver.", [])),
        "two_demos": _safe(lambda: mipro_programs.render_instruction_with_demos("You are a solver.", demos)),
    }
    try:
        config = importlib.import_module("optimizers.protocol.config")
        out["protocol_config"] = _public_constants(config)
    except Exception as exc:
        out["protocol_config"] = {"error": f"{type(exc).__name__}: {exc}"}
    return {"decoding": scrub.to_data(out)}


def _public_constants(module) -> dict:
    out = {}
    for name in sorted(vars(module)):
        if not name.isupper() or name.startswith("_"):
            continue
        value = getattr(module, name)
        if isinstance(value, (str, int, float, bool, list, tuple, dict, frozenset, set, type(None))):
            out[name] = value
    return out


def drive_method(cell: dict) -> dict:
    from tests.golden import methods

    return methods.drive_method(cell, TMP, SCRIPT)


DRIVERS = {
    "method": drive_method,
    "runner": drive_runner,
    "communication": drive_communication,
    "registry": drive_registry,
    "prompts": drive_prompts,
    "dataset": drive_dataset,
    "cli": drive_cli,
    "scorer": drive_scorer,
    "comm_parser": drive_comm_parser,
    "decoding": drive_decoding,
}
SERVER_KINDS = {"runner", "communication", "registry", "method"}


# ----------------------------------------------------------------------------- main
def _requests(cell: dict) -> tuple[list, str]:
    assert SERVER is not None
    SERVER.drain()
    records = sorted(SERVER.requests, key=lambda r: r["seq"])
    bodies = []
    for record in records:
        body = scrub.to_data(record["body"])
        if record["path"] != "/v1/chat/completions":
            body = {"__path__": scrub.scrub_text(record["path"]), **body}
        bodies.append(body)
    if cell.get("concurrent"):
        bodies.sort(key=scrub.canonical)
        if cell.get("dedupe_requests"):
            distinct = {scrub.canonical(body): body for body in bodies}
            return [distinct[key] for key in sorted(distinct)], "canonical-distinct"
        return bodies, "canonical"
    return bodies, "arrival"


def run_cell(cell: dict) -> dict:
    global SERVER
    scrub.configure({str(TMP / "home"): "<cell-home>", str(TMP): "<tmp>", str(SESSION): "<session>"})
    netguard.install()
    started = time.time()
    snapshot: dict[str, Any] = {"id": cell["id"], "kind": cell["kind"]}
    if cell["kind"] in SERVER_KINDS:
        if cell["kind"] == "method":
            from tests.golden import methods

            respond = methods.respond
        else:
            respond = responder.respond
        SERVER = FakeChatServer(lambda body: respond(body, SCRIPT), normalize=scrub.to_data).start()
        netguard.allow_port(SERVER.port)
        for name in (
            "VLLM_BASE_URL",
            "OPENAI_BASE_URL",
            "OPENAI_API_BASE",
            "REFLECTION_MODEL_BASE_URL",
            "TASK_ENDPOINTS",
        ):
            os.environ[name] = SERVER.base_url
    try:
        payload = DRIVERS[cell["kind"]](cell)
        snapshot["status"] = "ok"
        snapshot.update(payload)
    except Exception as exc:
        snapshot["status"] = "error"
        snapshot["error"] = scrub.scrub_text(f"{type(exc).__name__}: {exc}")
        snapshot["traceback_tail"] = [
            scrub.scrub_text(line) for line in traceback.format_exception(type(exc), exc, exc.__traceback__)[-3:]
        ]
    if SERVER is not None:
        snapshot["script"] = scrub.to_data(SCRIPT)
        snapshot["requests"], snapshot["request_order"] = _requests(cell)
        snapshot["n_requests"] = len(snapshot["requests"])
        if SERVER.errors:
            snapshot["server_errors"] = [scrub.scrub_text(e) for e in SERVER.errors]
        meta = {"max_inflight": SERVER.max_inflight}
        SERVER.stop()
    else:
        meta = {}
    meta["seconds"] = round(time.time() - started, 2)
    meta["blocked_network"] = sorted(set(scrub.scrub_text(b) for b in netguard.blocked_attempts()))
    return {"snapshot": snapshot, "meta": meta}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    cell = cellmod.cells_by_id()[args.cell]
    result = run_cell(cell)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, sort_keys=True))
    # Skip interpreter teardown: framework atexit hooks (telemetry flushers,
    # thread pools) can hang or print; all output is already on disk.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
