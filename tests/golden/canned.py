"""Scorer / parser snapshots on canned predictions (no model calls).

For each dataset: the bridge dataset ``metric`` (recorded under each optimizer) and
the extraction/scoring helpers of every topology runner are applied to the
same correct / wrong / malformed (and a few edge-case) predictions built from
the dataset's first evaluation instance.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Callable
from typing import Any

from tests.golden import cells as cellmod
from tests.golden import responder, scrub


def _safe(fn: Callable[[], Any]) -> Any:
    try:
        return {"value": scrub.to_data(fn(), long_limit=2000)}
    except Exception as exc:
        return {"raises": scrub.scrub_text(f"{type(exc).__name__}: {exc}")[:500]}


def _other_letter(letter: str) -> str:
    letters = "ABCD"
    return letters[(letters.index(letter) + 1) % 4] if letter in letters else "A"


def _wrong_bfcl(calls: list[dict]) -> list[dict]:
    wrong = []
    for call in calls:
        args = dict(call["arguments"])
        if args:
            first = sorted(args)[0]
            args[first] = "WRONG_VALUE"
        else:
            args["unexpected_argument"] = 1
        wrong.append({"name": call["name"], "arguments": args})
    return wrong


def _canonical(calls: list[dict]) -> list[dict]:
    return [{c["name"]: c["arguments"]} for c in calls]


def _wrong_apibank(call: str) -> str:
    if "=" in call:
        head, _, tail = call.partition("=")
        quote = "'" if tail.startswith("'") else ""
        rest = tail.split(",", 1)
        suffix = ("," + rest[1]) if len(rest) > 1 else ")]"
        return f"{head}={quote}WRONG_VALUE{quote}{suffix}"
    return call.replace("(", "(unexpected='x'", 1)


def texts_for(dataset: str, inst: dict, script: dict) -> dict[str, str]:
    """Canned model outputs (label -> text)."""
    if dataset == "gpqa":
        gold = str(inst.get("correct_letter") or inst.get("answer"))
        return {
            "correct": f"The analysis favours this option.\nAnswer: {gold}",
            "wrong": f"Answer: {_other_letter(gold)}",
            "malformed": "I am not sure which option is right.",
            "markdown_lowercase": f"**Answer:** ({gold.lower()})",
            "option_phrase": f"The correct option is {gold}.",
            "think_then_answer": f"<think>Answer: {_other_letter(gold)}</think>\nAnswer: {gold}",
        }
    if dataset == "hotpotqa":
        gold = str(inst.get("answer"))
        return {
            "correct": f"Answer: {gold}",
            "wrong": "Answer: Paris",
            "malformed": "",
            "partial": f"Answer: the {gold} indeed",
            "terminate": f"Answer: {gold}\nTERMINATE",
            "no_marker": gold,
        }
    if dataset == "math":
        gold = str(inst.get("answer"))
        return {
            "correct": f"\\boxed{{{gold}}}\nreasoning\n\\boxed{{{gold}}}",
            "wrong": "\\boxed{-12345}",
            "malformed": f"The answer is {gold}",
            "text_wrapped": f"\\boxed{{\\text{{{gold}}}}}",
            "spaced": f"\\boxed{{ {gold} }}",
        }
    if dataset in ("lcb", "apps"):
        fence = "```python\n{}\n```"
        return {
            "correct": script["final"],
            "wrong": fence.format("print(-1)"),
            "malformed": "Here is my idea: loop over the input and print the result.",
            "syntax_error": fence.format("def broken(:\n    pass"),
            "two_blocks": fence.format("print(-1)") + "\n" + script["final"],
        }
    if dataset == "bfcl":
        calls = script.get("domain_calls") or []
        return {
            "correct": responder.bfcl_final(calls),
            "wrong": responder.bfcl_final(_wrong_bfcl(calls)),
            "malformed": "I would call the function with the right arguments.",
            "name_arguments_shape": "```json\n"
            + json.dumps([{"name": c["name"], "arguments": c["arguments"]} for c in calls])
            + "\n```",
            "terminate": responder.bfcl_final(calls) + "\nTERMINATE",
        }
    if dataset == "apibank":
        gold = script["final"]
        return {
            "correct": gold,
            "wrong": _wrong_apibank(gold),
            "malformed": "Sure, I can help with that.",
            "with_prose": f"The next call is:\n{gold}",
        }
    if dataset == "toolhop":
        gold = str(inst.get("answer"))
        return {
            "correct": f"<answer>{gold}</answer>",
            "wrong": "<answer>definitely not the answer</answer>",
            "malformed": "",
            "untagged": gold,
        }
    if dataset == "swe":
        return {
            "correct": script["final"],
            "trivial": "```diff\n--- a/buggy.py\n+++ b/buggy.py\n```",
            "malformed": "I fixed the bug in buggy.py.",
        }
    raise KeyError(dataset)


def _prediction(dataset: str, label: str, text: str, script: dict):
    import dspy

    fields: dict[str, Any] = {"answer": text, "agent_trace": "", "winner": None, "vote_summary": ""}
    if dataset == "bfcl":
        calls = script.get("domain_calls") or []
        fields["tool_calls"] = (
            _canonical(calls)
            if label in ("correct", "terminate", "name_arguments_shape")
            else _canonical(_wrong_bfcl(calls))
            if label == "wrong"
            else []
        )
    else:
        fields["tool_calls"] = []
    if dataset == "swe":
        fields["patch"] = ""
    return dspy.Prediction(**fields)


def runner_helpers(dataset: str, module, inst: dict, texts: dict[str, str], script: dict) -> dict:
    out: dict[str, Any] = {}

    def call(name: str, *args, **kwargs):
        fn = getattr(module, name, None)
        if callable(fn):
            return _safe(lambda: fn(*args, **kwargs))
        return None

    def per_text(name: str):
        if callable(getattr(module, name, None)):
            out[name] = {label: call(name, text) for label, text in texts.items()}

    if dataset == "gpqa":
        per_text("extract_answer")
        per_text("strip_thinking")
    elif dataset == "hotpotqa":
        gold = str(inst.get("answer"))
        per_text("extract_answer")
        if callable(getattr(module, "extract_answer", None)):
            preds = {label: module.extract_answer(text) if text else None for label, text in texts.items()}
            out["exact_match_score"] = {label: call("exact_match_score", pred, gold) for label, pred in preds.items()}
            out["f1_score"] = {label: call("f1_score", pred, gold) for label, pred in preds.items()}
        out["normalize_answer"] = call("normalize_answer", gold)
    elif dataset == "math":
        gold = str(inst.get("answer"))
        per_text("extract_boxed")
        per_text("extract_answer")
        if callable(getattr(module, "extract_answer", None)):
            preds = {label: module.extract_answer(text) for label, text in texts.items()}
            out["exact_match_score"] = {label: call("exact_match_score", pred, gold) for label, pred in preds.items()}
            if callable(getattr(module, "is_equiv", None)):
                out["is_equiv"] = {label: call("is_equiv", pred, gold) for label, pred in preds.items()}
    elif dataset in ("lcb", "apps"):
        per_text("extract_code")
        tests = inst.get("tests") if dataset == "lcb" else inst.get("input_output")
        if dataset == "lcb":
            tests = list(tests or [])[:3]
        else:
            tests = {
                **(tests or {}),
                "inputs": (tests or {}).get("inputs", [])[:3],
                "outputs": (tests or {}).get("outputs", [])[:3],
            }
        if callable(getattr(module, "extract_code", None)) and callable(getattr(module, "run_tests", None)):
            results = {}
            for label, text in texts.items():
                code = module.extract_code(text)
                results[label] = _safe(lambda c=code: module.run_tests(c, tests, timeout_s=6)) if code else None
            out["run_tests"] = results
            for label, res in results.items():
                if res and "value" in res and isinstance(res["value"], dict) and "pass_rate" in res["value"]:
                    out.setdefault("exact_match_score", {})[label] = call(
                        "exact_match_score", res["value"]["pass_rate"]
                    )
    elif dataset == "bfcl":
        calls = script.get("domain_calls") or []
        outputs = {"correct": _canonical(calls), "wrong": _canonical(_wrong_bfcl(calls)), "empty": []}
        out["score_one"] = {
            label: call(
                "score_one", inst.get("function"), output, inst.get("ground_truth"), cellmod_bfcl_category(inst)
            )
            for label, output in outputs.items()
        }
        if callable(getattr(module, "to_canonical", None)):
            lc_calls = [{"name": c["name"], "args": c["arguments"], "id": f"call_{i}"} for i, c in enumerate(calls)]
            out["to_canonical"] = call("to_canonical", lc_calls)
    elif dataset == "apibank":
        per_text("extract_api_call")
        per_text("parse_api_call")
        if callable(getattr(module, "extract_api_call", None)):
            out["score_prediction"] = {
                label: call("score_prediction", inst, module.extract_api_call(text)) for label, text in texts.items()
            }
    elif dataset == "toolhop":
        gold = str(inst.get("answer"))
        per_text("extract_answer")
        out["score_answer"] = {label: call("score_answer", gold, text, "") for label, text in texts.items()}
        out["score_answer_prev_tool"] = call("score_answer", gold, "no answer", f"tool said {gold}")
    elif dataset == "swe":
        from tests.golden import worker

        out.update(worker.swe_helpers(module, inst))
    return out


def cellmod_bfcl_category(inst: dict) -> str:
    return str(inst.get("id", "simple_0")).rsplit("_", 1)[0]


def score_dataset(dataset: str, optimizer_setup, load_example, example_instance, script_for) -> dict:
    out: dict[str, Any] = {"registry_metric": {}, "runners": {}}
    for optimizer in cellmod.OPTIMIZERS:
        package, _, _ = optimizer_setup(optimizer)
        example = load_example(optimizer, dataset)
        inst = example_instance(example)
        script = script_for(dataset, inst)
        texts = texts_for(dataset, inst, script)
        metric_module = importlib.import_module(f"{package}.datasets.{dataset}")
        results = {}
        for label, text in texts.items():
            prediction = _prediction(dataset, label, text, script)
            res = _safe(lambda p=prediction, module=metric_module, item=example: module.metric(item, p))
            if "value" in res and isinstance(res["value"], dict):
                res = {"score": res["value"].get("score"), "feedback": res["value"].get("feedback")}
            results[label] = res
        out["registry_metric"][optimizer] = results
    # Topology runner helpers (one instance loaded through the single runner).
    from tests.golden import worker

    runner_cells = [c for c in cellmod.topology_cells() if c["dataset"] == dataset]
    instance_id = cellmod.first_eval_id(dataset)
    single = next(c for c in runner_cells if c["topology"] == "single")
    inst = worker.load_runner_instance(importlib.import_module(single["module"]), dataset, instance_id)
    script = script_for(dataset, inst)
    texts = texts_for(dataset, inst, script)
    out["texts"] = scrub.to_data(texts)
    for cell in runner_cells:
        try:
            module = importlib.import_module(cell["module"])
        except Exception as exc:
            out["runners"][cell["id"]] = {"import_error": scrub.scrub_text(f"{type(exc).__name__}: {exc}")}
            continue
        out["runners"][cell["id"]] = runner_helpers(dataset, module, inst, texts, script)
    if dataset == "bfcl":
        bfcl_calls = importlib.import_module("core.bfcl_calls")
        out["bfcl_calls.extract_canonical"] = {
            label: _safe(lambda t=text: bfcl_calls.extract_canonical(t)) for label, text in texts.items()
        }
    return scrub.to_data(out)


def communications() -> dict:
    cf = importlib.import_module("communications.communication_formats")
    datasets = ("hotpotqa", "lcb", "bfcl", "toolhop", "apibank", "swe")
    out: dict[str, Any] = {"contracts": {}, "parse": {}, "render": {}}
    for fmt in sorted(cf.FORMATS):
        out["contracts"][fmt] = {ds: cf.communication_contract(fmt, ds) for ds in datasets}
    samples = {
        "semi_ok": "[STATUS]\ncompleted\n[SUMMARY]\nDone.\n[EVIDENCE_OR_TESTS]\nChecked.\n[CONFIDENCE]\nhigh - sure\n[NEXT]\nUse it.\n\nAnswer: X",
        "semi_missing": "[STATUS]\nfinished\n[SUMMARY]\nDone.",
        "json_ok": 'JSON_REPORT:\n{"status": "completed", "summary": "s", "confidence": "low", "next": "n", "payload": {}}\nEND_JSON_REPORT\nAnswer: X',
        "json_fenced": '```json\n{"status": "blocked", "summary": "s", "confidence": "medium", "next": "n", "payload": []}\n```',
        "json_bad": "JSON_REPORT:\n{not json}\nEND_JSON_REPORT",
        "plain": "Answer: X",
    }
    for fmt in sorted(cf.FORMATS):
        out["parse"][fmt] = {
            label: _safe(lambda t=text, f=fmt: cf.parse_message(t, f)) for label, text in samples.items()
        }
    report = cf.normalize_report(
        "analyst",
        "Line one of the analysis.\nSecond line with evidence.\nAnswer: X",
        dataset="hotpotqa",
        topology="sequential",
        payload={"k": "v"},
    )
    out["normalize_report"] = report
    for fmt in sorted(cf.FORMATS):
        out["render"][fmt] = _safe(lambda f=fmt: cf.render_report(report, f))
    if hasattr(cf, "format_handoff"):
        out["format_handoff"] = {}
        for fmt in sorted(cf.FORMATS):
            out["format_handoff"][fmt] = _safe(
                lambda f=fmt: cf.format_handoff(
                    "analyst", "Line one.\nAnswer: X", fmt=f, dataset="hotpotqa", topology="sequential"
                )
            )
    return scrub.to_data(out)
