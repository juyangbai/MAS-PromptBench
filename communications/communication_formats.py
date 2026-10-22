"""Communications pairs: a topology runner run with one communication format.

``communications/<topology>/<dataset>/<dataset>_<format>.py`` calls :func:`install`
with its topology, dataset and format. The pair runs the topologies/ LangGraph
runner of the topology and dataset (:func:`runner_module`) as a module of its own
with ``COMMUNICATION_FORMAT`` preset (:func:`core.variant.module`), so the runner
applies the format itself: its ``COMMUNICATION`` policy
(:class:`core.communication.CommPolicy`) appends the format's contract to the
agents' prompts and renders their handoffs. The pair adds what the study
measures: ``solve`` records the runner's handoffs and scores its reports,
``run_one`` / ``run_batch`` write one record per instance (:mod:`core.batch`)
and ``main`` is the command line (:mod:`core.cli`). Run as a script, an entry
runs its command line. The format helpers live in :mod:`core.communication`
and are re-exported here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core import batch, cli, variant
from core.communication import (  # noqa: F401  (the format helpers, re-exported)
    FORMATS,
    STRICT_COMMUNICATION_FIELDS,
    append_contract,
    begin_handoff_recording,
    collect_reports,
    communication_contract,
    compact_output_fields,
    end_handoff_recording,
    format_handoff,
    normalize_report,
    parse_message,
    render_report,
)
from core.paths import RESULTS_DIR
from core.telemetry import langchain_ensemble_telemetry, normalize

RESULTS = RESULTS_DIR / "communications_baseline"
# Runner names an entry exposes as its own, where its runner has them.
RUNNER_NAMES = (
    "extract_answer",
    "exact_match_score",
    "f1_score",
    "extract_code",
    "run_tests",
    "format_prompt",
    "N_AGENTS",
    "N_ROUNDS",
    "dataset_summary",
    "score_one",
    "extract_canonical",
    "AST_CATEGORIES",
)


def runner_module(topology: str, dataset: str) -> str:
    """The topologies/ LangGraph runner of a pair."""
    framework = "" if topology == "independent" else "langgraph."
    return f"topologies.{topology}.{framework}{dataset}.langgraph_{dataset}"


def install(namespace: dict, *, topology: str, dataset: str, fmt: str) -> None:
    """Define a pair's names in ``namespace`` (its entry's ``globals()``); in a script, run its command line."""
    pair = _PAIRS[dataset](topology, fmt)
    namespace.update(pair.names())
    if namespace.get("__name__") == "__main__":
        raise SystemExit(pair.main())


class Pair:
    """The runner of ``(topology, dataset)`` with ``COMMUNICATION_FORMAT`` preset to ``fmt``, and its records."""

    dataset: str
    add_arguments = None  # the dataset's command-line options

    def __init__(self, topology: str, fmt: str):
        self.topology, self.fmt = topology, fmt
        self.runner = variant.module(runner_module(topology, self.dataset), COMMUNICATION_FORMAT=fmt)
        self.task = self.runner.task

    def names(self) -> dict[str, Any]:
        """The entry's public names: the pair's functions and constants, and the runner's helpers."""
        runner = self.runner
        names = {
            "COMMUNICATION_FORMAT": self.fmt,
            "TOPOLOGY": self.topology,
            "DATASET": self.dataset,
            "BASE_MODULE": runner.__name__,
            "VLLM_BASE_URL": runner.VLLM_BASE_URL,
            "MODEL_ID": runner.MODEL_ID,
            "load_instances": self.loader(),
            "solve": self.solve,
            "run_one": self.run_one,
            "run_batch": self.run_batch,
            "main": self.main,
        }
        names.update((name, getattr(runner, name)) for name in RUNNER_NAMES if hasattr(runner, name))
        return names

    def loader(self):
        """The pair's ``load_instances``: the runner's."""
        return self.runner.load_instances

    def solve(self, *args: Any, **kwargs: Any) -> dict:
        """The runner's solve with its handoffs recorded and its reports scored (:meth:`CommPolicy.solve`)."""
        return self.runner.COMMUNICATION.solve(self.runner.solve, *args, **kwargs)

    def run_one(self, instance: dict) -> dict:
        """Solve and score one instance; a failed solve raises."""
        out, latency_s, _ = batch.attempt(lambda: self.solve_instance(instance), propagate=True)
        return self.record(instance, out, round(latency_s, 2))

    def solve_instance(self, instance: dict) -> dict:
        """:meth:`solve` on the inputs of one instance."""
        raise NotImplementedError

    def record(self, instance: dict, out: dict, latency_s: float) -> dict:
        """The record of one solved instance."""
        raise NotImplementedError

    def error_fields(self, instance: dict) -> dict:
        """The dataset's fields of an error record."""
        return {}

    def error_record(self, instance: dict, exc: Exception) -> dict:
        """The record of an instance whose run raised ``exc``."""
        error = f"{type(exc).__name__}: {exc}"
        return {
            "id": instance.get("id"),
            "latency_s": 0,
            "em": 0.0,
            "communication_format": None,
            "communication_parse_ok": False,
            "communication_all_parse_ok": False,
            "communication_parse_rate": 0.0,
            "communication_required_report_count": 0,
            "communication_missing_roles": [],
            "communication_infra_error": error,
            "communication_parse_errors": [],
            "communication_reports": [],
            "error": error,
            **self.error_fields(instance),
        }

    def run_batch(self, instances: list[dict], out_path: Path | None = None, verbose: bool = True) -> dict:
        """Run every instance (a failure becomes its error record); ``{n, em, em_sum, total_s, per_instance}``."""

        def row(_, instance: dict) -> dict:
            try:
                return self.run_one(instance)
            except Exception as exc:
                return self.error_record(instance, exc)

        return batch.run_batch(
            instances,
            row,
            summarize=_summarize,
            out_path=out_path,
            verbose=verbose,
            progress=_progress_line,
            json_default=str,
        )

    def main(self, argv: list[str] | None = None) -> int:
        """The pair's command line (:func:`core.cli.main`)."""
        return cli.main(
            argv,
            description=f"communications pair {self.topology}/{self.dataset} [{self.fmt}]",
            load_instances=self.loader(),
            run_batch=self._run_cli_batch,
            add_arguments=self.add_arguments,
        )

    def _run_cli_batch(self, instances: list[dict], out_path: Path | None = None, category: str | None = None) -> dict:
        """:meth:`run_batch` into ``out_path`` (default ``results/communications_baseline/<pair>[/<category>]``)."""
        if out_path is None:
            folder = RESULTS / f"{self.topology}_{self.dataset}_{self.fmt}"
            out_path = (folder / category if category else folder) / "results.jsonl"
        summary = self.run_batch(instances, out_path=out_path)
        print(json.dumps({key: summary.get(key) for key in ("n", "em", "total_s")}, default=str))
        return summary


class HotpotQAPair(Pair):
    dataset = "hotpotqa"

    def solve_instance(self, instance: dict) -> dict:
        return self.solve(instance["question"])

    def record(self, instance: dict, out: dict, latency_s: float) -> dict:
        return self.task.record(
            instance,
            out.get("answer"),
            **self.task.meta(instance),
            latency_s=latency_s,
            **_report_fields(out),
            error=None,
        )

    def error_fields(self, instance: dict) -> dict:
        return {"question": instance.get("question"), "gold_answer": instance.get("answer"), "predicted_answer": None}


class LCBPair(Pair):
    dataset = "lcb"

    def solve_instance(self, instance: dict) -> dict:
        return self.solve(instance["problem"], starter_code=instance.get("starter_code") or None)

    def record(self, instance: dict, out: dict, latency_s: float) -> dict:
        code = out.get("code")
        scores = self.task.test_scores(code, instance["tests"], timeout_s=self.task.BATCH_TEST_TIMEOUT_S)
        return self.task.record(
            instance,
            code,
            {"winner": out.get("winner"), **scores},
            latency_s=latency_s,
            **_report_fields(out),
            error=None,
        )

    def error_fields(self, instance: dict) -> dict:
        return {"problem": str(instance.get("problem") or "")[:400], "predicted_code": None, "pass_rate": 0.0}


class ToolUsePair(Pair):
    """API-Bank and ToolHop: the runner's ``solve`` is one agent's call, ``solve_topology`` the topology's."""

    def solve(self, instance: dict, **kwargs: Any) -> dict:
        """The runner's ``solve_topology`` (labelled with the format) with its reports scored."""
        runner = self.runner
        return runner.COMMUNICATION.solve(
            runner.solve_topology,
            instance,
            style=f"{runner.STYLE}_communications_{self.fmt}",
            topology=runner.TOPOLOGY,
            role=runner.ROLE,
            prompt_suffix="",
            **kwargs,
        )

    def solve_instance(self, instance: dict) -> dict:
        return self.solve(instance)

    @staticmethod
    def team_fields(out: dict) -> dict:
        """The record fields of the team's vote."""
        return {key: out.get(key) for key in ("n_agents", "n_rounds", "winner", "buckets")}


class ToolHopPair(ToolUsePair):
    dataset = "toolhop"

    def record(self, instance: dict, out: dict, latency_s: float) -> dict:
        messages = out.get("messages") or []
        final = self.task.last_assistant_content(messages)
        predicted = out.get("predicted_answer")
        if predicted is None:
            predicted = self.task.extract_answer(final)
        if "correct" in out:
            correct = bool(out.get("correct"))
        else:
            correct = self.task.score_answer(
                str(instance.get("answer", "")), final, self.task.previous_tool_content(messages)
            )
        tool_calls = out.get("tool_calls")
        turns = out.get("turns")
        return {
            "id": instance["id"],
            "idx": instance["id"],
            "question": instance.get("question"),
            "gold_answer": instance.get("answer"),
            "predicted_answer": predicted,
            "answer_correct": int(correct),
            "correct": bool(correct),
            "em": float(bool(correct)),
            "latency_s": latency_s,
            **self.team_fields(out),
            "tool_calls": int(tool_calls if tool_calls is not None else _count_tool_calls(messages)),
            "turns": int(turns if turns is not None else _count_turns(messages)),
            **_report_fields(out),
            "error": None,
        }

    def error_fields(self, instance: dict) -> dict:
        return {"question": instance.get("question"), "gold_answer": instance.get("answer"), "predicted_answer": None}


class APIBankPair(ToolUsePair):
    dataset = "apibank"

    def record(self, instance: dict, out: dict, latency_s: float) -> dict:
        predicted = out.get("predicted_answer") or self.task.extract_api_call(out.get("raw") or "")
        scored = _runner_score(out) if "correct" in out else self.task.score_prediction(instance, predicted)
        return {
            "id": instance["id"],
            "idx": instance["id"],
            "file": instance.get("file"),
            "sample_id": instance.get("sample_id"),
            "question": self.task.format_chat_history(instance.get("chat_history") or []),
            "gold_api_call": instance.get("gold_api_call"),
            "gold_api_name": instance.get("ground_truth", {}).get("api_name"),
            "predicted_answer": predicted,
            "predicted_api_name": scored.get("predicted_api_name"),
            "predicted_params": scored.get("predicted_params"),
            "answer_correct": int(bool(scored.get("correct"))),
            "correct": bool(scored.get("correct")),
            "em": float(bool(scored.get("correct"))),
            "stage": scored.get("stage"),
            "latency_s": latency_s,
            **self.team_fields(out),
            "tool_calls": int(out.get("tool_calls") or 0),
            "turns": int(out.get("turns") or 1),
            **_report_fields(out),
            "error": scored.get("error"),
        }

    def error_fields(self, instance: dict) -> dict:
        return {
            "idx": instance.get("idx") or instance.get("id"),
            "question": str(instance.get("chat_history") or "")[:400],
            "gold_api_call": instance.get("gold_api_call"),
            "gold_api_name": (instance.get("ground_truth") or {}).get("api_name"),
            "predicted_answer": "",
            "answer_correct": 0,
            "correct": False,
            "stage": "solve",
        }


class BFCLPair(Pair):
    """BFCL: the instances carry their ground truth and category; failed solves and checks are recorded."""

    dataset = "bfcl"

    def add_arguments(self, parser) -> None:
        """``--category``: the BFCL subset."""
        self.task.add_arguments(parser)

    def loader(self):
        return self.load_instances

    def load_instances(
        self, category: str = "simple", limit: int | None = None, offset: int = 0, only: list[str] | None = None
    ) -> list[dict]:
        """The rows of one subset, each with its ``ground_truth`` (the possible answers) and ``category``."""
        pairs = self.task.load_pairs(category, limit, offset, only)
        return [
            {**row, "ground_truth": answers.get("ground_truth") or [], "category": category} for row, answers in pairs
        ]

    def run_one(self, instance: dict, out_dir: Path | None = None) -> dict:
        """Solve and check one instance; writes its trace under ``out_dir`` (if any)."""
        category = instance.get("category") or self.task.DEFAULT_CATEGORY
        summary = {"id": instance["id"], "category": category, "communication_format": self.fmt}
        call = {key: value for key, value in instance.items() if key not in ("ground_truth", "category")}
        out, latency_s, error = batch.attempt(lambda: self.solve(call))
        if error is not None:
            return {**summary, "em": 0.0, "valid": False, "error": error, "stage": "solve"}
        summary["latency_s"] = round(latency_s, 2)
        summary["model_output"] = out.get("model_output") or []
        summary["winner"] = out.get("winner")
        summary["tool_calls"] = len(summary["model_output"])
        summary.update(_telemetry(out))
        if out_dir is not None:
            batch.write_trace(Path(out_dir) / "traces" / f"{instance['id']}.txt", _trace_sections(out))
        summary.update(compact_output_fields(out))
        return self.verdict(summary, instance)

    def verdict(self, summary: dict, instance: dict) -> dict:
        """``summary`` with the AST checker's verdict on its ``model_output``."""
        try:
            report = self.runner.score_one(
                instance["function"],
                summary["model_output"],
                list(instance.get("ground_truth") or []),
                summary["category"],
            )
        except Exception as exc:
            return {**summary, "em": 0.0, "valid": False, "error": f"{type(exc).__name__}: {exc}", "stage": "score"}
        summary["valid"] = bool(report.get("valid"))
        summary["em"] = float(summary["valid"])
        summary["error_type"] = report.get("error_type")
        if not summary["valid"]:
            summary["score_error"] = (report.get("error") or [])[:3]
        summary["error"] = None
        return summary

    def error_fields(self, instance: dict) -> dict:
        return {"category": instance.get("category"), "model_output": [], "valid": False}


class SWEPair(Pair):
    """SWE-bench: the runner's ``run_one`` clones, solves and writes the artifacts."""

    dataset = "swe"

    def run_one(
        self,
        instance: dict,
        workdir_root: Path | None = None,
        out_dir: Path | None = None,
        eval_mode: str = "singularity",
    ) -> dict:
        """The runner's record of one instance (clones under ``workdir_root``, artifacts in ``out_dir``)."""
        record = self.runner.run_one(
            instance,
            Path(workdir_root or Path.home() / f"swe_work_communications_{self.topology}_{self.fmt}"),
            Path(out_dir or RESULTS / f"{self.topology}_swe_{self.fmt}"),
            eval_mode=eval_mode,
        )
        return {**record, "communication_format": self.fmt, "base_module": runner_module(self.topology, self.dataset)}


_PAIRS = {pair.dataset: pair for pair in (HotpotQAPair, LCBPair, ToolHopPair, APIBankPair, BFCLPair, SWEPair)}


def _telemetry(out: dict) -> dict:
    """Token and call counts of a solve: its own, else summed over its replicas' messages."""
    if out.get("telemetry"):
        return dict(out["telemetry"])
    if out.get("per_agent"):
        return normalize(langchain_ensemble_telemetry(out["per_agent"]))
    return {}


def _report_fields(out: dict) -> dict:
    return {**compact_output_fields(out), **_telemetry(out)}


def _runner_score(out: dict) -> dict:
    """An API-Bank topology's own verdict on its submitted call."""
    return {key: out.get(key) for key in ("correct", "predicted_api_name", "predicted_params", "stage", "error")}


def _count_tool_calls(messages: list[dict]) -> int:
    return sum(len(message.get("tool_calls") or []) for message in messages)


def _count_turns(messages: list[dict]) -> int:
    return sum(1 for message in messages if message.get("role") == "assistant")


def _trace_sections(out: dict) -> list[tuple[str, str]]:
    """A solve's trace: its stages' outputs, else its messages."""
    if out.get("by_stage"):
        return [(stage.upper(), text) for stage, text in out["by_stage"].items()]
    messages = [message for message in out.get("messages") or [] if isinstance(message, dict)]
    return [(message.get("source", "?"), message.get("content", "")) for message in messages]


def _summarize(records: list[dict]) -> dict:
    total = sum((float(record.get("em") or 0.0) for record in records), 0.0)
    n = len(records)
    return {"n": n, "em": total / n if n else 0.0, "em_sum": total}


def _progress_line(index: int, total: int, record: dict, records: list[dict]) -> str:
    return f"[{index + 1:>3}/{total}] {record.get('id')} em={record.get('em', 0):.0f} lat={record.get('latency_s', 0)}s"
