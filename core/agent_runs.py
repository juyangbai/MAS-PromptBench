"""Shared pieces of the API-Bank and ToolHop runners, whose agents call the endpoint directly.

A run is labelled by its *style*: the runner's ``<topology>_<framework>``, plus
``_communications_<format>`` for a communications run. Every agent's output is a
report dict (``role``, ``seed``, the text in ``raw`` / ``final_content`` /
``predicted_answer``, ``telemetry``); a topology combines the reports, and
:func:`run_rows` writes one record and one prediction per instance.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path

from core import batch
from core.communication import normalize_report, render_report
from core.paths import RESULTS_DIR

logger = logging.getLogger(__name__)

# Keys of a topology's output copied into the instance record.
OUTPUT_KEYS = (
    "n_agents", "n_rounds", "winner", "buckets", "per_agent", "per_peer", "by_stage", "stage_outputs", "workers", "manager",
)  # fmt: skip


def check_topology(topology: str, handled: str, *, suffixes: tuple[str, ...] = ("_openai",)) -> None:
    """Reject a call for another topology: each runner implements only its own."""
    key = topology
    for suffix in suffixes:
        key = key.replace(suffix, "")
    if key != handled:
        raise ValueError(f"this runner handles '{handled}'; received topology={topology!r}")


# Handoffs: what one agent reads of the others' outputs
def _report_text(report: dict) -> str:
    return str(
        report.get("raw")
        or report.get("final_content")
        or report.get("predicted_answer")
        or report.get("raw_tail")
        or ""
    )


def _fit_context_chunks(chunks: list[str], *, char_budget: int) -> str:
    selected: list[str] = []
    size = 0
    for chunk in reversed(chunks):
        extra = len(chunk) + (2 if selected else 0)
        if selected and size + extra > char_budget:
            continue
        selected.append(chunk)
        size += extra
        if size >= char_budget:
            break
    selected.reverse()
    return "\n\n".join(selected)


def reports_context(reports: list[dict], fmt: str | None = None, *, dataset: str, char_budget: int = 5000) -> str:
    """The agents' outputs as context for another agent, rendered in the communication format if any."""
    chunks = []
    for report in reports:
        label = report.get("role") or f"agent_{report.get('seed', '?')}"
        text = _report_text(report)
        if not text:
            continue
        if fmt:
            try:
                normalized = normalize_report(
                    str(label),
                    text[-700:],
                    dataset=dataset,
                    topology="handoff",
                    payload={"seed": report.get("seed")},
                )
                rendered = render_report(normalized, fmt)
                chunks.append(f"{label}:\n{rendered}")
            except Exception:
                chunks.append(f"{label}:\n{text[-1200:]}")
        else:
            chunks.append(f"{label}:\n{text[-1200:]}")
    if fmt:
        return _fit_context_chunks(chunks, char_budget=char_budget)
    context = "\n\n".join(chunks)
    return context[-char_budget:]


# Records
def write_json(path: Path, payload: dict) -> None:
    """Write ``payload`` as indented JSON (non-ASCII kept, other objects as ``str``)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


def run_rows(
    instances: list[dict],
    run_one: Callable[[dict, Path], dict],
    *,
    dataset: str,
    source: str,
    style: str,
    model_id: str,
    prediction: Callable[[dict, dict], dict],
    out_dir: Path | None = None,
    predictions: Path | None = None,
    team_size: int | None = None,
    hidden: tuple[str, ...] = (),
    verbose: bool = True,
) -> dict:
    """Run ``run_one(instance, out_dir)`` on every instance; returns ``{n, correct, accuracy, style[, team_size]}``.

    Each record is written to ``<out_dir>/results.jsonl`` and its
    ``prediction(instance, record)`` plus the model name to ``predictions``
    (default ``<out_dir>/predictions.jsonl``); both files start empty on every
    run. ``out_dir`` defaults to ``results/<dataset>/<style>``. The verbose
    progress leaves the record keys in ``hidden`` out.
    """
    out_dir = Path(out_dir) if out_dir else RESULTS_DIR / dataset / style
    out_dir.mkdir(parents=True, exist_ok=True)
    if verbose:
        team = f", N={team_size}" if team_size else ""
        logger.info("loaded %d instance(s) from %s (%s%s)", len(instances), source, style, team)

    def predicted(instance: dict, record: dict) -> dict:
        return {**prediction(instance, record), "model_name_or_path": model_id}

    def summarize(records: list[dict]) -> dict:
        correct = sum(int(bool(record.get("correct"))) for record in records)
        summary = {"n": len(records), "correct": correct, "accuracy": (correct / len(records)) if records else 0.0}
        return {**summary, "style": style, **({"team_size": team_size} if team_size else {})}

    def progress(index: int, total: int, record: dict, records: list[dict]) -> str:
        shown = {key: value for key, value in record.items() if key not in hidden}
        return f"  -> {json.dumps(shown, ensure_ascii=False, default=str)}"

    return batch.run_batch(
        instances,
        lambda index, instance: run_one(instance, out_dir),
        summarize=summarize,
        out_path=out_dir / "results.jsonl",
        outputs=[batch.Output(Path(predictions) if predictions else out_dir / "predictions.jsonl", predicted)],
        ensure_ascii=False,
        json_default=str,
        verbose=verbose,
        header=lambda index, total, instance: f"\n[{index + 1}/{total}] {instance['id']}",
        progress=progress,
        raw_summary=True,
    )
