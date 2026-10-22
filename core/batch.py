"""The runners' shared batch loop.

A runner supplies one callable per concern and :func:`run_batch` does the rest:

* ``row(index, instance) -> record`` solves and scores one instance (usually via
  :func:`attempt`, which times the call and turns a failure into an ``error``);
* ``summarize(records) -> dict`` computes the dataset's batch scores;
* ``header(index, total, instance) -> str`` and ``progress(index, total, record,
  records) -> str`` render the verbose progress before and after each row, logged
  at INFO; ``banner(summary) -> str`` renders the end-of-batch report, printed on
  ``stream`` (default stdout).

Records are written to ``out_path`` as JSON lines as soon as they exist, and each
:class:`Output` gets one line per instance too (e.g. a predictions file next to
the records). Every file is emptied when the batch starts, so a run never mixes
with an earlier one. The summary returned is ``{**summarize(records), "total_s":
..., "per_instance": records}``, or ``summarize(records)`` alone with ``raw_summary``.
A batch in which every row failed on infrastructure (the model server unreachable
or refusing the request) raises :class:`InfrastructureFailure` once it is written.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

logger = logging.getLogger(__name__)

# Exception types, as :func:`attempt` records them, that mean the model server could
# not be reached or refused the request: the row never got to the task.
INFRASTRUCTURE_ERRORS = frozenset(
    {
        "APIConnectionError",
        "APITimeoutError",
        "AuthenticationError",
        "PermissionDeniedError",
        "NotFoundError",
        "InternalServerError",
        "ServiceUnavailableError",
        "ConnectionError",
        "ConnectionRefusedError",
    }
)


class InfrastructureFailure(RuntimeError):
    """Every row of a batch failed on infrastructure (see :data:`INFRASTRUCTURE_ERRORS`)."""


def is_infrastructure_error(error: object) -> bool:
    """True when a record's ``error`` (``"<ExceptionType>: <message>"``) names an infrastructure exception."""
    return isinstance(error, str) and error.partition(":")[0] in INFRASTRUCTURE_ERRORS


@dataclass(frozen=True)
class Output:
    """A JSON-lines file with one line per instance: ``line(instance, record)``."""

    path: Path
    line: Callable[[dict, dict], dict]


def attempt(
    solve: Callable[[], dict],
    *,
    fallback: dict | None = None,
    propagate: bool = False,
) -> tuple[dict, float, str | None]:
    """Call ``solve`` and time it: ``(output, latency_s, error)``.

    On an exception the output is a copy of ``fallback`` (default ``{"answer": None}``)
    and ``error`` is ``"<ExceptionType>: <message>"``; with ``propagate`` the
    exception is re-raised instead.
    """
    start = time.time()
    try:
        out, error = solve(), None
    except Exception as exc:
        if propagate:
            raise
        out, error = dict(fallback or {"answer": None}), f"{type(exc).__name__}: {exc}"
    return out, time.time() - start, error


@contextmanager
def jsonl_writer(
    path: Path | None,
    *,
    default: Callable[[Any], Any] | None = None,
    ensure_ascii: bool = True,
) -> Iterator[Callable[[dict], None]]:
    """Empty ``path`` and yield ``write(record)`` adding one flushed JSON line to it (a no-op without a path)."""
    if path is None:
        yield lambda record: None
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as out:

        def write(record: dict) -> None:
            out.write(json.dumps(record, ensure_ascii=ensure_ascii, default=default) + "\n")
            out.flush()

        yield write


def run_batch(
    instances: list[dict],
    row: Callable[[int, dict], dict],
    *,
    summarize: Callable[[list[dict]], dict],
    out_path: Path | None = None,
    outputs: Sequence[Output] = (),
    ensure_ascii: bool = True,
    json_default: Callable[[Any], Any] | None = None,
    verbose: bool = True,
    stream: TextIO | None = None,
    header: Callable[[int, int, dict], str] | None = None,
    progress: Callable[[int, int, dict, list[dict]], str] | None = None,
    banner: Callable[[dict], str] | None = None,
    raw_summary: bool = False,
) -> dict:
    """Run ``row`` on every instance, write the records and return the summary.

    Raises :class:`InfrastructureFailure`, after the records and the report are
    written, when every row's ``error`` is an infrastructure error.
    """
    records: list[dict] = []
    start = time.time()

    def log(render: Callable[..., str] | None, *args) -> None:
        if verbose and render is not None:
            logger.info(render(*args))

    with ExitStack() as files:
        options = {"default": json_default, "ensure_ascii": ensure_ascii}
        write = files.enter_context(jsonl_writer(out_path, **options))
        extra = [(output.line, files.enter_context(jsonl_writer(output.path, **options))) for output in outputs]
        for index, instance in enumerate(instances):
            log(header, index, len(instances), instance)
            record = row(index, instance)
            records.append(record)
            for line, write_line in extra:
                write_line(line(instance, record))
            write(record)
            log(progress, index, len(instances), record, records)
    summary = summarize(records)
    if not raw_summary:
        summary = {**summary, "total_s": round(time.time() - start, 1), "per_instance": records}
    if verbose and banner is not None:
        print(banner(summary), file=stream, flush=True)
    if records and all(is_infrastructure_error(record.get("error")) for record in records):
        raise InfrastructureFailure(
            f"every row ({len(records)}) failed on infrastructure; the first error: {records[0]['error']}"
        )
    return summary


def write_trace(path: Path, sections: Iterable[tuple[str, str]]) -> None:
    """Write a text trace of ``(title, text)`` sections, each as ``=== title ===`` followed by the text."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as out:
        for title, text in sections:
            out.write(f"=== {title} ===\n{text}\n\n")
