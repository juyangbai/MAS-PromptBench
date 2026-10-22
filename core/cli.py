"""Command line shared by the runners.

Every runner accepts the same flags::

    python -m <runner> [--batch] [--limit N] [--offset K] [--only ID ...]
                       [--out-dir DIR] [--out PATH] [dataset options]

``--batch``        run the benchmark slice; without it, a runner with a canned
                   demo runs the demo instead
``--limit N``      evaluate at most N instances (after ``--offset``)
``--offset K``     skip the first K instances
``--only ID ...``  evaluate only these instance ids, whatever ``--limit`` says;
                   repeat the flag and/or list several ids after it
``--out-dir DIR``  output directory: ``DIR/predictions.jsonl`` plus the runner's
                   other artifacts (e.g. ``traces/``)
``--out PATH``     write the predictions JSONL to PATH instead of
                   ``DIR/predictions.jsonl``

A dataset adds its own options with ``add_arguments`` (e.g. a BFCL category) and
may set a default ``--limit``, an epilog (usage examples) and an :class:`InfoMode`
(a flag that prints a JSON report about the dataset instead of running). The
loader, the batch function, ``configure`` and the info report receive every
parsed option they declare a parameter for. A runner's output defaults are
explicit arguments of :func:`main`: without ``--out``/``--out-dir`` the
predictions go to ``predictions`` (None: not written) and ``run_batch`` keeps its
own default output directory.

Progress goes to stderr through :mod:`core.logs` (level ``LOG_LEVEL``, default
INFO); stdout carries the info report, the demo and the batch summary.
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from core import batch, logs

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InfoMode:
    """A flag that prints ``report(...)`` as JSON instead of running (e.g. ``--summary``)."""

    flag: str
    report: Callable[..., dict]
    help: str | None = None

    @property
    def dest(self) -> str:
        """The parsed option's name (``--smoke-dataset`` -> ``smoke_dataset``)."""
        return self.flag.lstrip("-").replace("-", "_")


def build_parser(
    description: str,
    add_arguments: Callable[[argparse.ArgumentParser], None] | None = None,
    *,
    default_limit: int | None = None,
    epilog: str | None = None,
    info: InfoMode | None = None,
) -> argparse.ArgumentParser:
    """The runner argument parser: the shared flags, the info flag and the dataset's options."""
    parser = argparse.ArgumentParser(
        description=description,
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter if epilog else argparse.HelpFormatter,
    )
    parser.add_argument("--batch", action="store_true", help="run the benchmark slice (default: the canned demo)")
    parser.add_argument("--limit", type=int, default=default_limit, metavar="N", help="evaluate at most N instances")
    parser.add_argument("--offset", type=int, default=0, metavar="K", help="skip the first K instances")
    parser.add_argument(
        "--only",
        action="extend",
        nargs="+",
        default=None,
        metavar="ID",
        help="evaluate only these instance ids, whatever --limit says (repeatable, several ids per flag)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="output directory for predictions.jsonl and the run's other artifacts",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        metavar="PATH",
        help="predictions JSONL path (default: <out-dir>/predictions.jsonl)",
    )
    if info is not None:
        parser.add_argument(info.flag, action="store_true", help=info.help)
    if add_arguments is not None:
        add_arguments(parser)
    return parser


def predictions_path(args: argparse.Namespace, default: Path | None = None) -> Path | None:
    """Where the predictions JSONL goes: ``--out``, else ``--out-dir``/predictions.jsonl, else ``default``."""
    if args.out is not None:
        return args.out
    if args.out_dir is not None:
        return args.out_dir / "predictions.jsonl"
    return default


def _accepted(fn: Callable, options: dict) -> dict:
    """The entries of ``options`` that ``fn`` declares as parameters."""
    params = inspect.signature(fn).parameters
    return {name: value for name, value in options.items() if name in params}


def main(
    argv: list[str] | None = None,
    *,
    description: str,
    load_instances: Callable[..., list[dict]],
    run_batch: Callable[..., dict],
    demo: Callable[[], None] | None = None,
    source: str = "",
    predictions: Path | None = None,
    add_arguments: Callable[[argparse.ArgumentParser], None] | None = None,
    default_limit: int | None = None,
    epilog: str | None = None,
    info: InfoMode | None = None,
    configure: Callable[..., None] | None = None,
    preflight: Callable[[], None] | None = None,
) -> int:
    """Parse the command line and run the info report, the demo or the batch; returns the exit status.

    ``configure`` runs first (e.g. to export settings that the options select).
    ``preflight`` runs before the demo or the batch (e.g. to exit at once when a
    dependency of the run is missing). The status is 1 when no instance is
    loaded or every row failed on infrastructure (:class:`core.batch.InfrastructureFailure`).
    """
    parser = build_parser(description, add_arguments, default_limit=default_limit, epilog=epilog, info=info)
    args = parser.parse_args(argv)
    logs.configure()
    options = vars(args)
    if args.only:
        options["limit"] = None
    if configure is not None:
        configure(**_accepted(configure, options))
    if info is not None and options[info.dest]:
        print(json.dumps(info.report(**_accepted(info.report, options)), indent=2, ensure_ascii=False, default=str))
        return 0
    if preflight is not None:
        preflight()
    if demo is not None and not args.batch:
        demo()
        return 0
    if source:
        logger.info("loading %s ...", source)
    instances = load_instances(**_accepted(load_instances, options))
    if not instances:
        logger.error("no instances loaded (check --limit/--offset/--only)")
        return 1
    logger.info("  loaded %d instance(s)", len(instances))
    out_path = predictions_path(args, predictions)
    try:
        run_batch(instances, **_accepted(run_batch, {**options, "out_path": out_path}))
    except batch.InfrastructureFailure as exc:
        logger.error("%s", exc)
        return 1
    if out_path is not None:
        logger.info("  predictions written to %s", out_path)
    return 0
