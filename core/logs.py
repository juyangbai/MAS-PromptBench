"""Console logging for the command-line entry points.

Library modules log through ``logging.getLogger(__name__)`` and never configure
logging. An entry point calls :func:`configure` once: the records of this
repository's packages (and of the ``__main__`` module) at or above the chosen
level go to stderr as bare messages. Third-party loggers are left untouched.

The level is the entry point's ``--log-level`` option where it has one (see
:func:`add_argument`), else the ``LOG_LEVEL`` environment variable, else INFO.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

LEVEL_ENV = "LOG_LEVEL"
LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
PACKAGES = ("core", "topologies", "teamsizes", "communications", "optimizers", "__main__")


class _StderrHandler(logging.StreamHandler):
    """Writes to whatever ``sys.stderr`` is when a record is emitted (tools and tests swap it)."""

    def __init__(self) -> None:
        logging.Handler.__init__(self)

    @property
    def stream(self):
        return sys.stderr


_HANDLER = _StderrHandler()
_HANDLER.setFormatter(logging.Formatter("%(message)s"))


def level(name: str | None = None) -> int:
    """The logging level ``name``, else ``$LOG_LEVEL``, else INFO."""
    chosen = (name or os.environ.get(LEVEL_ENV) or "INFO").strip().upper()
    if chosen not in LEVELS:
        raise ValueError(f"unknown log level {chosen!r}; use one of {', '.join(LEVELS)}")
    return getattr(logging, chosen)


def configure(name: str | None = None) -> None:
    """Send the repository's log records at :func:`level` ``(name)`` and above to stderr.

    Calling it again only changes the level.
    """
    threshold = level(name)
    for package in PACKAGES:
        logger = logging.getLogger(package)
        logger.setLevel(threshold)
        if _HANDLER not in logger.handlers:
            logger.addHandler(_HANDLER)


def add_argument(parser: argparse.ArgumentParser) -> None:
    """Add ``--log-level`` (default: ``$LOG_LEVEL``, else INFO) to ``parser``."""
    parser.add_argument(
        "--log-level",
        type=str.upper,
        choices=LEVELS,
        default=None,
        help=f"console log level (default: ${LEVEL_ENV}, else INFO)",
    )
