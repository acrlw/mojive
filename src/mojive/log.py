"""Small Loguru setup shared by every mojive runtime module."""

from __future__ import annotations

import sys
from contextlib import suppress
from typing import Any, TextIO

from loguru import logger

_FORMAT = "<dim>[mojive/{extra[component]}]</dim> <level>{level: <8}</level> {message}"
logger.disable("mojive")


def get_logger(component: str):
    return logger.bind(component=component)


def configure(
    *, verbose: bool = False, warnings_only: bool = False, stream: TextIO | None = None
) -> None:
    """Configure the Mojive logger once; application output remains on stdout."""
    logger.remove()
    logger.enable("mojive")
    logger.add(
        stream or sys.stderr,
        level="DEBUG" if verbose else "WARNING" if warnings_only else "INFO",
        format=_FORMAT,
        filter=lambda record: "component" in record["extra"],
        colorize=None,
        backtrace=verbose,
        diagnose=False,
    )


def add_output_sink(sink: Any) -> int:
    """Mirror Mojive runtime records into an editor-owned callable sink."""

    return int(
        logger.add(
            sink,
            level="DEBUG",
            format="{message}",
            filter=lambda record: "component" in record["extra"],
            colorize=False,
            backtrace=False,
            diagnose=False,
        )
    )


def remove_output_sink(sink_id: int) -> None:
    """Detach an editor output sink if it is still registered."""

    with suppress(ValueError):
        logger.remove(int(sink_id))
