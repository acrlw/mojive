"""Small Loguru setup shared by every forge-viewer runtime module."""

from __future__ import annotations

import sys
from typing import TextIO

from loguru import logger

_FORMAT = "<dim>[forge/{extra[component]}]</dim> <level>{level: <8}</level> {message}"
logger.disable("forge_viewer")


def get_logger(component: str):
    return logger.bind(component=component)


def configure(
    *, verbose: bool = False, warnings_only: bool = False, stream: TextIO | None = None
) -> None:
    """Configure the forge logger once; application output remains on stdout."""
    logger.remove()
    logger.enable("forge_viewer")
    logger.add(
        stream or sys.stderr,
        level="DEBUG" if verbose else "WARNING" if warnings_only else "INFO",
        format=_FORMAT,
        filter=lambda record: "component" in record["extra"],
        colorize=None,
        backtrace=verbose,
        diagnose=False,
    )
