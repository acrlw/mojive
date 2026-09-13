"""Cli: common."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from mojive.log import configure, get_logger

DEFAULT_BACKEND = "mujoco"
log = get_logger("cli")


def _setup_logging(json_mode: bool, verbose: bool) -> None:
    configure(verbose=verbose, warnings_only=json_mode)


def _resolve(name: str) -> Path:
    from mojive.scene.assets import resolve

    return resolve(name)


def _positive_int(value: str) -> int:
    try:
        result = int(value)
        if result > 0:
            return result
    except ValueError:
        pass
    raise argparse.ArgumentTypeError("value must be a positive integer")


def _positive_float(value: str) -> float:
    try:
        result = float(value)
        if math.isfinite(result) and result > 0.0:
            return result
    except ValueError:
        pass
    raise argparse.ArgumentTypeError("value must be finite and positive")
