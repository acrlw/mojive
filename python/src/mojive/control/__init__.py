"""Application operations and local RPC; transport imports do not start an application."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .application import CAMERA_DIRECTORY as CAMERA_DIRECTORY
    from .application import ControlApplication as ControlApplication

__all__ = ["CAMERA_DIRECTORY", "ControlApplication"]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(".application", __name__), name)
    globals()[name] = value
    return value
