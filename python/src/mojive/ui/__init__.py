"""Viewer interfaces with lazy widget exports for headless drawing adapters."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .viewport_widgets import (
        ToolHint,
        ToolHintRegistry,
        ViewportChromeRegistry,
        ViewportControl,
        draw_mouse_hint_glyph,
    )

__all__ = (
    "ToolHint",
    "ToolHintRegistry",
    "ViewportChromeRegistry",
    "ViewportControl",
    "draw_mouse_hint_glyph",
)


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(".viewport_widgets", __name__), name)
    globals()[name] = value
    return value
