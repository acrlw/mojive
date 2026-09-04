"""Programmatic configuration for the interactive viewer.

The configuration describes user-facing behavior rather than renderer or
physics implementation details.  Explicit values passed by an embedding
application override persisted desktop preferences for that viewer instance.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .render.backend import ShadowQuality


def _bool(value: object, default: bool) -> bool:
    return value if isinstance(value, bool) else default


def _position(value: object) -> tuple[float, float] | None:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        return None
    try:
        x, y = float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None
    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
        return None
    return x, y


@dataclass(frozen=True)
class CameraInputConfig:
    """Enable individual editor-camera input gestures."""

    orbit: bool = True
    pan: bool = True
    dolly: bool = True
    fly: bool = True
    view_cube: bool = True

    @classmethod
    def from_mapping(cls, value: object) -> CameraInputConfig:
        source = value if isinstance(value, Mapping) else {}
        defaults = cls()
        return cls(
            orbit=_bool(source.get("orbit"), defaults.orbit),
            pan=_bool(source.get("pan"), defaults.pan),
            dolly=_bool(source.get("dolly"), defaults.dolly),
            fly=_bool(source.get("fly"), defaults.fly),
            view_cube=_bool(source.get("view_cube"), defaults.view_cube),
        )


@dataclass(frozen=True)
class SelectionInputConfig:
    """Configure pointer gestures that change or focus scene selection."""

    pick: bool = True
    clear_on_empty: bool = True
    clear_with_escape: bool = True
    focus_on_double_click: bool = True
    pick_on_focus: bool = True

    @classmethod
    def from_mapping(cls, value: object) -> SelectionInputConfig:
        source = value if isinstance(value, Mapping) else {}
        defaults = cls()
        return cls(
            pick=_bool(source.get("pick"), defaults.pick),
            clear_on_empty=_bool(source.get("clear_on_empty"), defaults.clear_on_empty),
            clear_with_escape=_bool(source.get("clear_with_escape"), defaults.clear_with_escape),
            focus_on_double_click=_bool(
                source.get("focus_on_double_click"), defaults.focus_on_double_click
            ),
            pick_on_focus=_bool(source.get("pick_on_focus"), defaults.pick_on_focus),
        )


@dataclass(frozen=True)
class InteractionConfig:
    """Choose which built-in interactions may consume user input."""

    camera: CameraInputConfig = field(default_factory=CameraInputConfig)
    selection: SelectionInputConfig = field(default_factory=SelectionInputConfig)
    gizmo: bool = True
    perturb: bool = True
    playback_shortcuts: bool = True
    panel_shortcuts: bool = True

    @classmethod
    def from_mapping(cls, value: object) -> InteractionConfig:
        source = value if isinstance(value, Mapping) else {}
        defaults = cls()
        return cls(
            camera=CameraInputConfig.from_mapping(source.get("camera")),
            selection=SelectionInputConfig.from_mapping(source.get("selection")),
            gizmo=_bool(source.get("gizmo"), defaults.gizmo),
            perturb=_bool(source.get("perturb"), defaults.perturb),
            playback_shortcuts=_bool(source.get("playback_shortcuts"), defaults.playback_shortcuts),
            panel_shortcuts=_bool(source.get("panel_shortcuts"), defaults.panel_shortcuts),
        )


@dataclass(frozen=True)
class SelectionStyle:
    """Choose how the current logical selection is presented."""

    highlight: bool = True
    outline: bool = True
    gizmo: bool = True
    frame: bool = False
    label: bool = False
    bounds: bool = False

    @classmethod
    def from_mapping(cls, value: object) -> SelectionStyle:
        source = value if isinstance(value, Mapping) else {}
        defaults = cls()
        return cls(
            highlight=_bool(source.get("highlight"), defaults.highlight),
            outline=_bool(source.get("outline"), defaults.outline),
            gizmo=_bool(source.get("gizmo"), defaults.gizmo),
            frame=_bool(source.get("frame"), defaults.frame),
            label=_bool(source.get("label"), defaults.label),
            bounds=_bool(source.get("bounds"), defaults.bounds),
        )


@dataclass(frozen=True)
class PanelConfig:
    """Optional overrides for one panel's availability and initial state."""

    enabled: bool | None = None
    open: bool | None = None


@dataclass(frozen=True)
class LayoutConfig:
    """Control ImGui layout isolation for an embedded viewer."""

    persistence: bool = True
    path: str | Path | None = None
    reset: bool = False


@dataclass(frozen=True)
class ViewportOverlayConfig:
    """Configure movable viewport playback and tool capsules."""

    playback_scale: float = 1.0
    tool_scale: float = 1.0
    movable: bool = True
    playback_position: tuple[float, float] | None = None
    tool_position: tuple[float, float] | None = None

    @classmethod
    def from_mapping(cls, value: object) -> ViewportOverlayConfig:
        source = value if isinstance(value, Mapping) else {}
        defaults = cls()

        def scale(name: str, default: float) -> float:
            try:
                result = float(source.get(name, default))
            except (TypeError, ValueError):
                return default
            if not math.isfinite(result):
                return default
            return min(1.6, max(0.6, result))

        return cls(
            playback_scale=scale("playback_scale", defaults.playback_scale),
            tool_scale=scale("tool_scale", defaults.tool_scale),
            movable=_bool(source.get("movable"), defaults.movable),
            playback_position=_position(source.get("playback_position")),
            tool_position=_position(source.get("tool_position")),
        )


@dataclass(frozen=True)
class ViewerConfig:
    """Top-level behavior configuration for an interactive viewer."""

    interactions: InteractionConfig = field(default_factory=InteractionConfig)
    selection: SelectionStyle = field(default_factory=SelectionStyle)
    panels: Mapping[str, PanelConfig] = field(default_factory=dict)
    layout: LayoutConfig = field(default_factory=LayoutConfig)
    viewport_overlays: ViewportOverlayConfig = field(default_factory=ViewportOverlayConfig)
    shadow_quality: ShadowQuality | str | None = None


__all__ = [
    "CameraInputConfig",
    "InteractionConfig",
    "LayoutConfig",
    "PanelConfig",
    "SelectionInputConfig",
    "SelectionStyle",
    "ViewerConfig",
    "ViewportOverlayConfig",
]
