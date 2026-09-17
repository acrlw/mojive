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

from mojive.capture import CaptureSurface

from .types import ContactStyle, GeometryStyle

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
class CameraNavigationConfig:
    """Focus framing and bounded editor zoom, independent of the scene's cameras.

    Focus margin is relative to a tight fit; duration is in seconds. Linear zoom
    advances a fixed fraction of the last framed scene extent per wheel step.
    Distance limits are world units, including the equivalent perspective
    distance for orthographic zoom. These limits do not edit authored cameras.
    """

    focus_margin: float = 1.15
    focus_duration: float = 0.36
    focus_easing: str = "smoothstep"
    zoom_mode: str = "proportional"
    zoom_speed: float = 1.0
    min_distance: float = 0.001
    max_distance: float = 1e6

    def __post_init__(self) -> None:
        for name, (low, high) in CAMERA_NAVIGATION_RANGES.items():
            value = getattr(self, name)
            if not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f"{name} must be finite and between {low:g} and {high:g}")
        if self.min_distance >= self.max_distance:
            raise ValueError("minimum camera distance must be less than maximum distance")
        if self.focus_easing not in CAMERA_FOCUS_EASINGS:
            raise ValueError("unsupported focus easing")
        if self.zoom_mode not in ("proportional", "linear"):
            raise ValueError("unsupported zoom mode")

    @classmethod
    def from_mapping(cls, value: object) -> CameraNavigationConfig:
        source = value if isinstance(value, Mapping) else {}
        defaults = cls()
        values = {}
        for name, (low, high) in CAMERA_NAVIGATION_RANGES.items():
            try:
                number = float(source.get(name, getattr(defaults, name)))
            except (TypeError, ValueError, OverflowError):
                number = getattr(defaults, name)
            values[name] = (
                number
                if math.isfinite(number) and low <= number <= high
                else getattr(defaults, name)
            )
        if values["min_distance"] >= values["max_distance"]:
            values.update(min_distance=defaults.min_distance, max_distance=defaults.max_distance)
        for name, choices in (
            ("focus_easing", CAMERA_FOCUS_EASINGS),
            ("zoom_mode", ("proportional", "linear")),
        ):
            candidate = source.get(name, getattr(defaults, name))
            values[name] = candidate if candidate in choices else getattr(defaults, name)
        return cls(**values)


CAMERA_FOCUS_EASINGS = ("linear", "smoothstep", "smootherstep", "ease_out_cubic")
CAMERA_NAVIGATION_RANGES = {
    "focus_margin": (1.0, 5.0),
    "focus_duration": (0.0, 5.0),
    "zoom_speed": (0.05, 20.0),
    "min_distance": (0.001, 1e9),
    "max_distance": (0.001, 1e9),
}


@dataclass(frozen=True)
class CameraTrackingConfig:
    """World axes and position smoothing for the editor camera's tracking target.

    ``smoothing`` is the error half-life in wall-clock seconds: zero follows
    immediately. ``xy`` holds the camera height; ``xyz`` also follows height.
    """

    axes: str = "xy"
    smoothing: float = 0.25

    def __post_init__(self) -> None:
        if self.axes not in ("xy", "xyz"):
            raise ValueError("tracking axes must be 'xy' or 'xyz'")
        if not math.isfinite(self.smoothing) or self.smoothing < 0:
            raise ValueError("tracking smoothing must be finite and nonnegative")

    @classmethod
    def from_mapping(cls, value: object) -> CameraTrackingConfig:
        source = value if isinstance(value, Mapping) else {}
        defaults = cls()
        axes = source.get("axes", defaults.axes)
        if axes not in ("xy", "xyz"):
            axes = defaults.axes
        try:
            smoothing = float(source.get("smoothing", defaults.smoothing))
        except (TypeError, ValueError):
            smoothing = defaults.smoothing
        if not math.isfinite(smoothing) or smoothing < 0:
            smoothing = defaults.smoothing
        return cls(axes=axes, smoothing=smoothing)


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
    status_duration: float = 4.0

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

        try:
            duration = float(source.get("status_duration", defaults.status_duration))
        except (TypeError, ValueError):
            duration = defaults.status_duration
        if not math.isfinite(duration):
            duration = defaults.status_duration
        capsule_scale = scale("playback_scale", scale("tool_scale", defaults.playback_scale))
        return cls(
            status_duration=min(5.0, max(3.0, duration)),
            playback_scale=capsule_scale,
            tool_scale=capsule_scale,
            movable=_bool(source.get("movable"), defaults.movable),
            playback_position=_position(source.get("playback_position")),
            tool_position=_position(source.get("tool_position")),
        )


@dataclass(frozen=True)
class ViewportLayers:
    """Visibility of viewport overlays, shared by viewing, capture, and recording."""

    viewport_ui: bool = True
    gizmos: bool = True
    selection: bool = True
    helpers: bool = True
    perturbation: bool = True
    debug_3d: bool = True
    debug_2d: bool = True
    hidden_debug_layers: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, value: object) -> ViewportLayers:
        source = value if isinstance(value, Mapping) else {}
        hidden = source.get("hidden_debug_layers", ())
        return cls(
            **{
                name: _bool(source.get(name), True)
                for name in (
                    "viewport_ui",
                    "gizmos",
                    "selection",
                    "helpers",
                    "perturbation",
                    "debug_3d",
                    "debug_2d",
                )
            },
            hidden_debug_layers=tuple(
                dict.fromkeys(name for name in hidden if isinstance(name, str))
            )
            if isinstance(hidden, (tuple, list))
            else (),
        )


@dataclass(frozen=True)
class RecordingConfig:
    """Persisted interactive capture/video defaults, independent from display pacing."""

    countdown: float = 3.0
    fps: float = 60.0
    surface: CaptureSurface = CaptureSurface.VIEWPORT
    end_hold: float = 1.0
    run_simulation: bool = False
    rate_control: str = "quality"
    crf: int = 25
    bitrate_mbps: float = 12.0
    encoder_preset: str = "medium"
    pixel_format: str = "yuv420p"
    copy_to_clipboard: bool = True

    @classmethod
    def from_mapping(cls, value: object) -> RecordingConfig:
        source = value if isinstance(value, Mapping) else {}
        defaults = cls()

        def number(name, minimum, maximum):
            try:
                result = float(source.get(name, getattr(defaults, name)))
            except (TypeError, ValueError):
                return getattr(defaults, name)
            if not math.isfinite(result):
                return getattr(defaults, name)
            return min(maximum, max(minimum, result))

        def choice(name, values):
            result = source.get(name, getattr(defaults, name))
            return result if result in values else getattr(defaults, name)

        try:
            surface = CaptureSurface(source.get("surface", defaults.surface))
        except (ValueError, TypeError):
            surface = defaults.surface
        return cls(
            countdown=number("countdown", 0.0, 60.0),
            fps=number("fps", 1.0, 240.0),
            surface=surface,
            end_hold=number("end_hold", 0.0, 60.0),
            run_simulation=source.get("run_simulation") is True,
            rate_control=choice("rate_control", ("quality", "bitrate")),
            crf=round(number("crf", 0, 51)),
            bitrate_mbps=number("bitrate_mbps", 0.1, 500.0),
            encoder_preset=choice("encoder_preset", ("fast", "medium", "slow")),
            pixel_format=choice("pixel_format", ("yuv420p", "yuv444p")),
            copy_to_clipboard=_bool(source.get("copy_to_clipboard"), defaults.copy_to_clipboard),
        )


@dataclass(frozen=True)
class ViewerConfig:
    """Configure the full interactive viewer; feature reduction is explicit opt-in."""

    interactions: InteractionConfig = field(default_factory=InteractionConfig)
    selection: SelectionStyle = field(default_factory=SelectionStyle)
    panels: Mapping[str, PanelConfig] = field(default_factory=dict)
    layout: LayoutConfig = field(default_factory=LayoutConfig)
    viewport_overlays: ViewportOverlayConfig = field(default_factory=ViewportOverlayConfig)
    layers: ViewportLayers = field(default_factory=ViewportLayers)
    recording: RecordingConfig = field(default_factory=RecordingConfig)
    tracking: CameraTrackingConfig = field(default_factory=CameraTrackingConfig)
    shadow_quality: ShadowQuality | str | None = None
    threaded_physics: bool = True
    live_model_updates: bool | None = None
    # None retains the desktop panel set; an empty tuple loads no panels.
    builtin_panels: tuple[str, ...] | None = None
    debug_server: bool = True
    geometry_style: GeometryStyle | None = None
    navigation: CameraNavigationConfig = field(default_factory=CameraNavigationConfig)
    contact_style: ContactStyle | None = None

    @classmethod
    def minimal(cls, *panels: str) -> ViewerConfig:
        """Explicitly opt into navigation and selected panels without editing overlays or a socket.

        Normal viewer construction uses the full desktop configuration instead.
        Panel IDs are stable names such as ``inspector``, ``control``, ``joints``
        and ``camera``. Use ``dataclasses.replace`` to enable additional options.
        """
        return cls(
            builtin_panels=tuple(panels),
            debug_server=False,
            interactions=InteractionConfig(gizmo=False, perturb=False),
            selection=SelectionStyle(gizmo=False),
            layers=ViewportLayers(
                viewport_ui=False, gizmos=False, helpers=False, perturbation=False
            ),
            layout=LayoutConfig(persistence=False),
        )


__all__ = [
    "CameraInputConfig",
    "CameraNavigationConfig",
    "CameraTrackingConfig",
    "InteractionConfig",
    "LayoutConfig",
    "PanelConfig",
    "RecordingConfig",
    "SelectionInputConfig",
    "SelectionStyle",
    "ViewerConfig",
    "ViewportLayers",
    "ViewportOverlayConfig",
]
