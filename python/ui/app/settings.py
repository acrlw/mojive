"""Resolve programmatic viewer configuration and saved desktop preferences."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace

from mojive.config import (
    CameraNavigationConfig,
    CameraTrackingConfig,
    InteractionConfig,
    RecordingConfig,
    SelectionStyle,
    ViewerConfig,
    ViewportLayers,
    ViewportOverlayConfig,
)
from mojive.render.backend import ShadowQuality
from mojive.types import ContactStyle, GeometryStyle
from mojive.ui.input_bindings import InputBindings
from mojive.ui.perturb import DEFAULT_PERTURB_SCALE, MAX_PERTURB_SCALE
from mojive.ui.preferences import Preferences
from mojive.ui.viewcube import DEFAULT_SELECTION_PADDING
from mojive.ui.viewport_widgets import (
    DEFAULT_VIEWPORT_OVERLAY_SCALE,
    MAX_VIEWPORT_OVERLAY_SCALE,
    MIN_VIEWPORT_OVERLAY_SCALE,
)


def resolve_viewer_config(config: ViewerConfig | None, preferences: Preferences) -> ViewerConfig:
    """Explicit config replaces saved groups; optional None fields inherit saved values.

    In particular, passing ViewerConfig() opts into default interaction, navigation,
    recording and presentation groups. It still inherits geometry/contact style,
    shadow quality and live model updates unless those fields are supplied.
    """
    if config is None:
        config = ViewerConfig(
            interactions=InteractionConfig.from_mapping(preferences.get("interactions", {})),
            selection=SelectionStyle.from_mapping(preferences.get("selection_style", {})),
            navigation=CameraNavigationConfig.from_mapping(
                preferences.get("camera_navigation", {})
            ),
            viewport_overlays=ViewportOverlayConfig.from_mapping(
                preferences.get("viewport_overlays", {})
            ),
            layers=ViewportLayers.from_mapping(preferences.get("viewport_layers", {})),
            recording=RecordingConfig.from_mapping(preferences.get("recording", {})),
            tracking=CameraTrackingConfig.from_mapping(preferences.get("camera_tracking", {})),
        )
    geometry_style = config.geometry_style
    if geometry_style is None:
        try:
            geometry_style = GeometryStyle(**preferences.get("geometry_style", {}))
        except (TypeError, ValueError):
            geometry_style = GeometryStyle()
    contact_style = config.contact_style
    if contact_style is None:
        try:
            contact_style = ContactStyle(**preferences.get("contact_style", {}))
        except (TypeError, ValueError):
            contact_style = ContactStyle()
    try:
        shadow_quality = ShadowQuality(
            config.shadow_quality
            if config.shadow_quality is not None
            else preferences.get("shadow_quality", ShadowQuality.BALANCED.value)
        )
    except (TypeError, ValueError):
        shadow_quality = ShadowQuality.BALANCED
    return replace(
        config,
        viewport_overlays=ViewportOverlayConfig.from_mapping(asdict(config.viewport_overlays)),
        layers=ViewportLayers.from_mapping(asdict(config.layers)),
        recording=RecordingConfig.from_mapping(asdict(config.recording)),
        tracking=CameraTrackingConfig.from_mapping(asdict(config.tracking)),
        geometry_style=geometry_style,
        contact_style=contact_style,
        shadow_quality=shadow_quality,
        live_model_updates=bool(
            preferences.get("live_model_updates", False)
            if config.live_model_updates is None
            else config.live_model_updates
        ),
    )


@dataclass(frozen=True)
class EditorPreferences:
    """Resolved startup values for settings outside the programmatic viewer config."""

    take_pause_at_end: bool
    status_metric: str
    remember_precise_input_choices: bool
    precise_gizmo_absolute: bool
    precise_gizmo_angle_unit: str
    view_selection_padding: float
    perturb_force_scale: float
    perturb_torque_scale: float
    viewport_overlay_scale: float
    input_bindings: InputBindings

    @classmethod
    def resolve(cls, preferences: Preferences) -> EditorPreferences:
        angle_unit = preferences.get("precise_gizmo_angle_unit", "degrees")
        return cls(
            take_pause_at_end=preferences.get("take_pause_at_end", True) is not False,
            status_metric="steps" if preferences.get("status_metric") == "steps" else "time",
            remember_precise_input_choices=_boolean(
                preferences, "remember_precise_input_choices", True
            ),
            precise_gizmo_absolute=_boolean(preferences, "precise_gizmo_absolute", False),
            precise_gizmo_angle_unit=angle_unit
            if angle_unit in ("degrees", "radians")
            else "degrees",
            view_selection_padding=_number(
                preferences, "view_selection_padding", DEFAULT_SELECTION_PADDING
            ),
            perturb_force_scale=_perturb_scale(preferences, "perturb_force_scale"),
            perturb_torque_scale=_perturb_scale(preferences, "perturb_torque_scale"),
            viewport_overlay_scale=min(
                MAX_VIEWPORT_OVERLAY_SCALE,
                max(
                    MIN_VIEWPORT_OVERLAY_SCALE,
                    _number(preferences, "viewport_overlay_scale", DEFAULT_VIEWPORT_OVERLAY_SCALE),
                ),
            ),
            input_bindings=InputBindings.from_preferences(preferences.get("input_bindings", {})),
        )


def _boolean(preferences: Preferences, name: str, default: bool) -> bool:
    value = preferences.get(name, default)
    return value if isinstance(value, bool) else default


def _number(preferences: Preferences, name: str, default: float) -> float:
    try:
        return float(preferences.get(name, default))
    except (TypeError, ValueError):
        return default


def _perturb_scale(preferences: Preferences, name: str) -> float:
    value = _number(preferences, name, DEFAULT_PERTURB_SCALE)
    if not math.isfinite(value):
        value = DEFAULT_PERTURB_SCALE
    return min(MAX_PERTURB_SCALE, max(0.0, value))
