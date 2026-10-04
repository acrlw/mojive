"""Programmatic viewer configuration and embedding input contracts."""

from __future__ import annotations

from dataclasses import asdict

import pytest

from mojive import (
    THEME,
    CameraInputConfig,
    InputClaim,
    InteractionConfig,
    LayoutConfig,
    RecordingConfig,
    SelectionInputConfig,
    SelectionStyle,
    Theme,
    ViewerConfig,
    ViewportChromeColors,
    ViewportOverlayConfig,
)
from mojive.app.composition import _viewer_layout_path


def test_recording_encoding_defaults_round_trip_and_sanitize_saved_preferences():
    defaults = RecordingConfig()
    assert (defaults.rate_control, defaults.crf, defaults.encoder_preset) == (
        "quality",
        25,
        "medium",
    )
    assert RecordingConfig.from_mapping({"countdown": 0}).crf == 25
    config = RecordingConfig(
        rate_control="bitrate",
        crf=18,
        bitrate_mbps=8.5,
        encoder_preset="fast",
        pixel_format="yuv444p",
    )
    assert RecordingConfig.from_mapping(asdict(config)) == config
    invalid = RecordingConfig.from_mapping(
        {
            "rate_control": "invalid",
            "crf": -5,
            "bitrate_mbps": float("nan"),
            "encoder_preset": ["slow"],
            "pixel_format": "invalid",
        }
    )
    assert invalid.crf == 0 and invalid.bitrate_mbps == defaults.bitrate_mbps
    assert invalid.rate_control == "quality" and invalid.encoder_preset == "medium"
    assert invalid.pixel_format == "yuv420p"
    clipped = RecordingConfig.from_mapping({"crf": 999, "bitrate_mbps": -1})
    assert clipped.crf == 51 and clipped.bitrate_mbps == 0.1


def test_partial_preference_mapping_uses_documented_defaults() -> None:
    config = InteractionConfig.from_mapping(
        {
            "camera": {"fly": False},
            "selection": {"pick": False, "clear_on_empty": "invalid"},
            "gizmo": False,
        }
    )

    assert config.camera == CameraInputConfig(fly=False)
    assert config.selection == SelectionInputConfig(pick=False)
    assert config.gizmo is False
    assert config.perturb is True


def test_selection_style_mapping_ignores_non_boolean_values() -> None:
    assert SelectionStyle.from_mapping({"outline": False, "bounds": 1}) == SelectionStyle(
        outline=False
    )


def test_input_claim_normalizes_public_key_names() -> None:
    claim = InputClaim(keys=frozenset({"W", "Control", "1"}), mouse_buttons=frozenset({0}))

    assert claim.claims_key("w")
    assert claim.claims_key("ctrl")
    assert claim.claims_key("digit_1")
    assert claim.claims_button(0)
    assert not claim.claims_button(1)


def test_top_level_config_is_immutable_and_composable() -> None:
    config = ViewerConfig(
        interactions=InteractionConfig(camera=CameraInputConfig(fly=False)),
        selection=SelectionStyle(gizmo=False, frame=True),
    )

    assert config.interactions.camera.fly is False
    assert config.selection.frame is True


def test_public_theme_exposes_viewport_states_and_entity_palette() -> None:
    chrome = ViewportChromeColors(on_background=(0.2, 0.4, 0.6, 1.0))
    palette = ((0.8, 0.1, 0.2, 1.0),)
    theme = Theme(viewport=chrome, entity_palette=palette)

    assert theme.viewport.on_background == (0.2, 0.4, 0.6, 1.0)
    assert theme.entity_palette == palette
    assert THEME.viewport.surface[3] > 0.0


def test_viewport_overlay_mapping_validates_scales_and_positions() -> None:
    overlays = ViewportOverlayConfig.from_mapping(
        {
            "playback_scale": 0.1,
            "tool_scale": float("nan"),
            "movable": False,
            "playback_position": (0.25, 0.75),
            "tool_position": (2.0, 0.5),
        }
    )

    assert overlays.playback_scale == 0.6
    assert overlays.tool_scale == overlays.playback_scale
    assert overlays.movable is False
    assert overlays.playback_position == (0.25, 0.75)
    assert overlays.tool_position is None


def test_layout_policy_can_isolate_an_embedded_viewer(tmp_path) -> None:
    isolated = ViewerConfig(layout=LayoutConfig(persistence=False))
    custom = ViewerConfig(layout=LayoutConfig(path=tmp_path / "policy-eval.ini"))

    assert _viewer_layout_path(isolated, vsync=True) == ""
    assert _viewer_layout_path(custom, vsync=True) == str(tmp_path / "policy-eval.ini")
    assert _viewer_layout_path(custom, vsync=False) == ""


def test_camera_navigation_preferences_round_trip_and_recover_invalid_saved_values():
    from mojive import CameraNavigationConfig

    config = CameraNavigationConfig(
        focus_margin=1.8,
        focus_duration=0.6,
        focus_easing="smootherstep",
        zoom_mode="linear",
        zoom_speed=2,
        min_distance=0.05,
        max_distance=500,
    )
    assert CameraNavigationConfig.from_mapping(asdict(config)) == config
    assert (
        CameraNavigationConfig.from_mapping(
            {
                "focus_margin": float("nan"),
                "focus_duration": float("inf"),
                "focus_easing": [],
                "zoom_speed": "bad",
                "zoom_mode": "unknown",
                "min_distance": 1000,
                "max_distance": 1,
            }
        )
        == CameraNavigationConfig()
    )
    assert ViewerConfig(navigation=config).navigation == config


def test_camera_navigation_configuration_rejects_invalid_runtime_limits():
    from mojive import CameraNavigationConfig

    for values in (
        {"min_distance": 2, "max_distance": 1},
        {"max_distance": float("inf")},
        {"focus_duration": -1},
        {"focus_margin": float("nan")},
        {"focus_easing": "invalid"},
        {"zoom_speed": 0},
    ):
        with pytest.raises(ValueError):
            CameraNavigationConfig(**values)


@pytest.mark.parametrize("explicit", [False, True])
def test_config_resolution_preserves_group_and_optional_field_precedence(explicit):
    from mojive import CameraNavigationConfig, CameraTrackingConfig, ViewportLayers
    from mojive.render.backend import ShadowQuality
    from mojive.types import ContactStyle, GeometryStyle
    from mojive.ui.app.settings import resolve_viewer_config
    from mojive.ui.preferences import Preferences

    saved = ViewerConfig(
        interactions=InteractionConfig(gizmo=False),
        selection=SelectionStyle(outline=False),
        navigation=CameraNavigationConfig(focus_margin=2.0),
        viewport_overlays=ViewportOverlayConfig(playback_scale=1.2, tool_scale=1.2),
        layers=ViewportLayers(gizmos=False),
        recording=RecordingConfig(fps=48),
        tracking=CameraTrackingConfig(smoothing=0.8),
        geometry_style=GeometryStyle(visual_opacity=0.4),
        contact_style=ContactStyle(shape="sphere"),
        shadow_quality=ShadowQuality.HIGH,
        live_model_updates=True,
    )
    keys = {
        "interactions": "interactions",
        "selection": "selection_style",
        "navigation": "camera_navigation",
        "viewport_overlays": "viewport_overlays",
        "layers": "viewport_layers",
        "recording": "recording",
        "tracking": "camera_tracking",
        "geometry_style": "geometry_style",
        "contact_style": "contact_style",
    }
    preferences = Preferences(
        values={
            **{key: asdict(getattr(saved, name)) for name, key in keys.items()},
            "shadow_quality": "high",
            "live_model_updates": True,
        }
    )
    supplied = ViewerConfig() if explicit else None
    resolved = resolve_viewer_config(supplied, preferences)
    expected_groups = supplied if explicit else saved
    for name in (
        "interactions",
        "selection",
        "navigation",
        "viewport_overlays",
        "layers",
        "recording",
        "tracking",
    ):
        assert getattr(resolved, name) == getattr(expected_groups, name)
    for name in ("geometry_style", "contact_style", "shadow_quality", "live_model_updates"):
        assert getattr(resolved, name) == getattr(saved, name)
    if supplied is not None:
        assert supplied.live_model_updates is None
        assert supplied.geometry_style is None


def test_explicit_optional_config_overrides_saved_values_and_keeps_embedding_policy():
    from mojive.config import PanelConfig
    from mojive.render.backend import ShadowQuality
    from mojive.types import ContactStyle, GeometryStyle
    from mojive.ui.app.settings import resolve_viewer_config
    from mojive.ui.preferences import Preferences

    config = ViewerConfig(
        live_model_updates=False,
        shadow_quality=ShadowQuality.PERFORMANCE,
        geometry_style=GeometryStyle(),
        contact_style=ContactStyle(),
        panels={"settings": PanelConfig(open=True)},
        builtin_panels=("settings",),
        threaded_physics=False,
        debug_server=False,
        layout=LayoutConfig(persistence=False),
    )
    preferences = Preferences(
        values={
            "live_model_updates": True,
            "shadow_quality": "high",
            "geometry_style": {"visual_opacity": 0.1},
            "contact_style": {"shape": "sphere"},
        }
    )
    assert resolve_viewer_config(config, preferences) == config


@pytest.mark.parametrize("value", [None, [], "bad", {"unknown": True}])
def test_invalid_saved_config_groups_recover_without_rewriting_preferences(value):
    from mojive.render.backend import ShadowQuality
    from mojive.types import ContactStyle, GeometryStyle
    from mojive.ui.app.settings import resolve_viewer_config
    from mojive.ui.preferences import Preferences

    values = dict.fromkeys(
        (
            "interactions",
            "selection_style",
            "camera_navigation",
            "viewport_overlays",
            "viewport_layers",
            "recording",
            "camera_tracking",
            "geometry_style",
            "contact_style",
            "shadow_quality",
        ),
        value,
    )
    resolved = resolve_viewer_config(None, Preferences(values=values))
    defaults = ViewerConfig()
    assert resolved.interactions == defaults.interactions
    assert resolved.selection == defaults.selection
    assert resolved.navigation == defaults.navigation
    assert resolved.viewport_overlays == defaults.viewport_overlays
    assert resolved.layers == defaults.layers
    assert resolved.recording == defaults.recording
    assert resolved.tracking == defaults.tracking
    assert resolved.geometry_style == GeometryStyle()
    assert resolved.contact_style == ContactStyle()
    assert resolved.shadow_quality == ShadowQuality.BALANCED
    assert all(item is value for item in values.values())


def test_viewer_uses_resolved_values_and_one_store_for_later_preferences(tmp_path, monkeypatch):
    from mojive.adapters.static import StaticSceneAdapter
    from mojive.render.backend import NullBackend
    from mojive.scene import Scene
    from mojive.session import Session
    from mojive.ui.app import ViewerApp
    from mojive.ui.preferences import Preferences

    monkeypatch.setenv("MOJIVE_SETTINGS", str(tmp_path / "settings.json"))
    Preferences.load().update(
        {
            "interactions": {"gizmo": False},
            "precise_gizmo_absolute": True,
            "take_pause_at_end": False,
            "perturb_force_scale": 3.0,
            "perturb_torque_scale": 2.0,
        }
    )
    session = Session(StaticSceneAdapter(Scene()))
    try:
        app = ViewerApp(session, NullBackend())
        assert not app.interactions.gizmo
        assert not session.state_take_pause_at_end
        assert app.perturb.force_scale == 3.0
        assert app.perturb.torque_scale == 2.0
        assert app.preferences.values is app.localizer.preferences
        app.set_perturb_strength(4.0, 5.0)
        app.set_language("zh_CN")
        assert app.localizer.preference("perturb_force_scale") == 4.0
        assert app.preferences.get("language") == "zh_CN"
        assert Preferences.load().get("perturb_torque_scale") == 5.0
    finally:
        session.release()


@pytest.mark.parametrize(
    "attribute,value",
    [
        ("interactions", InteractionConfig(gizmo=False, perturb=False)),
        ("selection_style", SelectionStyle(gizmo=False)),
    ],
)
def test_failed_policy_save_keeps_applied_runtime_but_exposes_save_error(
    tmp_path, monkeypatch, attribute, value
):
    from pathlib import Path

    from mojive.adapters.static import StaticSceneAdapter
    from mojive.render.backend import NullBackend
    from mojive.scene import Scene
    from mojive.session import Session
    from mojive.ui.app import ViewerApp
    from mojive.ui.preferences import Preferences

    path = tmp_path / "settings.json"
    monkeypatch.setenv("MOJIVE_SETTINGS", str(path))
    Preferences.load().update({"unrelated": "preserved"})
    session = Session(StaticSceneAdapter(Scene()))
    try:
        app = ViewerApp(session, NullBackend())
        original_values = dict(app.preferences.values)
        original_file = path.read_bytes()
        gestures = []
        monkeypatch.setattr(app.gizmo, "cancel", lambda: gestures.append("gizmo"))
        monkeypatch.setattr(app.perturb, "end", lambda session: gestures.append("perturb"))
        session.perturb.active = True

        def reject_replace(self, target):
            raise OSError("settings disk unavailable")

        with monkeypatch.context() as failing:
            failing.setattr(Path, "replace", reject_replace)
            with pytest.raises(OSError, match="settings disk unavailable"):
                getattr(app, f"set_{attribute}")(value)
        assert getattr(app, attribute) == value
        assert app.preferences.values == original_values
        assert path.read_bytes() == original_file
        expected = ["gizmo", "perturb"] if attribute == "interactions" else ["gizmo"]
        assert gestures == expected
        gestures.clear()

        getattr(app, f"set_{attribute}")(value)
        assert getattr(app, attribute) == value
        assert Preferences.load().get(attribute) == asdict(value)
        assert gestures == (["gizmo", "perturb"] if attribute == "interactions" else ["gizmo"])
    finally:
        session.release()
