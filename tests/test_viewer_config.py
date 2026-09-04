"""Programmatic viewer configuration and embedding input contracts."""

from __future__ import annotations

from mojive import (
    CameraInputConfig,
    InputClaim,
    InteractionConfig,
    LayoutConfig,
    SelectionInputConfig,
    SelectionStyle,
    ViewerConfig,
)
from mojive.composition import _viewer_layout_path


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


def test_layout_policy_can_isolate_an_embedded_viewer(tmp_path) -> None:
    isolated = ViewerConfig(layout=LayoutConfig(persistence=False))
    custom = ViewerConfig(layout=LayoutConfig(path=tmp_path / "policy-eval.ini"))

    assert _viewer_layout_path(isolated, vsync=True) == ""
    assert _viewer_layout_path(custom, vsync=True) == str(tmp_path / "policy-eval.ini")
    assert _viewer_layout_path(custom, vsync=False) == ""
