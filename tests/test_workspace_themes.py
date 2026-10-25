"""Theme colors stay per-workspace, persist atomically, and preserve geometry."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from mojive.design.runtime import DesignRuntime
from mojive.design.specification import DEFAULT, validate_spec
from mojive.design.store import SpecificationStore
from mojive.design.themes import PRESETS, colors, select_theme, theme_name, themed_spec
from mojive.ui.design_palette import rgba8
from mojive.ui.theme import relative_luminance


def test_theme_switch_preserves_layout_type_and_module_overrides(tmp_path):
    store = SpecificationStore(tmp_path / "design.json")
    spec = deepcopy(DEFAULT)
    spec["parts"]["input_number"]["fontSize"] = 18
    spec["layout"]["inspector_width"] = 370
    spec["components"]["output"] = {"text_color": "#AA8899"}
    store.write(spec, save=True)
    runtime = DesignRuntime(store)
    select_theme(runtime, "light")
    assert theme_name(runtime.spec) == "light"
    assert runtime.spec["parts"] == spec["parts"]
    assert runtime.spec["layout"] == spec["layout"]
    assert runtime.spec["components"] == spec["components"]
    assert DesignRuntime(store).palette == runtime.palette
    select_theme(runtime, "dark")
    assert theme_name(runtime.spec) == "dark"
    assert runtime.spec["parts"] == spec["parts"]


def test_two_workspaces_do_not_mutate_each_others_colors(tmp_path):
    one = DesignRuntime(SpecificationStore(tmp_path / "one.json"))
    two = DesignRuntime(SpecificationStore(tmp_path / "two.json"))
    original = two.ink(rgba8("text"))
    select_theme(one, "light", persist=False)
    assert one.ink(rgba8("text")) != original
    assert two.ink(rgba8("text")) == original == rgba8("text")
    assert one.palette["on_primary"] != one.palette["child"]
    assert one.ink(rgba8("child")) == one.palette["child"]
    assert one.ink(rgba8("text")) == one.palette["text"]


@pytest.mark.parametrize("name", ("dark", "light"))
def test_named_themes_have_readable_text_and_action_contrast(name):
    palette = colors(themed_spec(DEFAULT, name))

    def luminance(role):
        value = palette[role]
        return relative_luminance(tuple(int(value[i : i + 2], 16) / 255 for i in (1, 3, 5)))

    for text, background in [
        ("text", "background"),
        ("text", "field"),
        ("secondary", "popup"),
        ("on_primary", "primary"),
        ("info", "child"),
        ("warning", "warning_surface"),
    ]:
        hi, lo = sorted((luminance(text), luminance(background)), reverse=True)
        assert (hi + 0.05) / (lo + 0.05) >= 4.5, (name, text, background)


def test_invalid_palette_rejected_and_legacy_snapshots_remain_valid():
    assert validate_spec(deepcopy(DEFAULT))
    for palette in ({"unknown": "#000000"}, {"text": "white"}, None):
        spec = deepcopy(DEFAULT)
        spec["palette"] = palette
        with pytest.raises(ValueError, match="palette"):
            validate_spec(spec)
    assert set(PRESETS["dark"]) == set(PRESETS["light"])


def test_theme_name_ignores_hex_letter_case():
    spec = themed_spec(DEFAULT, "dark")
    spec["tokens"]["border_color"] = "#ffffff0d"
    assert theme_name(spec) == "dark"


@pytest.mark.parametrize("role", ("field", "field_hovered", "field_active"))
def test_secondary_actions_separate_from_light_surfaces_without_changing_dark(role):
    from mojive.ui.interaction_paint import secondary_action_ink

    def rgba(value):
        return (*(int(value[i : i + 2], 16) for i in (1, 3, 5)), 255)

    for name in ("light", "dark"):
        palette = PRESETS[name]
        field, text = rgba(palette[role]), rgba(palette["text"])
        ink = secondary_action_ink(field, text)
        if name == "dark":
            assert ink == field
        else:
            field_luma, ink_luma, text_luma = (
                relative_luminance(tuple(v / 255 for v in color[:3]))
                for color in (field, ink, text)
            )
            assert text_luma < ink_luma < field_luma
            assert (ink_luma + 0.05) / (text_luma + 0.05) >= 4.5


@pytest.mark.parametrize("name", ("dark", "light"))
def test_neutral_painters_receive_cached_theme_and_nested_chrome(tmp_path, name):
    from mojive.ui.workspace.widgets import UI

    runtime = DesignRuntime(SpecificationStore(tmp_path / "design.json"))
    select_theme(runtime, name, persist=False)
    ui = SimpleNamespace(design=SimpleNamespace(runtime=runtime, family=lambda: "viewport"))
    theme = UI.control_theme(ui)
    assert UI.control_theme(ui) is theme
    assert theme.bg_frame == tuple(v / 255 for v in runtime.palette["field"])
    assert theme.border == tuple(v / 255 for v in runtime.palette["divider"])
    assert theme.viewport.off_foreground == theme.text
    assert theme.viewport.surface[:3] == theme.bg_child[:3]
    assert theme.viewport.on_background == theme.bg_frame_active
    assert theme.viewport.record == theme.danger
    spec = deepcopy(runtime.spec)
    spec["components"]["viewport"] = {"text_color": "#556677"}
    runtime.update_spec(spec)
    scoped = UI.control_theme(ui)
    assert scoped.viewport.off_foreground == scoped.text == (85 / 255, 102 / 255, 119 / 255, 1)
    assert UI.control_theme(ui) is scoped


@pytest.mark.parametrize("name", ("dark", "light"))
def test_gizmo_readout_markers_remain_legible_on_their_theme(name):
    from mojive.ui.gizmo.state import (
        JOINT_CURRENT_COLOR,
        JOINT_LOWER_LIMIT_COLOR,
        JOINT_UPPER_LIMIT_COLOR,
    )
    from mojive.ui.workspace.gizmo_readout import readout_marker_ink

    surface = tuple(v / 255 for v in rgba8("floating"))
    if name == "light":
        value = PRESETS[name]["floating"]
        surface = (*(int(value[i : i + 2], 16) / 255 for i in (1, 3, 5)), 1)
    for color in (JOINT_CURRENT_COLOR, JOINT_LOWER_LIMIT_COLOR, JOINT_UPPER_LIMIT_COLOR):
        ink = readout_marker_ink(color, surface)
        hi, lo = sorted((relative_luminance(ink), relative_luminance(surface)), reverse=True)
        assert (hi + 0.05) / (lo + 0.05) >= 3
        assert ink is readout_marker_ink(color, surface)
        assert sorted(range(3), key=lambda i: ink[i]) == sorted(range(3), key=lambda i: color[i])


def test_builtin_edit_notice_catalog_has_chinese_captions():
    from mojive.session import _SCENE_EDIT_COMMANDS
    from mojive.ui.localization import Localizer

    localizer = Localizer()
    localizer.set_language("zh_CN", persist=False)
    labels = (
        [command.__name__ for command in _SCENE_EDIT_COMMANDS]
        + [
            "Create " + shape
            for shape in (
                "box",
                "sphere",
                "cylinder",
                "cone",
                "capsule",
                "ellipsoid",
                "plane",
                "site",
            )
        ]
        + ["Apply model edits", "Delete model keyframes", "Resize geometry", "Edit model"]
    )
    for label in labels:
        assert localizer.text(label) != label
        assert localizer.text("Undo " + label) == "撤销：" + localizer.text(label)
    # User names and external diagnostics are not command captions.
    assert localizer.text("my_custom_entity") == "my_custom_entity"
