"""Guide specimens share workspace defaults while local edits remain isolated."""

from copy import deepcopy

import pytest
from examples.ui_design.native.guide_localization import translate
from examples.ui_design.native.guide_system import (
    GROUPS,
    PALETTE,
    ROLES,
    contrast,
    edit_group,
    grayscale_spec,
    proposal_spec,
    system_manifest,
)

from mojive.design.specification import DEFAULT, component_catalog, validate_spec
from mojive.ui.localization import Language, Localizer


@pytest.mark.parametrize("group", GROUPS)
def test_guide_group_edit_changes_only_its_declared_owner(group):
    original = proposal_spec()
    before = deepcopy(original)
    part = GROUPS[group]
    field = "fontSize" if "fontSize" in component_catalog()[part]["fields"] else "iconSize"
    edited = edit_group(original, group, field, 20)
    assert original == before
    expected = deepcopy(before)
    expected["parts"].setdefault(part, {})[field] = 20
    assert edited == expected
    assert edited["tokens"] == before["tokens"]


def test_guide_uses_production_defaults_without_a_second_appearance_profile(tmp_path):
    from mojive.design.runtime import DesignRuntime
    from mojive.design.store import SpecificationStore
    from mojive.ui.workspace.design import DEFAULT_DESIGN

    spec = proposal_spec()
    runtime = DesignRuntime(SpecificationStore(tmp_path / "native.json"))
    assert spec == runtime.spec
    assert spec is not DEFAULT
    assert runtime.dimensions(DEFAULT_DESIGN).control_height == spec["tokens"]["control_height"]
    original = deepcopy(DEFAULT)
    edit_group(spec, "Settings navigation", "paddingLeft", 24)
    assert original == DEFAULT


def test_guide_export_and_grayscale_keep_the_authored_proposal():
    baseline = deepcopy(DEFAULT)
    spec = proposal_spec()
    gray = grayscale_spec(spec)
    assert validate_spec(gray) == gray
    assert gray["tokens"]["sage_color"] != spec["tokens"]["sage_color"]
    assert baseline == DEFAULT
    assert (
        system_manifest(spec)["group_owners"]["Settings navigation"]["part"]
        == "settings_nav_button"
    )
    with pytest.raises(ValueError, match="no groupGap"):
        edit_group(spec, "Render metric", "groupGap", 8)


def test_guide_container_gap_is_separate_from_button_content_gap():
    original = proposal_spec()
    edited = edit_group(original, "Settings navigation", "container.rowGap", 24)
    assert edited["parts"]["settings_nav"]["rowGap"] == 24
    assert edited["parts"].get("settings_nav_button") == original["parts"].get(
        "settings_nav_button"
    )
    assert edited["parts"].get("scene_tree_row") == original["parts"].get("scene_tree_row")


def test_guide_language_changes_display_copy_without_changing_resource_identity():
    localizer = Localizer(Language.SIMPLIFIED_CHINESE)
    assert translate(localizer, "Scene filters") == "场景筛选"
    assert translate(localizer, "Navigation buttons") == "导航按钮"
    assert translate(localizer, "hinge_body") == "hinge_body"
    localizer.set_language(Language.ENGLISH, persist=False)
    assert translate(localizer, "Scene filters") == "Scene filters"


@pytest.mark.parametrize(
    "foreground,background,minimum",
    [
        ("text", "panel", 4.5),
        ("muted", "panel", 4.5),
        ("tertiary", "field", 4.5),
        ("canvas", "action", 4.5),
        ("action", "panel", 3),
    ],
)
def test_proposal_semantic_color_pairs_have_the_declared_contrast(foreground, background, minimum):
    assert contrast(PALETTE[ROLES[foreground]], PALETTE[ROLES[background]]) >= minimum


def test_guide_catalogue_includes_every_workspace_and_production_icon():
    from examples.ui_design.native.guide_icons import icon_catalogue

    from mojive.ui.icons import ICON_FAMILIES
    from mojive.ui.workspace.widget_icons import GLYPHS, PRODUCTION_GLYPHS

    entries = icon_catalogue()
    workspace = {entry.name for entry in entries if entry.family == "Workspace icons"}
    native = {entry.name for entry in entries if entry.native}
    assert workspace == GLYPHS.keys()
    assert native == PRODUCTION_GLYPHS.keys()
    assert {entry.name for entry in entries} >= {
        name for _family, icons in ICON_FAMILIES for _label, name in icons
    }
    chinese = Localizer(Language.SIMPLIFIED_CHINESE)
    assert all(translate(chinese, entry.label) != entry.label for entry in entries)
    assert len(system_manifest(proposal_spec())["icons"]) == len(entries)


@pytest.mark.parametrize("hovered,active", ((False, False), (True, False), (False, True)))
def test_switch_thumb_remains_distinct_from_track_in_authored_palette(tmp_path, hovered, active):
    from types import SimpleNamespace

    from imgui_bundle import imgui

    from mojive.design.runtime import DesignRuntime
    from mojive.design.store import SpecificationStore
    from mojive.ui.interaction_paint import ItemPaint, draw_switch

    runtime = DesignRuntime(SpecificationStore(tmp_path / "spec.json"))
    runtime.update_spec(proposal_spec())
    tracks, thumbs = [], []
    context = imgui.create_context()
    try:
        ui = SimpleNamespace(
            s=1,
            design=None,
            ink=runtime.ink,
            rect=lambda x, y, w, h, ink, **options: tracks.append(runtime.ink(ink)),
            draw=SimpleNamespace(
                add_circle_filled=lambda center, radius, ink, **options: thumbs.append(ink)
            ),
        )
        draw_switch(ui, 0, 0, 42, 24, 18, True, ItemPaint(hovered=hovered, active=active))
        thumb = imgui.color_convert_u32_to_float4(thumbs[0])
        thumb_rgb = tuple(round(c * 255) for c in (thumb.x, thumb.y, thumb.z))
        assert contrast(tracks[0], thumb_rgb) >= 3
        assert max(thumb_rgb) - min(thumb_rgb) < 24
    finally:
        imgui.destroy_context(context)
