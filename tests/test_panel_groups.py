"""Panel ownership isolates appearance, geometry, cached styles and saved overrides."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from mojive.design.groups import CURRENT_GROUPS, component_groups, group_part, group_scope
from mojive.design.native.controls import NativePrimitives
from mojive.design.native.geometry import CURRENT_NATIVE_PARTS, part_scope
from mojive.design.runtime import DesignRuntime
from mojive.design.specification import DEFAULT, component_catalog, resolve_part_value
from mojive.design.store import SpecificationStore

GROUPS = [(owner, base) for owner, group in component_groups().items() for base in group["parts"]]


@pytest.mark.parametrize("owner,base", GROUPS)
def test_local_group_edits_do_not_leak_through_shared_controls_or_style_cache(
    tmp_path, owner, base
):
    runtime = DesignRuntime(SpecificationStore(tmp_path / "design.json"))
    painter = NativePrimitives(SimpleNamespace(s=1), runtime, imgui=SimpleNamespace())
    part = group_part(owner, base)
    definition = component_catalog()[part]
    changes = {
        "fontSize": 23,
        "fontFamily": "portable",
        "textColor": "#f4b860",
        "iconSize": 27,
        "iconColor": "#4a9eff",
        "rowGap": 17,
        "columnGap": 19,
        "paddingLeft": 13,
        "marginRight": 11,
    }
    changes = {field: value for field, value in changes.items() if field in definition["fields"]}
    baseline = dict(painter.style(base).values)
    candidate = deepcopy(DEFAULT)
    candidate["parts"][part] = changes
    runtime.update_spec(candidate)
    with group_scope(owner):
        local = painter.style(base)
        assert part in local.part_ids
        for field, value in changes.items():
            assert local.get(field) == value
        with part_scope(base):
            child = painter.style("button")
            assert child.get("fontFamily") == "portable"
            assert child.get("textColor") == "#f4b860"
            assert child.get("fontSize") == 23
        assert painter.style(base) is local
    assert dict(painter.style(base).values) == baseline
    for peer in component_groups():
        if peer != owner and base in component_groups()[peer]["parts"]:
            with group_scope(peer):
                for field in changes:
                    assert painter.style(base).values[field] == resolve_part_value(
                        DEFAULT, base, field
                    )
            break
    runtime.update_spec(deepcopy(DEFAULT))
    with group_scope(owner):
        assert dict(painter.style(base).values) == baseline


def test_nested_settings_page_uses_local_content_and_restores_panel_ownership():
    with group_scope("Settings"), part_scope("panel_body"):
        parents = CURRENT_NATIVE_PARTS.get()
        with (
            pytest.raises(RuntimeError),
            group_scope("Settings/Camera"),
            part_scope("section_body"),
        ):
            assert "settings_camera_section_body" in CURRENT_NATIVE_PARTS.get()
            raise RuntimeError("Interrupted paint")
        assert CURRENT_NATIVE_PARTS.get() == parents
        assert CURRENT_GROUPS.get() == ("Settings",)
    assert not CURRENT_NATIVE_PARTS.get()
    assert not CURRENT_GROUPS.get()


@pytest.mark.parametrize(
    "owner,base,field,token,value",
    (
        ("Scene", "panel_body", "fontSize", "body_font_size", 21),
        ("Scene", "tree_row", "fontSize", "body_font_size", 21),
        ("Inspector", "section_header", "fontSize", "body_font_size", 21),
        ("Help", "window_title", "paddingX", "frame_padding_x", 12),
    ),
)
@pytest.mark.parametrize("family", (None, "settings"))
def test_owned_defaults_retain_global_and_family_token_edits(
    tmp_path, owner, base, field, token, value, family
):
    from mojive.design.native.geometry import NativePartStyle

    runtime = DesignRuntime(SpecificationStore(tmp_path / "design.json"))
    candidate = deepcopy(DEFAULT)
    if family is None:
        candidate["tokens"][token] = value
    else:
        candidate["components"][family] = {token: value}
    runtime.update_spec(candidate)
    assert NativePartStyle(runtime, base, family, authored_defaults=True).get(field, -1) == value
    with group_scope(owner):
        assert (
            NativePartStyle(runtime, base, family, authored_defaults=True).get(field, -1) == value
        )
