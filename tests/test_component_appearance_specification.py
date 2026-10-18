"""Component appearance edits share validation, help and atomic document history."""

from copy import deepcopy

import pytest

from mojive.design.editor import MISSING, SpecificationEditor
from mojive.design.help import get_field_help, get_part_help
from mojive.design.runtime import DesignRuntime
from mojive.design.specification import (
    DEFAULT,
    component_catalog,
    resolve_part_value,
    validate_spec,
)
from mojive.design.store import SpecificationStore, read_spec
from mojive.tools.design_profile import appearance_spec


def test_shared_appearance_rules_expose_only_declared_component_capabilities():
    catalog = component_catalog()
    for part, definition in catalog.items():
        fields = definition["fields"]
        if "text" in definition.get("appearance", ()):
            assert {"fontFamily", "fontSize", "textColor"} <= fields.keys()
        if "icon" in definition.get("appearance", ()):
            assert {"iconSize", "iconColor"} <= fields.keys()
        help_fields = get_part_help(part)["fields"]
        for field in fields:
            assert get_field_help(part, field) == help_fields[field]
    assert "iconSize" not in catalog["camera_body"]["fields"]
    assert "fontFamily" not in catalog["separator"]["fields"]


def test_direct_selection_groups_keep_generic_and_structural_classes_explicit():
    catalog = component_catalog()
    for part in (
        "settings_nav",
        "scene_tools",
        "pose_tools",
        "rail",
        "toolbox",
        "viewport_toolbar",
        "timeline_toolbar",
        "output_toolbar",
        "status_metrics",
    ):
        assert catalog[part]["selectionScope"] == "group"
    for part in ("button", "panel_header", "panel_body", "panel_tools", "button_row", "tabs"):
        assert "selectionScope" not in catalog[part]


@pytest.mark.parametrize("part", ("scene_tools", "pose_tools"))
@pytest.mark.parametrize("field,value", (("fontSize", 22), ("iconSize", 27), ("rowGap", 18)))
def test_panel_filter_roles_inherit_the_base_and_reset_without_changing_peers(
    tmp_path, part, field, value
):
    store = SpecificationStore(tmp_path / "design.json")
    store.write(DEFAULT, save=True)
    editor = SpecificationEditor(store.saved_path, store.live_path, read_spec)
    peer = "pose_tools" if part == "scene_tools" else "scene_tools"
    original = resolve_part_value(editor.spec, peer, field)
    base = value - 2
    assert editor.change(("parts", "panel_tools", field), base)
    assert resolve_part_value(editor.spec, part, field) == base
    assert resolve_part_value(editor.spec, peer, field) == base
    path = ("parts", part, field)
    assert editor.change(path, value)
    assert resolve_part_value(editor.spec, part, field) == value
    assert resolve_part_value(editor.spec, peer, field) == base
    assert editor.change(path, MISSING)
    assert resolve_part_value(editor.spec, part, field) == base
    assert editor.undo()
    assert resolve_part_value(editor.spec, part, field) == value
    assert editor.spec["parts"].get("settings_nav") == DEFAULT["parts"].get("settings_nav")
    assert original == resolve_part_value(DEFAULT, peer, field)


@pytest.mark.parametrize(
    "field,value",
    [
        ("iconColor", "inherit"),
        ("iconColor", "#123456"),
        ("iconColor", "#12345678"),
        ("fontFamily", "portable"),
        ("fontFamily", "mono"),
        ("textColor", "#abcdef80"),
    ],
)
def test_group_edits_reset_undo_and_save_without_touching_other_groups(tmp_path, field, value):
    store = SpecificationStore(tmp_path / "design.json")
    store.write(DEFAULT, save=True)
    editor = SpecificationEditor(store.saved_path, store.live_path, read_spec)
    part = "rail" if field.startswith("icon") else "viewport_toolbar"
    path = ("parts", part, field)
    assert editor.change(path, value)
    assert editor.effective(part)[field] == value
    assert editor.spec["tokens"] == DEFAULT["tokens"]
    assert editor.spec["parts"].get("toolbox") == DEFAULT["parts"].get("toolbox")
    editor.save()
    assert read_spec(store.saved_path)["parts"][part][field] == value
    assert editor.change(path, MISSING)
    assert resolve_part_value(editor.spec, part, field) == "inherit"
    assert editor.undo()
    assert editor.effective(part)[field] == value


@pytest.mark.parametrize("value", ["#abc", "#zzzzzz", "#1234567890", "red", None, 0, True, []])
def test_invalid_color_rejects_whole_write_without_overwriting_saved_spec(tmp_path, value):
    store = SpecificationStore(tmp_path / "design.json")
    store.write(DEFAULT, save=True)
    spec = deepcopy(DEFAULT)
    spec["parts"]["rail"] = {"iconSize": 30, "iconColor": value}
    with pytest.raises(ValueError):
        store.write(spec, save=True)
    assert store.latest()["spec"] == DEFAULT
    assert not store.live_path.exists()


def test_production_profile_preset_is_valid_and_keeps_global_defaults():
    value = appearance_spec()
    assert validate_spec(value) == value
    assert value["tokens"] == DEFAULT["tokens"]
    assert value["parts"]["rail"]["iconSize"] != value["parts"]["toolbox"]["iconSize"]


def test_each_surface_prepares_both_class_and_module_font_families(tmp_path):
    from types import SimpleNamespace

    runtime = DesignRuntime(SpecificationStore(tmp_path / "design.json"))
    value = deepcopy(DEFAULT)
    value["components"]["inspector"] = {"font_family": "portable"}
    value["parts"]["menu_trigger"] = {"fontFamily": "system"}
    runtime.update_spec(value)
    prepared = []
    ui = SimpleNamespace(prepare_fonts=lambda *args: prepared.append(args))
    runtime.prepare_fonts(ui)
    assert prepared == [(DEFAULT["tokens"]["font_family"], {"system", "portable"})]
