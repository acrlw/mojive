"""Shared design contract, persistence and independent workspace resolution."""

from copy import deepcopy

import pytest

from mojive.design.editor import MISSING, SpecificationEditor
from mojive.design.native.geometry import NativePartStyle, Rect
from mojive.design.resources import ROOT
from mojive.design.runtime import DesignRuntime
from mojive.design.specification import (
    DEFAULT,
    component_catalog,
    resolve_part_value,
    validate_spec,
)
from mojive.design.store import SpecificationConflict, SpecificationStore, read_spec
from mojive.ui.workspace.design import DEFAULT_DESIGN, DESIGN, use_design

FIELDS = [
    (part, field, rule)
    for part, definition in component_catalog().items()
    for field, rule in definition["fields"].items()
]


@pytest.mark.parametrize("part,field,rule", FIELDS, ids=[f"{p}.{f}" for p, f, _ in FIELDS])
def test_every_catalog_parameter_validates_and_resolves(part, field, rule):
    spec = deepcopy(DEFAULT)
    default = resolve_part_value(spec, part, field)
    if rule["type"] == "select":
        value = next(choice for choice in rule["choices"] if choice != default)
    else:
        value = rule["max"] if default != rule["max"] else rule["min"]
    spec.setdefault("parts", {}).setdefault(part, {})[field] = value
    validate_spec(spec)
    assert resolve_part_value(spec, part, field) == value
    restored = deepcopy(DEFAULT)
    assert resolve_part_value(restored, part, field) == default


def test_catalogue_and_assets_are_complete():
    assert len(component_catalog()) == 106
    assert len(FIELDS) == 1302
    for name in (
        "component-catalog.json",
        "design-spec.json",
        "tokens-schema.json",
        "parameter-help.json",
    ):
        assert (ROOT / name).is_file()
    validate_spec(deepcopy(DEFAULT))


@pytest.fixture
def store(tmp_path):
    store = SpecificationStore(tmp_path / "current-spec.json")
    store.write(DEFAULT, save=True)
    return store


def test_atomic_rejection_retains_last_valid_spec(store):
    before = store.latest()
    spec = deepcopy(DEFAULT)
    spec["parts"]["button"] = {"height": float("nan")}
    with pytest.raises(ValueError):
        store.write(spec)
    assert store.latest() == before
    store.live_path.write_text("{")
    latest = store.latest()
    assert latest["spec"] == DEFAULT
    assert latest["errors"][0]["source"] == "live-spec.json"


def test_two_editors_merge_different_fields_and_reject_stale_same_field(store):
    before = store.latest()["spec"]
    a, b = deepcopy(before), deepcopy(before)
    a.setdefault("parts", {}).setdefault("button", {})["height"] = 40
    b.setdefault("parts", {}).setdefault("button", {})["radius"] = 10
    store.merge(before, a)
    merged = store.merge(before, b)["spec"]
    assert merged["parts"]["button"] == {
        **before["parts"].get("button", {}),
        "height": 40,
        "radius": 10,
    }
    b["parts"]["button"]["height"] = 36
    with pytest.raises(SpecificationConflict):
        store.merge(before, b)
    assert store.latest()["spec"] == merged


def test_independent_specs_in_one_directory_do_not_share_drafts(tmp_path):
    first = SpecificationStore(tmp_path / "first.json")
    second = SpecificationStore(tmp_path / "second.json")
    for store in (first, second):
        store.write(DEFAULT, save=True)
    first_spec, second_spec = deepcopy(DEFAULT), deepcopy(DEFAULT)
    first_spec["parts"]["button"] = {"height": 40}
    second_spec["parts"]["button"] = {"height": 44}
    first.write(first_spec)
    second.write(second_spec)
    assert first.latest()["spec"] == first_spec
    assert second.latest()["spec"] == second_spec


def test_edit_reset_undo_save_and_external_change(store):
    editor = SpecificationEditor(store.saved_path, store.live_path, read_spec)
    editor.change(("parts", "button", "height"), 40)
    editor.change(("parts", "button", "height"), MISSING)
    assert editor.undo()
    assert editor.effective("button")["height"] == 40
    assert editor.redo()
    assert editor.effective("button")["height"] == resolve_part_value(DEFAULT, "button", "height")
    editor.undo()
    editor.save()
    assert read_spec(store.saved_path)["parts"]["button"]["height"] == 40
    other = deepcopy(editor.spec)
    other["parts"]["button"]["height"] = 44
    store.write(other)
    assert not editor.undo()
    assert editor.error
    assert store.latest()["spec"]["parts"]["button"]["height"] == 44


def test_dimension_context_does_not_leak_between_windows(store):
    runtime = DesignRuntime(store)
    spec = deepcopy(DEFAULT)
    spec["tokens"]["body_font_size"] = 18
    runtime.update_spec(spec)
    assert DESIGN.body_size == DEFAULT_DESIGN.body_size
    with use_design(runtime.dimensions):
        assert DESIGN.body_size == 18
        with use_design(lambda default: default):
            assert DESIGN.body_size == DEFAULT_DESIGN.body_size
        assert DESIGN.body_size == 18
    assert DESIGN.body_size == DEFAULT_DESIGN.body_size


def test_runtime_keeps_last_valid_view_when_both_files_are_invalid(store):
    runtime = DesignRuntime(store)
    spec = deepcopy(DEFAULT)
    spec["parts"]["button"] = {"height": 40}
    store.write(spec)
    runtime.poll(force=True)
    store.live_path.write_text("{")
    store.saved_path.write_text("{")
    assert not runtime.poll(force=True)
    assert runtime.resolved_part("button", "height") == 40
    assert runtime.error


@pytest.mark.parametrize(
    "key,value",
    [
        ("layout", {}),
        ("layout", []),
        ("native", None),
        ("tools", []),
        ("elements", []),
        ("elements", {"button": {"styles": []}}),
    ],
)
def test_malformed_sections_do_not_replace_a_running_spec(store, key, value):
    import json

    runtime = DesignRuntime(store)
    spec = deepcopy(DEFAULT)
    spec[key] = value
    with pytest.raises(ValueError):
        validate_spec(spec)
    store.live_path.write_text(json.dumps(spec))
    runtime.poll(force=True)
    assert runtime.spec == DEFAULT
    assert runtime.error


def test_catalogue_box_inheritance_and_condition(store):
    runtime = DesignRuntime(store)
    spec = deepcopy(DEFAULT)
    spec["parts"]["button"] = {
        "paddingX": 14,
        "widthMode": "fixed",
        "width": 100,
        "lineHeightMode": "auto",
        "lineHeight": 40,
    }
    runtime.update_spec(spec)
    style = NativePartStyle(runtime, "button")
    box = style.box(0, 0, 200, 28)
    assert box.frame.width == 100
    assert box.content.x == 14
    assert box.content.width == 72
    assert style.line_height() != 40
    assert Rect(1, 2, 3, 4).right == 4


def test_atomic_drafts_with_the_same_timestamp_refresh_live_styles(tmp_path):
    import os

    store = SpecificationStore(tmp_path / "spec.json")
    store.write(DEFAULT, save=True)
    runtime = DesignRuntime(store)
    stamp = store.saved_path.stat().st_mtime_ns + 1_000_000
    for align in ("start", "center", "end"):
        value = deepcopy(DEFAULT)
        value["parts"]["button_row"] = {"height": 80, "alignItems": align}
        store.write(value)
        os.utime(store.live_path, ns=(stamp, stamp))
        assert runtime.poll(force=True)
        assert runtime.resolved_part("button_row", "alignItems") == align
