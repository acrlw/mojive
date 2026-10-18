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
    elif rule["type"] == "color":
        value = "#12345678"
    else:
        value = rule["max"] if default != rule["max"] else rule["min"]
    spec.setdefault("parts", {}).setdefault(part, {})[field] = value
    validate_spec(spec)
    assert resolve_part_value(spec, part, field) == value
    restored = deepcopy(DEFAULT)
    assert resolve_part_value(restored, part, field) == default


def test_catalogue_and_assets_are_complete():
    catalog = component_catalog()
    assert {
        "scene_panel_header",
        "inspector_tabs",
        "settings_camera_settings_page",
        "status_backend",
    } <= catalog.keys()
    assert all(definition["fields"] for definition in component_catalog().values())
    for name in (
        "component-catalog.json",
        "design-spec.json",
        "tokens-schema.json",
        "parameter-help.json",
        "component-groups.json",
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


def test_runtime_loads_a_valid_design_spec_file_when_its_draft_is_invalid(tmp_path):
    store = SpecificationStore(tmp_path / "design-spec.json")
    store.write(DEFAULT, save=True)
    runtime = DesignRuntime(store)
    value = deepcopy(DEFAULT)
    value["parts"]["button"]["height"] = 40
    store.write(value, save=True)
    store.live_path.write_text("{")
    assert runtime.poll(force=True)
    assert runtime.resolved_part("button", "height") == 40
    assert runtime.error


@pytest.mark.parametrize("invalid", ("saved", "draft"))
def test_editor_and_runtime_share_the_latest_valid_snapshot(store, invalid):
    editor = SpecificationEditor(store.saved_path, store.live_path, read_spec)
    runtime = DesignRuntime(store)
    before_saved = deepcopy(editor.saved_spec)
    value = deepcopy(DEFAULT)
    value["parts"]["button"]["height"] = 40
    if invalid == "saved":
        store.write(value)
        store.saved_path.write_text("{")
    else:
        store.write(value, save=True)
        store.live_path.write_text("{")
    assert runtime.poll(force=True)
    assert editor.poll(force=True)
    assert editor.spec == runtime.spec == value
    assert editor.saved_spec == (before_saved if invalid == "saved" else value)
    assert editor.error and runtime.error


@pytest.mark.parametrize("recovery", ("restore", "import"))
def test_deleted_specifications_keep_editor_and_preview_on_the_last_valid_snapshot(
    store, tmp_path, recovery
):
    editor = SpecificationEditor(store.saved_path, store.live_path, read_spec)
    runtime = DesignRuntime(store)
    editor.change(("parts", "button", "height"), 40)
    assert runtime.poll(force=True)
    before = deepcopy(editor.spec)
    store.saved_path.unlink()
    store.live_path.unlink()
    assert not editor.poll(force=True)
    assert not runtime.poll(force=True)
    assert editor.spec == runtime.spec == before
    assert "missing" in editor.error and "missing" in runtime.error
    with pytest.raises(SpecificationConflict):
        editor.change(("parts", "button", "radius"), 10)
    if recovery == "restore":
        store.write(before, save=True)
    else:
        imported = SpecificationStore(tmp_path / "import.json")
        imported.write(before, save=True)
        editor.import_file(imported.saved_path)
    editor.poll(force=True)
    runtime.poll(force=True)
    assert editor.spec == runtime.spec == before
    assert not editor.error and not runtime.error


def test_empty_specification_directory_starts_with_defaults_without_an_error(tmp_path):
    store = SpecificationStore(tmp_path / "spec.json")
    editor = SpecificationEditor(store.saved_path, store.live_path, read_spec)
    runtime = DesignRuntime(store)
    assert editor.spec == runtime.spec == DEFAULT
    assert not editor.error and not runtime.error
    assert editor.change(("parts", "button", "height"), 40)
    assert runtime.poll(force=True)
    assert runtime.spec == editor.spec


def test_store_distinguishes_bundled_defaults_from_a_saved_design_spec(tmp_path):
    store = SpecificationStore(tmp_path / "design-spec.json")
    assert store.latest()["source"] is None
    store.write(DEFAULT, save=True)
    assert store.latest()["source"] == "design-spec.json"


def test_unchanged_runtime_poll_reuses_validated_snapshots_without_copying(store, monkeypatch):
    from mojive.design import store as store_module

    runtime = DesignRuntime(store)

    def unexpected(*_args, **_kwargs):
        raise AssertionError("Unchanged runtime polling must not read, validate or copy the spec")

    monkeypatch.setattr(store_module.json, "load", unexpected)
    monkeypatch.setattr(store_module, "_validate_loaded_spec", unexpected)
    monkeypatch.setattr(store_module, "deepcopy", unexpected)
    assert not runtime.poll(force=True)


def test_cached_snapshots_keep_return_values_and_saved_state_independent(store):
    first = store.latest()
    first["spec"]["parts"]["button"]["height"] = 40
    first["savedSpec"]["parts"]["button"]["height"] = 44
    first["errors"].append({"source": "external", "error": "caller-owned"})
    second = store.latest()
    assert second["spec"] == second["savedSpec"] == DEFAULT
    assert not second["errors"]


def test_invalid_snapshot_is_cached_and_revalidated_after_repair(store, monkeypatch):
    from mojive.design import store as store_module

    store.live_path.write_text("{broken")
    load, reads = store_module.json.load, []

    def count_reads(stream):
        reads.append(stream.name)
        return load(stream)

    monkeypatch.setattr(store_module.json, "load", count_reads)
    first = store.latest()
    assert first["errors"]
    assert store.latest()["errors"] == first["errors"]
    assert len(reads) == 1
    value = deepcopy(DEFAULT)
    value["parts"]["button"]["height"] = 40
    store.write(value)
    assert store.latest()["spec"] == value
    assert not store.latest()["errors"]
    assert len(reads) == 2


def test_revision_first_runtime_poll_still_observes_draft_errors_and_recovery(store):
    runtime = DesignRuntime(store)
    revision = runtime.revision
    store.live_path.write_text("{broken")
    assert not runtime.poll(force=True)
    assert runtime.error and runtime.revision == revision
    store.live_path.unlink()
    assert not runtime.poll(force=True)
    assert not runtime.error and runtime.revision == revision


@pytest.mark.parametrize("operation", ("change", "save"))
@pytest.mark.parametrize("unavailable", ("invalid", "missing"))
def test_editor_refuses_to_merge_into_unreadable_files_and_can_resume_after_repair(
    store, operation, unavailable
):
    editor = SpecificationEditor(store.saved_path, store.live_path, read_spec)
    editor.change(("parts", "button", "height"), 40)
    before = deepcopy(editor.spec)
    history = list(editor.undo_stack)
    if unavailable == "invalid":
        store.saved_path.write_text("{broken saved")
        store.live_path.write_text("{broken live")
    else:
        store.saved_path.unlink()
        store.live_path.unlink()
    with pytest.raises(SpecificationConflict, match="No valid design specification"):
        if operation == "save":
            editor.save()
        else:
            editor.change(("parts", "button", "radius"), 10)
    assert editor.spec == before
    assert editor.undo_stack == history
    if unavailable == "invalid":
        assert store.saved_path.read_text() == "{broken saved"
        assert store.live_path.read_text() == "{broken live"
    else:
        assert not store.saved_path.exists()
        assert not store.live_path.exists()
    store.write(before, save=True)
    assert editor.change(("parts", "button", "radius"), 10)
    assert store.latest()["spec"]["parts"]["button"]["height"] == 40
    assert store.latest()["spec"]["parts"]["button"]["radius"] == 10


@pytest.mark.parametrize("unavailable", ("invalid", "missing"))
def test_import_replaces_the_entire_specification_and_recovers_unavailable_files(
    store, tmp_path, unavailable
):
    editor = SpecificationEditor(store.saved_path, store.live_path, read_spec)
    editor.change(("parts", "button", "height"), 40)
    imported = SpecificationStore(tmp_path / "import.json")
    value = deepcopy(DEFAULT)
    value["parts"]["button"]["radius"] = 10
    imported.write(value, save=True)
    if unavailable == "invalid":
        store.saved_path.write_text("{")
        store.live_path.write_text("{")
    else:
        store.saved_path.unlink()
        store.live_path.unlink()
    editor.import_file(imported.saved_path)
    assert editor.spec == value
    assert store.latest()["spec"] == value
    assert not editor.undo_stack and not editor.redo_stack
    assert not editor.pending_sync
    editor.save()
    assert read_spec(store.saved_path) == value


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


def test_editor_refreshes_atomic_drafts_with_the_same_timestamp_and_size(store):
    import os

    editor = SpecificationEditor(store.saved_path, store.live_path, read_spec)
    value = deepcopy(DEFAULT)
    value["parts"]["button"]["height"] = 40
    store.write(value)
    assert editor.poll(force=True)
    stamp = store.live_path.stat().st_mtime_ns
    size = store.live_path.stat().st_size
    value["parts"]["button"]["height"] = 44
    store.write(value)
    os.utime(store.live_path, ns=(stamp, stamp))
    assert store.live_path.stat().st_size == size
    assert editor.poll(force=True)
    assert editor.effective("button")["height"] == 44


@pytest.mark.parametrize("operation", ("change", "save"))
def test_editor_does_not_cache_an_external_snapshot_after_its_own_write(
    store, monkeypatch, operation
):
    editor = SpecificationEditor(store.saved_path, store.live_path, read_spec)
    if operation == "save":
        editor.change(("parts", "button", "height"), 40)
    merge = store.merge

    def publish_after_merge(*args, **kwargs):
        envelope = merge(*args, **kwargs)
        value = deepcopy(envelope["spec"])
        value["parts"]["button"]["height"] = 44
        store.write(value)
        return envelope

    monkeypatch.setattr(editor.store, "merge", publish_after_merge)
    if operation == "save":
        editor.save()
    else:
        editor.change(("parts", "button", "height"), 40)
    assert editor.effective("button")["height"] == 40
    assert editor.poll(force=True)
    assert editor.effective("button")["height"] == 44


def test_editor_rechecks_a_snapshot_replaced_during_poll(store, monkeypatch):
    editor = SpecificationEditor(store.saved_path, store.live_path, read_spec)
    value = deepcopy(DEFAULT)
    value["parts"]["button"]["height"] = 40
    store.write(value)
    latest = editor.store.latest

    def replace_after_read():
        envelope = latest()
        value["parts"]["button"]["height"] = 44
        store.write(value)
        return envelope

    with monkeypatch.context() as reading:
        reading.setattr(editor.store, "latest", replace_after_read)
        assert editor.poll(force=True)
        assert editor.effective("button")["height"] == 40
    assert editor.poll(force=True)
    assert editor.effective("button")["height"] == 44
