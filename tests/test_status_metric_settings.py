"""Status metric spacing and width controls share live specification history."""

from copy import deepcopy

import pytest

from mojive.design.editor import MISSING, SpecificationEditor
from mojive.design.help import get_field_help, get_part_help
from mojive.design.native.geometry import NativePartStyle
from mojive.design.runtime import DesignRuntime
from mojive.design.specification import DEFAULT, component_catalog
from mojive.design.store import SpecificationStore, read_spec
from mojive.design.support import native_support

METRIC_DEFAULTS = {
    "labelGap": 4,
    "unitGap": 2,
    "deltaValueWidth": 0,
    "physicsValueWidth": 0,
    "stepsValueWidth": 0,
    "timeValueWidth": 0,
    "renderValueWidth": 0,
    "tooltipPaddingX": 10,
    "tooltipPaddingY": 8,
    "tooltipRowGap": 6,
}


@pytest.fixture
def metric_editor(tmp_path):
    store = SpecificationStore(tmp_path / "status-design.json")
    store.write(DEFAULT, save=True)
    editor = SpecificationEditor(store.saved_path, store.live_path, read_spec)
    return store, editor, DesignRuntime(store)


def test_numeric_controls_are_discoverable_as_production_settings():
    fields = component_catalog()["status_metrics"]["fields"]
    support = native_support()["status_metrics"]["fields"]
    help_fields = get_part_help("status_metrics")["fields"]
    for field in METRIC_DEFAULTS:
        assert fields[field]["group"] == ("悬停提示" if field.startswith("tooltip") else "数字排版")
        assert fields[field]["css"] == []
        assert support[field]["status"] == "applied"
        assert support[field]["consumers"] == [
            {
                "source": "ui/workspace/status_components.py",
                "description": "生产组件尺寸与排版",
            }
        ]
        assert get_field_help("status_metrics", field) == help_fields[field]
        assert help_fields[field]["unit"] == "逻辑像素"
        assert help_fields[field]["summary"] and help_fields[field]["reference"]


def test_time_format_round_trip_undo_and_reset_keep_existing_custom_width(metric_editor):
    store, editor, runtime = metric_editor
    definition = component_catalog()["status_time"]["fields"]["displayFormat"]
    assert definition["choices"] == ["seconds", "clock"]
    assert definition["default"] == "seconds"
    assert native_support()["status_time"]["fields"]["displayFormat"]["status"] == "applied"
    assert "99:59:59" in get_field_help("status_time", "displayFormat")["summary"]
    assert editor.change(("parts", "status_time", "valueWidth"), 92)
    assert editor.change(("parts", "status_time", "displayFormat"), "clock")
    editor.save()
    assert runtime.poll(force=True)
    assert NativePartStyle(runtime, "status_time").get("displayFormat") == "clock"
    assert read_spec(store.saved_path)["parts"]["status_time"] == {
        "valueWidth": 92,
        "displayFormat": "clock",
    }
    assert editor.undo()
    assert runtime.poll(force=True)
    time = NativePartStyle(runtime, "status_time")
    assert time.get("displayFormat") == "seconds" and time.get("valueWidth") == 92
    assert editor.redo()
    assert editor.change(("parts", "status_time", "displayFormat"), MISSING)
    assert runtime.poll(force=True)
    assert NativePartStyle(runtime, "status_time").get("displayFormat") == "seconds"
    for name in ("status_steps", "status_physics", "status_render"):
        assert editor.spec["parts"].get(name) == DEFAULT["parts"].get(name)


@pytest.mark.parametrize("part", ("backend", "steps", "time", "delta", "physics", "render"))
def test_divider_sides_inherit_common_spacing_and_keep_local_overrides(metric_editor, part):
    store, editor, runtime = metric_editor
    name = "status_" + part
    baseline = NativePartStyle(runtime, name).values
    assert baseline["leftGap"] == (0 if part == "backend" else 6)
    assert baseline["rightGap"] == (0 if part == "render" else 6)
    assert editor.change(("parts", "status_metrics", "columnGap"), 9)
    assert runtime.poll(force=True)
    inherited = NativePartStyle(runtime, name).values
    assert inherited["leftGap"] == (0 if part == "backend" else 9)
    assert inherited["rightGap"] == (0 if part == "render" else 9)
    for field, value in (("leftGap", 3), ("rightGap", 7), ("paddingRight", 5)):
        assert native_support()[name]["fields"][field]["status"] == "applied"
        assert editor.change(("parts", name, field), value)
    editor.save()
    assert runtime.poll(force=True)
    local = NativePartStyle(runtime, name).values
    assert (local["leftGap"], local["rightGap"], local["paddingRight"]) == (3, 7, 5)
    assert read_spec(store.saved_path)["parts"][name] == {
        "leftGap": 3,
        "rightGap": 7,
        "paddingRight": 5,
    }
    assert editor.undo()
    assert runtime.poll(force=True)
    undone = NativePartStyle(runtime, name).values
    assert (undone["leftGap"], undone["rightGap"], undone["paddingRight"]) == (3, 7, 2)


@pytest.mark.parametrize(
    "field,value",
    [
        ("labelGap", 0),
        ("unitGap", 1),
        ("deltaValueWidth", 25),
        ("physicsValueWidth", 18),
        ("stepsValueWidth", 32),
        ("timeValueWidth", 80),
        ("renderValueWidth", 32),
        ("tooltipPaddingX", 14),
        ("tooltipPaddingY", 10),
        ("tooltipRowGap", 8),
    ],
)
def test_live_edits_round_trip_reset_and_undo_only_the_metric_group(metric_editor, field, value):
    store, editor, runtime = metric_editor
    before = NativePartStyle(runtime, "status_metrics")
    assert {name: before.values[name] for name in METRIC_DEFAULTS} == METRIC_DEFAULTS

    path = ("parts", "status_metrics", field)
    assert editor.change(path, value)
    assert runtime.poll(force=True)
    values = NativePartStyle(runtime, "status_metrics").values
    assert values[field] == value
    assert {name: values[name] for name in METRIC_DEFAULTS if name != field} == {
        name: default for name, default in METRIC_DEFAULTS.items() if name != field
    }
    assert editor.spec["tokens"] == DEFAULT["tokens"]
    for group in ("statusbar", "status_context", "status_hints"):
        assert editor.spec["parts"].get(group) == DEFAULT["parts"].get(group)

    editor.save()
    assert read_spec(store.saved_path)["parts"]["status_metrics"][field] == value
    assert editor.change(path, MISSING)
    assert runtime.poll(force=True)
    assert NativePartStyle(runtime, "status_metrics").values[field] == METRIC_DEFAULTS[field]
    assert editor.undo()
    assert runtime.poll(force=True)
    assert NativePartStyle(runtime, "status_metrics").values[field] == value
    assert editor.redo()
    assert runtime.poll(force=True)
    assert NativePartStyle(runtime, "status_metrics").values[field] == METRIC_DEFAULTS[field]


@pytest.mark.parametrize("field", METRIC_DEFAULTS)
def test_numeric_width_zero_and_upper_bounds_survive_store_round_trip(metric_editor, field):
    store, editor, runtime = metric_editor
    maximum = component_catalog()["status_metrics"]["fields"][field]["max"]
    for value in (maximum, 0):
        assert editor.change(("parts", "status_metrics", field), value)
        editor.save()
        assert runtime.poll(force=True)
        assert read_spec(store.saved_path)["parts"]["status_metrics"][field] == value
        assert NativePartStyle(runtime, "status_metrics").values[field] == value


@pytest.mark.parametrize("field", METRIC_DEFAULTS)
@pytest.mark.parametrize("invalid", ["negative", "overflow", "fraction", "boolean", "nan"])
def test_invalid_numeric_layout_write_preserves_other_valid_edits(metric_editor, field, invalid):
    store, editor, _runtime = metric_editor
    maximum = component_catalog()["status_metrics"]["fields"][field]["max"]
    value = {
        "negative": -1,
        "overflow": maximum + 1,
        "fraction": 0.5,
        "boolean": True,
        "nan": float("nan"),
    }[invalid]
    before = store.latest()
    candidate = deepcopy(editor.spec)
    candidate["parts"]["button"]["height"] = 40
    candidate["parts"].setdefault("status_metrics", {})[field] = value
    with pytest.raises(ValueError):
        store.write(candidate, save=True)
    assert store.latest() == before
    assert not store.live_path.exists()


def test_guide_status_editor_displays_inherited_spacing_without_zero_fallback(
    metric_editor, monkeypatch
):
    from types import SimpleNamespace

    from examples.ui_design.native import guide_panels, guide_slices

    _store, editor, runtime = metric_editor
    assert editor.change(("parts", "status_metrics", "columnGap"), 9)
    assert editor.change(("parts", "status_physics", "paddingRight"), 7)
    assert runtime.poll(force=True)
    sliders = {}

    def slider(key, value, *_args):
        sliders[key.removeprefix("##status-")] = value
        return False, value

    monkeypatch.setattr(guide_panels, "telemetry", lambda *_args: None)
    monkeypatch.setattr(guide_slices.g, "set_cursor_screen_pos", lambda *_args: None)
    monkeypatch.setattr(guide_slices.g, "set_next_item_width", lambda *_args: None)
    monkeypatch.setattr(guide_slices.g, "slider_int", slider)
    viewer = SimpleNamespace(
        width=800,
        s=1,
        values={},
        text=lambda *_args, **_options: None,
        record=lambda *_args: None,
        rect=lambda x, y, width, height: SimpleNamespace(x=x, y=y, width=width, height=height),
        viewer_select=lambda _key, _x, _y, _width, value, _choices: (False, value),
        ui=SimpleNamespace(
            component_style=lambda part: NativePartStyle(runtime, part, authored_defaults=True)
        ),
    )
    guide_slices.status(viewer)
    assert sliders == {
        "paddingLeft": 2,
        "paddingRight": 7,
        "leftGap": 9,
        "rightGap": 9,
        "valueWidth": 0,
        "labelGap": 4,
        "unitGap": 2,
    }
