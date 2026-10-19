"""Optional property evidence leaves native geometry and editing state unchanged."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
from imgui_bundle import imgui

from mojive.design.groups import group_part, group_scope
from mojive.ui.workspace.property_style import NativePropertyPresentation, property_controls
from tests.test_component_appearance import appearance as appearance

pytestmark = pytest.mark.parametrize("appearance", (1, 1.2, 1.25, 2.25), indirect=True)


def _paint(ui, runtime, presenter):
    runtime.begin_frame()
    ui.design.begin_frame()
    ui.hits.clear()
    ui.numeric_edits.clear()
    imgui.set_cursor_screen_pos((30 * ui.s, 30 * ui.s))
    imgui.begin_child("Property diagnostics", (350 * ui.s, 600 * ui.s))
    try:
        with group_scope("Inspector"), ui.design.family_scope("inspector"), property_controls(ui):
            with ui.component_scope("section_body"):
                ui.component_box(
                    "button", 40 * ui.s, 40 * ui.s, 100 * ui.s, 30 * ui.s, key="local-box"
                )
            assert presenter.begin_table("properties")
            try:
                presenter.row("Mass")
                number = presenter.numeric("mass", 1.25, 0.01, 0, 10, "%.3f")
                presenter.readonly_value("Source", "Authored value\nSecond line")
            finally:
                presenter.end_table()
            p = imgui.get_cursor_screen_pos()
            vector = ui.vector("position", p.x, p.y, 300 * ui.s, "Position", (1, 2, 3), "m")
            draw = ui.draw
            return (
                number,
                vector,
                dict(ui.hits),
                dict(ui.numeric_edits),
                tuple(imgui.internal.get_current_window().dc.cursor_max_pos),
                tuple(
                    (tuple(vertex.pos), tuple(vertex.uv), vertex.col) for vertex in draw.vtx_buffer
                ),
                tuple(draw.idx_buffer),
            )
    finally:
        imgui.end_child()


ALL_RECORDS = {
    "local-box",
    "properties/row/Mass",
    "properties/row/Source",
    "property-value-Source",
    "section/properties",
    "section-body/properties",
    "position/controls",
    "position/row",
}


@pytest.mark.parametrize(
    ("mode", "expected"),
    (
        ("off", set()),
        ("all", ALL_RECORDS),
        ("fields", ALL_RECORDS),
        ("parent", ALL_RECORDS),
        ("row", {"properties/row/Mass", "properties/row/Source", "position/row"}),
        ("group", ALL_RECORDS),
        ("section", {"section/properties"}),
        ("control", {"local-box"}),
        ("unrelated", set()),
    ),
)
def test_property_observation_preserves_exact_paint_hits_edits_and_extent(
    appearance, monkeypatch, mode, expected
):
    ui, runtime, update, frame = appearance
    update(
        {
            "form_row": {"marginLeft": 3, "marginRight": 7},
            "section_body": {"paddingLeft": 4, "paddingRight": 5, "paddingBottom": 3},
        }
    )
    presenter = NativePropertyPresentation(ui)
    attempts, measures = [], []
    record, measure = ui.design.record, ui.measure

    def record_attempt(key, *args, **kwargs):
        if key in ALL_RECORDS:
            attempts.append(key)
        return record(key, *args, **kwargs)

    def measured(text, *args, **kwargs):
        if text in ("Authored value", "Second line"):
            measures.append(text)
        return measure(text, *args, **kwargs)

    monkeypatch.setattr(ui.design, "record", record_attempt)
    monkeypatch.setattr(ui, "measure", measured)

    def paint():
        attempts.clear()
        measures.clear()
        return _paint(ui, runtime, presenter)

    for _ in range(3):
        baseline = frame(paint)
    assert not attempts and not measures
    runtime.observe_components = mode == "all"
    runtime.track_field_reads = mode == "fields"
    part = {
        "parent": "section_body",
        "row": "form_row",
        "group": group_part("Inspector", "section_body"),
        "section": "section",
        "control": "section_body_button",
        "unrelated": "camera_body",
    }.get(mode)
    runtime.workbench = SimpleNamespace(part=part, pick=False) if part else None
    for _ in range(3):
        actual = frame(paint)
    assert actual == baseline
    assert set(attempts) == expected
    assert len(attempts) == len(expected)
    observed = {item["key"]: item for item in runtime.observed}
    assert expected <= observed.keys()
    if "property-value-Source" in expected:
        assert measures == ["Authored value", "Second line"]
        assert len(observed["property-value-Source"]["textRects"]) == 2
    else:
        assert not measures


def test_unobserved_numeric_group_publishes_hits_without_building_field_payload(appearance):
    ui, runtime, _update, frame = appearance

    class Fields(tuple):
        def __iter__(self):
            iterations.append(True)
            return super().__iter__()

    iterations = []

    def paint():
        runtime.begin_frame()
        layout = ui.numeric_layout(300 * ui.s, 3)
        layout = replace(layout, fields=Fields(layout.fields))
        iterations.clear()
        ui.record_numeric_group("actual-group", 20, 40, 300 * ui.s, layout)
        return ui.hits["actual-group"]

    before = frame(paint)
    assert not iterations and not runtime.observed
    runtime.observe_components = True
    assert frame(paint) == before
    assert iterations == [True]
    assert len(runtime.observed[-1]["fields"]) == 3
