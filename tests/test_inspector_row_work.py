"""Inspector row planning omits unused work while retaining native paint and evidence."""

from types import SimpleNamespace

import pytest
from imgui_bundle import imgui

from mojive.design.groups import group_scope
from mojive.ui.workspace.inspector_style import NativeInspectorPresentation
from mojive.ui.workspace.property_style import property_controls
from tests.test_component_appearance import appearance as appearance

pytestmark = pytest.mark.parametrize("appearance", (1, 1.2, 1.25, 2.25), indirect=True)


def _paint_row(ui, runtime, presenter, *, toggle=False, table="effort", width=420):
    runtime.begin_frame()
    ui.design.begin_frame()
    ui.hits.clear()
    ui.numeric_edits.clear()
    imgui.set_cursor_screen_pos((30 * ui.s, 30 * ui.s))
    imgui.begin_child("Inspector row", (width * ui.s, 450 * ui.s))
    try:
        with group_scope("Inspector"), ui.design.family_scope("inspector"), property_controls(ui):
            assert presenter.begin_table(table)
            try:
                presenter.row("Enabled" if toggle else "Mass", toggle=toggle)
                result = (
                    presenter.toggle("effort-value", True)
                    if toggle
                    else presenter.numeric("effort-value", 1.25, 0.01, 0, 10, "%.3f", 0)
                )
            finally:
                presenter.end_table()
            draw = ui.draw
            return (
                result,
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


@pytest.mark.parametrize("table", ("effort", "joint_properties_table"))
def test_toggle_rows_only_plan_the_shared_anchor_and_final_switch(appearance, monkeypatch, table):
    ui, runtime, _update, frame = appearance
    presenter = NativeInspectorPresentation(ui)
    presenter.ctx = SimpleNamespace(tr=str)
    requests, original = [], ui.form_layout

    def layout(*args, **kwargs):
        requests.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(ui, "form_layout", layout)
    for _ in range(3):
        requests.clear()
        snapshot = frame(lambda: _paint_row(ui, runtime, presenter, toggle=True, table=table))
    assert snapshot[0] == (False, True)
    assert "effort-value" in snapshot[1]
    assert len(requests) == (2 if table == "joint_properties_table" else 1)
    assert requests[-1][1]["field_part"] == "switch"
    assert "field_minimum" in requests[-1][1]
    if table == "joint_properties_table":
        assert requests[0][1] == {"axis": True, "minimum": 0}


def test_scalar_row_does_not_resolve_or_consume_unrelated_switch_geometry(appearance, monkeypatch):
    ui, runtime, _update, frame = appearance
    presenter = NativeInspectorPresentation(ui)
    presenter.ctx = SimpleNamespace(tr=str)
    runtime.track_field_reads = True
    original = ui.component_style

    def style(parts):
        assert "switch" not in ((parts,) if isinstance(parts, str) else parts)
        return original(parts)

    monkeypatch.setattr(ui, "component_style", style)
    for _ in range(3):
        snapshot = frame(lambda: _paint_row(ui, runtime, presenter))
    assert snapshot[0] == (False, 1.25)
    assert snapshot[2]["effort-value"].active is False
    assert "switch" not in runtime.used_fields


@pytest.mark.parametrize("toggle", (False, True))
@pytest.mark.parametrize(
    ("mode", "row_observed", "switch_observed"),
    (
        ("off", False, False),
        ("all", True, True),
        ("fields", True, True),
        ("row", True, False),
        ("switch", False, True),
        ("parent", True, True),
        ("unrelated", False, False),
    ),
)
def test_row_observation_is_lazy_and_preserves_exact_native_paint_hits_and_extent(
    appearance, monkeypatch, toggle, mode, row_observed, switch_observed
):
    ui, runtime, update, frame = appearance
    update(
        {
            "form_row": {"marginLeft": 3, "marginRight": 7},
            "switch": {"marginLeft": 4, "marginRight": 5, "switchWidth": 50},
        }
    )
    presenter = NativeInspectorPresentation(ui)
    presenter.ctx = SimpleNamespace(tr=str)
    attempts, original = [], ui.design.record

    def record(key, *args, **kwargs):
        if key == "effort/row/" + ("Enabled" if toggle else "Mass") or key.endswith("/label-row"):
            attempts.append(key)
        return original(key, *args, **kwargs)

    monkeypatch.setattr(ui.design, "record", record)

    def paint():
        attempts.clear()
        return _paint_row(ui, runtime, presenter, toggle=toggle, width=240)

    for _ in range(3):
        baseline = frame(paint)
    assert not attempts
    runtime.observe_components = mode == "all"
    runtime.track_field_reads = mode == "fields"
    selected = {
        "row": "form_row",
        "switch": "switch",
        "parent": "section_body",
        "unrelated": "camera_body",
    }.get(mode)
    runtime.workbench = SimpleNamespace(part=selected, pick=False) if selected else None
    for _ in range(3):
        actual = frame(paint)
    assert actual == baseline
    row_key = "effort/row/" + ("Enabled" if toggle else "Mass")
    expected = ([row_key] if row_observed else []) + (
        ["effort-value/label-row"] if toggle and switch_observed else []
    )
    assert attempts == expected
    records = {record["key"]: record for record in runtime.observed}
    for key in expected:
        assert key in records
        assert records[key]["labelRect"][2] >= 0
        assert records[key]["labelRect"][3] > 0
    if row_observed:
        assert records[row_key]["fields"]
