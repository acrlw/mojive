"""Prepared form layout styles retain the exact raw catalogue resolution contract."""

from copy import deepcopy

import pytest

from mojive.design.groups import group_scope
from mojive.design.native.geometry import NativePartStyle, part_scope
from mojive.design.native.presentation import DesignPresentation
from mojive.design.runtime import DesignRuntime
from mojive.design.store import SpecificationStore
from mojive.ui.workspace import numeric_fields
from mojive.ui.workspace.typography import TextRole, Typography
from tests.test_component_appearance import appearance as appearance

pytestmark = pytest.mark.parametrize("appearance", (1, 1.2, 1.25, 2.25), indirect=True)


def _fresh_style(ui, parts):
    return NativePartStyle(ui.design.runtime, parts, authored_defaults=True) if ui.design else None


def _form(ui, width, vector=True):
    return ui.form_layout(
        width * ui.s,
        3 if vector else 1,
        axis=vector,
        vector=vector,
        field_minimum=40 * ui.s,
        label_minimum=30 * ui.s,
    )


@pytest.mark.parametrize("vector", (False, True))
@pytest.mark.parametrize("width", (170, 640))
def test_cached_layout_matches_fresh_raw_styles_under_family_group_parent_and_typography_edits(
    appearance, monkeypatch, vector, width
):
    ui, runtime, update, frame = appearance
    update(
        {
            "form_row": {
                "marginLeft": 5,
                "marginRight": 7,
                "alignItems": "end",
                "fontSize": 25,
                "lineHeightMode": "fixed",
                "lineHeight": 31,
            },
            "vector_row": {"labelWidth": 87, "columnGap": 9, "rowGap": 11},
            "form_label": {"fontSize": 16, "marginTop": 2, "marginBottom": 3},
            "vector_group": {"marginLeft": 3, "columnGap": 7},
            "input_number": {"marginTop": 1, "marginBottom": 2},
        }
    )
    spec = deepcopy(runtime.spec)
    spec["components"]["inspector"] = {"body_font_size": 20, "numeric_gap": 13}
    runtime.update_spec(spec)
    ui.typography = Typography(numeric=TextRole("mono", 18))

    def draw():
        with (
            ui.design.family_scope("inspector"),
            group_scope("Inspector"),
            part_scope("form_row"),
        ):
            ui.design.primitives.family = "inspector"
            return _form(ui, width, vector)

    actual = frame(draw)
    assert frame(draw) == actual
    with monkeypatch.context() as fresh:
        fresh.setattr(ui, "_layout_style", lambda parts: _fresh_style(ui, parts))
        ui._form_layouts[2].clear()
        expected = frame(draw)
    assert actual == expected
    raw = ui._layout_style("form_label")
    assert raw.family is None and raw.authored_defaults
    assert raw.get("fontSize") == 16
    with pytest.raises(TypeError):
        raw.values["fontSize"] = 99


def test_warm_form_and_numeric_calls_reuse_their_raw_styles_and_revision_rebuilds(
    appearance, monkeypatch
):
    ui, runtime, _update, frame = appearance
    calls, constructor = [], numeric_fields.NativePartStyle

    def style(*args, **kwargs):
        calls.append(args[1])
        return constructor(*args, **kwargs)

    monkeypatch.setattr(numeric_fields, "NativePartStyle", style)
    for width in (640, 170, 640):
        frame(lambda width=width: _form(ui, width))
    assert len(calls) == 4
    runtime.update_spec(deepcopy(runtime.spec))
    frame(lambda: _form(ui, 640))
    assert len(calls) == 8


@pytest.mark.parametrize("vector", (False, True))
def test_warm_geometry_replays_the_same_actual_field_read_evidence(appearance, monkeypatch, vector):
    ui, runtime, _update, frame = appearance
    frame(lambda: _form(ui, 400, vector))
    runtime.track_field_reads = True
    runtime.begin_frame()
    actual = frame(lambda: _form(ui, 400, vector))
    reads = deepcopy(runtime.used_fields)
    assert reads and "form_row" in reads
    with monkeypatch.context() as fresh:
        fresh.setattr(ui, "_layout_style", lambda parts: _fresh_style(ui, parts))
        runtime.begin_frame()
        expected = frame(lambda: _form(ui, 400, vector))
        assert runtime.used_fields == reads
    assert actual == expected
    runtime.begin_frame()
    frame(lambda: _form(ui, 400, vector))
    assert runtime.used_fields == reads


def test_layout_cache_preserves_group_scoping_runtime_replacement_and_authored_mode(
    appearance, tmp_path
):
    ui, runtime, update, _frame = appearance
    update(
        {"inspector_section_body": {"marginLeft": 19}, "control_section_body": {"marginLeft": 7}}
    )
    with group_scope("Inspector"):
        inspector = ui._layout_style("section_body")
        assert inspector.edges("margin", ui.s)[0] == 19 * ui.s
    with group_scope("Control"):
        control = ui._layout_style("section_body")
        assert control is not inspector and control.edges("margin", ui.s)[0] == 7 * ui.s
    with group_scope("Inspector"):
        runtime.authored_defaults = False
        assert ui._layout_style("section_body") is inspector
        assert inspector.authored_defaults
    replacement = DesignRuntime(SpecificationStore(tmp_path / "replacement.json"))
    assert replacement.revision <= runtime.revision
    ui.design = DesignPresentation(ui, replacement)
    with group_scope("Inspector"):
        assert ui._layout_style("section_body") is not inspector
    ui.design = None
    assert ui._layout_style("section_body") is None


def test_raw_layout_cache_stays_bounded(appearance):
    ui, _runtime, _update, _frame = appearance
    for index in range(70):
        ui._layout_style(("unused-layout-" + str(index),))
    assert len(ui._numeric_layout_styles[2]) <= 64
    assert ui._layout_style("form_row") is ui._layout_style("form_row")


def test_form_geometry_reuses_warm_rows_and_invalidates_resizes_and_design_edits(appearance):
    ui, _runtime, update, frame = appearance
    initial = frame(lambda: _form(ui, 640))
    assert frame(lambda: _form(ui, 640)) is initial
    narrow = frame(lambda: _form(ui, 170))
    assert narrow != initial
    assert frame(lambda: _form(ui, 640)) is initial
    update({"vector_group": {"columnGap": 17}, "form_label": {"fontSize": 30}})
    edited = frame(lambda: _form(ui, 640))
    assert edited != initial
    assert frame(lambda: _form(ui, 640)) is edited
    for width in range(200, 340):
        frame(lambda width=width: _form(ui, width))
    assert len(ui._form_layouts[2]) <= 128
