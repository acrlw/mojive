"""Production frames only materialize geometry records for active diagnostics."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
from imgui_bundle import imgui

from mojive.adapters.base import NodeType
from mojive.design.groups import group_scope
from mojive.design.native.geometry import Rect
from mojive.design.native.presentation import DesignPresentation
from mojive.design.runtime import DesignRuntime
from mojive.design.store import SpecificationStore
from mojive.ui.theme import THEME
from mojive.ui.workspace import navigation as navigation_module
from mojive.ui.workspace.navigation import Navigation
from mojive.ui.workspace.widgets import UI


@pytest.fixture(params=(1.0, 1.5))
def observation_ui(request, tmp_path):
    context = imgui.create_context()
    io = imgui.get_io()
    io.set_ini_filename(None)
    io.display_size = (900 * request.param, 800 * request.param)
    io.delta_time = 1 / 60
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    ui = UI(
        SimpleNamespace(style_scale=request.param, font_report=SimpleNamespace(cjk_path=None)),
        portable_fonts=True,
    )
    runtime = DesignRuntime(SpecificationStore(tmp_path / "design.json"))
    ui.design = DesignPresentation(ui, runtime)

    def frame(paint):
        imgui.new_frame()
        imgui.set_next_window_pos((0, 0))
        imgui.set_next_window_size(io.display_size)
        imgui.begin("Observation payloads", flags=imgui.WindowFlags_.no_title_bar)
        ui.hits.clear()
        ui.numeric_edits.clear()
        runtime.observed.clear()
        ui.design.begin_frame()
        try:
            paint()
            vertices = tuple((v.pos.x, v.pos.y, v.uv.x, v.uv.y, v.col) for v in ui.draw.vtx_buffer)
            indices = tuple(ui.draw.idx_buffer)
            return dict(ui.hits), vertices, indices
        finally:
            imgui.end()
            imgui.render()

    try:
        frame(lambda: None)
        yield ui, runtime, frame
    finally:
        imgui.destroy_context(context)


def test_shared_controls_skip_records_without_changing_paint_or_hits(observation_ui, monkeypatch):
    ui, runtime, frame = observation_ui
    primitives = ui.design.primitives
    recorded = []
    record = primitives.record

    def capture(key, *args, **kwargs):
        recorded.append(key)
        return record(key, *args, **kwargs)

    monkeypatch.setattr(primitives, "record", capture)

    def rect(row):
        return Rect(40 * ui.s, (40 + row * 60) * ui.s, 240 * ui.s, 40 * ui.s)

    def paint():
        primitives.button("button", rect(0), "Camera", icon="camera")
        primitives.input_text("search", rect(1), "Search", icon="search")
        primitives.input_number("number", rect(2), 0.5, axis="X")
        primitives.select("select", rect(3), "First", ("First", "Second"))
        primitives.checkbox("checkbox", rect(4).x, rect(4).y, True, label="Visible")
        primitives.switch("switch", rect(5), True, label="Enabled")
        primitives.range("range", rect(6), 0.5, 0, 1)
        primitives.label("label", rect(7), "Position", unit="m", compound_parts=("vector",))
        primitives.number_shell("shell", rect(8), 0.5, unit="m")

    frame(paint)
    frame(paint)
    disabled = frame(paint)
    assert recorded == []
    assert primitives.observed == []
    assert runtime.observed == []
    runtime.observe_components = True
    enabled = frame(paint)
    assert enabled == disabled
    assert set(recorded) >= {
        "button",
        "search",
        "number",
        "select",
        "checkbox",
        "switch",
        "range",
        "label",
        "shell-shell",
    }
    records = {row["key"]: row for row in primitives.observed}
    assert records["label"]["compoundParts"] == ("vector",)
    assert records["button"]["icon"] == "camera"
    assert records["button"]["label"] == "Camera"
    assert records["number"]["axisWidth"] > 0
    assert records["shell-shell"]["unit"] == "m"
    assert any(row["key"] == "label/unit" for row in runtime.observed)
    assert all(row["hit"] == enabled[0][row["key"]] for row in records.values())


def test_disabled_control_observation_does_not_evaluate_diagnostic_fields(observation_ui):
    ui, _runtime, frame = observation_ui

    class UnusedParts:
        def __iter__(self):
            raise AssertionError("Disabled diagnostics evaluated compound parts")

    def paint():
        ui.design.primitives.label(
            "label", Rect(40, 40, 180, 28), "Position", compound_parts=UnusedParts()
        )

    hits, vertices, _indices = frame(paint)
    assert hits["label"][2] > 0
    assert vertices


def test_scene_payloads_are_lazy_and_preserve_scoped_observation(observation_ui, monkeypatch):
    ui, runtime, frame = observation_ui
    navigation = Navigation.__new__(Navigation)
    navigation.workspace = SimpleNamespace(ui=ui, s=ui.s)
    navigation.hierarchy = SimpleNamespace(_batch_selected=set())
    monkeypatch.setattr(navigation, "_scene_context", lambda *args: None)
    node = SimpleNamespace(node_id=7, name="Entity", type=NodeType.GEOM, visible=True)
    ctx = SimpleNamespace(
        session=SimpleNamespace(selected_node=None),
        theme=THEME,
        input_bindings=None,
        input_claim=None,
    )
    payloads = []

    def payload_list(value):
        assert runtime.observes(("tree_row", "scene_tree_row"))
        result = list(value)
        payloads.append(result)
        return result

    monkeypatch.setattr(navigation_module, "list", payload_list, raising=False)

    def paint():
        imgui.set_cursor_screen_pos((40 * ui.s, 40 * ui.s))
        with group_scope("Scene"):
            navigation.draw_scene_row(ctx, node, 1, True, 350 * ui.s)

    frame(paint)
    frame(paint)
    disabled = frame(paint)
    assert payloads == []
    assert runtime.observed == []
    # A selected group-local part enables records without the global observation switch.
    runtime.workbench = SimpleNamespace(part="scene_tree_row", pick=False)
    enabled = frame(paint)
    assert enabled == disabled
    assert payloads
    row = next(row for row in runtime.observed if row["key"] == "tree-row-7")
    assert row["parts"] == ("tree_row", "scene_tree_row")
    assert row["depth"] == 1
    assert row["selected"] is False
    assert row["outer"] == payloads[0]
    assert set(row["fieldRegions"]) == {"gap", "columnGap"}
    assert all(len(regions) == 2 for regions in row["fieldRegions"].values())


def test_scene_metrics_reuse_one_frame_without_crossing_style_scopes(observation_ui, monkeypatch):
    ui, runtime, frame = observation_ui
    navigation = Navigation.__new__(Navigation)
    navigation.workspace = SimpleNamespace(ui=ui, s=ui.s)
    spec = deepcopy(runtime.spec)
    spec["parts"].setdefault("tree_row", {})["rowGap"] = 3
    spec["parts"].setdefault("scene_tree_row", {})["rowGap"] = 13
    runtime.update_spec(spec)
    styles = []
    component_style = ui.component_style

    def resolve(parts):
        styles.append(parts)
        return component_style(parts)

    monkeypatch.setattr(ui, "component_style", resolve)
    previous = []

    def paint():
        geometry = navigation._scene_row_geometry()
        metrics = navigation._row_metrics
        resolved = len(styles)
        for _ in range(20):
            assert navigation._scene_row_geometry() is geometry
            assert navigation._row_metrics is metrics
        assert len(styles) == resolved
        assert metrics.row_gap == 3
        assert not previous or metrics is not previous[-1]
        previous.append(metrics)
        with group_scope("Scene"):
            navigation._scene_row_geometry()
            assert navigation._row_metrics.row_gap == 13
            grouped_metrics = navigation._row_metrics
            with ui.corner_scope(2 * ui.s, "icon_button"):
                navigation._scene_row_geometry()
                corner_metrics = navigation._row_metrics
                assert corner_metrics is not grouped_metrics
                with ui.component_scope("tree_row"):
                    style = ui.component_style("icon_button")
                expected = style.get("radius") if style.explicit("radius") else None
                assert corner_metrics.visibility_radius == expected
            navigation._scene_row_geometry()
            assert navigation._row_metrics is not corner_metrics
            assert navigation._row_metrics == grouped_metrics
        navigation._scene_row_geometry()
        assert navigation._row_metrics.row_gap == 3

    frame(paint)
    frame(paint)
