"""Production workspace controls exercised through ImGui without a GPU window."""

from types import SimpleNamespace

import pytest
from imgui_bundle import imgui

from mojive.ui.workspace.design import DESIGN, MONO_SIZE
from mojive.ui.workspace.widgets import UI, NumericEditState


@pytest.fixture(params=(1.0, 2.0))
def native_ui(request):
    context = imgui.create_context()
    io = imgui.get_io()
    io.set_ini_filename(None)
    io.display_size = (800 * request.param, 500 * request.param)
    io.delta_time = 1 / 60
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    ui = UI(
        SimpleNamespace(style_scale=request.param, font_report=SimpleNamespace(cjk_path=None)),
        portable_fonts=True,
    )
    state = SimpleNamespace(ui=ui, io=io, value=1.0, disabled=False)

    def frame(count=1):
        for _ in range(count):
            imgui.new_frame()
            imgui.set_next_window_pos((0, 0))
            imgui.set_next_window_size(io.display_size)
            imgui.begin("Production controls", flags=imgui.WindowFlags_.no_title_bar)
            ui.hits.clear()
            ui.numeric_edits.clear()
            imgui.begin_disabled(state.disabled)
            state.changed, state.value = ui.input(
                "number",
                40 * ui.s,
                40 * ui.s,
                110 * ui.s,
                state.value,
                axis="X",
                speed=0.1,
                reset_value=0.25,
            )
            imgui.end_disabled()
            ui.button("icon", 180 * ui.s, 40 * ui.s, icon="camera")
            ui.segmented("mode", 40 * ui.s, 100 * ui.s, 180, ("First", "Second"), 0)
            ui.switch("switch", 240 * ui.s, 100 * ui.s, True)
            imgui.end()
            imgui.render()

    def event(method, *args):
        getattr(io, method)(*args)
        frame()

    def point(key="number"):
        x, y, width, height = ui.hits[key]
        return x + width / 2, y + height / 2

    state.frame, state.event, state.point = frame, event, point
    try:
        frame(3)
        yield state
    finally:
        imgui.destroy_context(context)


def test_production_controls_keep_one_hit_and_edit_state_owner(native_ui):
    state, ui = native_ui, native_ui.ui
    assert set(ui.hits) >= {"number", "number-reset", "icon", "mode", "switch"}
    for key in ("number", "icon", "mode", "switch"):
        assert ui.hits[key][3] == pytest.approx(DESIGN.control_height * ui.s, abs=1)
    x, y = state.point()
    state.event("add_mouse_pos_event", x, y)
    state.event("add_mouse_button_event", 0, True)
    assert isinstance(ui.numeric_edits["number"], NumericEditState)
    assert ui.numeric_edits["number"].activated
    state.event("add_mouse_pos_event", x + 25 * ui.s, y)
    assert state.changed and state.value > 1.0
    state.event("add_mouse_button_event", 0, False)
    assert ui.numeric_edits["number"].committed
    state.event("add_mouse_pos_event", *state.point("number-reset"))
    state.event("add_mouse_button_event", 0, True)
    state.event("add_mouse_button_event", 0, False)
    assert state.value == 0.25 and ui.numeric_edits["number"].reset


@pytest.mark.parametrize("key,hint", (("text", ""), ("##text", "Search")))
def test_text_input_keeps_its_id_out_of_the_visible_field_bounds(native_ui, key, hint):
    ui = native_ui.ui
    imgui.new_frame()
    imgui.set_next_window_size((400 * ui.s, 200 * ui.s))
    imgui.begin("Text field bounds")
    try:
        changed, value = ui.input_text(key, 40 * ui.s, 40 * ui.s, 120 * ui.s, "Value", hint=hint)
        assert not changed and value == "Value"
        assert ui.hits[key] == pytest.approx((40 * ui.s, 40 * ui.s, 120 * ui.s, 28 * ui.s))
    finally:
        imgui.end()
        imgui.render()


def test_production_text_entry_cancellation_and_disabled_input(native_ui):
    state, ui = native_ui, native_ui.ui
    modifier = imgui.Key.mod_super if state.io.config_mac_osx_behaviors else imgui.Key.mod_ctrl
    state.event("add_key_event", modifier, True)
    state.event("add_mouse_pos_event", *state.point())
    state.event("add_mouse_button_event", 0, True)
    state.event("add_mouse_button_event", 0, False)
    state.event("add_key_event", modifier, False)
    state.event("add_input_characters_utf8", "5.25")
    state.event("add_key_event", imgui.Key.escape, True)
    assert state.value == 1.0 and ui.numeric_edits["number"].canceled
    state.event("add_key_event", imgui.Key.escape, False)
    state.disabled = True
    x, y = state.point()
    state.event("add_mouse_pos_event", x, y)
    state.event("add_mouse_button_event", 0, True)
    state.event("add_mouse_pos_event", x + 25 * ui.s, y)
    state.event("add_mouse_button_event", 0, False)
    assert state.value == 1.0 and not state.changed


@pytest.mark.parametrize("g3,optical", ((True, True), (True, False), (False, True)))
def test_production_switch_paints_with_live_capsule_settings(native_ui, tmp_path, g3, optical):
    from mojive.design.native.presentation import DesignPresentation
    from mojive.design.runtime import DesignRuntime
    from mojive.design.store import SpecificationStore

    runtime = DesignRuntime(SpecificationStore(tmp_path / "design.json"))
    native_ui.ui.design = DesignPresentation(native_ui.ui, runtime)
    runtime.spec["native"].update(g3=g3, g3_capsule_optical=optical)
    for smoothing in (0.0, 0.16, 0.618, 1.0):
        runtime.spec["native"]["g3_capsule_smoothing"] = smoothing
        native_ui.frame(2)
        assert native_ui.ui.hits["switch"][2] > 0


def test_numeric_measurement_reuses_alternating_formats(native_ui, monkeypatch):
    ui, calls = native_ui.ui, []
    measure = imgui.calc_text_size

    def count(text, *args, **kwargs):
        calls.append(text)
        return measure(text, *args, **kwargs)

    monkeypatch.setattr(imgui, "calc_text_size", count)
    short = ui.numeric_minimum(format="%.3f", sample_values=(1.0,))
    precise = ui.numeric_minimum(format="%.7f", sample_values=(1.0,))
    measured = len(calls)
    assert measured > 0
    for _ in range(10):
        assert ui.numeric_minimum(format="%.3f", sample_values=(1.0,)) == short
        assert ui.numeric_minimum(format="%.7f", sample_values=(1.0,)) == precise
    assert precise > short
    assert len(calls) == measured
    without_axis = ui.numeric_minimum(axis=False, format="%.3f", sample_values=(1.0,))
    assert without_axis + 18 * ui.s == short
    assert len(calls) == measured


def test_numeric_measurement_tracks_value_font_scale_and_framebuffer(native_ui, monkeypatch):
    ui = native_ui.ui
    measure = imgui.calc_text_size
    calls = []

    def count(text, *args, **kwargs):
        calls.append(text)
        return measure(text, *args, **kwargs)

    monkeypatch.setattr(imgui, "calc_text_size", count)
    baseline = ui.numeric_minimum(format="%.3f", sample_values=(1.0,))
    measured = len(calls)
    wider = ui.numeric_minimum(format="%.3f", sample_values=(1234567890123.0,))
    assert wider > baseline and len(calls) > measured
    measured = len(calls)
    original_font = ui.mono
    ui.mono = ui.body
    changed_font = ui.numeric_minimum(format="%.3f", sample_values=(1.0,))
    assert len(calls) > measured
    measured = len(calls)
    imgui.push_font(ui.body, MONO_SIZE * ui.s)
    expected = max(measure(value).x for value in ("0.123", "-999.999", "-0.012", "1.000"))
    imgui.pop_font()
    assert expected + 34 * ui.s <= changed_font <= expected + 34 * ui.s + 1
    ui.mono = original_font
    ui.s *= 1.5
    scaled = ui.numeric_minimum(format="%.3f", sample_values=(1.0,))
    assert scaled > baseline and len(calls) > measured
    measured = len(calls)
    native_ui.io.display_framebuffer_scale = (2, 2)
    ui.numeric_minimum(format="%.3f", sample_values=(1.0,))
    assert len(calls) > measured
    for index in range(100):
        ui.numeric_minimum(format="%.3f", sample_values=(index,))
    assert len(ui._numeric_measurements) <= 64


def test_status_slot_measurements_reuse_font_metrics(native_ui, monkeypatch):
    from mojive.ui.workspace.status_components import StatusGroups

    ui, calls = native_ui.ui, []
    measure = imgui.calc_text_size

    def count(text, *args, **kwargs):
        calls.append(text)
        return measure(text, *args, **kwargs)

    monkeypatch.setattr(imgui, "calc_text_size", count)
    groups = StatusGroups(ui)
    width = groups.text_budget("Render 8888.8 FPS")
    measured = len(calls)
    assert measured > 0
    for _ in range(20):
        assert StatusGroups(ui).text_budget("Render 8888.8 FPS") == width
    assert len(calls) == measured
    original = ui.component_style
    style = original("status_metrics")
    enlarged = SimpleNamespace(
        get=lambda field, fallback=None: 24 if field == "fontSize" else style.get(field, fallback)
    )
    monkeypatch.setattr(
        ui, "component_style", lambda part: enlarged if part == "status_metrics" else original(part)
    )
    assert groups.text_budget("Render 8888.8 FPS") > width
    assert len(calls) > measured


def test_status_metric_spacing_has_its_own_compact_default(native_ui):
    from mojive.ui.workspace.status_components import StatusGroups

    groups = StatusGroups(native_ui.ui)
    assert groups.gap("status_metrics") == 6 * native_ui.ui.s
    assert groups.gap("status_context") == 24 * native_ui.ui.s


def test_vector_layout_reuses_geometry_until_its_budget_changes(native_ui, monkeypatch):
    ui, calls = native_ui.ui, []
    form_layout = ui.form_layout

    def count(*args, **kwargs):
        calls.append((args, kwargs))
        return form_layout(*args, **kwargs)

    monkeypatch.setattr(ui, "form_layout", count)
    layout = ui.vector_layout(330 * ui.s, format="%.6g", key="transform")
    for _ in range(20):
        assert ui.vector_layout(330 * ui.s, format="%.6g", key="transform") is layout
    assert len(calls) == 1
    assert ui.vector_layout(331 * ui.s, format="%.6g", key="transform") is not layout
    assert len(calls) == 2
    ui.vector_layout(331 * ui.s, format="%.6g", key="transform", minimum=150 * ui.s)
    assert len(calls) == 3


def test_material_inspector_rgba_fields_consume_their_own_typography(
    native_ui, tmp_path, monkeypatch
):
    from copy import deepcopy

    from mojive import Scene
    from mojive import commands as cmd
    from mojive.adapters.static import StaticSceneAdapter
    from mojive.design.native.presentation import DesignPresentation
    from mojive.design.runtime import DesignRuntime
    from mojive.design.specification import DEFAULT
    from mojive.design.store import SpecificationStore
    from mojive.render.backend import NullBackend
    from mojive.session import Session
    from mojive.ui.app import ViewerApp
    from mojive.ui.imgui_draw import ImguiDraw2D
    from mojive.ui.panels import PanelContext
    from mojive.ui.workspace.entities import EntityProjection
    from mojive.ui.workspace.inspector import inspector
    from mojive.ui.workspace.menus import Menus
    from mojive.ui.workspace.shell import WorkspaceShell

    monkeypatch.setenv("MOJIVE_SETTINGS", str(tmp_path / "settings.json"))
    ui = native_ui.ui
    scene = Scene()
    shape = scene.box(name="colored", color=(0.125, 0.25, 0.5, 0.75))
    session = Session(StaticSceneAdapter(scene))
    app = ViewerApp(session, NullBackend())
    node = session.node_by_object_id(shape.object_id)
    assert session.submit(cmd.SelectNode(node.node_id)).ok
    projection = EntityProjection(app)
    app.entity, app.entities, app.initial_entities = (
        projection.entity,
        projection.entities,
        projection.initial_entities,
    )
    app.apply_document_edits = projection.apply_document_edits
    workspace = WorkspaceShell.__new__(WorkspaceShell)
    workspace.app, workspace.ui, workspace.s = app, ui, ui.s
    workspace.pinned, workspace.inspector_tab = None, "Material"
    workspace.open, workspace._numeric_edit = {"Inspector": True}, None
    workspace._panel_context = PanelContext(session, app.backend, style_scale=ui.s)
    workspace.menus = Menus(workspace, ())
    store = SpecificationStore(tmp_path / "design.json")
    spec = deepcopy(DEFAULT)
    spec["parts"]["rgba"] = {"fontSize": 20, "fontFamily": "portable", "textColor": "#f08844"}
    store.write(spec, save=True)
    runtime = DesignRuntime(store)
    runtime.observe_components = True
    ui.design = DesignPresentation(ui, runtime)
    painted, original = [], ImguiDraw2D.text

    def observe(draw, position, color, value, **kwargs):
        if value in ("0.125", "0.250", "0.500", "0.750"):
            painted.append((value, imgui.get_font_size(), imgui.get_font().font_id, color))
        return original(draw, position, color, value, **kwargs)

    monkeypatch.setattr(ImguiDraw2D, "text", observe)
    try:
        for _ in range(3):
            imgui.new_frame()
            imgui.set_next_window_size((800 * ui.s, 500 * ui.s))
            imgui.begin("Material Inspector")
            workspace.menus.begin_frame()
            inspector(workspace, 20 * ui.s, 20 * ui.s, 740 * ui.s, 440 * ui.s)
            imgui.end()
            imgui.render()
        assert {item[0] for item in painted} >= {"0.125", "0.250", "0.500", "0.750"}
        assert all(size == pytest.approx(20 * ui.s) for _, size, _, _ in painted)
        assert all(font == ui.body.font_id for _, _, font, _ in painted)
        assert all(
            color[:3] == pytest.approx((240 / 255, 136 / 255, 68 / 255))
            for _, _, _, color in painted
        )
        assert any(
            record["key"] == "material-color-channels" for record in ui.design.runtime.observed
        )
    finally:
        app.release()
