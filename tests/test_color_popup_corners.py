"""Live color popups own their corner geometry and one inset budget."""

from contextlib import contextmanager
from copy import deepcopy
from types import SimpleNamespace

import pytest
from imgui_bundle import imgui

from mojive.ui.workspace.color_editor import color_popup
from mojive.ui.workspace.menus import Menus
from tests import test_color_settings_layout
from tests.test_color_settings_layout import layout as layout


def _window():
    return imgui.get_current_context().open_popup_stack[0].window


@pytest.fixture(params=(1, 1.1, 1.2, 1.25, 2.25))
def production_color(request, tmp_path, monkeypatch):
    from mojive import Scene
    from mojive import commands as cmd
    from mojive.adapters.static import StaticSceneAdapter
    from mojive.design.native.presentation import DesignPresentation
    from mojive.design.runtime import DesignRuntime
    from mojive.design.store import SpecificationStore
    from mojive.render.backend import NullBackend
    from mojive.session import Session
    from mojive.ui.app import ViewerApp
    from mojive.ui.panels import PanelContext
    from mojive.ui.workspace.color_editor import ColorField
    from mojive.ui.workspace.entities import EntityProjection
    from mojive.ui.workspace.inspector_style import NativeInspectorPresentation
    from mojive.ui.workspace.widgets import UI

    monkeypatch.setenv("MOJIVE_SETTINGS", str(tmp_path / "settings.json"))
    context = imgui.create_context()
    io = imgui.get_io()
    io.set_ini_filename(None)
    io.delta_time = 1 / 60
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    io.display_size = (1000 * request.param, 1400 * request.param)
    scene = Scene()
    box = scene.box(name="color target", color=(0.125, 0.25, 0.5, 0.75))
    session = Session(StaticSceneAdapter(scene))
    app = ViewerApp(session, NullBackend())
    app.live_model_updates = True
    node = session.node_by_object_id(box.object_id)
    assert session.submit(cmd.SelectNode(node.node_id)).ok
    projection = EntityProjection(app)
    app.refresh_document = projection.refresh_document
    app.apply_document_edits = projection.apply_document_edits
    ui = UI(
        SimpleNamespace(style_scale=request.param, font_report=SimpleNamespace(cjk_path=None)),
        portable_fonts=True,
    )
    runtime = DesignRuntime(SpecificationStore(tmp_path / "color.json"))
    runtime.observe_components = True
    ui.design = DesignPresentation(ui, runtime)
    workspace = SimpleNamespace(ui=ui, s=ui.s, app=app)
    menu = Menus(workspace, ())
    field = ColorField()
    inspector = NativeInspectorPresentation(ui)
    inspector.node_id = node.node_id
    inspector.ctx = PanelContext(session, app.backend, style_scale=ui.s, translate=ui.translate)
    state = SimpleNamespace(
        ui=ui,
        runtime=runtime,
        entry="field",
        values=[0.125, 0.25, 0.5, 0.75],
        field=field,
        inspector=inspector,
        menu=menu,
    )

    def frame(count=1):
        for _ in range(count):
            projection.refresh_document()
            runtime.begin_frame()
            ui.hits.clear()
            ui.numeric_edits.clear()
            imgui.new_frame()
            imgui.set_next_window_pos((0, 0))
            imgui.set_next_window_size(io.display_size)
            imgui.begin("Production color host", flags=imgui.WindowFlags_.no_title_bar)
            imgui.set_cursor_screen_pos((30 * ui.s, 30 * ui.s))
            if state.entry == "field":
                changed, values = field.draw(ui, "color", state.values, width=300 * ui.s)
                if changed:
                    state.values = list(values)
            elif state.entry == "inspector":
                changed, values = inspector.color("##color", state.values)
                if changed:
                    state.values = list(values)
            else:
                menu.begin_frame()
                menu.color_control(projection.entity, 30 * ui.s, 30 * ui.s, 300 * ui.s)
                projection.apply_document_edits()
                menu.finish_color_edit()
            imgui.end()
            imgui.render()

    def open_popup(entry):
        state.entry = entry
        frame(3)
        key = {"field": "color-swatch", "inspector": "##color-picker", "menu": "material-color"}[
            entry
        ]
        x, y, width, height = ui.hits[key]
        io.add_mouse_pos_event(x + width / 2, y + height / 2)
        frame()
        io.add_mouse_button_event(0, True)
        frame()
        io.add_mouse_button_event(0, False)
        frame(6)
        if entry == "field":
            state.editor, state.key, state.done = field.editor, "color", "color-done"
        elif entry == "inspector":
            state.editor = inspector._color_editors[(node.node_id, "##color")]
            state.key = "##color-color-editor-" + str(node.node_id)
            state.done = state.key + "-done"
        else:
            state.editor, state.key, state.done = menu.color_editor, "color-editor", "done-color"
        state.editor.mode = 2
        frame(6)

    state.frame, state.open_popup = frame, open_popup
    try:
        yield state
    finally:
        app.release()
        imgui.destroy_context(context)


def _actual_geometry(state):
    window = _window()
    header = state.ui.hits[state.key + "-header"]
    done = state.ui.hits[state.done]
    insets = (
        header[0] - window.pos.x,
        header[1] - window.pos.y,
        window.pos.x + window.size.x - done[0] - done[2],
        window.pos.y + window.size.y - done[1] - done[3],
    )
    from mojive.ui.workspace.menu_geometry import native_epsilon

    return window, header, done, insets, native_epsilon(*header, *done, *window.pos, *window.size)


@pytest.mark.parametrize("entry", ("field", "inspector", "menu"))
def test_authored_color_radius_fits_actual_footer_and_channel_frames(production_color, entry):
    state, ui = production_color, production_color.ui
    spec = deepcopy(state.runtime.spec)
    spec["parts"]["color_editor"] = {"paddingX": 8, "paddingY": 8, "radius": 24}
    state.runtime.update_spec(spec)
    state.open_popup(entry)
    window, _header, done, insets, epsilon = _actual_geometry(state)
    container = next(r for r in reversed(state.runtime.observed) if r["kind"] == "color_popup")
    inner = container["innerRadius"] * ui.s
    assert window.window_rounding == pytest.approx(24 * ui.s, abs=epsilon)
    assert window.window_rounding == pytest.approx(inner + insets[0], abs=epsilon)
    assert inner <= min(done[2], done[3]) / 2 - 1
    for axis in "RGBA":
        field = ui.hits[state.key + axis + "-frame"]
        assert inner <= min(field[2], field[3]) / 2 - 1


@pytest.mark.parametrize("entry", ("field", "inspector", "menu"))
def test_default_production_color_popup_has_exact_native_insets_and_compact_corners(
    production_color, entry
):
    from mojive.ui.workspace.menu_geometry import native_unit

    state, ui = production_color, production_color.ui
    previous_rounding = imgui.get_style().popup_rounding
    state.open_popup(entry)
    window, header, done, insets, epsilon = _actual_geometry(state)
    assert insets == pytest.approx((native_unit(8 * ui.s),) * 4, abs=epsilon)
    assert window.window_rounding == pytest.approx(4 * ui.s + insets[0], abs=epsilon)
    container = next(r for r in reversed(state.runtime.observed) if r["kind"] == "color_popup")
    assert container["radiusDerived"] is True and container["radiusAuthored"] is False
    assert imgui.get_style().popup_rounding == previous_rounding
    original, current = (ui.hits[state.key + "-" + name] for name in ("original", "current"))
    assert original[0] == header[0]
    assert original[1] == current[1] == header[1]
    assert original[2:] == pytest.approx((28 * ui.s, 28 * ui.s), abs=epsilon)
    assert current[2:] == pytest.approx(original[2:], abs=epsilon)
    assert done[0] + done[2] == pytest.approx(header[0] + header[2], abs=epsilon)
    assert window.scroll_max.y == 0
    assert done[3] == pytest.approx(28 * ui.s, abs=epsilon)
    # The standard popup stays compact at fractional scale without breaking
    # the four-channel row into a taller grid or cropping numeric values.
    assert window.size.x / ui.s <= 282
    assert ui.hits[state.key + "-sv"][3] / ui.s <= 160 + epsilon
    channels = [ui.hits[state.key + axis + "-frame"] for axis in "RGBA"]
    assert len({round(rect[1], 3) for rect in channels}) == 1
    assert channels[-1][0] + channels[-1][2] <= header[0] + header[2] + epsilon


@pytest.mark.parametrize("entry", ("field", "inspector", "menu"))
@pytest.mark.parametrize(
    "padding,button_radius,color_radius,button_height",
    (
        ((6, 6, 6, 6), 3, None, 32),
        ((18, 18, 18, 18), 8, None, 28),
        ((5, 7, 11, 13), 6, None, 31),
        ((5, 7, 11, 13), 8, 24, 28),
        ((12, 12, 12, 12), 6, 0, 28),
    ),
)
def test_production_color_popup_preserves_asymmetric_insets_explicit_radius_and_footer_height(
    production_color, entry, padding, button_radius, color_radius, button_height
):
    from mojive.ui.workspace.menu_geometry import native_unit

    state, ui = production_color, production_color.ui
    spec = deepcopy(state.runtime.spec)
    fields = ("paddingLeft", "paddingTop", "paddingRight", "paddingBottom")
    spec["parts"]["color_editor"] = dict(zip(fields, padding, strict=True))
    if color_radius is not None:
        spec["parts"]["color_editor"]["radius"] = color_radius
    spec["parts"]["button"] = {"radius": button_radius, "height": button_height}
    state.runtime.update_spec(spec)
    state.open_popup(entry)
    window, _header, done, insets, epsilon = _actual_geometry(state)
    expected = (max(native_unit(value * ui.s) for value in padding),) * 4
    assert insets == pytest.approx(expected, abs=epsilon)
    radius = color_radius * ui.s if color_radius is not None else button_radius * ui.s + expected[0]
    assert window.window_rounding == pytest.approx(radius, abs=epsilon)
    container = next(r for r in reversed(state.runtime.observed) if r["kind"] == "color_popup")
    assert container["radiusDerived"] is (color_radius is None)
    assert container["radiusAuthored"] is (color_radius is not None)
    assert container["innerRadius"] * ui.s == pytest.approx(max(0, radius - expected[0]))
    assert ui.component_style("color_editor").edges("padding") == padding
    assert done[3] == pytest.approx(button_height * ui.s, abs=epsilon)


def test_color_popup_shadow_follows_the_actual_native_corner(layout, monkeypatch):
    state, ui = layout, layout.ui
    spec = deepcopy(state.runtime.spec)
    spec["parts"]["color_editor"] = {"radius": 24}
    spec["parts"]["menu"] = {"radius": 3}
    state.runtime.update_spec(spec)
    shadow_roundings = []
    draw = SimpleNamespace(
        push_clip_rect_full_screen=lambda: None,
        pop_clip_rect=lambda: None,
        add_rect=lambda *args, **kwargs: shadow_roundings.append(kwargs["rounding"]),
    )
    menu = SimpleNamespace(ui=ui, s=ui.s, observed={})

    @contextmanager
    def popup(ui, key):
        with color_popup(ui, key) as opened:
            if opened:
                with monkeypatch.context() as context:
                    context.setattr(imgui, "get_window_draw_list", lambda: draw)
                    Menus.shadow_and_record(menu, key, (key,))
            yield opened

    monkeypatch.setattr(test_color_settings_layout, "color_popup", popup)
    state.frame()
    actual_radius = _window().window_rounding
    assert actual_radius == 24 * ui.s
    assert shadow_roundings[-4:] == pytest.approx(
        [actual_radius + outset * ui.s for outset in (7, 5, 3, 1.5)]
    )
