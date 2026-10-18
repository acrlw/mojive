"""Native camera preview presentation and shared fly keys with CPU ImGui input."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from examples.ui_design.native.camera_preview import preview_region
from examples.ui_design.native.reference_menus import Menus
from examples.ui_design.native.scene_preview import Preview
from imgui_bundle import imgui

from mojive import commands as cmd
from mojive.adapters.base import NodeType
from mojive.render.backend import NullBackend
from mojive.types import ViewportImage
from mojive.ui import gestures as gs
from mojive.ui.input_bindings import InputAction
from mojive.ui.workspace.widgets import UI


class ImageBackend(NullBackend):
    def __init__(self, *args):
        super().__init__()
        self.size = (320, 180)

    def resize(self, width, height):
        self.size = (width, height)

    def create_peer(self, width, height):
        peer = ImageBackend()
        peer.resize(width, height)
        return peer

    def render(self, *args, **kwargs):
        return ViewportImage(19, *self.size)


@pytest.fixture(params=(1.0, 2.0))
def cameras(request, monkeypatch, tmp_path):
    monkeypatch.setenv("MOJIVE_SETTINGS", str(tmp_path / "settings.json"))
    monkeypatch.setattr("mojive.render.opengl.backend.OpenGLBackend", ImageBackend)
    context = imgui.create_context()
    io = imgui.get_io()
    io.set_ini_filename(None)
    io.display_size = (900 * request.param, 600 * request.param)
    io.delta_time = 1 / 60
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    ui = UI.__new__(UI)
    ui.s = request.param
    ui.body = ui.bold = ui.mono = io.fonts.add_font_default()
    ui.hits = {}
    ui.translate = str
    window = SimpleNamespace(
        style_scale=request.param,
        apply_theme=lambda value: None,
        make_current=lambda: None,
        viewport_texture_ref=lambda image: imgui.ImTextureRef(image.texture_id),
    )
    app = Preview(window, "opengl")
    app._numeric_cancel_frame = -1
    app.viewport_layers = replace(app.viewport_layers, gizmos=False)
    study = SimpleNamespace(
        ui=ui,
        s=ui.s,
        window=window,
        preview=app,
        _numeric_edit=None,
        rects={},
        input_overlays=[],
        timeline_toggles=0,
    )
    study.menus = Menus(study, ())
    app.study = study
    controls = {}
    input_text = imgui.input_text

    def text_input(identifier, *args):
        result = input_text(identifier, *args)
        lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
        controls[identifier] = (lo.x, lo.y, hi.x - lo.x, hi.y - lo.y)
        return result

    monkeypatch.setattr(imgui, "input_text", text_input)
    state = SimpleNamespace(
        app=app,
        study=study,
        ui=ui,
        typed="",
        edit=False,
        last_keys=None,
        viewport=(20 * ui.s, 20 * ui.s, 800 * ui.s, 500 * ui.s),
        obstacles=[],
    )

    def viewport(name):
        app._draw_reference_camera_preview(name)
        imgui.end()

    study.viewport = viewport

    def frame(count=1, *, contents=False):
        for _ in range(count):
            ui.hits.clear()
            imgui.new_frame()
            imgui.set_next_window_pos((0, 0))
            imgui.set_next_window_size(io.display_size)
            imgui.begin("Viewport", flags=imgui.WindowFlags_.no_title_bar)
            app._viewport_rect = state.viewport
            study.rects = {"overlays": list(state.obstacles)}
            study.input_overlays = list(state.obstacles)
            if state.edit:
                imgui.set_cursor_screen_pos((30 * ui.s, 35 * ui.s))
                _, state.typed = imgui.input_text("camera-field", state.typed)
            state.input_state = app._input_state()
            app._poll_application_shortcuts()
            state.last_keys = app._poll_keys()
            app._poll_camera(gs.InputState(), state.last_keys, 1 / 60)
            name, view = app.camera_preview.selected_camera(app.session)
            app.camera_preview.update(
                app.backend, app.session.source, 0, app.session.frame, view, (320, 180)
            )
            if contents:
                app._draw_viewport_contents(name)
            else:
                app._draw_reference_camera_preview(name)
                imgui.end()
            imgui.render()

    def event(method, *args):
        getattr(io, method)(*args)
        frame()

    def click_rect(rect):
        x, y, width, height = rect
        event("add_mouse_pos_event", x + width / 2, y + height / 2)
        event("add_mouse_button_event", 0, True)
        event("add_mouse_button_event", 0, False)

    def click(key):
        click_rect(ui.hits[key])

    state.frame, state.event, state.click, state.click_rect = frame, event, click, click_rect
    state.controls = controls
    frame(3)
    try:
        yield state
    finally:
        app.release()
        imgui.destroy_context(context)


def select_camera(state):
    node = next(node for node in state.app.session.nodes if node.type is NodeType.CAMERA)
    assert state.app.session.submit(cmd.SelectNode(node.node_id)).ok
    state.app.camera_preview.set_enabled(True)
    state.frame(3)
    return node


def test_native_camera_preview_is_drawn_in_viewport_and_close_disables_it(cameras):
    state = cameras
    assert not state.app.camera_preview.enabled
    assert "camera_preview" not in state.study.rects
    select_camera(state)
    state.frame(contents=True)
    assert state.app.camera_preview._image.texture_id == 19
    bounds = state.app.camera_preview.bounds
    assert bounds is not None
    a, b, c, d = bounds
    assert state.study.rects["camera_preview"] == pytest.approx((a, b, c - a, d - b))
    assert state.study.rects["camera_preview"] in state.study.input_overlays
    state.click("camera-preview-close")
    assert not state.app.camera_preview.enabled
    state.frame(2)
    assert state.app.camera_preview.bounds is None
    assert "camera_preview" not in state.study.rects


def test_native_camera_preview_drag_pin_and_lock_use_shared_state(cameras):
    state = cameras
    node = select_camera(state)
    x, y, width, height = state.ui.hits["camera-preview-title"]
    before = state.app.camera_preview.bounds
    state.event("add_mouse_pos_event", x + width / 2, y + height / 2)
    state.event("add_mouse_button_event", 0, True)
    state.event(
        "add_mouse_pos_event", x + width / 2 - 45 * state.ui.s, y + height / 2 - 30 * state.ui.s
    )
    state.event("add_mouse_button_event", 0, False)
    state.frame(2)
    assert state.app.camera_preview.bounds[0] < before[0]
    assert state.app.camera_preview.bounds[1] < before[1]
    state.click("camera-preview-pin")
    assert state.app.camera_preview.pinned
    assert state.app.session.submit(cmd.Select(0)).ok
    state.frame(2)
    assert state.app.camera_preview.bounds is not None

    state.click("camera-preview-lock")
    assert state.app.camera_preview.locked and not state.app.camera_preview.pinned
    assert (
        state.app.camera_preview._camera_id
        == state.app.session.cameras[node.camera_index].camera_id
    )
    state.frame(2)
    assert state.app.camera_preview.bounds is not None


def test_native_preview_title_has_padding_inside_its_hover_region(cameras, monkeypatch):
    state = cameras
    node = select_camera(state)
    text = state.ui.text
    captions = []

    def observe(x, y, label, *args, **kwargs):
        if label == node.name:
            captions.append((x, y, kwargs.get("max_width")))
        return text(x, y, label, *args, **kwargs)

    monkeypatch.setattr(state.ui, "text", observe)
    state.frame(3)
    x, y, width, height = state.ui.hits["camera-preview-title"]
    text_x, text_y, text_width = captions[-1]
    assert text_x - x == pytest.approx(8 * state.ui.s)
    assert x + width - text_x - text_width == pytest.approx(8 * state.ui.s)
    assert text_y > y and text_y + 14 * state.ui.s < y + height


@pytest.mark.parametrize(
    "key,axis,sign",
    (("w", 0, 1), ("s", 0, -1), ("a", 1, -1), ("d", 1, 1), ("q", 2, 1), ("e", 2, -1)),
)
def test_native_fly_uses_shared_wasd_qe_without_changing_gizmo_or_snap(cameras, key, axis, sign):
    state = cameras
    app = state.app
    assert app.input_bindings.key_id(InputAction.GIZMO_TRANSLATE) == "g"
    assert app.input_bindings.key_id(InputAction.GIZMO_ROTATE) == "r"
    assert app.input_bindings.key_id(InputAction.GIZMO_SPACE) == "b"
    before = app.camera.view().eye.copy()
    mode, snap = app.gizmo.mode, app._snap_latched
    state.event("add_mouse_pos_event", 100 * state.ui.s, 150 * state.ui.s)
    state.event("add_key_event", getattr(imgui.Key, key), True)
    assert state.last_keys.fly[axis] == sign
    state.frame(4)
    state.event("add_key_event", getattr(imgui.Key, key), False)
    assert np.linalg.norm(app.camera.view().eye - before) > 0.001
    assert app.gizmo.mode == mode
    assert app._snap_latched == snap


def test_native_text_edit_claims_fly_keys(cameras):
    state = cameras
    state.edit = True
    state.frame(3)
    state.click_rect(state.controls["camera-field"])
    state.frame(2)
    before = state.app.camera.view().eye.copy()
    state.event("add_key_event", imgui.Key.w, True)
    state.event("add_input_characters_utf8", "w")
    state.frame(4)
    state.event("add_key_event", imgui.Key.w, False)
    assert state.typed == "w"
    assert state.last_keys.fly == (0, 0, 0)
    np.testing.assert_allclose(state.app.camera.view().eye, before)


def test_native_camera_preview_compact_actions_and_bounds(cameras):
    state = cameras
    s = state.ui.s
    state.viewport = (20 * s, 20 * s, 160 * s, 200 * s)
    select_camera(state)
    state.frame(3)
    assert state.app.camera_preview.bounds is not None
    x, y, width, height = state.study.rects["camera_preview"]
    for key in ("camera-preview-title", "camera-preview-more", "camera-preview-close"):
        a, b, w, h = state.ui.hits[key]
        assert x <= a < a + w <= x + width
        assert y <= b < b + h <= y + height
    state.click("camera-preview-more")
    state.frame(2)
    state.click("camera-preview-lock")
    assert state.app.camera_preview.locked
    state.frame(2)
    assert state.app.camera_preview.bounds is not None


def test_native_preview_image_keeps_padding_and_camera_aspect(cameras, monkeypatch):
    state = cameras
    images = []
    draw_image = imgui.image

    def observe_image(*args, **kwargs):
        draw_image(*args, **kwargs)
        lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
        images.append((lo.x, lo.y, hi.x, hi.y))

    monkeypatch.setattr(imgui, "image", observe_image)
    select_camera(state)
    state.frame(3)
    left, top, right, bottom = state.app.camera_preview.bounds
    x, y, end_x, end_y = images[-1]
    assert x - left == pytest.approx(6 * state.ui.s)
    assert right - end_x == pytest.approx(6 * state.ui.s)
    assert bottom - end_y == pytest.approx(6 * state.ui.s)
    assert y - top == pytest.approx(30 * state.ui.s)
    assert (end_x - x) / (end_y - y) == pytest.approx(16 / 9)
    for key in ("camera-preview-pin", "camera-preview-lock", "camera-preview-close"):
        a, b, width, height = state.ui.hits[key]
        assert top < b < b + height < y
        assert a + width <= right - 6 * state.ui.s


def test_camera_preview_default_header_keeps_shared_pin_and_lock_inputs(cameras, monkeypatch):
    state = cameras
    button = imgui.button
    buttons = {}

    def observe(label, *args, **kwargs):
        pressed = button(label, *args, **kwargs)
        lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
        buttons[label] = (lo.x, lo.y, hi.x - lo.x, hi.y - lo.y)
        return pressed

    monkeypatch.setattr(imgui, "button", observe)
    monkeypatch.setattr(
        state.app,
        "_draw_reference_camera_preview",
        lambda name: state.app.camera_preview.draw(state.study.window, state.viewport, name),
    )
    node = select_camera(state)
    assert "Camera · " + node.name in buttons
    state.click_rect(buttons["Pin"])
    assert state.app.camera_preview.pinned
    state.click_rect(buttons["Lock"])
    assert state.app.camera_preview.locked and not state.app.camera_preview.pinned


def test_native_preview_hides_with_viewport_ui_without_disabling_camera_state(cameras):
    state = cameras
    select_camera(state)
    state.click("camera-preview-pin")
    state.app.viewport_layers = replace(state.app.viewport_layers, viewport_ui=False)
    state.frame(2)
    assert state.app.camera_preview.enabled and state.app.camera_preview.pinned
    assert state.app.camera_preview.bounds is None
    assert "camera_preview" not in state.study.rects
    state.app.viewport_layers = replace(state.app.viewport_layers, viewport_ui=True)
    state.frame(2)
    assert state.app.camera_preview.bounds is not None


def test_compact_preview_owns_the_space_and_clears_cube_hit_regions(cameras):
    state = cameras
    s = state.ui.s
    state.viewport = (235.628 * s, 36 * s, 108.244 * s, 122.5 * s)
    state.obstacles = [(245.628 * s, 46 * s, 36 * s, 36 * s)]
    select_camera(state)
    state.frame(3)
    assert state.app.view_cube.preview_active
    assert state.app.view_cube.scale_for(state.viewport, s) == 0
    assert state.app.view_cube.balls == []
    assert not state.input_state.over_view_cube
    bounds = state.app.camera_preview.bounds
    assert bounds is not None
    assert bounds[2] - bounds[0] >= 56 * s
    for key in ("camera-preview-more", "camera-preview-close"):
        x, y, width, height = state.ui.hits[key]
        assert bounds[0] <= x < x + width <= bounds[2]
        assert bounds[1] <= y < y + height <= bounds[3]
    state.app.camera_preview.set_enabled(False)
    state.frame(3)
    assert not state.app.view_cube.preview_active
    assert state.app.view_cube.scale_for(state.viewport, s) > 0
    assert state.app.view_cube.balls


@pytest.mark.parametrize("scale", (1.0, 2.0))
@pytest.mark.parametrize(
    "size", ((310, 220, True), (600, 400, True), (160, 150, False), (80, 60, False))
)
def test_preview_region_fits_between_viewport_controls(scale, size):
    width, height, visible = size
    viewport = (40 * scale, 30 * scale, width * scale, height * scale)
    obstacles = [
        (50 * scale, 40 * scale, 36 * scale, 36 * scale),
        ((40 + width - 115) * scale, 38 * scale, 105 * scale, 105 * scale),
    ]
    region = preview_region(viewport, obstacles, scale)
    assert (region is not None) == visible
    if region is None:
        return
    x, y, w, h = region
    vx, vy, vw, vh = viewport
    assert vx <= x < x + w <= vx + vw
    assert vy <= y < y + h <= vy + vh
    for ox, oy, ow, oh in obstacles:
        assert x + w <= ox or ox + ow <= x or y + h <= oy or oy + oh <= y
