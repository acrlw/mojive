"""Reference controls share native font metrics and compiled icon bounds."""

from itertools import pairwise
from types import SimpleNamespace

import pytest
from examples.ui_design.native.docking_study import PANELS, Study
from examples.ui_design.native.reference_menus import Menus
from examples.ui_design.native.scene_preview import ReferenceViewCube
from examples.ui_design.native.widgets import (
    CONTROL_THEME,
    GLYPHS,
    PANEL,
    UI,
)
from imgui_bundle import imgui

from mojive import CameraView
from mojive.config import ViewportLayers
from mojive.ui.controls import segmented_control, segmented_control_width
from mojive.ui.imgui_draw import ImguiDraw2D
from mojive.ui.input_bindings import DEFAULT_INPUT_BINDINGS
from mojive.ui.localization import Language, Localizer


@pytest.fixture(params=(1.0, 2.25))
def controls(request):
    context = imgui.create_context()
    io = imgui.get_io()
    io.set_ini_filename(None)
    io.display_size = (800 * request.param, 400 * request.param)
    io.delta_time = 1 / 60
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    report = SimpleNamespace(cjk_path=None)
    if request.node.callspec.params.get("language") == "zh_CN":
        from mojive.text.sources import resolve_font_sources

        sources = resolve_font_sources(allow_download=False)
        if len(sources) < 2:
            pytest.skip("CJK font is unavailable")
        report.cjk_path, report.cjk_index = sources[1].path, sources[1].index
    ui = UI(SimpleNamespace(style_scale=request.param, font_report=report))

    def frame(paint=lambda: None):
        imgui.new_frame()
        imgui.set_next_window_pos((0, 0))
        imgui.set_next_window_size(io.display_size)
        imgui.begin("Control alignment", flags=imgui.WindowFlags_.no_title_bar)
        result = paint()
        imgui.end()
        imgui.render()
        return result

    frame()
    try:
        yield ui, frame
    finally:
        imgui.destroy_context(context)


def vertex_center(draw_list, first):
    vertices = [vertex.pos for vertex in list(draw_list.vtx_buffer)[first:]]
    assert vertices
    return (
        (min(vertex.x for vertex in vertices) + max(vertex.x for vertex in vertices)) / 2,
        (min(vertex.y for vertex in vertices) + max(vertex.y for vertex in vertices)) / 2,
    )


@pytest.mark.parametrize("font", ("body", "bold", "mono"))
def test_text_cap_height_is_centered_in_the_requested_line_box(controls, font):
    ui, frame = controls

    def paint():
        first = len(ui.draw.vtx_buffer)
        ui.text(40 * ui.s, 50 * ui.s, "H", bold=font == "bold", mono=font == "mono")
        return vertex_center(ui.draw, first)

    _, center_y = frame(paint)
    assert center_y == pytest.approx(56.5 * ui.s, abs=0.01)


@pytest.mark.parametrize("language", ("en", "zh_CN"))
@pytest.mark.parametrize(
    "axes,width,columns",
    (
        ("RGBA", 110, 1),
        ("RGBA", 180, 1),
        ("RGBA", 198, 2),
        ("RGBA", 320, 2),
        ("RGBA", 400, 4),
        ("RGB", 110, 1),
        ("RGB", 180, 1),
        ("RGB", 198, 2),
        ("RGB", 320, 3),
        ("RGB", 400, 3),
    ),
)
def test_numeric_group_layout_keeps_rgba_precision_visible_and_fields_equal(
    controls, monkeypatch, language, axes, width, columns
):
    ui, frame = controls
    painted = []
    text = ImguiDraw2D.text
    values = (0.2, 0.0041472, 0.000123, 1.0)[: len(axes)]

    def observe(draw, position, color, value, **kwargs):
        painted.append(value)
        return text(draw, position, color, value, **kwargs)

    monkeypatch.setattr(ImguiDraw2D, "text", observe)

    def paint():
        layout = ui.numeric_layout(width * ui.s, len(axes), column_options=(4, 2, 1))
        for axis, value, (x, y, w, _) in zip(axes, values, layout.fields, strict=True):
            changed, result = ui.input(
                "rgba-" + axis, 40 * ui.s + x, 50 * ui.s + y, w, value, axis=axis, format="%.7f"
            )
            assert not changed and result == value
        return layout

    layout = frame(paint)
    fields = [ui.hits["rgba-" + axis + "-frame"] for axis in axes]
    assert layout.columns == columns
    assert len({round(field[2], 4) for field in fields}) == 1
    assert all(value in painted for value in ("0.2", "0.0041472", "0.000123", "1")[: len(axes)])
    assert all(not value.endswith("…") for value in painted)
    for x, y, w, h in fields:
        assert 40 * ui.s <= x and x + w <= (40 + width) * ui.s + 0.5
        assert h == pytest.approx(28 * ui.s, abs=1)
        assert y + h <= 50 * ui.s + layout.height + 0.5
    for first, second in pairwise(fields):
        if first[1] == second[1]:
            assert second[0] - first[0] - first[2] == pytest.approx(4 * ui.s, abs=1)
        else:
            assert second[1] - first[1] - first[3] == pytest.approx(8 * ui.s, abs=1)


@pytest.mark.parametrize("language", ("en", "zh_CN"))
@pytest.mark.parametrize("width", (228, 320, 360, 700))
def test_transform_rows_share_label_placement_field_width_and_measured_height(
    controls, language, width
):
    ui, frame = controls
    ui.translate = Localizer(Language(language)).text

    def paint():
        layout = ui.vector_layout(width * ui.s)
        y = 30 * ui.s
        for key, label in (("position", "Position"), ("rotation", "Rotation")):
            ui.vector(key, 40 * ui.s, y, width * ui.s, label, (0.0, 0.0, 0.0), layout=layout)
            y += layout.height + 8 * ui.s
        return layout

    layout = frame(paint)
    first = [ui.hits["position" + axis + "-frame"] for axis in "XYZ"]
    second = [ui.hits["rotation" + axis + "-frame"] for axis in "XYZ"]
    assert layout.label_above is (width < 360)
    assert {round(field[2], 4) for field in first + second} == {round(first[0][2], 4)}
    assert [field[0] for field in first] == [field[0] for field in second]
    assert second[0][1] - first[0][1] == pytest.approx(layout.height + 8 * ui.s)
    assert second[0][1] - first[-1][1] - first[-1][3] >= 8 * ui.s - 1


def test_icons_center_their_complete_visible_bounds_at_control_sizes(controls):
    ui, frame = controls

    def paint():
        for name in GLYPHS:
            for size in (14, 16, 18):
                first = len(ui.draw.vtx_buffer)
                ui.icon(name, 80 * ui.s, 100 * ui.s, size)
                center = vertex_center(ui.draw, first)
                expected = ((80 + size / 2) * ui.s, (100 + size / 2) * ui.s)
                assert center == pytest.approx(expected, abs=0.75 * ui.s), (name, size)

    frame(paint)


@pytest.mark.parametrize("control", ("button", "section"))
def test_control_text_and_icon_share_one_vertical_center(controls, monkeypatch, control):
    ui, frame = controls
    centers = {}
    icon, text = ui.icon, ui.text

    def observe_icon(*args, **kwargs):
        first = len(ui.draw.vtx_buffer)
        icon(*args, **kwargs)
        centers["icon"] = vertex_center(ui.draw, first)[1]

    def observe_text(*args, **kwargs):
        first = len(ui.draw.vtx_buffer)
        text(*args, **kwargs)
        centers["text"] = vertex_center(ui.draw, first)[1]

    monkeypatch.setattr(ui, "icon", observe_icon)
    monkeypatch.setattr(ui, "text", observe_text)
    if control == "button":
        frame(lambda: ui.button("aligned", 40 * ui.s, 50 * ui.s, 90, icon="folder", label="H"))
        expected = 64 * ui.s
    else:
        frame(lambda: ui.section("H", 40 * ui.s, 50 * ui.s, 180 * ui.s, open=True))
        expected = 69 * ui.s
    assert centers["text"] == pytest.approx(expected, abs=0.01)
    assert centers["icon"] == pytest.approx(expected, abs=0.6 * ui.s)


@pytest.mark.parametrize("nested,enabled", ((False, False), (True, True), (True, False)))
def test_disabled_button_uses_opaque_muted_ink_and_real_disabled_hits(
    controls, monkeypatch, nested, enabled
):
    ui, frame = controls
    ink, icon = [], ui.icon
    tint = (200, 90, 50, 255)

    def observe(*args, **kwargs):
        first = len(ui.draw.vtx_buffer)
        icon(*args, **kwargs)
        ink[:] = [vertex.col for vertex in list(ui.draw.vtx_buffer)[first:]]

    monkeypatch.setattr(ui, "icon", observe)

    def paint():
        imgui.begin_disabled(nested)
        clicked = ui.button(
            "disabled-ink",
            40 * ui.s,
            50 * ui.s,
            120,
            icon="joint",
            label="H",
            icon_color=tint,
            enabled=enabled,
        )
        disabled = bool(imgui.get_item_flags() & imgui.ItemFlags_.disabled)
        imgui.end_disabled()
        return clicked, disabled

    assert frame(paint) == (False, True)
    assert max(color >> 24 for color in ink) == 255
    opacity = imgui.get_style().disabled_alpha
    expected = tuple(
        round(a * opacity + b * (1 - opacity)) for a, b in zip(tint[:3], PANEL[:3], strict=True)
    )
    opaque_colors = {
        (color & 255, (color >> 8) & 255, (color >> 16) & 255)
        for color in ink
        if color >> 24 == 255
    }
    assert opaque_colors == {expected}
    x, y, w, h = ui.hits["disabled-ink"]
    assert (w, h) == (120 * ui.s, 28 * ui.s)
    io = imgui.get_io()
    io.add_mouse_pos_event(x + w / 2, y + h / 2)
    frame(paint)
    io.add_mouse_button_event(0, True)
    assert frame(paint) == (False, True)
    io.add_mouse_button_event(0, False)
    assert frame(paint) == (False, True)
    assert imgui.get_style().alpha == 1


def test_shared_segment_callback_preserves_native_rendering_and_item_identity(controls):
    ui, frame = controls
    observed = []

    def observe(index):
        lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
        observed.append((index, imgui.get_item_id(), (lo.x, lo.y, hi.x, hi.y)))

    def paint(callback=None):
        result = segmented_control(
            "shared-choice",
            ("Default", "Visual", "Collision", "Both"),
            2,
            width=390 * ui.s,
            on_item=callback,
        )
        return result, tuple((v.pos.x, v.pos.y, v.col) for v in ui.draw.vtx_buffer)

    original = frame(paint)
    recorded = frame(lambda: paint(observe))
    assert recorded == original
    assert [entry[0] for entry in observed] == list(range(4))
    assert len({entry[1] for entry in observed}) == 4
    assert all(entry[2][2] > entry[2][0] and entry[2][3] > entry[2][1] for entry in observed)


@pytest.mark.parametrize("enabled", (True, False))
def test_segment_icons_use_native_contours_and_caption_alignment(controls, monkeypatch, enabled):
    ui, frame = controls
    centers, opacities = [], []
    icon, text = ui.icon, ui.text

    def observe(method, *args, **kwargs):
        first = len(ui.draw.vtx_buffer)
        method(*args, **kwargs)
        centers.append(vertex_center(ui.draw, first)[1])
        opacities.append(max(v.col >> 24 for v in list(ui.draw.vtx_buffer)[first:]))

    monkeypatch.setattr(ui, "icon", lambda *a, **kw: observe(icon, *a, **kw))
    monkeypatch.setattr(ui, "text", lambda *a, **kw: observe(text, *a, **kw))
    frame(
        lambda: ui.segmented(
            "icon-choice",
            40 * ui.s,
            50 * ui.s,
            180,
            ("H", "H"),
            0,
            icons=("folder", "camera"),
            enabled=enabled,
        )
    )
    assert centers == pytest.approx([64 * ui.s] * 4, abs=0.6 * ui.s)
    alpha = 1 if enabled else imgui.get_style().disabled_alpha
    assert opacities == [round(alpha * 255)] * 4


@pytest.fixture
def segments(controls, monkeypatch):
    ui, draw_frame = controls
    io = imgui.get_io()
    io.config_flags |= imgui.ConfigFlags_.nav_enable_keyboard
    state = SimpleNamespace(
        ui=ui, width=390, layout="equal", selected=0, enabled=True, items={}, colors={}
    )
    labels = ("Default", "Visual", "Collision", "Both")
    keys = tuple("choice-" + label.casefold() for label in labels)
    button = imgui.button

    def record_button(label, *args, **kwargs):
        clicked = button(label, *args, **kwargs)
        if "##choice-" in label:
            state.items[label.partition("##")[0]] = (
                imgui.get_item_id(),
                imgui.is_item_hovered(),
                imgui.is_item_active(),
            )
            state.colors[label.partition("##")[0]] = imgui.get_color_u32(imgui.Col_.text)
        return clicked

    monkeypatch.setattr(imgui, "button", record_button)

    def paint():
        ui.hits.clear()
        state.items.clear()
        state.colors.clear()
        state.selected = ui.segmented(
            "choice",
            40 * ui.s,
            50 * ui.s,
            state.width,
            labels,
            state.selected,
            keys=keys,
            enabled=state.enabled,
            layout=state.layout,
        )
        state.vertices = tuple((v.pos.x, v.pos.y, v.col) for v in ui.draw.vtx_buffer)

    def frame(count=1):
        for _ in range(count):
            draw_frame(paint)

    def event(method, *args):
        getattr(io, method)(*args)
        frame()

    def click(key):
        x, y, w, h = ui.hits[key]
        event("add_mouse_pos_event", x + w / 2, y + h / 2)
        event("add_mouse_button_event", 0, True)
        event("add_mouse_button_event", 0, False)
        frame(2)

    state.frame, state.event, state.click = frame, event, click
    frame(3)
    return state


@pytest.mark.parametrize("width", (390, 180))
def test_native_segments_keep_one_contiguous_shell_and_complete_labels(segments, width):
    state, ui = segments, segments.ui
    state.width = width
    state.frame(2)
    rects = [ui.hits["choice-" + label.casefold()] for label in state.items]
    gx, gy, gw, gh = ui.hits["choice"]
    assert (gx, gy, gw) == pytest.approx((40 * ui.s, 50 * ui.s, width * ui.s))
    assert gh == pytest.approx((28 if width == 390 else 112) * ui.s)
    for index, (x, y, w, h) in enumerate(rects):
        label = tuple(state.items)[index]
        assert ui.measure(label) + 16 * ui.s <= w
        assert h == pytest.approx(28 * ui.s)
        if index:
            px, py, pw, ph = rects[index - 1]
            assert (x, y) == pytest.approx((px + pw, py) if width == 390 else (px, py + ph))
    assert (
        sum(
            c == imgui.get_color_u32(imgui.ImVec4(*CONTROL_THEME.primary_bright))
            for c in state.colors.values()
        )
        == 1
    )


@pytest.mark.parametrize("language", ("en", "zh_CN"))
@pytest.mark.parametrize("width", (228, 180, 130))
def test_content_segments_fit_native_labels_before_falling_back(segments, language, width):
    state, ui = segments, segments.ui
    ui.translate = Localizer(Language(language)).text
    state.width, state.layout = width, "content"
    state.frame(2)
    labels = tuple(state.items)
    rects = [ui.hits["choice-" + name] for name in ("default", "visual", "collision", "both")]
    inline = width == 228 or (language == "zh_CN" and width == 180)
    assert ui.hits["choice"][3] == pytest.approx((28 if inline else 112) * ui.s)
    assert len(state.items) == 4
    padding = []
    for index, ((x, y, w, h), label) in enumerate(zip(rects, labels, strict=True)):
        required = ui.measure(label) + 12 * ui.s
        assert required <= w
        assert h == pytest.approx(28 * ui.s)
        padding.append(w - required)
        if index:
            px, py, pw, ph = rects[index - 1]
            assert (x, y) == pytest.approx((px + pw, py) if inline else (px, py + ph))
    if inline:
        assert max(padding) - min(padding) <= 2
        assert rects[-1][0] + rects[-1][2] == pytest.approx((40 + width) * ui.s)
        if language == "en":
            assert rects[2][2] > rects[3][2]
    imgui.push_font(ui.body, 13 * ui.s)
    imgui.push_style_var(imgui.StyleVar_.frame_padding, (6 * ui.s, 7 * ui.s))
    measured = segmented_control_width(labels, layout="content")
    imgui.pop_style_var()
    imgui.pop_font()
    assert (measured <= width * ui.s) is inline


@pytest.mark.parametrize(
    "layout,width", (("equal", 390), ("equal", 180), ("content", 228), ("content", 130))
)
def test_native_segments_use_real_hover_press_and_exclusive_selection(segments, layout, width):
    state, ui = segments, segments.ui
    state.width, state.layout = width, layout
    state.frame(2)
    x, y, w, h = ui.hits["choice-collision"]
    state.event("add_mouse_pos_event", x + w / 2, y + h / 2)
    assert state.items["Collision"][1]
    hovered = imgui.get_color_u32(imgui.ImVec4(*CONTROL_THEME.bg_frame_hovered))
    assert any(
        color == hovered and x <= vx <= x + w and y <= vy <= y + h
        for vx, vy, color in state.vertices
    )
    state.event("add_mouse_button_event", 0, True)
    assert state.items["Collision"][2]
    assert state.selected == 0
    pressed = imgui.get_color_u32(imgui.ImVec4(*CONTROL_THEME.bg_frame_active))
    assert any(
        color == pressed and x <= vx <= x + w and y <= vy <= y + h
        for vx, vy, color in state.vertices
    )
    state.event("add_mouse_button_event", 0, False)
    state.frame(2)
    assert state.selected == 2
    highlighted = imgui.get_color_u32(imgui.ImVec4(*CONTROL_THEME.primary_bright))
    assert [label for label, color in state.colors.items() if color == highlighted] == ["Collision"]
    state.enabled = False
    state.click("choice-both")
    assert state.selected == 2


@pytest.mark.parametrize(
    "layout,width", (("equal", 390), ("equal", 180), ("content", 228), ("content", 130))
)
def test_native_segments_preserve_directional_keyboard_navigation_and_activation(
    segments, layout, width
):
    state = segments
    state.width, state.layout = width, layout
    state.frame(2)
    state.click("choice-default")
    state.event("add_mouse_pos_event", -100, -100)
    direction = (
        imgui.Key.right_arrow
        if state.ui.hits["choice"][3] == 28 * state.ui.s
        else imgui.Key.down_arrow
    )
    state.event("add_key_event", direction, True)
    state.event("add_key_event", direction, False)
    state.frame(2)
    assert imgui.get_current_context().nav_id == state.items["Visual"][0]
    state.event("add_key_event", imgui.Key.space, True)
    state.event("add_key_event", imgui.Key.space, False)
    state.frame(2)
    assert state.selected == 1


def test_native_segments_allow_no_choice_until_user_selects(segments):
    state = segments
    state.selected = -1
    state.frame(2)
    highlighted = imgui.get_color_u32(imgui.ImVec4(*CONTROL_THEME.primary_bright))
    assert state.selected == -1
    assert all(color != highlighted for color in state.colors.values())
    state.click("choice-collision")
    assert state.selected == 2


@pytest.fixture
def viewport_controls(controls, tmp_path):
    ui, _ = controls
    io = imgui.get_io()
    study = Study.__new__(Study)
    study.ui, study.s = ui, ui.s
    study.rects = {}
    study.open = dict.fromkeys(PANELS, False)
    study.open.update(Inspector=True, Timeline=True)
    study.output_directory = tmp_path
    study.window = SimpleNamespace(painter=lambda: None, viewport_texture_ref=None)
    study.cube = ReferenceViewCube(selection_padding=1.2)
    state = SimpleNamespace(study=study, width=140, height=130)
    gizmo = SimpleNamespace(space="world")
    gizmo.set_space = lambda value: setattr(gizmo, "space", value)
    session = SimpleNamespace(
        paused=True,
        state_take_playing=False,
        state_take_recording=False,
        state_take_loop_enabled=False,
        state_take_times=[],
        can_step_back=False,
        speed=1.0,
        adapter=SimpleNamespace(caps=SimpleNamespace(simulation=True)),
        frame=SimpleNamespace(time=0.0),
    )
    app = SimpleNamespace(
        _viewport_image=None,
        viewport_surface=SimpleNamespace(
            draw_image=lambda *_: (20 * ui.s, 50 * ui.s, state.width * ui.s, state.height * ui.s)
        ),
        _draw_scene_overlays=lambda _: None,
        _draw_reference_camera_preview=lambda _: None,
        camera_preview=SimpleNamespace(enabled=False),
        input_bindings=DEFAULT_INPUT_BINDINGS,
        viewport_layers=ViewportLayers(),
        model_edits=SimpleNamespace(active=False),
        camera=SimpleNamespace(orthographic=False),
        tool="select",
        entity=None,
        gizmo=gizmo,
        _snap_latched=False,
        session=session,
    )
    study.preview = app
    study.menus = Menus(study, [])

    def draw_overlays(_):
        study.cube.update(CameraView(), app._viewport_rect, io.mouse_pos, ui.s)
        study.cube.draw(ImguiDraw2D())

    app._draw_scene_overlays = draw_overlays

    def frame(count=1):
        for _ in range(count):
            io.display_size = (
                max(800, state.width + 80) * ui.s,
                max(400, state.height + 80) * ui.s,
            )
            imgui.new_frame()
            study.menus.begin_frame()
            ui.hits.clear()
            imgui.set_next_window_pos((0, 0))
            imgui.set_next_window_size(io.display_size)
            imgui.begin("Viewport", flags=imgui.WindowFlags_.no_title_bar)
            study.viewport()
            imgui.render()

    def click(key):
        x, y, w, h = ui.hits[key]
        io.add_mouse_pos_event(x + w / 2, y + h / 2)
        frame()
        io.add_mouse_button_event(0, True)
        frame()
        io.add_mouse_button_event(0, False)
        frame(3)

    state.frame, state.click = frame, click
    frame(3)
    return state


@pytest.mark.parametrize(
    "size",
    ((110, 90), (110, 130), (140, 90), (140, 130), (140, 300), (110, 180), (230, 150), (310, 90)),
)
def test_compact_viewport_controls_keep_real_hit_regions_inside_the_image(viewport_controls, size):
    state = viewport_controls
    state.width, state.height = size
    state.frame(2)
    ui, study = state.study.ui, state.study
    assert "viewport-more" in ui.hits
    assert not any(
        key in ui.hits for key in ("projection-button", "shading", "tool-select", "play")
    )
    x, y, w, h = study.rects["viewport_content"]
    for cx, cy, cw, ch in ui.hits.values():
        assert x <= cx and y <= cy and cx + cw <= x + w and cy + ch <= y + h
    overlays = study.rects["overlays"]
    assert len(overlays) == 2
    for index, (ox, oy, ow, oh) in enumerate(overlays):
        assert x <= ox and y <= oy and ox + ow <= x + w and oy + oh <= y + h
        for px, py, pw, ph in overlays[index + 1 :]:
            assert ox + ow <= px or px + pw <= ox or oy + oh <= py or py + ph <= oy
    cube = study.cube
    front = min(cube.balls, key=lambda ball: ball.depth)
    hovered = cube.update(CameraView(), study.rects["viewport_content"], front.screen, ui.s)
    assert hovered is not None and (hovered.axis, hovered.sign) == (front.axis, front.sign)


def test_compact_viewport_menu_changes_tools_frame_and_snap_with_native_input(viewport_controls):
    state = viewport_controls
    for name in ("move", "rotate", "scale", "select"):
        state.click("viewport-more")
        state.click("tool-" + name)
        assert state.study.preview.tool == name
    state.click("viewport-more")
    state.click("frame")
    assert state.study.space == "Body"
    state.click("viewport-more")
    state.click("snap")
    assert state.study.snap


@pytest.mark.parametrize("path", (("Camera",), ("Shading",), ("Playback", "Playback speed")))
def test_compact_viewport_keeps_existing_native_popup_paths(viewport_controls, path):
    state = viewport_controls
    state.study.menus.preview(path)
    state.frame(4)
    assert "/".join(path) in state.study.menus.observed


@pytest.mark.parametrize("size", ((310, 348), (400, 300), (500, 600)))
def test_regular_viewport_preserves_the_toolbar_layout_and_nonoverlapping_controls(
    viewport_controls, size
):
    state = viewport_controls
    state.width, state.height = size
    state.frame(2)
    hits = state.study.ui.hits
    assert "viewport-more" not in hits
    assert all(
        key in hits for key in ("projection-button", "shading", "tool-select", "frame", "snap")
    )
    x, y, w, h = state.study.rects["viewport_content"]
    for key, (cx, cy, cw, ch) in hits.items():
        assert x <= cx and y <= cy and cx + cw <= x + w and cy + ch <= y + h, key
        for other, (ox, oy, ow, oh) in hits.items():
            if key != other:
                assert cx + cw <= ox or ox + ow <= cx or cy + ch <= oy or oy + oh <= cy, (
                    key,
                    other,
                )
    overlays = state.study.rects["overlays"]
    for index, (ox, oy, ow, oh) in enumerate(overlays):
        assert x <= ox and y <= oy and ox + ow <= x + w and oy + oh <= y + h
        for px, py, pw, ph in overlays[index + 1 :]:
            assert ox + ow <= px or px + pw <= ox or oy + oh <= py or py + ph <= oy
