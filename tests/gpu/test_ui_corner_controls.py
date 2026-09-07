"""Exercise the production corner controls through real ImGui input and rendering."""

from pathlib import Path

import numpy as np
import pytest
from imgui_bundle import imgui
from PIL import Image

from mojive.ui import theme
from mojive.ui.window import Window, WindowConfig

pytestmark = pytest.mark.gpu


def test_corner_sliders_update_independently_and_keep_fractional_values(monkeypatch):
    # Keep every probe slider visible regardless of the desktop's default scale.
    monkeypatch.setenv("MOJIVE_UI_SCALE", "1")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]))
    from design.tools import render_ui_feasibility as probe

    window = Window(
        WindowConfig(width=1600, height=1000, docking=False, ini_path="", show_on_start=False)
    )
    rectangles = {}
    slider = imgui.slider_float

    def record_slider(label, *args, **kwargs):
        result = slider(label, *args, **kwargs)
        if label.startswith("##corner-") or label == "##imgui-corner-radius":
            lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
            rectangles[label] = (lo.x, lo.y, hi.x, hi.y)
        return result

    monkeypatch.setattr(imgui, "slider_float", record_slider)
    try:
        probe._apply_concept_theme(window.style_scale)
        state = probe.ProbeState(page="Geometry", geometry_tab="Corners")
        process_inputs = window._impl.process_inputs
        queued = []

        def inputs():
            process_inputs()
            for x, y, pressed in queued:
                imgui.get_io().add_mouse_pos_event(x, y)
                imgui.get_io().add_mouse_button_event(0, pressed)
            queued.clear()

        monkeypatch.setattr(window._impl, "process_inputs", inputs)

        def frame():
            window.begin_frame()
            probe._draw_workspace(window, state)
            window.end_frame()

        for _ in range(4):
            frame()
        assert set(rectangles) == {f"##corner-{name}" for _, name in probe.CORNER_CONTROLS} | {
            "##imgui-corner-radius"
        }
        for index, (_, name) in enumerate(probe.CORNER_CONTROLS):
            before = {key: getattr(state, key) for _, key in probe.CORNER_CONTROLS}
            lo_x, lo_y, hi_x, hi_y = rectangles[f"##corner-{name}"]
            fraction = 0.23 if index % 2 else 0.79
            x, y = lo_x + (hi_x - lo_x) * fraction, (lo_y + hi_y) * 0.5
            queued.append((x, y, False))
            frame()
            queued.append((x, y, True))
            frame()
            frame()
            queued.append((x, y, False))
            frame()
            assert abs(getattr(state, name) - fraction) < 0.06
            assert 0.0 < getattr(state, name) < 1.0
            for other, value in before.items():
                if other != name:
                    assert getattr(state, other) == value
        padding = imgui.get_style().window_padding
        spacing = (padding.x, padding.y)
        before = {key: getattr(state, key) for _, key in probe.CORNER_CONTROLS}
        x0, y0, x1, y1 = rectangles["##imgui-corner-radius"]
        x, y = x0 + (x1 - x0) * 0.28, (y0 + y1) * 0.5
        for pressed in (False, True, True, False):
            queued.append((x, y, pressed))
            frame()
        assert 4.0 < state.imgui_rounding < 5.0
        padding = imgui.get_style().window_padding
        assert (padding.x, padding.y) == spacing
        assert before == {key: getattr(state, key) for _, key in probe.CORNER_CONTROLS}
        for name in probe.theme_mod.CONTROL_ROUNDING:
            expected = state.imgui_rounding * window.style_scale
            assert getattr(imgui.get_style(), name) == pytest.approx(expected)
        assert imgui.get_style().grab_rounding == pytest.approx(
            max(0.0, imgui.get_style().frame_rounding - 2.0)
        )
        exported = probe._geometry_values_text(state)
        assert f"imgui_rounding={state.imgui_rounding}," in exported
        assert all(
            f"{name}={getattr(state, name)}," in exported for _, name in probe.CORNER_CONTROLS
        )
    finally:
        window.close()


@pytest.mark.parametrize("vertical", (False, True))
def test_slider_grab_visibly_tracks_radius_without_early_width_saturation(vertical, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]))
    from tests.curve_assertions import distance_to_path

    window = Window(
        WindowConfig(width=420, height=280, docking=False, ini_path="", show_on_start=False)
    )
    outlines = {}
    destination = Path("output/g3-controls")
    destination.mkdir(parents=True, exist_ok=True)
    try:
        color = imgui.ImVec4(0.25, 0.9, 0.4, 1.0)
        packed = imgui.color_convert_float4_to_u32(color)
        for radius in (0, 4, 8, 10, 12, 16):
            for _ in range(3):
                window.begin_frame()
                theme.apply_corner_radius(imgui, radius)
                # Keep the outer frame unsaturated through radius 12 so this
                # specifically detects premature clamping by the narrow thumb.
                imgui.get_style().frame_padding = imgui.ImVec2(4, 8)
                imgui.set_next_window_pos(imgui.ImVec2(0, 0))
                imgui.set_next_window_size(imgui.ImVec2(420, 280))
                imgui.begin("Slider radius", flags=imgui.WindowFlags_.no_saved_settings.value)
                imgui.text(f"Corner radius: {radius} px")
                imgui.push_style_color(imgui.Col_.slider_grab, color)
                imgui.push_style_color(imgui.Col_.slider_grab_active, color)
                if vertical:
                    imgui.v_slider_float("##radius", imgui.ImVec2(28, 150), 0.5, 0.0, 1.0, "")
                else:
                    imgui.set_next_item_width(360)
                    imgui.slider_float("##radius", 0.5, 0.0, 1.0, "")
                imgui.pop_style_color(2)
                outline = np.array(
                    [
                        (v.pos.x, v.pos.y)
                        for v in imgui.get_window_draw_list().vtx_buffer
                        if v.col == packed
                    ]
                )
                imgui.end()
                pixels = window.end_frame(readback=True)
            outlines[radius] = outline
            name = "vertical" if vertical else "horizontal"
            Image.fromarray(pixels[::-1]).save(destination / f"slider-{name}-{radius:02d}.png")
        for outline in outlines.values():
            assert len(outline) >= 4
            # The minimum thumb is square, so its inner-frame radius fits without
            # a second width-based scale. Radius edits keep value mapping stable.
            width, height = np.ptp(outline, axis=0)
            assert width == pytest.approx(height, abs=0.6)
            assert outline.min(axis=0) == pytest.approx(outlines[0].min(axis=0), abs=0.6)
            assert outline.max(axis=0) == pytest.approx(outlines[0].max(axis=0), abs=0.6)
        # The old half-width clamp made radii 8 and 10 produce identical thumbs.
        for lower, upper in ((4, 8), (8, 10), (10, 12), (12, 16)):
            assert distance_to_path(outlines[upper], outlines[lower]).max() > 0.08
    finally:
        window.close()


@pytest.mark.parametrize("vertical", (False, True))
@pytest.mark.parametrize("integer", (False, True))
def test_slider_grab_endpoints_and_center_match_mouse_input(vertical, integer, monkeypatch):
    window = Window(
        WindowConfig(width=420, height=280, docking=False, ini_path="", show_on_start=False)
    )
    queued = []
    process_inputs = window._impl.process_inputs

    def inputs():
        process_inputs()
        for x, y, pressed in queued:
            imgui.get_io().add_mouse_pos_event(x, y)
            imgui.get_io().add_mouse_button_event(0, pressed)
        queued.clear()

    monkeypatch.setattr(window._impl, "process_inputs", inputs)
    value = 5 if integer else 0.5
    color = imgui.ImVec4(0.25, 0.9, 0.4, 1.0)
    packed = imgui.color_convert_float4_to_u32(color)

    def frame():
        nonlocal value
        window.begin_frame()
        theme.apply_corner_radius(imgui, 16)
        imgui.set_next_window_pos(imgui.ImVec2(0, 0))
        imgui.set_next_window_size(imgui.ImVec2(420, 280))
        imgui.begin("Slider input", flags=imgui.WindowFlags_.no_saved_settings.value)
        imgui.push_style_color(imgui.Col_.slider_grab, color)
        imgui.push_style_color(imgui.Col_.slider_grab_active, color)
        if vertical:
            slider = imgui.v_slider_int if integer else imgui.v_slider_float
            _, value = slider("##value", imgui.ImVec2(28, 180), value, 0, 10 if integer else 1)
        else:
            imgui.set_next_item_width(360)
            slider = imgui.slider_int if integer else imgui.slider_float
            _, value = slider("##value", value, 0, 10 if integer else 1)
        imgui.pop_style_color(2)
        lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
        bounds = np.array(((lo.x, lo.y), (hi.x, hi.y)))
        vertices = np.array(
            [(v.pos.x, v.pos.y) for v in imgui.get_window_draw_list().vtx_buffer if v.col == packed]
        )
        grab = np.array((vertices.min(axis=0), vertices.max(axis=0)))
        imgui.end()
        window.end_frame()
        return bounds, grab

    try:
        for _ in range(3):
            bounds, grab = frame()
        axis = int(vertical)
        for fraction in (0.0, 0.5, 1.0):
            point = bounds.mean(axis=0)
            point[axis] = bounds[0, axis] + (bounds[1, axis] - bounds[0, axis]) * fraction
            point[axis] = np.clip(point[axis], bounds[0, axis] + 0.5, bounds[1, axis] - 0.5)
            for pressed in (False, True, True, False):
                queued.append((*point, pressed))
                bounds, grab = frame()
            expected = (1 - fraction if vertical else fraction) * (10 if integer else 1)
            assert value == pytest.approx(expected, abs=0.01)
            assert np.all(grab[0] >= bounds[0] + 1.5)
            assert np.all(grab[1] <= bounds[1] - 1.5)
            if fraction == 0.5:
                assert grab.mean(axis=0) == pytest.approx(bounds.mean(axis=0), abs=0.6)
            else:
                end = int(fraction)
                assert abs(grab[end, axis] - bounds[end, axis]) == pytest.approx(2, abs=0.6)
    finally:
        window.close()


@pytest.mark.parametrize("radius", (8.0, 16.0))
@pytest.mark.parametrize("height", (24.0, 40.0))
def test_tab_focus_uses_the_control_radius_and_outset(radius, height, monkeypatch):
    from tests.curve_assertions import distance_to_path

    from mojive.curves2d import smooth_rect_points

    window = Window(
        WindowConfig(width=480, height=340, docking=False, ini_path="", show_on_start=False)
    )
    queued = []
    process_inputs = window._impl.process_inputs

    def inputs():
        process_inputs()
        for pressed in queued:
            imgui.get_io().add_key_event(imgui.Key.tab, pressed)
        queued.clear()

    monkeypatch.setattr(window._impl, "process_inputs", inputs)
    focus_color = imgui.ImVec4(0.9, 0.3, 0.7, 1.0)
    packed = imgui.color_convert_float4_to_u32(focus_color)
    destination = Path("output/g3-controls")
    destination.mkdir(parents=True, exist_ok=True)
    seen = set()
    try:
        for index in range(14):
            if index in (5, 10):
                queued.append(True)
            if index in (6, 11):
                queued.append(False)
            window.begin_frame()
            theme.apply_corner_radius(imgui, radius)
            style = imgui.get_style()
            style.child_rounding = 28
            style.set_color_(int(imgui.Col_.nav_cursor), focus_color)
            imgui.set_next_window_pos((0, 0))
            imgui.set_next_window_size((480, 340))
            imgui.begin("Keyboard focus")
            imgui.text("Tab: button and child panel")
            imgui.button("Rounded button", (350, height))
            button = imgui.get_item_id()
            lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
            bounds = {button: ("button", (lo.x, lo.y), (hi.x, hi.y), radius)}
            child = imgui.get_id("Child")
            imgui.begin_child("Child", (350, 180), imgui.ChildFlags_.borders)
            imgui.button("Inside child")
            imgui.end_child()
            lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
            bounds[child] = ("child", (lo.x, lo.y), (hi.x, hi.y), 28)
            focus = np.array(
                [
                    (v.pos.x, v.pos.y)
                    for v in imgui.get_window_draw_list().vtx_buffer
                    if v.col == packed
                ]
            )
            context = imgui.get_current_context()
            focused = context.nav_id
            imgui.end()
            pixels = window.end_frame(readback=True)
            if index not in (8, 13):
                continue
            assert context.nav_cursor_visible
            assert focused == (button if index == 8 else child)
            name, lo, hi, rounding = bounds[focused]
            assert len(focus) > 8
            Image.fromarray(pixels[::-1]).save(
                destination / f"focus-{name}-{height:g}-r{radius:g}.png"
            )
            # Compare the actual focus stroke against the widget's boundary.
            # Keep the gap uniform even when a small button saturates its radius
            # or a child panel uses a different radius from FrameRounding.
            effective = min(rounding, (hi[1] - lo[1]) / 2 - 1)
            boundary = smooth_rect_points(*lo, *hi, effective, smoothing=0.0)
            # Texture-based AA emits pairs straddling the stroke centerline.
            centerline = focus.reshape(-1, 2, 2).mean(axis=1)
            gap = distance_to_path(centerline, boundary)
            assert gap.min() > 2.9
            assert gap.max() < 4.4
            seen.add(name)
        assert seen == {"button", "child"}
    finally:
        window.close()


def test_native_menu_highlight_keeps_text_padding(monkeypatch):
    window = Window(
        WindowConfig(width=440, height=320, docking=False, ini_path="", show_on_start=False)
    )
    process_inputs = window._impl.process_inputs
    pointer = [-100.0, -100.0]

    def inputs():
        process_inputs()
        imgui.get_io().add_mouse_pos_event(*pointer)

    monkeypatch.setattr(window._impl, "process_inputs", inputs)
    bounds = {}
    color = imgui.ImVec4(0.35, 0.7, 0.5, 1.0)
    packed = imgui.color_convert_float4_to_u32(color)
    shapes = {}
    text_shapes = {}
    destination = Path("output/g3-controls")
    destination.mkdir(parents=True, exist_ok=True)
    try:
        for index in range(16):
            active = (index - 4) // 4
            if 0 <= active < 3:
                pointer[:] = bounds[active]
            window.begin_frame()
            theme.apply(imgui)
            imgui.get_style().set_color_(int(imgui.Col_.header_hovered), color)
            imgui.set_next_window_pos((0, 0))
            imgui.set_next_window_size((440, 320))
            imgui.begin("Menu acceptance")
            if index == 0:
                imgui.open_popup("Menu")
            if imgui.begin_popup("Menu"):
                for item, label in enumerate(("First command", "Middle command", "Last command")):
                    imgui.menu_item(label, "", False)
                    lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
                    bounds[item] = ((lo.x + hi.x) / 2, (lo.y + hi.y) / 2)
                points = np.array(
                    [
                        (v.pos.x, v.pos.y)
                        for v in imgui.get_window_draw_list().vtx_buffer
                        if v.col == packed
                    ]
                )
                text_points = np.array(
                    [
                        (v.pos.x, v.pos.y)
                        for v in imgui.get_window_draw_list().vtx_buffer
                        if v.col == imgui.get_color_u32(imgui.Col_.text)
                    ]
                )
                imgui.end_popup()
            imgui.end()
            pixels = window.end_frame(readback=True)
            if index in (7, 11, 15):
                shapes[active] = points
                text_shapes[active] = text_points
                Image.fromarray(pixels[::-1]).save(destination / f"menu-row-{active}.png")
        for index, points in shapes.items():
            assert len(points) == 4
            lo, hi = points.min(axis=0), points.max(axis=0)
            text = text_shapes[index]
            assert text[:, 0].min() - lo[0] >= 3.0
            assert hi[0] - text[:, 0].max() >= 3.0
    finally:
        window.close()


@pytest.mark.parametrize("control", ("number", "text", "search"))
@pytest.mark.parametrize("scale", (1.0, 1.5))
def test_input_focus_follows_the_full_control_contour(control, scale, monkeypatch):
    from tests.curve_assertions import distance_to_path

    from mojive.curves2d import smooth_rect_points
    from mojive.ui.panels import search_input

    monkeypatch.setenv("MOJIVE_UI_SCALE", str(scale))
    window = Window(
        WindowConfig(width=540, height=260, docking=False, ini_path="", show_on_start=False)
    )
    packed = imgui.color_convert_float4_to_u32(imgui.ImVec4(0.9, 0.3, 0.7, 1.0))
    try:
        for index in range(7):
            window.begin_frame()
            imgui.push_style_color(imgui.Col_.nav_cursor, packed)
            imgui.set_next_window_pos((0, 0))
            imgui.set_next_window_size((540, 260))
            imgui.begin("Input focus")
            imgui.text("Keyboard focus follows the field, including search icons")
            origin = imgui.get_cursor_screen_pos()
            lo = (origin.x, origin.y)
            hi = (origin.x + 300 * scale, origin.y + imgui.get_frame_height())
            imgui.set_next_item_width(300 * scale)
            if index == 2:
                imgui.set_keyboard_focus_here()
            if control == "number":
                imgui.input_double("##input", 0.0, 0.0, 0.0, "%.6f")
            elif control == "text":
                imgui.input_text("##input", "Text")
            else:
                search_input("##input", "", hint="Search hierarchy")
            vertices = np.array(
                [
                    (v.pos.x, v.pos.y)
                    for v in imgui.get_window_draw_list().vtx_buffer
                    if v.col == packed
                ]
            )
            imgui.end()
            imgui.pop_style_color()
            pixels = window.end_frame(readback=True)
        assert imgui.get_current_context().nav_cursor_visible
        assert len(vertices) > 8
        boundary = smooth_rect_points(*lo, *hi, imgui.get_style().frame_rounding, smoothing=0.0)
        centerline = vertices.reshape(-1, 2, 2).mean(axis=1)
        gap = distance_to_path(centerline, boundary)
        assert gap.min() > 2.9
        assert gap.max() < 4.4
        destination = Path("output/g3-controls")
        destination.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pixels[::-1]).save(destination / f"focus-{control}-{scale:g}x.png")
    finally:
        window.close()
