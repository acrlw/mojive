"""Shared controls retain native input and contiguous bounds after reflow."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest
from imgui_bundle import imgui

from mojive.ui.controls import segmented_control
from mojive.ui.theme import THEME
from mojive.ui.window import Window, WindowConfig

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("count,width", [(1, 320), (2, 320), (3, 480), (5, 100)])
@pytest.mark.parametrize("scale", [1.0, 2.5])
def test_segmented_options_reflow_and_keep_pointer_keyboard_and_disabled_input(
    monkeypatch, count, width, scale
):
    window = Window(
        WindowConfig(
            title="Shared controls",
            width=800,
            height=800,
            ui_scale=scale,
            show_on_start=False,
            docking=False,
            ini_path="",
            vsync=False,
        )
    )
    labels = tuple(f"Option {i}" for i in range(count))
    selected, disabled, focus = 0, False, False
    rectangles = []
    button = imgui.button

    def record(label, *args, **kwargs):
        result = button(label, *args, **kwargs)
        lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
        rectangles.append((lo.x, lo.y, hi.x, hi.y))
        return result

    monkeypatch.setattr(imgui, "button", record)

    def frame():
        nonlocal selected
        rectangles.clear()
        window.begin_frame()
        imgui.set_next_window_pos((0, 0))
        imgui.set_next_window_size((800, 800))
        imgui.begin("##controls", None, imgui.WindowFlags_.no_decoration)
        imgui.begin_disabled(disabled)
        if focus:
            imgui.set_keyboard_focus_here()
        selected = segmented_control("choices", labels, selected, width=width)
        imgui.end_disabled()
        imgui.end()
        window.end_frame()

    def click(index):
        lo_x, lo_y, hi_x, hi_y = rectangles[index]
        io.add_mouse_pos_event((lo_x + hi_x) * 0.5, (lo_y + hi_y) * 0.5)
        frame()
        io.add_mouse_button_event(0, True)
        frame()
        io.add_mouse_button_event(0, False)
        frame()

    try:
        io = imgui.get_io()
        io.config_flags |= imgui.ConfigFlags_.nav_enable_keyboard
        frame()
        frame()
        assert len(rectangles) == count
        for left, right in pairwise(rectangles):
            if left[1] == right[1]:
                assert left[2] == pytest.approx(right[0], abs=1)
            else:
                assert left[3] == pytest.approx(right[1], abs=1)
                assert left[0] == right[0]
        click(count - 1)
        assert selected == count - 1
        disabled = True
        click(0)
        assert selected == count - 1
        disabled = False
        io.add_key_event(imgui.Key.tab, True)
        frame()
        io.add_key_event(imgui.Key.tab, False)
        frame()
        focus = True
        frame()
        focus = False
        frame()
        io.add_key_event(imgui.Key.space, True)
        frame()
        io.add_key_event(imgui.Key.space, False)
        frame()
        assert selected == 0
    finally:
        window.close()


@pytest.mark.parametrize("scale", [1.0, 1.25, 2.5])
def test_segment_boundaries_have_no_background_crack_and_focus_stays_inside(monkeypatch, scale):
    window = Window(
        WindowConfig(
            width=800,
            height=240,
            ui_scale=scale,
            show_on_start=False,
            docking=False,
            ini_path="",
            vsync=False,
        )
    )
    native = imgui.button
    rectangles, focus = [], []
    focus_color = (1.0, 0.1, 0.7, 1.0)
    packed = imgui.color_convert_float4_to_u32(imgui.ImVec4(*focus_color))

    def record(label, *args, **kwargs):
        result = native(label, *args, **kwargs)
        lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
        rectangles.append((lo.x, lo.y, hi.x, hi.y))
        return result

    monkeypatch.setattr(imgui, "button", record)
    try:
        for focused in (-1, 0, 1, 2):
            for _ in range(2):
                rectangles.clear()
                window.begin_frame()
                imgui.set_next_window_pos((0, 0))
                imgui.set_next_window_size((800, 240))
                imgui.begin("##surface", None, imgui.WindowFlags_.no_decoration)
                imgui.set_cursor_pos((30.0, 30.0))
                context = imgui.get_current_context()
                context.nav_id = imgui.get_id(f"##surface-{focused}")
                context.nav_cursor_visible = focused >= 0
                imgui.push_style_color(imgui.Col_.nav_cursor, focus_color)
                # Icon labels retain hidden IDs, avoiding glyphs near either join.
                segmented_control(
                    "surface", ("", "", ""), 1, width=517.5, icons=("persp", "ortho", "persp")
                )
                imgui.pop_style_color()
                focus = np.array(
                    [
                        (v.pos.x, v.pos.y)
                        for v in imgui.get_window_draw_list().vtx_buffer
                        if v.col == packed
                    ]
                )
                imgui.end()
                pixels = window.end_frame(readback=True)[::-1]
            if focused == -1:
                # Item rectangles use logical points; framebuffer readback uses pixels.
                framebuffer_scale = imgui.get_io().display_framebuffer_scale
                y = int((rectangles[0][1] + rectangles[0][3]) * 0.5 * framebuffer_scale.y)
                for rect in rectangles[:-1]:
                    x = round(rect[2] * framebuffer_scale.x)
                    strip = pixels[y, x - 2 : x + 3, :3]
                    floor = np.array(THEME.bg_frame[:3]) * 255 - 2
                    assert np.all(strip >= floor), strip
            else:
                assert len(focus) > 0
                x0, y0, x1, y1 = rectangles[focused]
                assert np.all(focus >= (x0, y0))
                assert np.all(focus <= (x1, y1))
                # Inner joins stay square even when the outside end is rounded.
                inner_x = x1 if focused == 0 else x0
                assert np.min(np.linalg.norm(focus - (inner_x, y0), axis=1)) < 3.0 * scale
    finally:
        window.close()
