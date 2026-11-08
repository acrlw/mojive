"""Custom controls retain one inset keyboard-focus contour and native input."""

import pytest
from imgui_bundle import imgui

from mojive.design.native.geometry import Rect
from mojive.ui.compound_fields import end_content_child
from tests.test_component_appearance import appearance as appearance


@pytest.mark.parametrize("control", ("switch", "checkbox", "range"))
def test_keyboard_focus_is_inset_once_and_mouse_resumes_without_outline(appearance, control):
    ui, _runtime, _update, frame = appearance
    io = imgui.get_io()
    io.config_flags |= imgui.ConfigFlags_.nav_enable_keyboard
    vertices = []

    def paint():
        ui.design.begin_frame()
        imgui.push_style_color(imgui.Col_.nav_cursor, (1, 0, 1, 1))
        draw = imgui.get_window_draw_list()
        start = len(draw.vtx_buffer)
        rect = Rect(40 * ui.s, 40 * ui.s, 240 * ui.s, 28 * ui.s)
        primitive = getattr(ui.design.primitives, control)
        if control == "range":
            primitive("focused", rect, 0.5, 0, 1)
        elif control == "checkbox":
            primitive("focused", rect.x, rect.y, True, label="Enabled")
        else:
            primitive("focused", rect, True, label="Enabled")
        vertices[:] = [
            (v.pos.x, v.pos.y)
            for v in list(draw.vtx_buffer)[start:]
            if v.col & 0xFFFFFF == 0xFF00FF and v.col >> 24
        ]
        imgui.pop_style_color()

    for _ in range(3):
        frame(paint)
    assert not vertices
    io.add_key_event(imgui.Key.tab, True)
    frame(paint)
    io.add_key_event(imgui.Key.tab, False)
    for _ in range(3):
        frame(paint)
    assert vertices
    x, y, width, height = ui.hits["focused"]
    # The native default cursor expands outside the control; the component
    # contour stays inside, including its antialias fringe.
    assert all(x <= vx <= x + width and y <= vy <= y + height for vx, vy in vertices)
    io.add_mouse_pos_event(x + width / 2, y + height / 2)
    frame(paint)
    io.add_mouse_button_event(0, True)
    frame(paint)
    io.add_mouse_button_event(0, False)
    for _ in range(3):
        frame(paint)
    assert not vertices


def test_content_region_does_not_paint_a_control_focus_border(appearance):
    ui, _runtime, _update, frame = appearance
    context = imgui.get_current_context()
    _phase = None

    def paint():
        ui.design.begin_frame()
        imgui.push_style_color(imgui.Col_.nav_cursor, (1, 0, 1, 1))
        parent = context.current_window
        start = len(parent.draw_list.vtx_buffer)
        imgui.begin_child("Content region", (300 * ui.s, 180 * ui.s))
        child = context.current_window
        if _phase == "control":
            context.nav_window = child
            context.nav_id = child.get_id("child-control")
            context.nav_cursor_visible = True
        pos = imgui.get_cursor_screen_pos()
        ui.button("child-control", pos.x, pos.y, 120, 28, label="Enabled")
        control_focus = [
            v for v in child.draw_list.vtx_buffer if v.col & 0xFFFFFF == 0xFF00FF and v.col >> 24
        ]
        if _phase == "region":
            context.nav_window = parent
            context.nav_id = child.child_id
            context.nav_cursor_visible = True
        focused = context.nav_id
        end_content_child()
        assert context.current_window == parent
        assert context.nav_id == focused
        assert tuple(imgui.get_style().color_(imgui.Col_.nav_cursor)) == (1, 0, 1, 1)
        assert not [
            v
            for v in list(parent.draw_list.vtx_buffer)[start:]
            if v.col & 0xFFFFFF == 0xFF00FF and v.col >> 24
        ]
        if _phase == "control":
            assert control_focus
        imgui.pop_style_color()

    for _ in range(3):
        frame(paint)
    for _phase in ("region", "control"):
        frame(paint)
