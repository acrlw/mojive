"""Native scrollbar geometry keeps visible content clear without replacing input."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
from imgui_bundle import imgui

from mojive.ui.scrollbar_visibility import NativeScrollbarVisibility
from mojive.ui.theme import THEME, apply_colors, pack_u32
from mojive.ui.workspace.popup_style import menu_popup_style
from mojive.ui.workspace.widget_style import install_style
from mojive.ui.workspace.widgets import UI
from mojive.ui.workspace.window_chrome import panel_surface


@pytest.mark.parametrize("scale", (1.0, 1.25, 2.0, 2.25))
@pytest.mark.parametrize("axis", (imgui.internal.Axis.x, imgui.internal.Axis.y))
@pytest.mark.parametrize("surface", ("panel", "popup"))
def test_native_scrollbar_has_balanced_clearance_and_retains_track_click(scale, axis, surface):
    from contextlib import nullcontext

    context = imgui.create_context()
    io = imgui.get_io()
    io.set_ini_filename(None)
    io.display_size = (800 * scale, 600 * scale)
    io.delta_time = 1 / 60
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    ui = UI(
        SimpleNamespace(style_scale=scale, font_report=SimpleNamespace(cjk_path=None)),
        portable_fonts=True,
    )
    color = imgui.color_convert_float4_to_u32(imgui.ImVec4(0.125, 0.75, 0.25, 1)) & 0xFFFFFF
    horizontal = axis == imgui.internal.Axis.x
    flag = (
        imgui.WindowFlags_.always_horizontal_scrollbar
        if horizontal
        else imgui.WindowFlags_.always_vertical_scrollbar
    )

    def frame():
        imgui.new_frame()
        imgui.set_next_window_pos((0, 0))
        imgui.set_next_window_size(io.display_size)
        imgui.begin("Host", flags=imgui.WindowFlags_.no_scrollbar)
        with menu_popup_style(ui) if surface == "popup" else nullcontext():
            for key in (
                imgui.Col_.scrollbar_grab,
                imgui.Col_.scrollbar_grab_hovered,
                imgui.Col_.scrollbar_grab_active,
            ):
                imgui.push_style_color(key, (0.125, 0.75, 0.25, 1))
            imgui.push_style_var(imgui.StyleVar_.window_padding, (0, 0))
            imgui.begin_child("Content", (300 * scale, 180 * scale), window_flags=flag)
            imgui.pop_style_var()
            window = imgui.internal.get_current_window()
            track = imgui.internal.get_window_scrollbar_rect(window, axis)
            bounds = (track.min.x, track.min.y, track.max.x, track.max.y)
            available = imgui.get_content_region_avail()
            pos = imgui.get_cursor_screen_pos()
            content_end = pos.y + available.y if horizontal else pos.x + available.x
            imgui.dummy((700 * scale, available.y) if horizontal else (available.x, 700 * scale))
            scroll = window.scroll.x if horizontal else window.scroll.y
            points = [
                (vertex.pos.x, vertex.pos.y)
                for host in imgui.get_current_context().windows
                for vertex in host.draw_list.vtx_buffer
                if vertex.col & 0xFFFFFF == color
            ]
            imgui.end_child()
            imgui.pop_style_color(3)
        imgui.end()
        imgui.render()
        return bounds, content_end, points, scroll

    try:
        install_style(scale)
        for _ in range(4):
            bounds, content_end, points, _scroll = frame()
        assert points
        cross = 1 if horizontal else 0
        start, end = bounds[cross], bounds[cross + 2]
        visible_start = min(point[cross] for point in points)
        visible_end = max(point[cross] for point in points)
        # Include native half-point antialiasing when measuring the actual mesh.
        assert visible_start - content_end == pytest.approx(8 * scale - 0.5, abs=0.6)
        assert end - visible_end == pytest.approx(8 * scale - 0.5, abs=0.6)
        assert visible_end - visible_start == pytest.approx(3 * scale + 1, abs=0.6)
        assert end - start == pytest.approx(19 * scale)
        point = (
            (bounds[2] - 10 * scale, (bounds[1] + bounds[3]) / 2)
            if horizontal
            else ((bounds[0] + bounds[2]) / 2, bounds[3] - 10 * scale)
        )
        io.add_mouse_pos_event(*point)
        frame()
        frame()
        io.add_mouse_button_event(0, True)
        frame()
        io.add_mouse_button_event(0, False)
        assert frame()[3] > 0
    finally:
        imgui.destroy_context(context)


@pytest.mark.parametrize("scale", (1.0, 1.2, 2.25))
@pytest.mark.parametrize("surface", ("panel", "popup"))
@pytest.mark.parametrize("shown", (False, True))
def test_theme_reapplication_reveals_only_thumb_without_a_separate_track_surface(
    scale, surface, shown
):
    context = imgui.create_context()
    try:
        io = imgui.get_io()
        io.set_ini_filename(None)
        io.display_size, io.delta_time = (800 * scale, 600 * scale), 1 / 60
        io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
        ui = UI(
            SimpleNamespace(style_scale=scale, font_report=SimpleNamespace(cjk_path=None)),
            portable_fonts=True,
        )
        # Production startup and live profile changes apply theme colors after
        # installing the workspace style. Give the former track a distinct ink.
        theme = replace(THEME, bg_child=(0.6, 0.1, 0.4, 1))
        apply_colors(imgui, theme)
        thumb = (0.125, 0.75, 0.25, 1)
        for color in (
            imgui.Col_.scrollbar_grab,
            imgui.Col_.scrollbar_grab_hovered,
            imgui.Col_.scrollbar_grab_active,
        ):
            imgui.get_style().set_color_(color, thumb)
        visibility = NativeScrollbarVisibility(clock=lambda: 0)
        visibility.register("Settings")
        if shown:
            visibility.notify_scroll("Settings")
        for _ in range(4):
            imgui.new_frame()
            with visibility.scope(), visibility.panel_scope("Settings"):
                style = (
                    menu_popup_style(ui)
                    if surface == "popup"
                    else panel_surface(ui, name="Settings", docked=False)
                )
                with style:
                    if surface == "popup" and not imgui.is_popup_open("Scroll popup"):
                        imgui.open_popup("Scroll popup")
                    imgui.set_next_window_pos((40 * scale, 40 * scale))
                    imgui.set_next_window_size((300 * scale, 180 * scale))
                    flags = imgui.WindowFlags_.always_vertical_scrollbar
                    if surface == "popup":
                        assert imgui.begin_popup("Scroll popup", flags)
                    else:
                        imgui.begin("Settings", flags=flags | imgui.WindowFlags_.no_title_bar)
                    window = imgui.internal.get_current_window()
                    visibility.bind_current_window()
                    imgui.dummy((100 * scale, 700 * scale))
                    if surface == "popup":
                        imgui.end_popup()
                    else:
                        imgui.end()
            imgui.render()
        colors = {vertex.col for vertex in window.draw_list.vtx_buffer}
        assert pack_u32(theme.bg_child) not in colors
        assert (pack_u32(thumb) in colors) is shown
        assert window.scrollbar_y and window.scroll_max.y > 0
        assert window.scrollbar_sizes.x == pytest.approx(19 * scale)
        assert imgui.get_style().color_(imgui.Col_.scrollbar_bg).w == 0
    finally:
        imgui.destroy_context(context)
