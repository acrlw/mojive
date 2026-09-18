"""Feasibility page navigation and full workspace composition."""

from __future__ import annotations

from imgui_bundle import imgui

from mojive.ui import theme as theme_mod
from mojive.ui.window import Window

from ..ui_capsule_geometry import end_padding
from ..ui_redesign import draw_redesign
from .components import draw_components
from .fixtures import CONCEPT_THEME, VIEW_A, VIEW_B, WORKSPACE_CANVAS_SIZE
from .geometry import _draw_geometry_page
from .gizmos import _draw_joint_gizmo
from .layout import _flags, _virtual_canvas_size
from .panels import (
    _draw_hierarchy_gallery,
    _draw_inspector_gallery,
    _draw_output,
    _draw_panel_page,
    _draw_status_strip,
    _draw_workspace_right_dock,
)
from .state import ProbeState
from .widgets import (
    _circular_icon_button,
    _draw_hint_bar,
    _draw_joint_value_input,
    _draw_playback,
    _draw_tool_column,
    _draw_value_input,
    _hint_bar_width,
)


def _draw_viewport(size, scale: float, state: ProbeState) -> tuple[float, float, float, float]:
    child_flags = _flags(imgui.ChildFlags_.borders)
    window_flags = _flags(imgui.WindowFlags_.no_scrollbar, imgui.WindowFlags_.no_scroll_with_mouse)
    if not imgui.begin_child("Viewport###ProbeViewport", size, child_flags, window_flags):
        imgui.end_child()
        return (0.0, 0.0, 0.0, 0.0)

    lo = imgui.get_window_pos()
    window_size = imgui.get_window_size()
    x0, y0 = float(lo.x), float(lo.y)
    x1, y1 = x0 + float(window_size.x), y0 + float(window_size.y)
    draw = state.painter()
    cell = 28.0 * scale
    row = 0
    y = y0
    while y < y1:
        column = 0
        x = x0
        while x < x1:
            draw.rect_filled(
                (x, y),
                (min(x + cell, x1), min(y + cell, y1)),
                VIEW_A if (row + column) % 2 == 0 else VIEW_B,
            )
            x += cell
            column += 1
        y += cell
        row += 1

    draw.text((x0 + 14 * scale, y0 + 11 * scale), CONCEPT_THEME.text_disabled, "joint_types.xml")
    if state.show_construction_notes:
        state_radius = state.overlay_icon_radius + state.overlay_radial_step
        shell_radius = state_radius + state.overlay_radial_step
        draw.text(
            (x0 + 14 * scale, y0 + 42 * scale),
            CONCEPT_THEME.text_disabled,
            f"Geometry  icon Ø{state.overlay_icon_radius * 2} amber · "
            f"state Ø{state_radius * 2} green · shell {shell_radius * 2} · "
            f"centers {state.overlay_center_step}",
        )
    shell_radius = state.overlay_icon_radius + state.overlay_radial_step * 2
    if state.show_playback:
        playback_width = (end_padding(state) * 2.0 + state.overlay_center_step * 3.0) * scale
        _draw_playback(
            draw,
            (x0 + (window_size.x - playback_width) * 0.5, y0 + 18 * scale),
            scale,
            state,
        )
    if state.show_tool_column:
        _draw_tool_column(draw, (x0 + 18 * scale, y0 + 92 * scale), scale, state)
    if state.show_joint_gizmos:
        _draw_joint_gizmo(
            draw,
            (x0 + 250 * scale, y0 + 250 * scale),
            scale,
            state,
            item_id="workspace-joint",
            show_limit_labels=False,
        )
    if state.show_context_hints:
        hint_width = _hint_bar_width(draw, scale, state)
        hint_height = (state.hint_control_height + state.hint_padding_y * 2.0) * scale
        _draw_hint_bar(
            draw,
            (
                x0 + (window_size.x - hint_width) * 0.5,
                y1 - hint_height - 16.0 * scale,
            ),
            scale,
            state,
        )

    imgui.end_child()
    return (x0, y0, x1, y1)


def _draw_workspace_canvas(available, scale: float, state: ProbeState):
    flags = _flags(
        imgui.WindowFlags_.horizontal_scrollbar,
        imgui.WindowFlags_.always_vertical_scrollbar,
    )
    if not imgui.begin_child(
        "Editor preview canvas###ProbeWorkspaceCanvas",
        available,
        imgui.ChildFlags_.none.value,
        flags,
    ):
        imgui.end_child()
        return None

    cursor = imgui.get_cursor_pos()
    origin = imgui.get_cursor_screen_pos()
    width, height = _virtual_canvas_size(available, scale, WORKSPACE_CANVAS_SIZE)
    spacing = imgui.get_style().item_spacing
    status_height = 34.0 * scale
    main_height = height - status_height - spacing.y
    left_width = 330.0 * scale
    right_width = 390.0 * scale
    center_width = width - left_width - right_width - spacing.x * 2.0
    bottom_height = 250.0 * scale
    viewport_height = main_height - bottom_height - spacing.y
    right_top_height = main_height * 0.50
    right_bottom_height = main_height - right_top_height - spacing.y

    left_x = origin.x
    center_x = left_x + left_width + spacing.x
    right_x = center_x + center_width + spacing.x
    top_y = origin.y

    imgui.set_cursor_screen_pos(imgui.ImVec2(left_x, top_y))
    _draw_hierarchy_gallery(imgui.ImVec2(left_width, main_height), state, scale)

    imgui.set_cursor_screen_pos(imgui.ImVec2(center_x, top_y))
    viewport_rect = _draw_viewport(imgui.ImVec2(center_width, viewport_height), scale, state)
    imgui.set_cursor_screen_pos(imgui.ImVec2(center_x, top_y + viewport_height + spacing.y))
    _draw_output(imgui.ImVec2(center_width, bottom_height), state, scale)

    imgui.set_cursor_screen_pos(imgui.ImVec2(right_x, top_y))
    _draw_workspace_right_dock(imgui.ImVec2(right_width, right_top_height), state, scale)
    imgui.set_cursor_screen_pos(imgui.ImVec2(right_x, top_y + right_top_height + spacing.y))
    _draw_inspector_gallery(imgui.ImVec2(right_width, right_bottom_height), scale)

    status_origin = (origin.x, origin.y + main_height + spacing.y)
    _draw_status_strip(
        state.painter(),
        status_origin,
        width,
        status_height,
        scale,
        selected="a_sphere",
        running=state.playing,
        fps="60.0" if state.playing else "119.8",
        backend=state.renderer,
    )

    imgui.set_cursor_pos(imgui.ImVec2(cursor.x + width - 1.0, cursor.y + height - 1.0))
    imgui.dummy(imgui.ImVec2(1.0, 1.0))
    imgui.end_child()
    return viewport_rect


def _draw_workspace(window: Window, state: ProbeState) -> None:
    state.painter = state.redesign.painter = window.painter
    scale = window.style_scale
    theme_mod.apply_corner_radius(imgui, state.imgui_rounding, scale)
    display = imgui.get_io().display_size
    imgui.set_next_window_pos(imgui.ImVec2(0.0, 0.0))
    imgui.set_next_window_size(display)
    flags = _flags(
        imgui.WindowFlags_.no_title_bar,
        imgui.WindowFlags_.no_resize,
        imgui.WindowFlags_.no_move,
        imgui.WindowFlags_.no_collapse,
        imgui.WindowFlags_.no_saved_settings,
        imgui.WindowFlags_.menu_bar,
    )
    opened, _ = imgui.begin("Mojive UI Probe", True, flags)
    if not opened:
        imgui.end()
        return

    if imgui.begin_menu_bar():
        for label in ("File", "Edit", "Entity", "View", "Window", "Help"):
            if imgui.begin_menu(label):
                imgui.menu_item("Design probe", "", False, False)
                imgui.end_menu()
        if imgui.begin_menu("Probe"):
            for page in ("Workspace", "Panels", "Geometry", "Components", "Redesign"):
                clicked, _ = imgui.menu_item(page, "", state.page == page)
                if clicked:
                    state.page = page
            imgui.separator()
            clicked, _ = imgui.menu_item(
                "Preview Icon Library",
                "",
                state.preview_icon_library,
            )
            if clicked:
                state.preview_icon_library = not state.preview_icon_library
            imgui.set_item_tooltip("Use Icon Library candidates in the actual feasibility layouts")
            clicked, _ = imgui.menu_item(
                "Apply icon X/Y offsets", "", state.apply_icon_offsets, state.preview_icon_library
            )
            if clicked:
                state.apply_icon_offsets = not state.apply_icon_offsets
            imgui.set_item_tooltip(
                "Compare candidate UI previews with and without their manual or automatic offsets. "
                "Enable Preview Icon Library first. Stored offsets and library editing stay unchanged."
            )
            clicked, _ = imgui.menu_item(
                "Link mirrored icon offsets", "", state.link_mirrored_icon_offsets
            )
            if clicked:
                state.link_mirrored_icon_offsets = not state.link_mirrored_icon_offsets
            imgui.set_item_tooltip(
                "Subsequent manual edits reflect X and copy Y to the paired glyph. "
                "Existing values are preserved until an edit or reset."
            )
            imgui.separator()
            _, state.imgui_rounding = imgui.slider_float(
                "ImGui corner radius",
                state.imgui_rounding,
                0.0,
                16.0,
                "%.2f px",
                imgui.SliderFlags_.always_clamp.value,
            )
            imgui.separator()
            imgui.text_disabled("Viewport overlays")
            for label, attribute in (
                ("Playback", "show_playback"),
                ("Tool column", "show_tool_column"),
                ("Joint gizmos", "show_joint_gizmos"),
                ("Context hints", "show_context_hints"),
                ("Value input", "value_open"),
                ("Joint value input", "joint_value_open"),
            ):
                value = bool(getattr(state, attribute))
                clicked, _ = imgui.menu_item(label, "", value)
                if clicked:
                    setattr(state, attribute, not value)
            imgui.separator()
            imgui.text_disabled("Construction")
            all_construction = bool(
                state.show_icon_bounds
                and state.show_state_circles
                and state.show_construction_notes
            )
            clicked, _ = imgui.menu_item("All construction", "", all_construction)
            if clicked:
                target = not all_construction
                state.show_icon_bounds = target
                state.show_state_circles = target
                state.show_construction_notes = target
            for label, attribute in (
                ("Icon bounds", "show_icon_bounds"),
                ("State circles", "show_state_circles"),
                ("Geometry notes", "show_construction_notes"),
                ("G3 transitions", "highlight_g3"),
            ):
                value = bool(getattr(state, attribute))
                clicked, _ = imgui.menu_item(label, "", value)
                if clicked:
                    setattr(state, attribute, not value)
            imgui.end_menu()
        imgui.begin_disabled(state.page == "Components")
        clicked, _ = imgui.menu_item(
            (
                "Icons preview ON###icon-library-preview"
                if state.preview_icon_library
                else "Icons preview OFF###icon-library-preview"
            ),
            "",
            state.preview_icon_library,
        )
        if clicked:
            state.preview_icon_library = not state.preview_icon_library
        imgui.end_disabled()
        imgui.set_item_tooltip(
            "Components compares production glyphs with local offsets"
            if state.page == "Components"
            else "Substitute candidates throughout the current page"
        )
        imgui.end_menu_bar()

    available = imgui.get_content_region_avail()
    if state.page == "Components":
        draw_components(state, scale)
        imgui.end()
        return
    if state.page == "Redesign":
        draw_redesign(
            state.redesign,
            scale,
            state,
            circular_button=_circular_icon_button,
        )
        imgui.end()
        return
    if state.page == "Panels":
        _draw_panel_page(available, state, scale)
        imgui.end()
        return
    if state.page == "Geometry":
        _draw_geometry_page(available, scale, state)
        imgui.end()
        viewport = imgui.get_main_viewport()
        _draw_joint_value_input(
            (
                viewport.work_pos.x + 410.0 * scale,
                viewport.work_pos.y + 250.0 * scale,
            ),
            scale,
            state,
        )
        return

    viewport_rect = _draw_workspace_canvas(available, scale, state)
    imgui.end()

    if viewport_rect is None:
        return

    _draw_value_input(
        (viewport_rect[2] - 326.0 * scale, viewport_rect[1] + 74.0 * scale),
        scale,
        state,
    )
    _draw_joint_value_input(
        (viewport_rect[0] + 420.0 * scale, viewport_rect[1] + 250.0 * scale),
        scale,
        state,
    )
