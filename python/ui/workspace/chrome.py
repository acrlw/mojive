"""Native workspace chrome; shared scene edits use Session commands."""

from __future__ import annotations

from imgui_bundle import imgui

from .constants import ICONS, LEFT_PANELS, PANELS
from .design import BODY_SIZE, DESIGN
from .widgets import (
    DIM,
    MUTED,
    SAGE,
)


def sidebar(workspace, name, direction, size):
    flags = (
        imgui.WindowFlags_.no_decoration
        | imgui.WindowFlags_.no_docking
        | imgui.WindowFlags_.no_move
        | imgui.WindowFlags_.no_saved_settings
    )
    if name == "Application":
        flags |= imgui.WindowFlags_.menu_bar
    imgui.push_style_var(imgui.StyleVar_.window_padding, (0, 0))
    visible = imgui.internal.begin_viewport_side_bar(
        name, imgui.get_main_viewport(), direction, size * workspace.s, flags
    )
    return visible


def end_sidebar(workspace):
    imgui.end()
    imgui.pop_style_var()


def chrome(workspace):
    u = workspace.ui
    s = workspace.s
    height = DESIGN.application_height
    action_y = (height - DESIGN.control_height) / 2
    imgui.push_style_var(imgui.StyleVar_.frame_padding, (8 * s, (height - BODY_SIZE) * s / 2))
    if workspace.sidebar("Application", imgui.Dir.up, height):
        p = imgui.get_window_pos()
        w = imgui.get_window_width()
        x = p.x
        y = p.y
        imgui.push_clip_rect((x, y), (x + w, y + height * s), False)
        workspace.menus.bar()
        center_y = workspace.menus.menu_center_y
        brand_width = u.measure("M", 18)
        u.text(x + (44 * s - brand_width) / 2, center_y - 9 * s, "M", SAGE, 18, bold=True)
        workspace.rects["application-logo"] = [x + 8 * s, center_y - 9 * s, 28 * s, 18 * s]
        if w > 950 * s:
            tx = x + w * 0.49
            u.role_icon("file", tx, y + action_y * s, 14 * s, 28 * s, role="inline", c=DIM)
            path = workspace.app.session.asset_path
            u.text(
                tx + 22 * s,
                y + (height - BODY_SIZE) * s / 2,
                path.name if path else "Untitled",
                MUTED,
                BODY_SIZE,
            )
        if u.button(
            "toggle-inspector",
            x + w - 57 * s,
            y + action_y * s,
            26,
            28,
            icon="inspect",
            active=workspace.open["Inspector"],
            tooltip="Toggle Inspector",
        ):
            workspace.open["Inspector"] = not workspace.open["Inspector"]
        settings_clicked = u.button(
            "preferences",
            x + w - 29 * s,
            y + action_y * s,
            26,
            28,
            icon="settings",
            tooltip="Settings · F9",
            active=workspace.open["Settings"],
        )
        if settings_clicked:
            workspace.activate("Settings")
        workspace.menus.dialogs()
        imgui.pop_clip_rect()
    workspace.end_sidebar()
    imgui.pop_style_var()
    if workspace.sidebar("Navigation", imgui.Dir.left, DESIGN.sidebar_width):
        p = imgui.get_window_pos()
        h = imgui.get_window_height()
        compact = h < 470 * s
        for n, (name, icon) in enumerate(zip(PANELS, ICONS, strict=True)):
            yy = (
                p.y + (8 + n * 42) * s - imgui.get_scroll_y()
                if compact or n < 7
                else p.y + h - (3 - (n - 7)) * 42 * s
            )
            if u.button(
                "rail-" + name,
                p.x + 6 * s,
                yy,
                36,
                36,
                icon=icon,
                icon_role="activity",
                active=(
                    workspace.left_active == name
                    if name in LEFT_PANELS
                    else workspace.bottom_active == name
                )
                and workspace.open[name],
                tooltip=name,
            ):
                workspace.toggle_activity(name)
        if compact:
            imgui.set_cursor_screen_pos((p.x, p.y + 440 * s - imgui.get_scroll_y()))
            imgui.dummy((1, 1))
    workspace.end_sidebar()
