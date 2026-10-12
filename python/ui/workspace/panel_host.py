"""Native workspace panel host; shared scene edits use Session commands."""

from __future__ import annotations

import numpy as np
from imgui_bundle import imgui

from mojive import commands as cmd

from .constants import BOTTOM_PANELS, LEFT_PANELS
from .design import BODY_SIZE, DESIGN
from .property_style import property_controls
from .widgets import (
    TEXT,
)


def panel(workspace, name, ctx):
    panel = workspace.app.panels.get(name)
    if not workspace.open[name]:
        if name == "Inspector":
            presentation = getattr(workspace, "inspector_presentation", None)
            if presentation is not None and presentation.color_edit_open:
                if panel._edit_transaction:
                    ctx.submit(cmd.CancelEditTransaction())
                    panel._edit_transaction = False
                presentation.color_edit_open = False
                presentation.color_edit_owner = None
        panel.finish_frame(ctx)
        return False
    if getattr(workspace, "new_panel", None) == name:
        if name == "Settings":
            viewport = imgui.get_main_viewport()
            size = (
                min(820 * workspace.s, viewport.size.x - 32 * workspace.s),
                min(620 * workspace.s, viewport.size.y - 64 * workspace.s),
            )
            imgui.set_next_window_dock_id(0, imgui.Cond_.always)
            imgui.set_next_window_pos(
                (
                    viewport.pos.x + (viewport.size.x - size[0]) / 2,
                    viewport.pos.y + (viewport.size.y - size[1]) / 2,
                ),
                imgui.Cond_.appearing,
            )
            imgui.set_next_window_size(size, imgui.Cond_.appearing)
        elif name in LEFT_PANELS:
            scene = imgui.internal.find_window_by_name(workspace.left_active)
            target = scene.dock_id if scene is not None else getattr(workspace, "left", 0)
            imgui.set_next_window_dock_id(target, imgui.Cond_.first_use_ever)
        workspace.new_panel = None
    if name == "Inspector" and workspace.float_next:
        viewport = imgui.get_main_viewport()
        size = (
            min(320 * workspace.s, viewport.work_size.x - 24 * workspace.s),
            min(570 * workspace.s, viewport.work_size.y - 24 * workspace.s),
        )
        imgui.set_next_window_dock_id(0, imgui.Cond_.always)
        imgui.set_next_window_pos(
            (
                viewport.work_pos.x + (viewport.work_size.x - size[0]) / 2,
                viewport.work_pos.y + (viewport.work_size.y - size[1]) / 2,
            ),
            imgui.Cond_.always,
        )
        imgui.set_next_window_size(size, imgui.Cond_.always)
        workspace.float_next = False
    if workspace.focus_next == name or workspace.bottom_focus == name:
        imgui.set_next_window_focus()
        workspace.focus_next = None
        window = imgui.internal.find_window_by_name(name)
        if (
            window is not None
            and window.dock_node is not None
            and window.dock_node.tab_bar is not None
        ):
            window.dock_node.tab_bar.next_selected_tab_id = window.tab_id
    wc = imgui.WindowClass()
    wc.dock_node_flags_override_set = (
        imgui.internal.DockNodeFlagsPrivate_.no_window_menu_button
        | imgui.internal.DockNodeFlagsPrivate_.no_close_button
    )
    window = imgui.internal.find_window_by_name(name)
    solo_inspector = name == "Inspector" and (
        window is None or window.dock_node is None or len(window.dock_node.windows) <= 1
    )
    if (name in LEFT_PANELS and not workspace.inspector_grouped_with(name)) or solo_inspector:
        wc.dock_node_flags_override_set |= imgui.internal.DockNodeFlagsPrivate_.no_tab_bar
    imgui.set_next_window_class(wc)
    title = ctx.tr(name) + "###" + name if ctx.tr(name) != name else name
    flags = (
        imgui.WindowFlags_.no_title_bar if name in (*LEFT_PANELS, "Inspector", "Settings") else 0
    )
    if name == "Settings":
        imgui.set_next_window_dock_id(0, imgui.Cond_.always)
        flags |= imgui.WindowFlags_.no_docking
    if name in LEFT_PANELS:
        flags |= imgui.WindowFlags_.no_scrollbar | imgui.WindowFlags_.no_scroll_with_mouse
    visible, workspace.open[name] = imgui.begin(title, True, flags)
    if visible:
        if name in BOTTOM_PANELS:
            window = imgui.internal.get_current_window()
            if window.dock_node is not None and window.dock_node.selected_tab_id == window.tab_id:
                workspace.bottom_active = name
                if workspace.bottom_focus == name:
                    workspace.bottom_focus = None
        workspace.widths[name] = imgui.get_content_region_avail().x
        workspace._panel_context = ctx
        workspace.ui.value_context = ctx
        p = imgui.get_cursor_screen_pos()
        size = imgui.get_content_region_avail()
        x, y, w, h = p.x, p.y, size.x, size.y
        if name in LEFT_PANELS:
            workspace.panel_header(name, x, y, w)
            y += 38 * workspace.s
            h -= 38 * workspace.s
            imgui.set_cursor_screen_pos((x, y))
            imgui.push_style_var(imgui.StyleVar_.window_padding, (0, 0))
            imgui.begin_child("Panel content", (w, max(workspace.s, h)))
            imgui.pop_style_var()
            p = imgui.get_cursor_screen_pos()
            size = imgui.get_content_region_avail()
            x, y, w, h = p.x, p.y, size.x, size.y
        draw = {
            "Scene": workspace.scene_panel,
            "Inspector": workspace.inspector,
            "Control": workspace.control_panel,
            "Assets": workspace.assets_panel,
            "Pose": workspace.pose_panel,
            "Cameras": workspace.camera_panel,
            "Layers": workspace.layers_panel,
            "Timeline": workspace.timeline,
        }.get(name)
        with property_controls(workspace.ui):
            if name == "Settings":
                workspace.settings.draw(ctx)
            elif name in ("Scene", "Inspector", "Timeline"):
                draw(x, y, w, h)
            elif draw is not None:
                draw(x, y, w)
            elif name in ("Sensors", "Output", "Statistics", "Plot"):
                panel.draw(ctx)
        if name in LEFT_PANELS:
            imgui.end_child()
        workspace.app.apply_document_edits()
    if name == "Inspector":
        panel.finish_frame(
            ctx,
            keep_edit_transaction=bool(
                getattr(
                    getattr(workspace, "inspector_presentation", None), "color_edit_open", False
                )
            ),
        )
    else:
        panel.finish_frame(ctx)
    panel.open = workspace.open[name]
    imgui.end()
    return visible and workspace.open[name]


def panel_header(workspace, name, x, y, w):
    u, s = workspace.ui, workspace.s
    actions = {
        "Scene": (
            ("search", "search", "Search scene"),
            ("frame", "frame", "Frame all"),
            ("add", "plus", "Add"),
        ),
        "Control": (("reset", "reset", "Reset controls"),),
        "Pose": (
            ("search", "search", "Search joints"),
            ("sort", "sort", "Sort by name"),
            ("copy", "copy", "Copy state"),
        ),
        "Cameras": (("frame", "frame", "Frame all"),),
        "Layers": (("gizmos", "move", "Transform handles"),),
    }.get(name, ())
    if name == "Pose" and not hasattr(workspace, "navigation"):
        from .navigation import Navigation

        workspace.navigation = Navigation(workspace)
    enabled = [item for item in actions if item[0] in workspace.header_actions[name]]
    # Keep the title readable as dock panels narrow; remaining actions stay in More.
    slots = max(0, int((w / s - 74) / 28) - 1)
    visible = enabled[:slots]
    right = x + w - 28 * s
    imgui.push_clip_rect((x, y), (right - len(visible) * 28 * s - 4 * s, y + 30 * s), True)
    u.text(x, y + DESIGN.control_padding_y(BODY_SIZE) * s, name, TEXT, BODY_SIZE, bold=True)
    imgui.pop_clip_rect()
    for index, (key, icon, label) in enumerate(visible):
        xx = right - (len(visible) - index) * 28 * s
        active = (
            workspace.navigation.pose_action_active(key)
            if name == "Pose"
            else (key == "search" and workspace.search_visible)
            or (key == "gizmos" and workspace.app.gizmos)
        )
        if u.button(
            "header-" + name + "-" + key, xx, y, 26, 28, icon=icon, tooltip=label, active=active
        ):
            workspace.header_action(name, key)
    popup = "Panel actions###" + name
    filtered = name == "Scene" and workspace.filter != "All"
    hint = u.translate("Filter") + ": " + u.translate(workspace.filter) if filtered else "More"
    if u.button(
        "header-" + name + "-more", right, y, 26, 28, icon="more", tooltip=hint, active=filtered
    ):
        imgui.open_popup(popup)
    with workspace.menus.style():
        if imgui.begin_popup(popup):
            for key, _icon, label in actions:
                clicked, _ = imgui.menu_item(u.translate(label), "", False)
                if clicked:
                    workspace.header_action(name, key)
            if name == "Scene":
                imgui.separator()
                if imgui.begin_menu(u.translate("Filter")):
                    from .navigation import Navigation

                    if not hasattr(workspace, "navigation"):
                        workspace.navigation = Navigation(workspace)
                    for kind, count in workspace.navigation.scene_filters(workspace._panel_context):
                        label = f"{u.translate(kind.title())} {count}"
                        clicked, _ = imgui.menu_item(label, "", workspace.filter.casefold() == kind)
                        if clicked:
                            workspace.filter = kind.title()
                    imgui.end_menu()
            if actions:
                imgui.separator()
                if imgui.begin_menu(u.translate("Header actions")):
                    for key, _icon, label in actions:
                        changed, checked = imgui.checkbox(
                            u.translate(label), key in workspace.header_actions[name]
                        )
                        if changed:
                            if checked:
                                workspace.header_actions[name].add(key)
                            else:
                                workspace.header_actions[name].discard(key)
                    imgui.end_menu()
            imgui.separator()
            clicked, _ = imgui.menu_item(workspace.app.localizer.text("Hide panel"), "", False)
            if clicked:
                workspace.open[name] = False
            imgui.end_popup()
    if name == "Scene":
        if workspace.header_add_pending or imgui.is_popup_open("Add"):
            workspace.menus.popup("Add", (right, y, 26 * s, 28 * s), workspace.header_add_pending)
        workspace.header_add_pending = False
    u.line(x - 12 * s, y + 32 * s, w + 24 * s)


def header_action(workspace, name, key):
    if name == "Pose":
        workspace.navigation.pose_action(key)
    elif key == "search":
        workspace.search_visible = not workspace.search_visible
        if not workspace.search_visible:
            workspace.query = ""
    elif key == "frame":
        workspace.app._frame_scene(animate=True)
    elif key == "add":
        workspace.header_add_pending = True
    elif name == "Control" and key == "reset":
        session = workspace.app.session
        if session.adapter.caps.write_ctrl and session.frame.ctrl is not None:
            session.submit(cmd.SetCtrlVector(np.zeros_like(session.frame.ctrl)))
    elif key == "gizmos":
        workspace.app.gizmos = not workspace.app.gizmos
