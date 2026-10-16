"""Native workspace panel host; shared scene edits use Session commands."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from imgui_bundle import imgui

from mojive import commands as cmd
from mojive.ui.workspace.component_layout import control_frame_size, control_size
from mojive.ui.workspace.constants import BOTTOM_PANELS, LEFT_PANELS
from mojive.ui.workspace.design import BODY_SIZE, DESIGN, current_design, use_design
from mojive.ui.workspace.property_style import property_controls
from mojive.ui.workspace.widgets import (
    TEXT,
)


def panel(workspace, name, ctx):
    visibility = getattr(workspace.app, "scrollbar_visibility", None)
    if visibility is not None:
        visibility.set_visible(name, workspace.open[name])
    if visibility is None or not workspace.open[name]:
        return _panel(workspace, name, ctx)
    with visibility.panel_scope(name, window_name=name):
        return _panel(workspace, name, ctx)


def _panel(workspace, name, ctx):
    panel = workspace.app.panels.get(name)
    u = workspace.ui
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
    panel_style = u.component_style("panel")
    imgui.push_style_var(
        imgui.StyleVar_.window_rounding,
        panel_style.get("radius", DESIGN.container_radius) * workspace.s,
    )
    visible, workspace.open[name] = imgui.begin(title, True, flags)
    visibility = getattr(workspace.app, "scrollbar_visibility", None)
    if visibility is not None:
        visibility.bind_current_window()
    imgui.pop_style_var()
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
        outer = u.component_box("panel", x, y, w, h, key="panel-" + name, observe=False)
        window = imgui.get_current_context().current_window
        background_command = next(
            (command for command in window.draw_list.cmd_buffer if command.elem_count > 0), None
        )
        u.record_component_box(
            "panel",
            outer,
            key="panel-" + name,
            windowRect=[window.pos.x, window.pos.y, window.size.x, window.size.y],
            docked=bool(window.dock_id),
            nativeBackgroundClip=list(background_command.clip_rect)
            if background_command is not None
            else None,
            consumedFields={
                "panel": [
                    "widthMode",
                    "width",
                    "heightMode",
                    "height",
                    "radius",
                    "marginLeft",
                    "marginRight",
                    "marginTop",
                    "marginBottom",
                    *(("rowGap",) if name in (*LEFT_PANELS, "Inspector") else ()),
                ]
            },
            spacing="header-content" if name in (*LEFT_PANELS, "Inspector") else None,
        )
        x, y, w, h = outer.content.tuple()
        if name in LEFT_PANELS:
            body_top = workspace.panel_header(name, x, y, w)
            extra_gap = panel_style.get("rowGap", 0) * workspace.s
            body_top += extra_gap
            h -= body_top - y
            y = body_top
        if name != "Inspector":
            body = u.component_box(
                "panel_body",
                x,
                y,
                w,
                max(workspace.s, h),
                key="panel-body-" + name,
                observe=False,
            )
            u.record_component_box(
                "panel_body",
                body,
                key="panel-body-" + name,
                ownedSpacingPart={
                    "Scene": "tree_row",
                    "Statistics": "statistics_list",
                    "Output": "output_list",
                    "Timeline": "timeline_sheet",
                    "Sensors": "sensor",
                }.get(name),
            )
            x, y, w, h = body.content.tuple()
            imgui.set_cursor_screen_pos((x, y))
            imgui.push_style_var(imgui.StyleVar_.window_padding, (0, 0))
            # A stable gutter prevents numeric reflow from changing its own
            # available width when the new row count removes vertical overflow.
            flags = (
                imgui.WindowFlags_.always_vertical_scrollbar
                if name in ("Pose", "Control", "Cameras")
                else 0
            )
            imgui.begin_child("Panel content", (w, max(workspace.s, h)), window_flags=flags)
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
        body_style = u.component_style("panel_body")
        dimensions = replace(current_design(), row_gap=body_style.get("rowGap", DESIGN.row_gap))
        with (
            use_design(lambda _: dimensions),
            u.component_scope("panel_body"),
            property_controls(workspace.ui),
        ):
            if name == "Settings":
                workspace.settings.draw(ctx)
            elif name in ("Scene", "Inspector", "Timeline"):
                draw(x, y, w, h)
            elif draw is not None:
                draw(x, y, w)
            elif name in ("Sensors", "Output", "Statistics", "Plot"):
                panel.draw(ctx)
        if name != "Inspector":
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
    style = u.component_style("panel_header")
    padding = style.edges("padding", s)
    margin = style.edges("margin", s)
    font = style.get("fontSize", BODY_SIZE)
    gap = style.get("columnGap", style.get("gap", 2)) * s
    icon_size = style.get("iconSize", 16)
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
        from mojive.ui.workspace.navigation import Navigation

        workspace.navigation = Navigation(workspace)
    enabled = [item for item in actions if item[0] in workspace.header_actions[name]]
    sizes = {
        key: control_size(u, "header-" + name + "-" + key, 26, 28, icon=icon, icon_size=icon_size)
        for key, icon, _label in (*enabled, ("more", "more", "More"))
    }
    height = max(
        style.get("height", 30) * s,
        max(max(font, style.line_height(font * 1.4)) * s, *(v[1] for v in sizes.values()))
        + padding[1]
        + padding[3],
    )
    box = u.component_box(
        "panel_header", x, y, w, height + margin[1] + margin[3], key="panel-header-" + name
    )
    x, y, w, _ = box.content.tuple()
    # Keep the title readable as dock panels narrow; remaining actions stay in More.
    more_width = min(w, sizes["more"][0])
    available = max(0, w - 74 * s - more_width - gap)
    visible = []
    for item in enabled:
        required = sizes[item[0]][0] + gap
        if required > available:
            break
        visible.append(item)
        available -= required
    right = x + w - more_width
    action_x = right - sum(sizes[item[0]][0] + gap for item in visible)
    from mojive.design.native.geometry import Rect

    title_rect = Rect(x, y, max(0, action_x - x - 4 * s), box.content.height)
    u.component_text("panel_header", title_rect, name, TEXT, bold=True)
    vertical = style.get("alignItems", "center")
    factor = 0 if vertical in ("start", "top") else 1 if vertical in ("end", "bottom") else 0.5
    for key, icon, label in visible:
        width, height = sizes[key]
        button_y = y + max(0, box.content.height - height) * factor
        frame = control_frame_size(u, "header-" + name + "-" + key, width, height, icon=icon)
        active = (
            workspace.navigation.pose_action_active(key)
            if name == "Pose"
            else (key == "search" and workspace.search_visible)
            or (key == "gizmos" and workspace.app.gizmos)
        )
        if u.button(
            "header-" + name + "-" + key,
            action_x,
            button_y,
            *frame,
            icon=icon,
            tooltip=label,
            active=active,
            icon_size=icon_size,
            width_limit=frame[0] * s,
        ):
            workspace.header_action(name, key)
        action_x += width + gap
    popup = "Panel actions###" + name
    filtered = name == "Scene" and workspace.filter != "All"
    hint = u.translate("Filter") + ": " + u.translate(workspace.filter) if filtered else "More"
    more_frame = control_frame_size(
        u, "header-" + name + "-more", more_width, sizes["more"][1], icon="more"
    )
    if u.button(
        "header-" + name + "-more",
        right,
        y + max(0, box.content.height - sizes["more"][1]) * factor,
        *more_frame,
        icon="more",
        tooltip=hint,
        active=filtered,
        icon_size=icon_size,
        width_limit=more_frame[0] * s,
    ):
        imgui.open_popup(popup)
    with workspace.menus.local_popup(popup) as opened:
        if opened:
            for key, _icon, label in actions:
                clicked, _ = workspace.menus.local_item(
                    label, key="header-menu-" + name + "-" + key, icon=_icon
                )
                if clicked:
                    workspace.header_action(name, key)
            if name == "Scene":
                imgui.separator()
                with workspace.menus.local_submenu(
                    "Filter", key="header-menu-" + name + "-filters"
                ) as filters_open:
                    if filters_open:
                        from mojive.ui.workspace.navigation import Navigation

                        if not hasattr(workspace, "navigation"):
                            workspace.navigation = Navigation(workspace)
                        for kind, count in workspace.navigation.scene_filters(
                            workspace._panel_context
                        ):
                            label = f"{u.translate(kind.title())} {count}"
                            clicked, _ = workspace.menus.local_item(
                                label,
                                checked=workspace.filter.casefold() == kind,
                                key="header-filter-" + kind,
                            )
                            if clicked:
                                workspace.filter = kind.title()
            if actions:
                imgui.separator()
                with workspace.menus.local_submenu(
                    "Header actions", key="header-menu-" + name + "-actions"
                ) as actions_open:
                    if actions_open:
                        for key, _icon, label in actions:
                            pos = imgui.get_cursor_screen_pos()
                            changed, checked = u.checkbox(
                                "##header-action-" + name + "-" + key,
                                pos.x,
                                pos.y,
                                key in workspace.header_actions[name],
                                label=u.translate(label),
                            )
                            if changed:
                                if checked:
                                    workspace.header_actions[name].add(key)
                                else:
                                    workspace.header_actions[name].discard(key)
            imgui.separator()
            clicked, _ = workspace.menus.local_item(
                "Hide panel", key="header-menu-" + name + "-hide", icon="close"
            )
            if clicked:
                workspace.open[name] = False
    if name == "Scene":
        if workspace.header_add_pending or imgui.is_popup_open("Add"):
            anchor = u.hits.get("header-Scene-add", u.hits["header-Scene-more"])
            workspace.menus.popup("Add", anchor, workspace.header_add_pending)
        workspace.header_add_pending = False
    u.line(box.frame.x - 12 * s, box.outer.bottom + 2 * s, box.frame.width + 24 * s)
    return box.outer.bottom + 8 * s


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
