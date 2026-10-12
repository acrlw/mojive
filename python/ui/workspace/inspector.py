"""Native workspace inspector; shared scene edits use Session commands."""

from __future__ import annotations

import numpy as np
from imgui_bundle import imgui

from .design import BODY_SIZE, DESIGN, TITLE_SIZE
from .inspector_components import draw_component
from .property_style import property_toggle
from .widgets import (
    MUTED,
    SAGE,
    TEXT,
    color,
)


def inspector(workspace, x, y, w, h):
    u = workspace.ui
    s = workspace.s
    e = (
        next((e for e in workspace.app.entities if e["identity"] == workspace.pinned), None)
        if workspace.pinned
        else workspace.app.entity
    )
    if workspace.pinned and e is None:
        workspace.pinned = None
        e = workspace.app.entity
    imgui.set_cursor_screen_pos((x, y))
    imgui.invisible_button("inspector-header", (max(s, w - 60 * s), 30 * s))
    u.hits["inspector-header"] = (x, y, max(s, w - 60 * s), 30 * s)
    if imgui.is_item_active() and imgui.is_mouse_dragging(0):
        window = imgui.internal.get_current_window()
        if window.dock_node is not None:
            imgui.internal.dock_context_queue_undock_window(imgui.get_current_context(), window)
        else:
            imgui.internal.start_mouse_moving_window(window)
    u.role_icon(
        e["icon"] if e else "inspect",
        x,
        y,
        20 * s,
        30 * s,
        role="title",
        c=e["icon_color"] if e else MUTED,
    )
    imgui.push_clip_rect((x, y), (x + w - 64 * s, y + 30 * s), True)
    u.text(
        x + 28 * s,
        y + (30 - TITLE_SIZE) * s / 2,
        e["name"] if e else "Inspector",
        TEXT,
        TITLE_SIZE,
        bold=True,
        max_width=max(0, w - 92 * s),
        translate=e is None,
    )
    imgui.pop_clip_rect()
    if (
        e
        and u.measure(e["name"], TITLE_SIZE, translate=False) > max(0, w - 92 * s)
        and imgui.is_item_hovered()
    ):
        u.tooltip(e["name"], translate=False)
    if u.button(
        "pin",
        x + w - 56 * s,
        y + s,
        26,
        28,
        icon="pin",
        icon_role="subtle",
        active=bool(workspace.pinned),
        enabled=e is not None,
        tooltip="Pin Inspector",
    ):
        workspace.pinned = None if workspace.pinned else e["identity"]
    if u.button(
        "inspector-close",
        x + w - 26 * s,
        y + s,
        26,
        28,
        icon="close",
        icon_role="subtle",
        tooltip="Close",
    ):
        workspace.open["Inspector"] = False
    if not e:
        u.text(x, y + 44 * s, "Select an entity to inspect.", MUTED)
        imgui.set_cursor_screen_pos((x, y + 44 * s))
        imgui.dummy((0, 28 * s))
        return
    tabs = ["Properties", "Physics", "Material", "Joint"]
    tab_icons = ("inspect", "body", "shading", "pose")
    tab_minimums = [u.measure(u.translate(tab)) + 12 * s for tab in tabs]
    compact_tabs = w < sum(tab_minimums)
    tab_x = x
    for n, t in enumerate(tabs):
        tw = w / 4 if compact_tabs else tab_minimums[n] + (w - sum(tab_minimums)) / 4
        tx = tab_x
        if u.button(
            "tab-" + t,
            tx,
            y + 38 * s,
            tw / s,
            32,
            label="" if compact_tabs else t,
            icon=tab_icons[n] if compact_tabs else None,
            tooltip=t if compact_tabs else None,
            align="center",
            text_color=SAGE if workspace.inspector_tab == t else MUTED,
            icon_color=SAGE if workspace.inspector_tab == t else MUTED,
        ):
            workspace.inspector_tab = t
        if workspace.inspector_tab == t:
            u.draw.add_line(
                (tx + 8 * s, y + 70 * s), (tx + tw - 8 * s, y + 70 * s), color(SAGE), 2 * s
            )
        tab_x += tw
    body_y = y + 82 * s
    imgui.set_cursor_screen_pos((x, body_y))
    imgui.push_style_var(imgui.StyleVar_.window_padding, (0, 0))
    imgui.begin_child("Inspector content", (w, max(s, h - 82 * s)))
    imgui.pop_style_var()
    p = imgui.get_cursor_screen_pos()
    bx, yy = p.x, p.y
    bw = imgui.get_content_region_avail().x
    original = next(
        (value for value in workspace.app.initial_entities if value["identity"] == e["identity"]),
        None,
    )
    if draw_component(workspace, workspace._panel_context, e, workspace.inspector_tab):
        yy = max(yy, imgui.get_cursor_screen_pos().y)
    elif workspace.inspector_tab == "Properties":
        if u.section("TRANSFORM", bx, yy, bw - 32 * s, workspace.expanded["TRANSFORM"]):
            workspace.expanded["TRANSFORM"] = not workspace.expanded["TRANSFORM"]
        if u.button(
            "restore-transform",
            bx + bw - 28 * s,
            yy + 6 * s,
            26,
            26,
            icon="reset",
            tooltip="Restore transform",
            enabled=not workspace.playing and e["posable"] and original is not None,
        ):
            for key in ("position", "rotation", "scale"):
                e[key] = original[key]
            workspace.app.apply_document_edits()
        if workspace.expanded["TRANSFORM"]:
            layout = u.vector_layout(
                bw,
                format="%.3g",
                values=(*e["position"], *e["rotation"], *e["scale"]),
                key=f"transform-{e['node_id']}",
            )
            row_y = yy + 42 * s
            for key, label, unit in (
                ("position", "Position", "m"),
                ("rotation", "Rotation", "°"),
                ("scale", "Scale", ""),
            ):
                imgui.begin_disabled(
                    workspace.playing or not e["scalable" if key == "scale" else "posable"]
                )
                changed, val = u.vector(
                    "##" + e["id"] + key,
                    bx,
                    row_y,
                    bw,
                    label,
                    e[key],
                    unit,
                    speed=0.5 if key == "rotation" else 0.01,
                    lo=0.01 if key == "scale" else 0,
                    hi=100 if key == "scale" else 0,
                    format="%.3g",
                    reset_values=(1.0, 1.0, 1.0) if key == "scale" else (0.0, 0.0, 0.0),
                    layout=layout,
                )
                imgui.end_disabled()
                edit_key = "##" + e["id"] + key
                state = workspace.begin_numeric_edit(edit_key, deferred=key == "scale")
                if (
                    changed
                    and not state.canceled
                    and all(np.isfinite(val))
                    and (key != "scale" or (min(val) >= 0.01 and max(val) <= 100))
                ):
                    e[key] = val
                workspace.app.apply_document_edits()
                workspace.finish_numeric_edit(edit_key, state)
                row_y += layout.height + DESIGN.row_gap * s
            yy = row_y + 16 * s
        else:
            yy += 40 * s
        if u.section("DISPLAY", bx, yy, bw, workspace.expanded["DISPLAY"]):
            workspace.expanded["DISPLAY"] = not workspace.expanded["DISPLAY"]
        if workspace.expanded["DISPLAY"]:
            imgui.set_cursor_screen_pos((bx, yy + 38 * s))
            e["hidden"] = not property_toggle(u, "entity-visible", "Visible", not e["hidden"])
            workspace.app.gizmos = property_toggle(
                u, "gizmos", "Transform handles", workspace.app.gizmos
            )
            yy = imgui.get_cursor_screen_pos().y
        else:
            yy += 40 * s
        if u.section("IDENTITY", bx, yy, bw, workspace.expanded["IDENTITY"]):
            workspace.expanded["IDENTITY"] = not workspace.expanded["IDENTITY"]
        yy += 42 * s
        if workspace.expanded["IDENTITY"]:
            u.text(bx, yy, "Object ID", MUTED, BODY_SIZE)
            u.text(bx + 100 * s, yy, str(e["object_id"]), TEXT, BODY_SIZE)
            u.text(bx, yy + 30 * s, "Type", MUTED, BODY_SIZE)
            u.text(bx + 100 * s, yy + 30 * s, e["type"], TEXT, BODY_SIZE)
            yy += 64 * s
    elif workspace.inspector_tab == "Material":
        u.section("SURFACE", bx, yy, bw)
        imgui.begin_disabled(not e["colorable"])
        workspace.menus.color_control(e, bx, yy + 48 * s, bw)
        layout = u.numeric_layout(bw, 4, column_options=(4, 2, 1))
        for n, axis in enumerate("RGBA"):
            dx, dy, field_w, _ = layout.fields[n]
            key = "##" + e["id"] + "color" + axis
            changed, value = u.input(
                key,
                bx + dx,
                yy + 112 * s + dy,
                field_w,
                e["rgba"][n],
                axis=axis,
                speed=0.005,
                lo=0,
                hi=1,
                format="%.3g",
                reset_value=original["rgba"][n] if original else (1.0 if axis == "A" else 0.0),
            )
            state = workspace.begin_numeric_edit(key)
            if changed and not state.canceled:
                e["rgba"][n] = value
                e["color"] = "#" + "".join(f"{round(v * 255):02x}" for v in e["rgba"][:3])
            workspace.app.apply_document_edits()
            workspace.finish_numeric_edit(key, state)
        imgui.end_disabled()
        yy += 136 * s + layout.height
        imgui.set_cursor_screen_pos((bx, yy))
        # Shared material ownership includes shape resources, textures, PBR,
        # assignments, and model asset actions beyond the native color shortcut.
        draw_component(workspace, workspace._panel_context, e, "Material", material_details=True)
        yy = imgui.get_cursor_screen_pos().y
    imgui.set_cursor_screen_pos((bx, yy))
    imgui.dummy((1, 1))
    imgui.end_child()
