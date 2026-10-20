"""Native workspace inspector; shared scene edits use Session commands."""

from __future__ import annotations

import numpy as np
from imgui_bundle import imgui

from mojive.ui.workspace.design import BODY_SIZE, DESIGN
from mojive.ui.workspace.inspector_components import draw_component
from mojive.ui.workspace.property_style import property_toggle, switch_form_layout
from mojive.ui.workspace.section_columns import section_column
from mojive.ui.workspace.widgets import (
    MUTED,
    TEXT,
)
from mojive.ui.workspace.window_chrome import fit_native_content_width


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
    from mojive.ui.workspace.inspector_heading import draw_heading

    body_y = draw_heading(workspace, e, x, y, w)
    body_y += u.metric("panel", "rowGap", 0) * s
    body = u.component_box(
        "panel_body", x, body_y, w, max(s, h - (body_y - y)), key="panel-body-Inspector"
    )
    x, body_y, w, body_h = body.content.tuple()
    if not e:
        style = u.component_style("empty")
        line = max(28, style.line_height()) * s
        u.component_text(
            "empty", (x, body_y, w, line), "Select an object in the scene or viewport.", MUTED
        )
        imgui.set_cursor_screen_pos((x, body_y))
        imgui.dummy((0, line))
        return
    imgui.set_cursor_screen_pos((x, body_y))
    imgui.push_style_var(imgui.StyleVar_.window_padding, (0, 0))
    # Reserve the gutter even when reflow removes overflow; otherwise the same
    # panel width can retain either stacked or inline XYZ after a resize.
    imgui.begin_child(
        "Inspector content",
        (w, max(s, body_h)),
        window_flags=imgui.WindowFlags_.always_vertical_scrollbar,
    )
    imgui.pop_style_var()
    fit_native_content_width(imgui.internal.get_current_window())
    try:
        with section_column(u) as column:
            _draw_body(workspace, e, column.x, column.y, column.width)
    finally:
        imgui.end_child()


def _draw_body(workspace, e, bx, yy, bw):
    u, s = workspace.ui, workspace.s
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
            rows = (
                ("position", "Position", "m"),
                ("rotation", "Rotation", "°"),
                ("scale", "Scale", ""),
            )
            layout = u.vector_layout(
                bw,
                format="%.3f",
                label_minimum=max(
                    u.vector_caption_width(u.translate(label), unit) for _, label, unit in rows
                ),
                key=f"transform-{e['node_id']}",
            )
            row_y = yy + u.section_offset(42) * s
            for key, label, unit in rows:
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
                    format="%.3f",
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
            yy += u.section_offset(40) * s
        if u.section("DISPLAY", bx, yy, bw, workspace.expanded["DISPLAY"]):
            workspace.expanded["DISPLAY"] = not workspace.expanded["DISPLAY"]
        if workspace.expanded["DISPLAY"]:
            imgui.set_cursor_screen_pos((bx, yy + u.section_offset(38) * s))
            label_width = switch_form_layout(u, bw, "Transform handles").label_width
            e["hidden"] = not property_toggle(
                u, "entity-visible", "Visible", not e["hidden"], label_width=label_width
            )
            workspace.app.gizmos = property_toggle(
                u, "gizmos", "Transform handles", workspace.app.gizmos, label_width=label_width
            )
            yy = imgui.get_cursor_screen_pos().y
        else:
            yy += u.section_offset(40) * s
        if u.section("IDENTITY", bx, yy, bw, workspace.expanded["IDENTITY"]):
            workspace.expanded["IDENTITY"] = not workspace.expanded["IDENTITY"]
        yy += u.section_offset(42) * s
        if workspace.expanded["IDENTITY"]:
            u.text(bx, yy, "Object ID", MUTED, BODY_SIZE)
            u.text(bx + 100 * s, yy, str(e["object_id"]), TEXT, BODY_SIZE)
            u.text(bx, yy + 30 * s, "Type", MUTED, BODY_SIZE)
            u.text(bx + 100 * s, yy + 30 * s, e["type"], TEXT, BODY_SIZE)
            yy += 64 * s
        imgui.set_cursor_screen_pos((bx, yy))
        draw_component(workspace, workspace._panel_context, e, "Properties", details="geometry")
        yy = imgui.get_cursor_screen_pos().y
    elif workspace.inspector_tab == "Material":
        u.section("SURFACE", bx, yy, bw)
        imgui.begin_disabled(not e["colorable"])
        workspace.menus.color_control(e, bx, yy + 48 * s, bw)
        with u.component_scope("rgba"):
            layout = u.numeric_layout(
                bw,
                4,
                column_options=(4, 1),
                parts="rgba",
                minimum=u.color_channel_minimum(format="%.3f"),
            )
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
                    format="%.3f",
                    reset_value=original["rgba"][n] if original else (1.0 if axis == "A" else 0.0),
                )
                state = workspace.begin_numeric_edit(key)
                if changed and not state.canceled:
                    e["rgba"][n] = value
                    e["color"] = "#" + "".join(f"{round(v * 255):02x}" for v in e["rgba"][:3])
                workspace.app.apply_document_edits()
                workspace.finish_numeric_edit(key, state)
            u.record_numeric_group(
                "material-color-channels", bx, yy + 112 * s, bw, layout, parts="rgba"
            )
        imgui.end_disabled()
        yy += 136 * s + layout.height
        imgui.set_cursor_screen_pos((bx, yy))
        # Material controls share the registered panel's transactions and asset actions.
        draw_component(workspace, workspace._panel_context, e, "Material", details="appearance")
        yy = imgui.get_cursor_screen_pos().y
    imgui.set_cursor_screen_pos((bx, yy))
    imgui.dummy((1, 1))
