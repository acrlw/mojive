"""Native workspace scene panels; shared scene edits use Session commands."""

from __future__ import annotations

from dataclasses import replace
from math import degrees, isfinite

import numpy as np
from imgui_bundle import imgui

from mojive import commands as cmd
from mojive.design.native.geometry import part_scope
from mojive.ui.layers import LAYER_LABELS, LAYER_SECTIONS, debug_layer_group
from mojive.ui.panels import (
    activate_edit_gizmo,
    copy_state_vector,
    pointer_pressed,
    publish_focus_item_hint,
)
from mojive.ui.pointer_bindings import PointerAction
from mojive.ui.workspace.component_layout import component_flow, control_frame_size, control_size
from mojive.ui.workspace.design import BODY_SIZE, DESIGN
from mojive.ui.workspace.property_style import (
    NativePropertyPresentation,
    advance,
    property_section,
    property_toggle,
)
from mojive.ui.workspace.widgets import (
    ACCENT,
    INPUT,
    MUTED,
    TEXT,
)


def scene_panel(workspace, x, y, w, h):
    from mojive.ui.workspace.navigation import Navigation

    if not hasattr(workspace, "navigation"):
        workspace.navigation = Navigation(workspace)
    workspace.navigation.draw_scene(workspace._panel_context, x, y, w, h)


@part_scope("button_row")
def _control_actions(workspace, x, y, width, enabled):
    ui, scale = workspace.ui, workspace.s
    style = ui.component_style("button_row")
    padding, margin = style.edges("padding", scale), style.edges("margin", scale)
    available = max(scale, width - padding[0] - padding[2] - margin[0] - margin[2])
    actions = (
        ("reset-actuators", "Reset controls", "reset", "Reset controls"),
        ("open-pose", "Pose", "pose", "Open pose editor"),
    )
    sizes = [
        control_size(
            ui,
            key,
            min(available / scale, ui.measure(label) / scale + 40),
            28,
            label=label,
            icon=icon,
        )
        for key, label, icon, _tooltip in actions
    ]
    box, flow = component_flow(
        ui,
        "button_row",
        x,
        y,
        width,
        sizes,
        key="control-actions",
        column_gap=DESIGN.item_spacing_x,
        row_gap=DESIGN.row_gap,
    )
    for (key, label, icon, tooltip), rect in zip(actions, flow.rectangles, strict=True):
        frame = control_frame_size(ui, key, rect.width, rect.height, label=label, icon=icon)
        if ui.button(
            key,
            rect.x,
            rect.y,
            *frame,
            label=label,
            icon=icon,
            width_limit=frame[0] * scale,
            filled=key == "reset-actuators",
            tooltip=tooltip,
            enabled=enabled if key == "reset-actuators" else True,
        ):
            if key == "reset-actuators" and enabled:
                workspace.app.session.submit(
                    cmd.SetCtrlVector(np.zeros_like(workspace.app.session.frame.ctrl))
                )
            elif key == "open-pose":
                workspace.activate("Pose")
    return box.outer.bottom


def _control_detail(ui, key, x, y, width, text, *, unavailable=False):
    """Use one measured caption row for metadata, availability and read-only reasons."""
    part = "input_number" if unavailable else "control_metadata"
    style, scale = ui.component_style(part), ui.s
    font_size = style.get("fontSize", 12)
    padding = style.edges("padding", scale, (8, 3, 8, 3))
    height = style.line_height(font_size * 1.4) * scale + padding[1] + padding[3]
    box = ui.component_box(part, x, y, width, height, key=key, padding=padding)
    if unavailable:
        ui.rect(*box.frame.tuple(), INPUT, style.get("radius", DESIGN.control_radius))
    ui.component_text(part, box.content, text, MUTED, size=font_size, translate=False)
    imgui.set_cursor_screen_pos((box.frame.x, box.frame.y))
    imgui.invisible_button(key, (max(scale, box.frame.width), max(scale, box.frame.height)))
    ui.hits[key] = box.frame.tuple()
    if imgui.is_item_hovered(imgui.HoveredFlags_.delay_normal):
        ui.tooltip(text, translate=False)
    return box.outer.bottom


def _control_range(actuator):
    """A declared limit is usable only when both finite endpoints are ordered."""
    low, high = actuator.ctrl_range
    return (
        (low, high)
        if actuator.ctrl_limited and isfinite(low) and isfinite(high) and low < high
        else None
    )


def control_action_enabled(workspace, key):
    session = workspace.app.session
    if key == "copy-ctrl":
        return session.frame.ctrl is not None
    if key == "copy-act":
        return session.adapter.caps.state_snapshots
    return True


def control_action_active(workspace, key):
    panel = workspace.app.panels.get("control")
    return (key == "search" and workspace.control_search_visible) or (
        key == "sort" and panel._sort_by_name
    )


def control_action(workspace, key):
    session = workspace.app.session
    panel = workspace.app.panels.get("control")
    if not control_action_enabled(workspace, key):
        return
    if key == "search":
        workspace.control_search_visible = not workspace.control_search_visible
        if not workspace.control_search_visible:
            panel._search = ""
    elif key == "sort":
        panel._sort_by_name = not panel._sort_by_name
    elif key == "copy-ctrl":
        copy_state_vector(session.frame.ctrl)
    elif key == "copy-act":
        state = session.adapter.capture_state()
        copy_state_vector(None if state is None else state.act)
    elif key == "reset" and session.adapter.caps.write_ctrl and session.frame.ctrl is not None:
        session.submit(cmd.SetCtrlVector(np.zeros_like(session.frame.ctrl)))


@part_scope("panel_tools")
def _control_search(workspace, panel, x, y, width):
    if not workspace.control_search_visible:
        return y
    ui = workspace.ui
    _, panel._search = ui.search(
        "##control-search", x, y, width, panel._search, hint=ui.translate("Search actuators")
    )
    tools = ui.component_style("panel_tools")
    return imgui.get_item_rect_max().y + tools.get("rowGap", tools.get("gap", 10)) * ui.s


def control_panel(workspace, x, y, w):
    u, s = workspace.ui, workspace.s
    session = workspace.app.session
    ctx = workspace._panel_context
    panel = workspace.app.panels.get("control")
    if panel.presentation is None:
        panel.presentation = NativePropertyPresentation(u, prefix="control")
    panel._snapshot(ctx)
    if panel._selection_revision != session.selection_revision:
        panel._selected_address = -1
        panel._selection_revision = session.selection_revision
    y = _control_search(workspace, panel, x, y, w)
    rows = panel.actuator_rows(ctx)
    joint_nodes = panel._joint_nodes
    values = session.frame.ctrl
    enabled = session.adapter.caps.write_ctrl
    complete = values is not None and all(
        actuator.ctrl_address >= 0
        and actuator.ctrl_address + actuator.ctrl_count <= len(values)
        and np.isfinite(
            values[actuator.ctrl_address : actuator.ctrl_address + actuator.ctrl_count]
        ).all()
        for actuator in session.actuators
    )
    row = 0
    imgui.set_cursor_screen_pos((x, y))
    expanded = property_section(u, ctx.tr("actuators"), imgui.TreeNodeFlags_.default_open)
    yy = imgui.get_cursor_screen_pos().y
    if session.actuators and expanded and not enabled:
        reason = (
            "Read only · external control"
            if session.adapter.caps.external_clock
            else "Read only · control writes unavailable"
        )
        yy = _control_detail(u, "control-read-only", x, yy, w, ctx.tr(reason)) + DESIGN.row_gap * s
    for actuator, component in rows if expanded else ():
        bounds = _control_range(actuator)
        address = actuator.ctrl_address + component
        suffix = f"[{component}]" if actuator.ctrl_count > 1 else ""
        target = session.node(actuator.target_node_id) or joint_nodes.get(actuator.joint)
        label = (actuator.name or f"act{actuator.actuator_id}") + suffix
        key = "control-target-" + str(address)
        _, heading_height = control_size(u, key, w / s, DESIGN.control_height, label=label)
        button_width, button_height = control_frame_size(u, key, w, heading_height, label=label)
        pressed = u.button(
            key,
            x,
            yy,
            button_width,
            button_height,
            width_limit=button_width * s,
            label=label,
            text_color=ACCENT if panel._selected_address == address else TEXT,
            active=panel._selected_address == address,
            translate=False,
            align="left",
        )
        hovered = imgui.is_item_hovered()
        focused = hovered and pointer_pressed(ctx, PointerAction.PANEL_FOCUS)
        if hovered and target is not None:
            publish_focus_item_hint(ctx)
        if target is not None and (pressed or focused):
            ctx.submit(cmd.SelectNode(target.node_id))
            if focused:
                joint = next(
                    (item for item in session.joints if item.joint_id == target.joint_index),
                    None,
                )
                activate_edit_gizmo(ctx, target, joint)
                if joint is not None and ctx.focus_joint is not None:
                    ctx.focus_joint(joint.joint_id)
                elif ctx.focus_node is not None:
                    ctx.focus_node(target.node_id)
        if pressed or focused:
            panel._selection_revision = session.selection_revision
            panel._selected_address = address
        target_name = target.name if target is not None else ctx.tr("Target not specified")
        if bounds is not None:
            shown = (
                tuple(degrees(v) for v in bounds)
                if actuator.unit == "rad" and panel._angular_degrees
                else bounds
            )
            unit = "°" if actuator.unit == "rad" and panel._angular_degrees else actuator.unit
            limits = f"{shown[0]:.4g} … {shown[1]:.4g}" + (f" {unit}" if unit else "")
        else:
            limits = ctx.tr("Invalid control range" if actuator.ctrl_limited else "Unbounded")
        detail = target_name + " · " + limits
        value_y = (
            _control_detail(
                u,
                "control-meta-" + str(address),
                x,
                yy + heading_height,
                w,
                detail,
            )
            + DESIGN.numeric_gap * s
        )
        available = (
            values is not None and 0 <= address < len(values) and isfinite(float(values[address]))
        )
        row += 1
        if not available:
            yy = (
                _control_detail(
                    u,
                    "##ctrl-" + str(address),
                    x,
                    value_y,
                    w,
                    ctx.tr("Control value unavailable"),
                    unavailable=True,
                )
                + DESIGN.row_gap * s
            )
            continue
        current = float(values[address])
        editable = enabled and (not actuator.ctrl_limited or bounds is not None)
        imgui.begin_disabled(not editable)
        edit = u.rail(
            "##ctrl-" + str(address),
            x,
            value_y,
            w,
            current,
            bounds,
            initial=float(panel._initial_ctrl[address])
            if 0 <= address < len(panel._initial_ctrl)
            and isfinite(float(panel._initial_ctrl[address]))
            else current,
            fmt="%.3f",
            show_reset=False,
            unit=actuator.unit,
            angular_degrees=panel._angular_degrees,
            toggle_unit=panel._toggle_angle_unit,
        )
        imgui.end_disabled()
        if edit.changed and editable:
            session.submit(cmd.SetCtrl(address, edit.value))
        yy = (
            u.hits["##ctrl-" + str(address)][1]
            + u.hits["##ctrl-" + str(address)][3]
            + DESIGN.row_gap * s
        )
    if not session.actuators or (expanded and not rows):
        u.text(
            x,
            yy + DESIGN.control_padding_y(BODY_SIZE) * s,
            "No matching actuators" if session.actuators else "No actuators in this scene",
            MUTED,
            BODY_SIZE,
        )
        u.hits["control-empty"] = (x, yy, w, DESIGN.control_height * s)
        yy += (DESIGN.control_height + DESIGN.row_gap) * s
    yy = _control_actions(workspace, x, yy, w, enabled and complete and row > 0)
    advance(u, x, yy + DESIGN.row_gap * s)
    if session.equality_constraints and property_section(
        u, ctx.tr("equality"), imgui.TreeNodeFlags_.default_open
    ):
        panel._equality(ctx, panel.presentation)


def assets_panel(workspace, x, y, w):
    u = workspace.ui
    from mojive.ui.workspace.material_library import draw_material_library

    draw_material_library(workspace, x, y, w)
    # The palette is a native shortcut; the registered panel retains model asset
    # inventory, import, assignment, and lifecycle ownership.
    panel = workspace.app.panels.get("assets")
    ctx = workspace._panel_context
    if panel._selected is not None and panel._selected != getattr(
        workspace, "_focused_asset", None
    ):
        imgui.set_next_item_open(True, imgui.Cond_.always)
    workspace._focused_asset = panel._selected
    if panel.presentation is None:
        panel.presentation = NativePropertyPresentation(u, prefix="assets")
    if property_section(u, ctx.tr("Model assets"), 0):
        panel.draw(ctx)


def pose_panel(workspace, x, y, w, *, node_id=None):
    from mojive.ui.workspace.navigation import Navigation

    if not hasattr(workspace, "navigation"):
        workspace.navigation = Navigation(workspace)
    workspace.navigation.draw_pose(workspace._panel_context, x, y, w, node_id=node_id)


def layers_panel(workspace, x, y, w):
    u, s, ctx = workspace.ui, workspace.s, workspace._panel_context
    yy = y

    def visibility_row(key, label, visible, tooltip=""):
        imgui.set_cursor_screen_pos((x, yy))
        return property_toggle(u, key, label, visible, width=w, tooltip=tooltip)

    descriptions = dict(LAYER_LABELS)
    debug_open = False
    for index, (title, controls) in enumerate(LAYER_SECTIONS):
        imgui.set_cursor_screen_pos((x, yy + (DESIGN.row_gap * s if index else 0)))
        opened = property_section(
            u,
            ctx.tr(title),
            imgui.TreeNodeFlags_.default_open,
            item_id="layers-section-" + controls[0][0],
            divider=index > 0,
        )
        yy = imgui.get_cursor_screen_pos().y
        if title == "Debug display":
            debug_open = opened
        for name, label in controls if opened else ():
            layers = workspace.app.viewport_layers
            value = visibility_row(
                "layer-" + name,
                ctx.tr(label),
                getattr(layers, name),
                ctx.tr(descriptions[name]),
            )
            if value != getattr(layers, name):
                ctx.set_viewport_layers(replace(layers, **{name: value}))
            yy = imgui.get_cursor_screen_pos().y
    draw = ctx.backend.debug
    named = (
        tuple(
            layer
            for layer in draw.layers()
            if debug_layer_group(layer.name) in ("debug_3d", "debug_2d")
            and (
                layer.primitives or layer.name in workspace.app.viewport_layers.hidden_debug_layers
            )
        )
        if draw is not None and debug_open
        else ()
    )
    named_open = True
    if named:
        imgui.set_cursor_screen_pos((x, yy + DESIGN.row_gap * s))
        named_open = property_section(
            u,
            ctx.tr("Named drawing layers"),
            imgui.TreeNodeFlags_.default_open,
            item_id="layers-section-named",
            divider=False,
        )
        yy = imgui.get_cursor_screen_pos().y
    for layer in named if named_open else ():
        layers = workspace.app.viewport_layers
        visible = layer.name not in layers.hidden_debug_layers
        imgui.begin_disabled(not getattr(layers, debug_layer_group(layer.name)))
        value = visibility_row(
            "drawing-layer-" + layer.name, layer.name.removeprefix("canvas2d:"), visible
        )
        imgui.end_disabled()
        if value != visible:
            hidden = tuple(name for name in layers.hidden_debug_layers if name != layer.name)
            if not value:
                hidden += (layer.name,)
            ctx.set_viewport_layers(replace(layers, hidden_debug_layers=hidden))
        yy = imgui.get_cursor_screen_pos().y
    imgui.set_cursor_screen_pos((x, yy))
    imgui.dummy((1, 1))


def timeline(workspace, x, y, w, h):
    from mojive.ui.workspace.timeline_panel import WorkspaceTimeline

    if workspace.timeline_panel is None:
        workspace.timeline_panel = WorkspaceTimeline(workspace)
    workspace.timeline_panel.draw(workspace._panel_context, x, y, w, h)
