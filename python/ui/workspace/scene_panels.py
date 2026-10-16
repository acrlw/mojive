"""Native workspace scene panels; shared scene edits use Session commands."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from imgui_bundle import imgui

from mojive import commands as cmd
from mojive.ui.layers import LAYER_LABELS, debug_layer_group
from mojive.ui.panels import activate_edit_gizmo, pointer_pressed, publish_focus_item_hint
from mojive.ui.pointer_bindings import PointerAction
from mojive.ui.workspace.component_layout import control_frame_size, control_size
from mojive.ui.workspace.design import BODY_SIZE, DESIGN
from mojive.ui.workspace.property_style import (
    NativePropertyPresentation,
    advance,
    property_section,
    property_toggle,
)
from mojive.ui.workspace.widgets import (
    ACCENT,
    MUTED,
)


def scene_panel(workspace, x, y, w, h):
    from mojive.ui.workspace.navigation import Navigation

    if not hasattr(workspace, "navigation"):
        workspace.navigation = Navigation(workspace)
    workspace.navigation.draw_scene(workspace._panel_context, x, y, w, h)


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
    joint_nodes = {node.joint_index: node for node in session.nodes if node.joint_index >= 0}
    values = session.frame.ctrl
    enabled = session.adapter.caps.write_ctrl and values is not None
    row = 0
    imgui.set_cursor_screen_pos((x, y))
    expanded = property_section(u, ctx.tr("actuators"), imgui.TreeNodeFlags_.default_open)
    yy = imgui.get_cursor_screen_pos().y
    for actuator in session.actuators if expanded else ():
        lo, hi = actuator.ctrl_range if actuator.ctrl_limited else (-1.0, 1.0)
        for component in range(actuator.ctrl_count):
            address = actuator.ctrl_address + component
            suffix = f"[{component}]" if actuator.ctrl_count > 1 else ""
            target = session.node(actuator.target_node_id) or joint_nodes.get(actuator.joint)
            label, key = actuator.name + suffix, "control-target-" + str(address)
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
                text_color=ACCENT if panel._selected_address == address else MUTED,
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
            current = 0.0 if values is None else float(values[address])
            imgui.begin_disabled(not enabled)
            edit = u.rail(
                "##ctrl-" + str(address),
                x,
                yy + heading_height + DESIGN.numeric_gap * s,
                w,
                current,
                (lo, hi) if actuator.ctrl_limited else None,
                initial=float(panel._initial_ctrl[address])
                if address < len(panel._initial_ctrl)
                else current,
                fmt="%.3f",
                show_reset=False,
                unit=actuator.unit,
                angular_degrees=panel._angular_degrees,
                toggle_unit=panel._toggle_angle_unit,
            )
            imgui.end_disabled()
            if edit.changed and enabled:
                session.submit(cmd.SetCtrl(address, edit.value))
            row += 1
            yy = (
                u.hits["##ctrl-" + str(address)][1]
                + u.hits["##ctrl-" + str(address)][3]
                + DESIGN.row_gap * s
            )
    if not session.actuators:
        u.text(
            x,
            yy + DESIGN.control_padding_y(BODY_SIZE) * s,
            "No actuators in this scene",
            MUTED,
            BODY_SIZE,
        )
        u.hits["control-empty"] = (x, yy, w, DESIGN.control_height * s)
        yy += (DESIGN.control_height + DESIGN.row_gap) * s
    _, reset_height = control_size(
        u, "reset-actuators", w / s, 28, label="Reset controls", icon="reset"
    )
    button_width, button_height = control_frame_size(
        u, "reset-actuators", w, reset_height, label="Reset controls"
    )
    if (
        u.button(
            "reset-actuators",
            x,
            yy,
            button_width,
            button_height,
            width_limit=button_width * s,
            icon="reset",
            label="Reset controls",
            filled=True,
            enabled=enabled and row > 0,
        )
        and enabled
    ):
        session.submit(cmd.SetCtrlVector(np.zeros_like(values)))
    yy += reset_height + DESIGN.row_gap * s
    _, pose_height = control_size(u, "open-pose", w / s, 28, label="Open pose editor", icon="pose")
    button_width, button_height = control_frame_size(
        u, "open-pose", w, pose_height, label="Open pose editor"
    )
    if u.button(
        "open-pose",
        x,
        yy,
        button_width,
        button_height,
        width_limit=button_width * s,
        icon="pose",
        label="Open pose editor",
        filled=True,
    ):
        workspace.activate("Pose")
    advance(u, x, yy + pose_height + DESIGN.row_gap * s)
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

    labels = (
        "Viewport UI",
        "Gizmos",
        "Selection",
        "Helpers",
        "Perturbation",
        "3D drawings",
        "2D drawings",
    )
    for (name, description), label in zip(LAYER_LABELS, labels, strict=True):
        layers = workspace.app.viewport_layers
        value = visibility_row(
            "layer-" + name, ctx.tr(label), getattr(layers, name), ctx.tr(description)
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
        if draw is not None
        else ()
    )
    named_open = True
    if named:
        imgui.set_cursor_screen_pos((x, yy + DESIGN.row_gap * s))
        named_open = property_section(
            u, ctx.tr("Named drawing layers"), imgui.TreeNodeFlags_.default_open
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
