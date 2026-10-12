"""Native workspace scene panels; shared scene edits use Session commands."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from imgui_bundle import imgui

from mojive import commands as cmd
from mojive.ui.layers import LAYER_LABELS, debug_layer_group
from mojive.ui.panels import activate_edit_gizmo, pointer_pressed, publish_focus_item_hint
from mojive.ui.pointer_bindings import PointerAction

from .design import BODY_SIZE, DESIGN
from .property_style import NativePropertyPresentation, advance, property_section, property_toggle
from .widgets import (
    LINE,
    MUTED,
    SAGE,
    TEXT,
    color,
)


def scene_panel(workspace, x, y, w, h):
    from .navigation import Navigation

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
            pressed = u.button(
                "control-target-" + str(address),
                x,
                yy,
                w / s,
                DESIGN.control_height,
                label=actuator.name + suffix,
                text_color=SAGE if panel._selected_address == address else MUTED,
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
                yy + (DESIGN.control_height + DESIGN.numeric_gap) * s,
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
    if (
        u.button(
            "reset-actuators",
            x,
            yy,
            w / s,
            28,
            icon="reset",
            label="Reset controls",
            filled=True,
            enabled=enabled and row > 0,
        )
        and enabled
    ):
        session.submit(cmd.SetCtrlVector(np.zeros_like(values)))
    if u.button(
        "open-pose",
        x,
        yy + (DESIGN.control_height + DESIGN.row_gap) * s,
        w / s,
        28,
        icon="pose",
        label="Open pose editor",
        filled=True,
    ):
        workspace.activate("Pose")
    advance(u, x, yy + 2 * (DESIGN.control_height + DESIGN.row_gap) * s)
    if session.equality_constraints and property_section(
        u, ctx.tr("equality"), imgui.TreeNodeFlags_.default_open
    ):
        panel._equality(ctx, panel.presentation)


def assets_panel(workspace, x, y, w):
    u = workspace.ui
    s = workspace.s
    u.section("MATERIAL LIBRARY", x, y, w)
    palette = [
        ("Sage ceramic", "#9cbf8d"),
        ("Amber satin", "#e3bc67"),
        ("Slate matte", "#829cb7"),
        ("Coral polymer", "#d58979"),
    ]
    for n, (name, hexcolor) in enumerate(palette):
        yy = y + (38 + n * (48 + DESIGN.row_gap)) * s
        u.rect(x, yy, w, 48 * s, (37, 41, 46, 255), 6)
        if u.button("material-" + name, x, yy, w / s, 48, active=workspace.material == n):
            workspace.material = n
        u.draw.add_rect(
            (x, yy),
            (x + w, yy + 48 * s),
            color((*SAGE[:3], 90) if workspace.material == n else LINE),
            6 * s,
            thickness=s,
        )
        u.draw.add_circle_filled(
            (x + 28 * s, yy + 24 * s),
            16 * s,
            color((*[int(hexcolor[i : i + 2], 16) for i in (1, 3, 5)], 255)),
        )
        u.text(x + 56 * s, yy + 17 * s, name, TEXT, BODY_SIZE)
        if workspace.material == n:
            u.role_icon(
                "check",
                x + w - 26 * s,
                yy + 10 * s,
                16 * s,
                28 * s,
                role="inline",
                c=SAGE,
            )
    if u.button(
        "apply-material",
        x,
        y + (38 + 4 * (48 + DESIGN.row_gap)) * s,
        w / s,
        28,
        icon="check",
        label="Apply material",
        align="center",
        active=True,
        solid=True,
        enabled=workspace.app.entity is not None and workspace.app.entity["colorable"],
    ):
        workspace.app.entity["color"] = palette[workspace.material][1]
        workspace.app.apply_document_edits()
    advance(u, x, y + (38 + 4 * (48 + DESIGN.row_gap) + DESIGN.control_height + DESIGN.row_gap) * s)
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
    from .navigation import Navigation

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
        yy += (DESIGN.control_height + DESIGN.row_gap) * s
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
    if named:
        u.section(ctx.tr("Named drawing layers"), x, yy + 8 * s, w)
        yy += 50 * s
    for layer in named:
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
        yy += (DESIGN.control_height + DESIGN.row_gap) * s
    imgui.set_cursor_screen_pos((x, yy))
    imgui.dummy((1, 1))


def timeline(workspace, x, y, w, h):
    from .timeline_panel import WorkspaceTimeline

    if workspace.timeline_panel is None:
        workspace.timeline_panel = WorkspaceTimeline(workspace)
    workspace.timeline_panel.draw(workspace._panel_context, x, y, w, h)
