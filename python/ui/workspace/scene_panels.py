"""Native workspace scene panels; shared scene edits use Session commands."""

from __future__ import annotations

from dataclasses import replace
from math import isfinite

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
from mojive.ui.workspace.component_layout import control_frame_size, control_size
from mojive.ui.workspace.design import DESIGN
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
    if ui.measure(
        text, font_size, translate=False, style=style
    ) > box.content.width and imgui.is_item_hovered(imgui.HoveredFlags_.delay_normal):
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
    if key == "reset":
        values = session.frame.ctrl
        return (
            bool(session.actuators)
            and session.adapter.caps.write_ctrl
            and values is not None
            and all(
                actuator.ctrl_address >= 0
                and actuator.ctrl_address + actuator.ctrl_count <= len(values)
                and np.isfinite(
                    values[actuator.ctrl_address : actuator.ctrl_address + actuator.ctrl_count]
                ).all()
                for actuator in session.actuators
            )
        )
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


@part_scope("entity_card")
def control_panel(workspace, x, y, w):
    from .scalar_card import scalar_card, scalar_value_width

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
    imgui.set_cursor_screen_pos((x, y))
    yy = y
    if session.actuators and not enabled:
        reason = (
            "Read only · external control"
            if session.adapter.caps.external_clock
            else "Read only · control writes unavailable"
        )
        yy = _control_detail(u, "control-read-only", x, yy, w, ctx.tr(reason)) + DESIGN.row_gap * s
    from mojive.design.native.geometry import Rect

    from .entity_cards import card_controls, card_surface

    outer_x, outer_width = x, w
    card_style = u.component_style("entity_card")
    pads = card_style.edges("padding", s, (8, 8, 8, 8))
    card_gap = card_style.get("gap", 12) * s
    x, w = x + pads[0], w - pads[0] - pads[2]
    for actuator, component in rows:
        card_top = yy
        yy += pads[1]
        bounds = _control_range(actuator)
        address = actuator.ctrl_address + component
        suffix = f"[{component}]" if actuator.ctrl_count > 1 else ""
        target = session.node(actuator.target_node_id) or joint_nodes.get(actuator.joint)
        label = (actuator.name or f"act{actuator.actuator_id}") + suffix
        key = "control-target-" + str(address)
        _, heading_height = control_size(u, key, w / s, DESIGN.control_height, label=label)
        card_height = u.rail_height(unit=bool(actuator.unit)) * (2 if bounds else 1) + (
            4 * s if bounds else 0
        )
        card_surface(
            u,
            Rect(outer_x, card_top, outer_width, card_height + pads[1] + pads[3]),
            selected=panel._selected_address == address,
        )
        with card_controls(u, pads):
            button_width, button_height = control_frame_size(
                u, key, max(s, w - scalar_value_width(u, w) - 8 * s), heading_height, label=label
            )
            pressed = u.button(
                key,
                x,
                yy,
                button_width,
                button_height,
                width_limit=button_width * s,
                label=label,
                text_color=ACCENT if panel._selected_address == address else TEXT,
                active=False,
                translate=False,
                align="left",
                tooltip="Invalid control range"
                if actuator.ctrl_limited and bounds is None
                else (target.name if target is not None else ""),
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
            value_y = yy
            available = (
                values is not None
                and 0 <= address < len(values)
                and isfinite(float(values[address]))
            )
            if not available:
                yy = (
                    _control_detail(
                        u,
                        "##ctrl-" + str(address),
                        x,
                        value_y + heading_height + 4 * s,
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
            edit = scalar_card(
                u,
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
                + pads[3]
                + card_gap
            )
    if not rows:
        from .property_style import panel_empty

        yy = panel_empty(
            u,
            "control-empty",
            "No matching actuators" if session.actuators else "No actuators in this scene",
            outer_x,
            yy,
            outer_width,
        )
    advance(u, x, yy + DESIGN.row_gap * s)
    if session.equality_constraints and property_section(
        u, ctx.tr("equality"), imgui.TreeNodeFlags_.default_open
    ):
        panel._equality(ctx, panel.presentation)


def assets_panel(workspace, x, y, w):
    from mojive.ui.workspace.resource_panel import resource_panel

    resource_panel(workspace, x, y, w)


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
