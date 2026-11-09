"""Native joint pose fields backed by atomic generalized-position commands."""

from __future__ import annotations

from contextlib import nullcontext

import numpy as np
from imgui_bundle import imgui

from mojive import commands as cmd
from mojive import math3d
from mojive.adapters.base import NodeType
from mojive.design.native.geometry import Rect
from mojive.ui.panels import copyable_name_item, pointer_pressed, publish_focus_item_hint
from mojive.ui.panels.inspector import _nearest_euler_degrees
from mojive.ui.pointer_bindings import PointerAction
from mojive.ui.workspace.component_layout import control_frame_size, control_size
from mojive.ui.workspace.design import BODY_SIZE, DESIGN
from mojive.ui.workspace.scalar_card import scalar_card, scalar_value_width
from mojive.ui.workspace.widgets import MUTED, NumericEditState

_QPOS_SIZE = {"hinge": 1, "slide": 1, "ball": 4, "free": 7}


def joint_icon(kind):
    """Use motion-specific symbols for the four supported joint types."""
    return "joint-" + kind if kind in _QPOS_SIZE else "joint"


def _selected_joint(selected, joint):
    if selected is None:
        return False
    return (
        selected.joint_index == joint.joint_id
        if selected.type is NodeType.JOINT
        else selected.body_index == joint.body
    )


def joint_pose_command(joint, qpos, *, position=None, rotation=None):
    """Convert displayed XYZ degrees to a normalized wxyz joint quaternion.

    Ball coordinates are relative to the joint frame. Free coordinates already
    describe the world pose, including any composed-model placement.
    """
    size = _QPOS_SIZE.get(joint.type)
    start = joint.qpos_adr
    if size is None or start < 0 or qpos is None or start + size > len(qpos):
        raise ValueError("Joint positions are unavailable")
    if joint.type not in ("ball", "free") or (position is None and rotation is None):
        raise ValueError("A ball or free joint pose is required")
    indices, values = [], []
    if position is not None:
        if joint.type != "free":
            raise ValueError("Only free joints have position coordinates")
        coordinates = np.asarray(position, np.float64).reshape(3)
        if not np.all(np.isfinite(coordinates)):
            raise ValueError("Joint position must be finite")
        indices.extend(range(start, start + 3))
        values.extend(coordinates)
    if rotation is not None:
        angles = np.asarray(rotation, np.float64).reshape(3)
        if not np.all(np.isfinite(angles)):
            raise ValueError("Joint rotation must be finite")
        quaternion = np.asarray(
            math3d.mat3_to_quat(math3d.euler_xyz_to_mat3(np.radians(np.remainder(angles, 360)))),
            np.float64,
        )
        quaternion /= np.linalg.norm(quaternion)
        offset = start + (3 if joint.type == "free" else 0)
        if float(np.dot(quaternion, qpos[offset : offset + 4])) < 0:
            quaternion *= -1
        indices.extend(range(offset, offset + 4))
        values.extend(quaternion)
    return cmd.SetQposBatch(np.asarray(indices, np.intp), np.asarray(values, np.float64))


class NativePoseEditor:
    """Own joint value presentation; Navigation owns browsing and selection."""

    def __init__(self, workspace):
        self.workspace = workspace
        self._generation = None
        self._nodes = {}
        self._rotations = {}
        self._failed_gestures = set()
        self._active_joint = None

    def draw(
        self,
        ctx,
        x,
        y,
        w,
        *,
        joints,
        select_joint,
        initial_qpos=None,
        show_names=True,
        joint_color=None,
        compact_poses=False,
    ):
        u, s = self.workspace.ui, self.workspace.s
        if initial_qpos is None:
            initial_qpos = getattr(ctx.session.source, "initial_qpos", ())
        generation = (ctx.session.document_id, ctx.session.structure_generation)
        if self._generation != generation:
            self._generation = generation
            self._rotations.clear()
            self._nodes = {
                node.joint_index: node for node in ctx.session.nodes if node.type is NodeType.JOINT
            }
            self._active_joint = None
        window = imgui.internal.get_current_window()
        clip_top, clip_bottom = window.clip_rect.min.y, window.clip_rect.max.y
        if not imgui.is_any_item_active():
            self._active_joint = None
        qpos = ctx.session.frame.qpos
        style = u.component_style("joint")
        pads, margin = style.edges("padding", s), style.edges("margin", s)
        inner_width = u.component_box("joint", 0, 0, w, 0, observe=False).content.width
        gap = style.get("rowGap", style.get("gap", 8)) * s
        item_gap = style.get("gap", 8) * s
        vector_layout = None
        scalar_heights = {}
        with u.component_scope("joint"):
            heading_height = (
                control_size(
                    u, "pose-joint", inner_width / s, 28, label="Joint", icon="joint-hinge"
                )[1]
                if show_names
                else 0
            )
            body_column = self._body_column(inner_width, heading_height, show_names=show_names)
        heading_gap = gap if show_names else 0
        for joint in joints:
            top = y
            scalar = show_names and joint.type in ("hinge", "slide")
            size = _QPOS_SIZE.get(joint.type)
            available = (
                qpos is not None
                and size is not None
                and joint.qpos_adr >= 0
                and joint.qpos_adr + size <= len(qpos)
            )
            with u.component_scope("joint"):
                field_gap = gap
                if available and joint.type in ("ball", "free"):
                    if vector_layout is None:
                        minimum = (
                            u._numeric_sample_width(("-0.000",), flat=True)
                            if compact_poses
                            else u.numeric_minimum(format="%.3f", compact=True)
                        )
                        vector_layout = u.vector_layout(
                            body_column.width,
                            format="%.3f",
                            label_above=True if compact_poses else None,
                            minimum=minimum,
                        )
                    row_height = vector_layout.height
                    field_height = row_height * (2 if joint.type == "free" else 1) + field_gap * (
                        1 if joint.type == "free" else 0
                    )
                else:
                    if available and joint.limited not in scalar_heights:
                        scalar_heights[joint.limited] = u.rail_extent(
                            body_column.width, bounded=joint.limited
                        )
                    if not available:
                        field_height = 28 * s
                    elif scalar:
                        field_height = (
                            u.rail_height(unit=joint.type == "hinge") if joint.limited else 0
                        )
                    else:
                        field_height = scalar_heights[joint.limited]
            height = (
                heading_height
                + heading_gap
                + field_height
                + pads[1]
                + pads[3]
                + margin[1]
                + margin[3]
            )
            bottom = top + height + item_gap
            # Keep the full scroll extent without submitting offscreen text, icons,
            # quaternion conversions, or controls. An active drag keeps its owner alive.
            if (bottom < clip_top or top > clip_bottom) and self._active_joint != (
                window.id_,
                joint.joint_id,
            ):
                y = bottom
                continue
            node = self._nodes.get(joint.joint_id)
            box = u.component_box(
                "joint", x, top, w, height, key=f"joint-card-{joint.joint_id}", observe=False
            )
            from .entity_cards import card_controls, card_surface

            if show_names:
                card_surface(
                    u, box.frame, selected=_selected_joint(ctx.session.selected_node, joint)
                )
            bx, y, bw, _ = box.content.tuple()
            fields = Rect(
                bx + body_column.x,
                y + heading_height + heading_gap,
                min(body_column.width, max(0, bw - body_column.x)),
                field_height,
            )
            with u.component_scope("joint"):
                heading_width = bw
                if scalar:
                    heading_width = max(
                        s,
                        body_column.x
                        + body_column.width
                        - scalar_value_width(u, body_column.width, show_reset=False)
                        - 8 * s,
                    )
            regions = {
                "gap": [x, box.outer.bottom, w, item_gap],
                "rowGap": [
                    bx,
                    y + heading_height,
                    bw,
                    heading_gap,
                ],
            }
            u.record_component_box(
                "joint",
                box,
                key=f"joint-card-{joint.joint_id}",
                fieldRegions=regions,
                bodyColumn=list(fields.tuple()),
                headingContentColumn=[fields.x, y, fields.width, heading_height]
                if show_names
                else None,
                alignmentAxis="leading-content" if show_names else "unheaded",
            )
            with (
                u.component_scope("joint"),
                card_controls(u, pads) if show_names else nullcontext(),
            ):
                if show_names:
                    self._heading(
                        ctx,
                        joint,
                        node,
                        select_joint,
                        bx,
                        y,
                        bw,
                        joint_color,
                        heading_height,
                        heading_width,
                    )
                    y += heading_height + heading_gap
                field_x, field_y, field_width, _ = fields.tuple()
                enabled = (
                    available
                    and ctx.session.paused
                    and not ctx.session.state_take_playing
                    and ctx.session.adapter.caps.write_qpos
                )
                if not available:
                    u.component_text("joint", fields.tuple(), "qpos not produced this frame", MUTED)
                elif joint.type in ("hinge", "slide"):
                    self._scalar(
                        ctx,
                        joint,
                        node,
                        select_joint,
                        box.content.x if scalar else field_x,
                        box.content.y if scalar else field_y,
                        box.content.width if scalar else field_width,
                        enabled,
                        initial_qpos,
                        card=scalar,
                    )
                else:
                    self._pose(
                        ctx,
                        joint,
                        node,
                        select_joint,
                        field_x,
                        field_y,
                        field_width,
                        enabled,
                        vector_layout,
                        gap,
                        flat=compact_poses,
                    )
            y = bottom
            u.hits[f"pose-fields-{joint.joint_id}"] = (x, top, w, y - top)
        if joints:
            publish_focus_item_hint(ctx)
        else:
            from .property_style import panel_empty

            y = panel_empty(u, "pose-empty", "No matching joints", x, y, w)
        imgui.set_cursor_screen_pos((x, y))
        imgui.dummy((1, 1))
        return y

    def _body_column(self, width, heading_height, *, show_names):
        """Align body fields with the heading's icon column without its name indent."""
        column = Rect(0, 0, width, 0)
        if not show_names:
            return column
        u, scale = self.workspace.ui, self.workspace.s
        key, label = "pose-joint", "Joint"
        parts = u.design.button_parts(key, label) if u.design is not None else "button"
        style = u.component_style(parts)
        frame_width, frame_height = control_frame_size(u, key, width, heading_height, label=label)
        box = style.box(
            0,
            0,
            frame_width * scale,
            frame_height * scale,
            scale=scale,
            padding=(8 * scale, 0, 8 * scale, 0),
        )
        return column.inset(box.content.x, 0, box.padding[2] + box.margin[2], 0)

    def _heading(
        self,
        ctx,
        joint,
        node,
        select_joint,
        x,
        y,
        w,
        color,
        height,
        name_width,
    ):
        u = self.workspace.ui
        key, label = "pose-joint-" + str(joint.joint_id), joint.name or f"joint{joint.joint_id}"
        width, frame_height = control_frame_size(u, key, name_width, height, label=label)
        pressed = u.button(
            key,
            x,
            y,
            width,
            h=frame_height,
            width_limit=width * u.s,
            icon=joint_icon(joint.type),
            label=label,
            active=False,
            icon_color=color,
            translate=False,
            align="left",
        )
        focused = imgui.is_item_hovered() and pointer_pressed(ctx, PointerAction.PANEL_FOCUS)
        if imgui.is_item_clicked() or focused or (pressed and not imgui.is_mouse_released(0)):
            select_joint(ctx, joint, node, focused=focused)
        # The heading button owns the clipped-name hover hint.
        copyable_name_item(ctx, joint.name, name_width, tooltip=False)

    def _rotation(self, ctx, joint):
        qpos = ctx.session.frame.qpos
        offset = joint.qpos_adr + (3 if joint.type == "free" else 0)
        quaternion = np.asarray(qpos[offset : offset + 4], np.float64)
        if not np.all(np.isfinite(quaternion)) or np.linalg.norm(quaternion) <= 1e-12:
            return None
        matrix = math3d.quat_to_mat3(quaternion)
        previous = self._rotations.get(joint.joint_id)
        rotation = (
            previous[1].copy()
            if previous is not None and np.allclose(matrix, previous[0], atol=2e-5, rtol=0)
            else _nearest_euler_degrees(matrix, previous[1] if previous is not None else None)
        )
        self._rotations[joint.joint_id] = (matrix.copy(), rotation.copy())
        return rotation

    def _scalar(
        self, ctx, joint, node, select_joint, x, y, w, enabled, initial_qpos, *, card=False
    ):
        u = self.workspace.ui
        angular = joint.type == "hinge"
        lo, hi = joint.range if joint.limited else (0, 0)
        key = "##pose-joint-" + str(joint.joint_id)
        imgui.begin_disabled(not enabled)
        imgui.begin_group()
        painter = (lambda *args, **kwargs: scalar_card(u, *args, **kwargs)) if card else u.rail
        edit = painter(
            key,
            x,
            y,
            w,
            float(ctx.session.frame.qpos[joint.qpos_adr]),
            (lo, hi) if joint.limited else None,
            initial=float(initial_qpos[joint.qpos_adr])
            if len(initial_qpos) > joint.qpos_adr
            else float(ctx.session.frame.qpos[joint.qpos_adr]),
            show_reset=False,
            **({} if card else {"fmt": "%.3f"}),
            unit="rad" if angular else "m",
            angular_degrees=True,
        )
        imgui.end_group()
        active, deactivated = imgui.is_item_active(), imgui.is_item_deactivated()
        state = u.numeric_edits[key] = NumericEditState(
            activated=imgui.is_item_activated(),
            active=active,
            deactivated=deactivated,
            committed=imgui.is_item_deactivated_after_edit(),
            canceled=(active or deactivated) and imgui.is_key_pressed(imgui.Key.escape, False),
        )
        imgui.end_disabled()
        self._edit(
            ctx,
            joint,
            node,
            select_joint,
            key,
            state,
            edit.changed,
            enabled,
            lambda: cmd.SetQpos(joint.qpos_adr, edit.value),
        )
        return y + 36 * u.s

    def _pose(self, ctx, joint, node, select_joint, x, y, w, enabled, layout, gap, *, flat=False):
        u = self.workspace.ui
        qpos = ctx.session.frame.qpos
        start = joint.qpos_adr
        offset = start + (3 if joint.type == "free" else 0)
        rotation = self._rotation(ctx, joint)
        if rotation is None:
            u.text(
                x, y + DESIGN.control_padding_y(BODY_SIZE) * u.s, "Invalid joint quaternion", MUTED
            )
            return y + 36 * u.s
        fields = (
            (("position", qpos[start : start + 3], "m", 0.01, "%.3f"),)
            if joint.type == "free"
            else ()
        )
        fields += (("rotation", rotation, "°", 0.5, "%.3f"),)
        initial = ctx.session.source.initial_qpos
        for name, values, unit, speed, fmt in fields:
            key = f"##pose-joint-{joint.joint_id}-{name}"
            if name == "rotation":
                reset = (
                    _nearest_euler_degrees(math3d.quat_to_mat3(initial[offset : offset + 4]))
                    if initial is not None and offset + 4 <= len(initial)
                    else (0, 0, 0)
                )
            else:
                reset = (
                    initial[start : start + 3]
                    if initial is not None and start + 3 <= len(initial)
                    else (0, 0, 0)
                )
            imgui.begin_disabled(not enabled)
            changed, edited = u.vector(
                key,
                x,
                y,
                w,
                name.title(),
                values,
                unit,
                speed=speed,
                format=fmt,
                reset_values=reset,
                layout=layout,
                flat=flat,
            )
            imgui.end_disabled()
            state = u.numeric_edits[key]
            applied = self._edit(
                ctx,
                joint,
                node,
                select_joint,
                key,
                state,
                changed,
                enabled,
                lambda name=name, edited=edited: joint_pose_command(joint, qpos, **{name: edited}),
            )
            if applied and name == "rotation":
                self._rotations[joint.joint_id] = (
                    math3d.euler_xyz_to_mat3(np.radians(edited)),
                    np.asarray(edited, np.float64),
                )
            y += layout.height + gap
        return y

    def _edit(self, ctx, joint, node, select_joint, key, state, changed, enabled, command):
        if not enabled:
            return False
        edit_key = f"{key}##window-{imgui.internal.get_current_window().id_}"
        owner = (imgui.internal.get_current_window().id_, joint.joint_id)
        if state.active or state.activated:
            self._active_joint = owner
        elif state.deactivated and self._active_joint == owner:
            self._active_joint = None
        if state.activated:
            self._failed_gestures.discard(edit_key)
            select_joint(ctx, joint, node)
        # The same joint may be visible in both Pose and Inspector. Only the
        # window that owns the gesture may finish its transaction.
        self.workspace.ui.numeric_edits[edit_key] = state
        if ctx.session.adapter.caps.edit_history and edit_key not in self._failed_gestures:
            try:
                self.workspace.begin_numeric_edit(edit_key, label="Pose edit")
            except RuntimeError as error:
                ctx.report(str(error), level="error")
                self._failed_gestures.add(edit_key)
            if state.activated and self.workspace._numeric_edit != edit_key:
                self._failed_gestures.add(edit_key)
        if edit_key in self._failed_gestures:
            if state.canceled or (state.deactivated and not state.active):
                self._failed_gestures.discard(edit_key)
            return False
        applied = False
        if changed and not state.canceled:
            try:
                update = command()
            except (TypeError, ValueError) as error:
                ctx.report(str(error), level="error")
            else:
                applied = ctx.submit(update).ok
        self.workspace.finish_numeric_edit(edit_key, state)
        return applied
