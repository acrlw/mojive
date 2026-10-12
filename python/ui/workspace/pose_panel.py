"""Native joint pose fields backed by atomic generalized-position commands."""

from __future__ import annotations

import numpy as np
from imgui_bundle import imgui

from mojive import commands as cmd
from mojive import math3d
from mojive.adapters.base import NodeType
from mojive.ui.panels import copyable_name_item, pointer_pressed, publish_focus_item_hint
from mojive.ui.panels.inspector import _nearest_euler_degrees
from mojive.ui.pointer_bindings import PointerAction

from .design import BODY_SIZE, DESIGN
from .widgets import MUTED, NumericEditState

_QPOS_SIZE = {"hinge": 1, "slide": 1, "ball": 4, "free": 7}


def joint_icon(kind):
    """Use motion-specific symbols for the four supported joint types."""
    return "joint-" + kind if kind in _QPOS_SIZE else "joint"


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

    def draw(self, ctx, x, y, w, *, joints, select_joint, show_names=True, joint_color=None):
        u, s = self.workspace.ui, self.workspace.s
        generation = (ctx.session.document_id, ctx.session.structure_generation)
        if self._generation != generation:
            self._generation = generation
            self._rotations.clear()
            self._nodes = {
                node.joint_index: node for node in ctx.session.nodes if node.type is NodeType.JOINT
            }
        aliases = {
            name: next((joint.joint_id for joint in ctx.session.joints if joint.name == name), -1)
            for name in ("hinge_limited", "slide")
        }
        for joint in joints:
            top = y
            node = self._nodes.get(joint.joint_id)
            if show_names:
                self._heading(ctx, joint, node, select_joint, x, y, w, joint_color)
                y += 34 * s
            qpos = ctx.session.frame.qpos
            size = _QPOS_SIZE.get(joint.type)
            available = (
                qpos is not None
                and size is not None
                and joint.qpos_adr >= 0
                and joint.qpos_adr + size <= len(qpos)
            )
            enabled = (
                available
                and ctx.session.paused
                and not ctx.session.state_take_playing
                and ctx.session.adapter.caps.write_qpos
            )
            if not available:
                u.text(
                    x,
                    y + DESIGN.control_padding_y(BODY_SIZE) * s,
                    "qpos not produced this frame",
                    MUTED,
                )
                y += 36 * s
            elif joint.type in ("hinge", "slide"):
                y = self._scalar(ctx, joint, node, select_joint, aliases, x, y, w, enabled)
                if show_names:
                    y += 6 * s
            else:
                y = self._pose(ctx, joint, node, select_joint, x, y, w, enabled)
            u.hits[f"pose-fields-{joint.joint_id}"] = (x, top, w, y - top)
        if joints:
            publish_focus_item_hint(ctx)
        else:
            u.text(x, y + 8 * s, "No matching joints", MUTED)
            y += 32 * s
        imgui.set_cursor_screen_pos((x, y))
        imgui.dummy((1, 1))
        return y

    def _heading(self, ctx, joint, node, select_joint, x, y, w, color):
        u = self.workspace.ui
        pressed = u.button(
            "pose-joint-" + str(joint.joint_id),
            x,
            y,
            w / u.s,
            icon=joint_icon(joint.type),
            label=joint.name or f"joint{joint.joint_id}",
            active=node is not None and ctx.session.selected_node is node,
            icon_color=color,
            translate=False,
            align="left",
        )
        focused = imgui.is_item_hovered() and pointer_pressed(ctx, PointerAction.PANEL_FOCUS)
        if imgui.is_item_clicked() or focused or (pressed and not imgui.is_mouse_released(0)):
            select_joint(ctx, joint, node, focused=focused)
        copyable_name_item(ctx, joint.name, w)

    def _scalar(self, ctx, joint, node, select_joint, aliases, x, y, w, enabled):
        u = self.workspace.ui
        angular = joint.type == "hinge"
        factor = 180 / np.pi if angular else 1
        lo, hi = joint.range if joint.limited else (0, 0)
        suffix = (
            "hinge"
            if aliases["hinge_limited"] == joint.joint_id
            else "slide"
            if aliases["slide"] == joint.joint_id
            else "joint-" + str(joint.joint_id)
        )
        key = "##pose-" + suffix
        imgui.begin_disabled(not enabled)
        imgui.begin_group()
        changed, value = u.slider(
            key,
            x,
            y,
            w,
            float(ctx.session.frame.qpos[joint.qpos_adr]) * factor,
            lo * factor,
            hi * factor,
            "%.1f°" if angular else "%.3f m",
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
            changed,
            enabled,
            lambda: cmd.SetQpos(joint.qpos_adr, value / factor),
        )
        return y + 36 * u.s

    def _pose(self, ctx, joint, node, select_joint, x, y, w, enabled):
        u = self.workspace.ui
        qpos = ctx.session.frame.qpos
        start = joint.qpos_adr
        offset = start + (3 if joint.type == "free" else 0)
        quaternion = np.asarray(qpos[offset : offset + 4], np.float64)
        valid = np.all(np.isfinite(quaternion)) and np.linalg.norm(quaternion) > 1e-12
        if not valid:
            u.text(
                x, y + DESIGN.control_padding_y(BODY_SIZE) * u.s, "Invalid joint quaternion", MUTED
            )
            return y + 36 * u.s
        matrix = math3d.quat_to_mat3(quaternion)
        previous = self._rotations.get(joint.joint_id)
        rotation = (
            previous[1].copy()
            if previous is not None and np.allclose(matrix, previous[0], atol=2e-5, rtol=0)
            else _nearest_euler_degrees(matrix, previous[1] if previous is not None else None)
        )
        self._rotations[joint.joint_id] = (matrix.copy(), rotation.copy())
        layout = u.vector_layout(w)
        fields = (
            (("position", qpos[start : start + 3], "m", 0.01, "%.6g"),)
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
            self._label_hint(ctx, joint, name, key, x, y, layout)
            y += layout.height + 8 * u.s
        return y

    def _label_hint(self, ctx, joint, name, key, x, y, layout):
        u = self.workspace.ui
        bounds = (x, y, layout.label_width, 20 * u.s)
        u.hits[key + "-label"] = bounds
        if imgui.is_window_hovered() and imgui.is_mouse_hovering_rect(
            (x, y), (x + bounds[2], y + bounds[3])
        ):
            u.tooltip(
                "Position uses world coordinates."
                if name == "position"
                else "Euler angles (XYZ), in world coordinates."
                if joint.type == "free"
                else "Euler angles (XYZ), relative to the joint frame."
            )

    def _edit(self, ctx, joint, node, select_joint, key, state, changed, enabled, command):
        if not enabled:
            return False
        edit_key = f"{key}##window-{imgui.internal.get_current_window().id_}"
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
