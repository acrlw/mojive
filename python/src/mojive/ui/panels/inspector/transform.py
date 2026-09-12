"""Inspector: transform."""

from __future__ import annotations

import numpy as np
from imgui_bundle import imgui

from mojive import commands as cmd
from mojive import math3d
from mojive.adapters.base import (
    SceneNode,
)
from mojive.ui.panels import (
    PanelContext,
    begin_kv_table,
    labeled,
)

from .fields import (
    _vector_fields,
)
from .support import (
    _free_velocity,
    _has_free_velocity,
    _nearest_euler_degrees,
    _node_pose,
    _pose_editable,
    gizmo_refusal_reason,
)


class _Transform:
    """Private transform methods of InspectorPanel; state belongs to its owner."""

    def _transform(self, ctx: PanelContext, node: SceneNode) -> None:
        self.show_transform = imgui.collapsing_header(
            ctx.tr("transform"), imgui.TreeNodeFlags_.default_open
        )
        if not self.show_transform:
            return
        frame = ctx.session.frame
        pos, mat = _node_pose(frame, node)
        if pos is None:
            imgui.text_disabled(ctx.tr("no pose this frame"))
            return
        mat = np.eye(3, dtype=np.float32) if mat is None else np.asarray(mat).reshape(3, 3)
        euler = self._continuous_euler(node.node_id, mat)
        editable = _pose_editable(
            ctx.session.adapter.caps.write_pose, ctx.session.paused, node.posable
        )
        scale_editable = ctx.session.paused and ctx.session.scale_target(node.node_id) is not None
        self._transform_velocity = _has_free_velocity(ctx.session.joints, node.body_index)
        velocity = _free_velocity(ctx.session.frame.qvel, ctx.session.joints, node.body_index)
        label_width = (
            max(
                imgui.calc_text_size(ctx.tr("linear velocity")).x,
                imgui.calc_text_size(ctx.tr("angular velocity")).x,
            )
            + 10.0 * ctx.style_scale
            if velocity is not None
            else 0.0
        )

        edits = _vector_fields(
            ctx,
            node,
            "insp_transform",
            (
                (ctx.tr("position"), pos, 0.01, "%.3f", None),
                (ctx.tr("rotation"), euler, 0.5, "%.1f", None),
                (ctx.tr("Scale"), ctx.session.scale_factors(node.node_id), 0.01, "%.3f", (1, 1, 1)),
            ),
            editable=(editable, editable, scale_editable),
            label_width=label_width,
        )
        (pos_changed, new_pos), (rot_changed, new_euler), (scale_changed, new_scale) = edits
        if scale_changed:
            # The shared pending-edit workflow previews scale and bakes it once on Apply.
            ctx.submit(cmd.SetScale(node.node_id, new_scale))
        if scale_editable:
            imgui.text_disabled(ctx.tr("Apply to bake scale."))
            imgui.set_item_tooltip(ctx.tr("Apply bakes Scale into dimensions and resets it to 1."))
        if velocity is not None:
            _vector_fields(
                ctx,
                node,
                "insp_transform_velocity",
                (
                    (ctx.tr("linear velocity"), velocity[0], 0.0, "%.3f", None),
                    (ctx.tr("angular velocity"), velocity[1], 0.0, "%.3f", None),
                ),
                editable=False,
                label_width=label_width,
            )

        if pos_changed or rot_changed:
            rotation = math3d.euler_xyz_to_mat3(np.radians(new_euler))
            if rot_changed:
                self._rotation_euler[:] = new_euler
                self._rotation_matrix[:] = rotation
            self._submit_edit(
                ctx,
                cmd.SetPose(
                    node.node_id,
                    np.asarray(new_pos, np.float32),
                    rotation,
                ),
            )

    def _continuous_euler(self, node_id: int, matrix) -> np.ndarray:
        matrix = np.asarray(matrix, np.float64).reshape(3, 3)
        same_node = node_id == self._rotation_node
        if same_node and np.allclose(matrix, self._rotation_matrix, atol=2e-5):
            return self._rotation_euler.copy()
        reference = self._rotation_euler if same_node else None
        self._rotation_node = node_id
        self._rotation_euler[:] = _nearest_euler_degrees(matrix, reference)
        self._rotation_matrix[:] = matrix
        return self._rotation_euler.copy()

    def _gizmo_reason(self, ctx: PanelContext, node: SceneNode) -> None:
        if ctx.gizmo is not None and not ctx.gizmo.enabled:
            return
        caps = ctx.session.adapter.caps
        availability = ctx.gizmo.evaluate(ctx.session, node) if ctx.gizmo is not None else None
        reason = (
            availability.reason
            if availability is not None
            else gizmo_refusal_reason(ctx.session.paused, node.posable)
        )
        if availability is None and not caps.write_pose:
            imgui.separator()
            imgui.text_colored(
                imgui.ImVec4(*ctx.theme.warning),
                f"{caps.name} cannot edit this transform",
            )
            return
        active = availability.ok if availability is not None else reason is None
        if active:
            return
        if ctx.gizmo is not None and ctx.gizmo.read_only_frame_available(ctx.session, node):
            return
        imgui.separator()
        imgui.text_colored(imgui.ImVec4(*ctx.theme.warning), ctx.tr("gizmo hidden"))
        imgui.text_wrapped(reason)

    def _velocity(self, ctx: PanelContext, node: SceneNode) -> None:
        self.show_velocity = imgui.collapsing_header(ctx.tr("velocity"))
        if not self.show_velocity:
            return
        qvel = ctx.session.frame.qvel
        if qvel is None:
            imgui.text_disabled(ctx.tr("waiting for the next frame (qvel is produced on demand)"))
            return
        dofs = ctx.session.joints_for_body(node.body_index)
        if not dofs:
            imgui.text_disabled(ctx.tr("no joint on this body"))
            return
        if begin_kv_table("insp_vel"):
            for j in dofs:
                lo = j.qvel_adr
                hi = min(lo + max(1, j.dof), len(qvel))
                labeled(j.name or f"dof{lo}", "  ".join(f"{v:+.4f}" for v in qvel[lo:hi]))
            imgui.end_table()
