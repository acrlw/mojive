"""Inverse-kinematics target and solver controls."""

from __future__ import annotations

import numpy as np
from imgui_bundle import imgui

from ... import commands as cmd
from ... import math3d
from ...adapters.base import FrameNeeds, IkOptions, NodeKind
from . import Panel, PanelContext


class IkPanel(Panel):
    name = "IK"
    default_open = False
    shortcut = "F12"

    def __init__(self) -> None:
        super().__init__()
        self.position = True
        self.rotation = False
        self.max_iterations = 64
        self.tolerance = 1e-4
        self.damping = 1e-3
        self.step_limit = 0.25
        self._node_id = -1
        self._target_position = np.zeros(3, np.float64)
        self._target_euler = np.zeros(3, np.float64)
        self._locked: set[int] = set()
        self._weights: list[float] = []

    def frame_needs(self) -> FrameNeeds:
        return FrameNeeds(poses=True, qpos=True)

    def draw(self, ctx: PanelContext) -> None:
        session = ctx.session
        node = session.selected_node
        if not session.adapter.caps.inverse_kinematics:
            imgui.text_disabled(f"{session.adapter.caps.name} has no IK solver")
            return
        if node is None or not node.ik_target:
            imgui.text_disabled("select a body or site")
            return
        pose = _node_pose(session.frame, node)
        if pose is None:
            imgui.text_disabled("waiting for target pose")
            return
        if node.node_id != self._node_id:
            self._node_id = node.node_id
            self._set_target(*pose)
            self._locked.clear()
            self._weights = [1.0] * len(session.joints)

        imgui.text(f"target: {node.name}")
        _changed, self.position = imgui.checkbox("position", self.position)
        imgui.same_line()
        _changed, self.rotation = imgui.checkbox("rotation", self.rotation)
        imgui.same_line()
        if imgui.button(f"gizmo frame: {ctx.gizmo.space}"):
            ctx.gizmo.toggle_space()
        imgui.set_item_tooltip("Target fields use world coordinates; the gizmo uses this frame")
        imgui.text_disabled("Use G/R and X/Y/Z in the viewport for interactive IK")

        changed, values = imgui.drag_float3(
            "target position", self._target_position, 0.005, 0.0, 0.0, "%.4f"
        )
        if changed:
            self._target_position[:] = values
        changed, values = imgui.drag_float3(
            "target rotation", self._target_euler, 0.25, 0.0, 0.0, "%.2f°"
        )
        if changed:
            self._target_euler[:] = values

        _changed, self.max_iterations = imgui.slider_int("iterations", self.max_iterations, 1, 256)
        _changed, self.tolerance = imgui.drag_float(
            "tolerance", self.tolerance, 1e-5, 1e-7, 0.1, "%.6f"
        )
        _changed, self.damping = imgui.drag_float("damping", self.damping, 1e-4, 1e-6, 1.0, "%.5f")
        _changed, self.step_limit = imgui.drag_float(
            "step limit", self.step_limit, 0.01, 0.01, 2.0, "%.3f"
        )

        if imgui.collapsing_header("joint constraints"):
            if len(self._weights) != len(session.joints):
                self._weights = [1.0] * len(session.joints)
            for index, joint in enumerate(session.joints):
                locked = index in self._locked
                changed, locked = imgui.checkbox(f"##lock_{index}", locked)
                if changed:
                    self._locked.discard(index) if not locked else self._locked.add(index)
                imgui.same_line()
                imgui.set_next_item_width(120.0 * ctx.style_scale)
                changed, weight = imgui.slider_float(
                    f"{joint.name}##weight_{index}", self._weights[index], 0.0, 2.0, "%.2f"
                )
                if changed:
                    self._weights[index] = weight
                if joint.limited:
                    imgui.same_line()
                    imgui.text_disabled(
                        f"[{np.degrees(joint.range[0]):.0f}, {np.degrees(joint.range[1]):.0f}]°"
                        if joint.kind == "hinge"
                        else f"[{joint.range[0]:.3g}, {joint.range[1]:.3g}]"
                    )

        disabled = not session.paused or not (self.position or self.rotation)
        imgui.begin_disabled(disabled)
        if imgui.button("solve"):
            ctx.submit(
                cmd.SolveIk(
                    node.node_id,
                    self._target_position.copy(),
                    math3d.euler_xyz_to_mat3(np.radians(self._target_euler)),
                    IkOptions(
                        position=self.position,
                        rotation=self.rotation,
                        max_iterations=self.max_iterations,
                        tolerance=self.tolerance,
                        damping=self.damping,
                        step_limit=self.step_limit,
                        locked_joints=tuple(sorted(self._locked)),
                        joint_weights=tuple(self._weights),
                    ),
                )
            )
        imgui.same_line()
        if imgui.button("use current pose"):
            self._set_target(*pose)
        imgui.same_line()
        if imgui.button("undo"):
            ctx.submit(cmd.UndoIk())
        imgui.end_disabled()

        result = session.ik_result
        if result is not None:
            color = ctx.theme.primary if result.converged else ctx.theme.warning
            imgui.text_colored(
                imgui.ImVec4(*color),
                f"{result.iterations} iterations · {result.position_error:.4g} m · "
                f"{np.degrees(result.rotation_error):.3g}°",
            )
            if result.message:
                imgui.text_disabled(result.message)

    def _set_target(self, position: np.ndarray, rotation: np.ndarray) -> None:
        self._target_position[:] = position
        self._target_euler[:] = np.degrees(math3d.mat3_to_euler_xyz(rotation))


def _node_pose(frame, node):
    if node.kind is NodeKind.SITE and frame.site_xpos is not None:
        index = node.site_index
        if 0 <= index < len(frame.site_xpos):
            rotation = (
                np.eye(3)
                if frame.site_xmat is None
                else np.asarray(frame.site_xmat[index]).reshape(3, 3)
            )
            return np.asarray(frame.site_xpos[index], np.float64), rotation
    if frame.body_xpos is not None and 0 <= node.body_index < len(frame.body_xpos):
        rotation = (
            np.eye(3)
            if frame.body_xmat is None
            else np.asarray(frame.body_xmat[node.body_index]).reshape(3, 3)
        )
        return np.asarray(frame.body_xpos[node.body_index], np.float64), rotation
    return None
