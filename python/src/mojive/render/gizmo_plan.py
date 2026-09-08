"""Backend-neutral 3D gizmo draw planning."""

from __future__ import annotations

import numpy as np

from ..gizmo import (
    ACTIVE_HANDLE_COLOR,
    AXIS_COLORS,
    AXIS_HANDLES,
    CENTER_COLOR,
    CENTER_RADIUS,
    CENTER_SHELL_RADIUS,
    CONTRAST_EDGE_COLOR,
    CONTRAST_EDGE_PT,
    HOVER_COLOR,
    PLANE_ACTIVE_ALPHA,
    PLANE_ALPHA,
    PLANE_HANDLES,
    ROTATE_AXIS_HANDLES,
    SIZE_PT,
    TRACKBALL_RADIUS,
    GizmoFrame,
    GizmoHandle,
    GizmoMode,
    axis_hover_color,
    axis_rotation,
    display_handles,
    handle_projection_alpha,
    rotation_half_basis,
    rotation_handle_color,
    rotation_ring_is_full,
    screen_rotation_basis,
    trackball_color,
)
from ..types import CameraView

# Pipeline variants (compare, depth write, cull) per handle group.
_PIPE_PLANE = ("less", False, "none")
_PIPE_HANDLE = ("less", True, "back")
_PIPE_OUTLINE = ("less", False, "back")
_PIPE_OVERLAY = ("always", False, "back")

_MESHES = (
    "arrow",
    "arrow_edge",
    "plane",
    "ring",
    "half_ring",
    "ring_edge",
    "half_ring_edge",
    "screen_ring",
    "screen_ring_edge",
    "trackball",
    "center",
)


class GizmoPlanner:
    """Shared handle visibility, transforms and draw order for GPU backends."""

    def plan(self, frame, camera, view_proj, proj, height):
        if frame is None:
            return []
        origin = np.ones(4, np.float64)
        origin[:3] = frame.position
        clip = np.asarray(view_proj, np.float64) @ origin
        p11 = float(proj[1, 1])
        if clip[3] <= 0 or abs(p11) <= 1e-9:
            return []
        scale = float(frame.size_px) * 2.0 * float(clip[3]) / (p11 * max(height, 1))
        if scale <= 0:
            return []
        return (
            self._plan_translate(frame, camera, scale)
            if frame.mode is GizmoMode.TRANSLATE
            else self._plan_rotate(frame, camera, scale)
        )

    def _plan_translate(self, frame: GizmoFrame, camera: CameraView, scale: float) -> list:
        visible = display_handles(frame)
        plans = []
        for axis, handle in enumerate(PLANE_HANDLES):
            if handle in visible:
                alpha = handle_projection_alpha(
                    frame, handle, camera, frame.position, frame.rotation[:, axis]
                )
                if alpha <= 0.0:
                    continue
                opacity = PLANE_ACTIVE_ALPHA if frame.active is handle else PLANE_ALPHA * alpha
                plans.append(
                    (
                        "plane",
                        _PIPE_PLANE,
                        _model(frame.position, axis_rotation(frame.rotation, axis), scale),
                        self._color(frame, handle, axis, alpha=opacity),
                        0.0,
                    )
                )
        for axis, handle in enumerate(AXIS_HANDLES):
            if handle in visible:
                alpha = handle_projection_alpha(
                    frame, handle, camera, frame.position, frame.rotation[:, axis]
                )
                if alpha <= 0.0:
                    continue
                if frame.outline_color is not None:
                    plans.append(
                        (
                            "arrow_edge",
                            _PIPE_OUTLINE,
                            _model(frame.position, axis_rotation(frame.rotation, axis), scale),
                            self._outline_color(frame, alpha),
                            CENTER_SHELL_RADIUS,
                        )
                    )
                plans.append(
                    (
                        "arrow",
                        _PIPE_HANDLE,
                        _model(frame.position, axis_rotation(frame.rotation, axis), scale),
                        self._color(frame, handle, axis, alpha),
                        CENTER_SHELL_RADIUS,
                    )
                )
        if GizmoHandle.SCREEN in visible:
            plans.append(
                (
                    "center",
                    _PIPE_OVERLAY,
                    _model(
                        frame.position,
                        np.eye(3),
                        scale * (CENTER_RADIUS + CONTRAST_EDGE_PT / SIZE_PT),
                    ),
                    CONTRAST_EDGE_COLOR,
                    0.0,
                )
            )
            plans.append(
                (
                    "center",
                    _PIPE_OVERLAY,
                    _model(frame.position, np.eye(3), scale * CENTER_RADIUS),
                    HOVER_COLOR if self._hot(frame, GizmoHandle.SCREEN) else CENTER_COLOR,
                    0.0,
                )
            )
        return plans

    def _plan_rotate(self, frame: GizmoFrame, camera: CameraView, scale: float) -> list:
        visible = display_handles(frame)
        plans = []
        if GizmoHandle.ROTATE_TRACKBALL in visible:
            plans.append(
                (
                    "trackball",
                    _PIPE_OVERLAY,
                    _model(
                        frame.position,
                        screen_rotation_basis(camera),
                        scale * TRACKBALL_RADIUS,
                    ),
                    trackball_color(frame),
                    0.0,
                )
            )
        for axis, handle in enumerate(ROTATE_AXIS_HANDLES):
            if handle not in visible:
                continue
            if frame.active_rotation_overlay and frame.active is handle:
                continue
            full = rotation_ring_is_full(frame, handle)
            alpha = handle_projection_alpha(
                frame, handle, camera, frame.position, frame.rotation[:, axis]
            )
            if alpha <= 0.0:
                continue
            rotation = (
                axis_rotation(frame.rotation, axis)
                if full
                else rotation_half_basis(camera, frame.position, frame.rotation, axis)
            )
            if frame.outline_color is not None:
                plans.append(
                    (
                        "ring_edge" if full else "half_ring_edge",
                        _PIPE_OUTLINE,
                        _model(frame.position, rotation, scale),
                        self._outline_color(frame, alpha),
                        0.0,
                    )
                )
            plans.append(
                (
                    "ring" if full else "half_ring",
                    _PIPE_HANDLE,
                    _model(frame.position, rotation, scale),
                    rotation_handle_color(frame, handle, axis, alpha),
                    0.0,
                )
            )
        if GizmoHandle.ROTATE_SCREEN in visible and not (
            frame.active_rotation_overlay and frame.active is GizmoHandle.ROTATE_SCREEN
        ):
            basis = screen_rotation_basis(camera)
            plans.append(
                (
                    "screen_ring_edge",
                    _PIPE_OVERLAY,
                    _model(frame.position, basis, scale),
                    CONTRAST_EDGE_COLOR,
                    0.0,
                )
            )
            plans.append(
                (
                    "screen_ring",
                    _PIPE_OVERLAY,
                    _model(frame.position, basis, scale),
                    HOVER_COLOR if self._hot(frame, GizmoHandle.ROTATE_SCREEN) else CENTER_COLOR,
                    0.0,
                )
            )
        return plans

    @staticmethod
    def _hot(frame: GizmoFrame, handle: GizmoHandle) -> bool:
        return frame.active is handle or frame.hovered is handle

    def _color(self, frame: GizmoFrame, handle: GizmoHandle, axis: int, alpha: float = 1.0):
        base = AXIS_COLORS[axis] if frame.handle_color is None else frame.handle_color
        if frame.handle_color is not None and frame.active is handle:
            color = ACTIVE_HANDLE_COLOR.copy()
        elif frame.handle_color is not None and frame.hovered is handle:
            color = np.asarray(axis_hover_color(base), np.float32)
        else:
            color = HOVER_COLOR.copy() if self._hot(frame, handle) else np.asarray(base).copy()
        color[3] = alpha
        return color

    @staticmethod
    def _outline_color(frame: GizmoFrame, alpha: float) -> np.ndarray:
        color = np.asarray(frame.outline_color, np.float32).copy()
        color[3] *= float(alpha)
        return color


def _model(position, rotation, scale: float) -> np.ndarray:
    m = np.zeros((4, 4), np.float32)
    m[:3, :3] = np.asarray(rotation, np.float32) * float(scale)
    m[:3, 3] = position
    m[3, 3] = 1.0
    return m
