"""Gizmo: drawing."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from mojive.drawing.curves import arc_ribbon_mesh, smooth_affine_corners
from mojive.interaction.gizmo import (
    ACTIVE_COLOR,
    ACTIVE_HANDLE_COLOR,
    ALL_HANDLE_MASK,
    AXIS_COLORS,
    AXIS_END,
    AXIS_HANDLES,
    AXIS_START,
    CENTER_COLOR,
    CENTER_RADIUS,
    CENTER_SHELL_RADIUS,
    CONTRAST_EDGE_COLOR,
    CONTRAST_EDGE_PT,
    HOVER_COLOR,
    JOINT_HANDLE_COLOR,
    PLANE_ACTIVE_ALPHA,
    PLANE_ALPHA,
    PLANE_CORNER_RADIUS_PT,
    PLANE_HANDLES,
    RING_RADIUS,
    RING_SEGMENTS,
    RING_WIDTH_PT,
    ROTATE_AXIS_HANDLES,
    ROTATE_HANDLES,
    SCREEN_RING_RADIUS,
    SCREEN_RING_WIDTH_PT,
    SIZE_PT,
    TRACKBALL_RADIUS,
    GizmoHandle,
    GizmoMode,
    GizmoSpace,
    GizmoStyle,
    axis_active_color,
    axis_arrow_polygon,
    axis_dark_color,
    axis_handle_alpha,
    axis_hover_color,
    dimension_axis_polygon,
    display_handles,
    handle_mask,
    handle_projection_alpha,
    masked_axis_start,
    paint_order,
    plane_corners,
    plane_direction,
    prepare_projection,
    project,
    project_rotation_arc,
    rotation_handle_color,
    rotation_ring,
    rotation_ring_alpha,
    rotation_ring_is_full,
    trackball_color,
    visibility,
    world_scale,
)
from mojive.types import CameraView
from mojive.ui.draw2d import Draw2D

if TYPE_CHECKING:
    from mojive.session import Session


from .joint_labels import _draw_joint_value_label
from .projection import _RotationDialProjector, _ScreenRotationDialProjector
from .state import (
    JOINT_ACTIVE_DARK_COLOR,
    JOINT_RANGE_COLOR,
    JointLimitHit,
    _axis_of,
    _joint_current_tick_color,
    _joint_drag_label_color,
    _with_alpha,
)


class _Drawing:
    """Private drawing methods of ObjectGizmo; state belongs to its owner."""

    def publish(
        self,
        backend: Any,
        session: Session,
        cam: CameraView,
        rect: tuple[float, float, float, float],
        *,
        ui_scale: float,
        style_scale: float,
        yielding: bool,
        interactive: bool,
    ) -> bool:
        self._joint_range = None
        self._hinge_axis = False
        node = session.selected_node
        self._verdict = self.evaluate(session, node)
        self._display_only = bool(
            self._mode is not GizmoMode.DIMENSIONS
            and not self._verdict.ok
            and self.read_only_frame_available(session, node)
        )
        self._interactive = bool(interactive and not self._display_only)
        self._visible = (
            (self.enabled or self.model_placement_model_id >= 0)
            and not yielding
            and (self._verdict.ok or self._display_only)
        )
        if not self._visible:
            self._display_only = False
            self._clear_translation_guide(backend)
            if backend.caps.gizmo:
                backend.set_gizmo(None)
            self._drawn = False
            return False
        target, _reason = (None, "") if self._display_only else self._joint_target(session, node)
        dimension_target = None
        if not self._display_only and self._mode is GizmoMode.DIMENSIONS:
            dimension_target, _reason = self._dimension_target(session, node)
        pose = (
            self._dimension_pose(session, node, dimension_target)
            if dimension_target is not None
            else self._target_pose(session, node, target)
        )
        if pose is None:
            self._drawn = False
            return False
        pos, mat = pose
        self._hinge_axis = target is not None and target.joint.type == "hinge"
        self._joint_range = self._joint_range_state(session, target)
        mode = (
            GizmoMode.TRANSLATE
            if self._display_only
            else (target.mode if target is not None else self._mode)
        )
        self._handle_mask = (
            handle_mask(*AXIS_HANDLES, GizmoHandle.SCREEN)
            if self._display_only
            else self._dimension_handle_mask(dimension_target.dimensions)
            if dimension_target is not None
            else target.handles
            if target is not None
            else ALL_HANDLE_MASK
        )
        active = self._active is not GizmoHandle.NONE
        if active:
            basis = self._start_basis
            if self._active in ROTATE_HANDLES:
                pos = self._start_pos
        else:
            basis = self._target_basis(mat, target)
        projection = prepare_projection(cam)
        scale = world_scale(
            cam,
            pos,
            rect[3],
            SIZE_PT * float(style_scale),
            prepared=projection,
        )
        self._axis_mask, self._plane_mask = visibility(
            cam,
            pos,
            basis,
            rect,
            scale,
            prepared=projection,
        )
        frame = self._frame
        frame.mode = mode
        frame.style = self._style
        frame.space = GizmoSpace.BODY if mode is GizmoMode.DIMENSIONS else self._space
        np.copyto(frame.position, pos, casting="unsafe")
        np.copyto(frame.rotation, basis, casting="unsafe")
        frame.size_px = SIZE_PT * float(ui_scale)
        frame.hovered = self._hovered if self._interactive else GizmoHandle.NONE
        frame.active = self._active
        frame.active_rotation_overlay = self._using and self._active in ROTATE_HANDLES
        frame.axis_mask = self._axis_mask
        frame.plane_mask = self._plane_mask
        frame.handle_mask = self._handle_mask
        frame.handle_color = (
            JOINT_HANDLE_COLOR
            if target is not None and target.joint.type in ("hinge", "slide")
            else None
        )
        frame.outline_color = None
        frame.active_projection_fade = target is not None
        self._publish_translation_guide(backend, ui_scale)
        if self._style is GizmoStyle.FLAT or mode is GizmoMode.DIMENSIONS:
            if backend.caps.gizmo:
                backend.set_gizmo(None)
            self._drawn = False
            return True
        if not backend.caps.gizmo:
            self._drawn = False
            return False
        self._drawn = bool(backend.set_gizmo(frame))
        return self._drawn

    def draw_overlay(self, cam, rect, overlay: Draw2D, *, style_scale: float = 1.0) -> None:
        self._joint_limit_hits = ()
        if not self._visible:
            return
        if self._keyboard and not self._snapping:
            self._draw_axis_constraint(overlay, cam, rect, style_scale)
        if self._style is GizmoStyle.FLAT or self._frame.mode is GizmoMode.DIMENSIONS:
            self._draw_flat(overlay, cam, rect, style_scale)
            self._drawn = True
        joint_range_below_dial = bool(
            self._joint_range is not None
            and self._joint_range.joint_type == "hinge"
            and self._using
            and self._snapping
            and self._active in ROTATE_HANDLES
        )
        range_projection = self._joint_range_projection(
            cam,
            rect,
            style_scale,
            self._joint_range,
            self._frame.position,
            self._frame.rotation,
        )
        if self._joint_range is not None and not joint_range_below_dial:
            self._draw_joint_range(
                overlay,
                cam,
                rect,
                style_scale,
                phase="geometry",
                prepared=range_projection,
            )
        if self._hinge_axis and not self._using:
            self._draw_rotation_axis_guide(overlay, cam, rect, style_scale)
        self._joint_precision = self._joint_precision_projection(
            rect,
            style_scale,
            self._joint_range,
            range_projection,
        )
        if self.joint_precision_visible:
            self._draw_joint_precision(overlay, style_scale)
        if (
            self._using
            and not self._joint_precision_active
            and self._snapping
            and self._active in AXIS_HANDLES
            and self._mode is not GizmoMode.DIMENSIONS
        ):
            self._draw_translation_snap_ruler(overlay, cam, rect, style_scale)
        if (
            self._using
            and not self._joint_precision_active
            and self._active not in ROTATE_HANDLES
            and self._mode is not GizmoMode.DIMENSIONS
            and not self._guide_gpu
        ):
            if (
                self._active_joint is not None
                and self._active_joint.type == "slide"
                and not self._snapping
            ):
                # Limited slides already include these markers in their range.
                if self._joint_range is None:
                    self._draw_joint_translation_guide(overlay, cam, rect, style_scale)
            else:
                self._draw_translation_guide(overlay, cam, rect, style_scale)
        rotation_dial_projector = None
        if (
            self._using
            and not self._joint_precision_active
            and self._active in ROTATE_HANDLES
            and self._active is not GizmoHandle.ROTATE_TRACKBALL
        ):
            projector = (
                _ScreenRotationDialProjector
                if self._active is GizmoHandle.ROTATE_SCREEN
                else _RotationDialProjector
            )
            rotation_dial_projector = projector(
                cam,
                rect,
                self._start_pos,
                self._axis,
                self._rotation_start_vec,
                SIZE_PT * style_scale,
            )
            self._draw_rotation_axis_guide(overlay, cam, rect, style_scale)
            if self._snapping:
                self._draw_rotation_snap_ticks(
                    overlay, cam, rect, style_scale, rotation_dial_projector
                )
            if joint_range_below_dial:
                # Snap ticks are a ruler beneath the joint arc, not spikes
                # painted over its silhouette.
                self._draw_joint_range(
                    overlay,
                    cam,
                    rect,
                    style_scale,
                    phase="geometry",
                    prepared=range_projection,
                )
            self._draw_rotation_guide(overlay, cam, rect, style_scale, rotation_dial_projector)
        if self._using and self._label and not self._joint_precision_active:
            self._draw_value_label(overlay, cam, rect, style_scale, rotation_dial_projector)

    @staticmethod
    def draw_joint_limit_label(
        overlay: Draw2D,
        hit: JointLimitHit,
        style_scale: float,
    ) -> tuple[float, float, float, float]:
        """Draw the delayed, non-interactive value label for one endpoint tick."""

        return _draw_joint_value_label(
            overlay,
            hit.label_anchor,
            hit.semantic_color,
            hit.label,
            style_scale,
            above=hit.label_above,
            align_right=hit.label_align_right,
        )

    def _draw_flat(self, overlay: Draw2D, cam, rect, style_scale: float) -> None:
        # Scalar joint manipulation is represented by its range axis or arc.
        # Slide joints add their translation arrow after the range so the
        # visible affordance and the regular axis hit region stay in sync.
        if self._joint_range is not None:
            return
        frame = self._frame
        origin = np.asarray(frame.position, np.float64)
        rotation = np.asarray(frame.rotation, np.float64)
        projection = prepare_projection(cam)
        scale = world_scale(
            cam,
            origin,
            rect[3],
            SIZE_PT * style_scale,
            prepared=projection,
        )
        visible = display_handles(frame)

        # The draw list paints in submission order with no depth buffer, so
        # each handle group draws far-to-near (painter's order) to put the
        # nearer handle on top where handles overlap.
        planes = [axis for axis, handle in enumerate(PLANE_HANDLES) if handle in visible]
        for k in paint_order(cam, origin, [plane_direction(rotation, axis) for axis in planes]):
            axis = planes[k]
            handle = PLANE_HANDLES[axis]
            alpha = handle_projection_alpha(frame, handle, cam, origin, rotation[:, axis])
            if alpha <= 0.0:
                continue
            screen = project(
                cam,
                plane_corners(origin, rotation, scale, axis),
                rect,
                prepared=projection,
            )
            if np.any(screen[:, 2] <= 0.0):
                continue
            opacity = PLANE_ACTIVE_ALPHA if frame.active is handle else PLANE_ALPHA * alpha
            points = smooth_affine_corners(
                screen[:, :2], PLANE_CORNER_RADIUS_PT * style_scale, frame.corner_smoothing
            )
            overlay.convex_fill(points, self._flat_color(handle, axis, opacity))

        axes = [axis for axis, handle in enumerate(AXIS_HANDLES) if handle in visible]
        for k in paint_order(cam, origin, [rotation[:, axis] for axis in axes]):
            axis = axes[k]
            handle = AXIS_HANDLES[axis]
            alpha = handle_projection_alpha(frame, handle, cam, origin, rotation[:, axis])
            if alpha <= 0.0:
                continue
            screen = project(
                cam,
                (
                    origin + rotation[:, axis] * scale * AXIS_START,
                    origin + rotation[:, axis] * scale * AXIS_END,
                ),
                rect,
                prepared=projection,
            )
            if np.any(screen[:, 2] <= 0.0):
                continue
            start = masked_axis_start(
                screen[0, :2],
                screen[1, :2],
                CENTER_SHELL_RADIUS * SIZE_PT * style_scale,
            )
            color = self._flat_color(handle, axis, alpha)
            if frame.mode is GizmoMode.DIMENSIONS:
                points = dimension_axis_polygon(
                    start, screen[1, :2], style_scale, smoothing=frame.corner_smoothing
                )
                overlay.concave_fill(points, color)
            else:
                points = axis_arrow_polygon(
                    start, screen[1, :2], style_scale, smoothing=frame.corner_smoothing
                )
                if len(points):
                    overlay.concave_fill(points, color)

        if GizmoHandle.SCREEN in visible or frame.mode is GizmoMode.DIMENSIONS:
            center = project(cam, (origin,), rect, prepared=projection)[0]
            if center[2] > 0.0:
                color = HOVER_COLOR if self._hot(GizmoHandle.SCREEN) else CENTER_COLOR
                radius = CENTER_RADIUS * SIZE_PT * style_scale
                if frame.mode is GizmoMode.DIMENSIONS and frame.handle_mask == handle_mask(
                    GizmoHandle.SCREEN
                ):
                    overlay.circle(
                        center[:2],
                        SCREEN_RING_RADIUS * SIZE_PT * style_scale,
                        color,
                        SCREEN_RING_WIDTH_PT * style_scale,
                        segments=RING_SEGMENTS,
                    )
                overlay.circle_filled(
                    center[:2],
                    radius + CONTRAST_EDGE_PT * style_scale,
                    CONTRAST_EDGE_COLOR,
                    segments=24,
                )
                overlay.circle_filled(center[:2], radius, color, segments=24)

        if GizmoHandle.ROTATE_TRACKBALL in visible:
            center = project(cam, (origin,), rect, prepared=projection)[0]
            if center[2] > 0.0:
                overlay.circle_filled(
                    center[:2],
                    TRACKBALL_RADIUS * SIZE_PT * style_scale,
                    trackball_color(frame),
                    segments=RING_SEGMENTS,
                )

        for axis, handle in enumerate(ROTATE_AXIS_HANDLES):
            if handle not in visible:
                continue
            if frame.active_rotation_overlay and frame.active is handle:
                continue
            full = rotation_ring_is_full(frame, handle)
            alpha = handle_projection_alpha(frame, handle, cam, origin, rotation[:, axis])
            if alpha <= 0.0:
                continue
            if full:
                ring = rotation_ring(cam, origin, rotation, scale, axis, full=True)
                screen = project(cam, ring, rect, prepared=projection)
            else:
                screen = project_rotation_arc(
                    cam, origin, rotation, scale, axis, rect, prepared=projection
                )
            if np.any(screen[:, 2] <= 0.0):
                continue
            ring_color = rotation_handle_color(frame, handle, axis, alpha)
            ring_width = RING_WIDTH_PT * style_scale
            if full:
                overlay.polyline(
                    screen[:, :2],
                    ring_color,
                    ring_width,
                    closed=True,
                )
                continue
            stroke, indices = arc_ribbon_mesh(
                screen[:, :2],
                None,
                None,
                ring_width,
                round_caps=True,
                smoothing=0.0,
            )
            if len(stroke):
                # Submit the translucent ribbon and both caps as one fill so
                # their overlap cannot accumulate alpha at either endpoint.
                overlay.indexed_fill(stroke, indices, ring_color, outline=stroke)
            else:
                overlay.polyline(
                    screen[:, :2],
                    ring_color,
                    ring_width,
                    cap="round",
                    smoothing=0.0,
                )

        if GizmoHandle.ROTATE_SCREEN in visible and not (
            frame.active_rotation_overlay and frame.active is GizmoHandle.ROTATE_SCREEN
        ):
            center = project(cam, (origin,), rect, prepared=projection)[0]
            if center[2] > 0.0:
                color = HOVER_COLOR if self._hot(GizmoHandle.ROTATE_SCREEN) else CENTER_COLOR
                radius = SCREEN_RING_RADIUS * SIZE_PT * style_scale
                overlay.circle(
                    center[:2],
                    radius,
                    CONTRAST_EDGE_COLOR,
                    (SCREEN_RING_WIDTH_PT + 2.0 * CONTRAST_EDGE_PT) * style_scale,
                    segments=RING_SEGMENTS,
                )
                overlay.circle(
                    center[:2],
                    radius,
                    color,
                    SCREEN_RING_WIDTH_PT * style_scale,
                    segments=RING_SEGMENTS,
                )

    def _flat_color(self, handle: GizmoHandle, axis: int, alpha: float = 1.0):
        base = self._handle_color(axis)
        if self._frame.handle_color is not None and self._active is handle:
            color = ACTIVE_HANDLE_COLOR
        elif self._frame.handle_color is not None and self._interactive and self._hovered is handle:
            color = axis_hover_color(base)
        else:
            color = HOVER_COLOR if self._hot(handle) else base
        return float(color[0]), float(color[1]), float(color[2]), float(alpha)

    def _handle_color(self, axis: int) -> np.ndarray:
        color = self._frame.handle_color
        return AXIS_COLORS[axis] if color is None else np.asarray(color, np.float32)

    def _active_rotation_palette(self):
        """Return fill, pressed, and dark colors for the active rotation handle."""

        axis = _axis_of(self._active)
        if axis < 0:
            return ACTIVE_COLOR, HOVER_COLOR, ACTIVE_COLOR
        if self._joint_range is not None and self._joint_range.joint_type == "hinge":
            active_joint = np.asarray(ACTIVE_HANDLE_COLOR, np.float32)
            return active_joint, active_joint, JOINT_ACTIVE_DARK_COLOR
        base = self._handle_color(axis)
        return base, axis_active_color(base), axis_dark_color(base)

    def _hinge_range_color(self):
        if self._active is GizmoHandle.ROTATE_Z:
            return ACTIVE_HANDLE_COLOR
        if self._interactive and self._hovered is GizmoHandle.ROTATE_Z:
            return axis_hover_color(JOINT_RANGE_COLOR)
        return JOINT_RANGE_COLOR

    def _hot(self, handle: GizmoHandle) -> bool:
        return self._active is handle or (self._interactive and self._hovered is handle)

    def _draw_value_label(
        self,
        overlay: Draw2D,
        cam,
        rect,
        style_scale: float,
        dial: _RotationDialProjector | _ScreenRotationDialProjector | None,
    ) -> None:
        pad = 6.0 * style_scale
        gap = 14.0 * style_scale
        anchor_world = self._frame.position
        anchor = None
        if self._active in ROTATE_HANDLES:
            ring_radius = (
                SCREEN_RING_RADIUS if self._active is GizmoHandle.ROTATE_SCREEN else RING_RADIUS
            )
            if dial is not None:
                angle = self._rotation_angle
                joint_range = self._joint_range
                if joint_range is not None and joint_range.joint_type == "hinge":
                    dial = _RotationDialProjector(
                        cam,
                        rect,
                        self._start_pos,
                        self._axis,
                        self._start_basis[:, 0],
                        SIZE_PT * style_scale,
                    )
                    angle = joint_range.current
                anchor = dial.points(ring_radius, (angle,))[0]
        if anchor is None:
            anchor = project(cam, (anchor_world,), rect)[0]
        if anchor[2] <= 0.0:
            return
        width_f, height_f = overlay.text_size(self._label)
        if self._active_joint is None or self._active is GizmoHandle.ROTATE_TRACKBALL:
            semantic_color = None
        elif self._joint_range is not None:
            state = self._joint_range
            if state.joint_type == "hinge":
                alpha = rotation_ring_alpha(cam, self._start_pos, self._axis)
                range_color = _with_alpha(self._hinge_range_color(), alpha)
            else:
                origin = np.asarray(self._frame.position, np.float64)
                axis = np.asarray(self._frame.rotation, np.float64).reshape(3, 3)[:, 2]
                alpha = axis_handle_alpha(cam, origin, axis)
                range_color = self._flat_color(GizmoHandle.Z, 2, alpha)
            semantic_color = _joint_current_tick_color(range_color)
        elif self._active in ROTATE_AXIS_HANDLES and self._joint_range is None:
            _fill, semantic_color, _dark = self._active_rotation_palette()
        else:
            semantic_color = _joint_drag_label_color(None)
        dot_radius = 3.0 * style_scale if semantic_color is not None else 0.0
        dot_gap = 6.0 * style_scale if semantic_color is not None else 0.0
        prefix_width = dot_radius * 2.0 + dot_gap
        width, height = width_f + 2.0 * pad + prefix_width, height_f + 2.0 * pad
        x = float(np.clip(anchor[0] + gap, rect[0] + 4.0, rect[0] + rect[2] - width - 4.0))
        y = float(np.clip(anchor[1] + gap, rect[1] + 4.0, rect[1] + rect[3] - height - 4.0))
        overlay.rect_filled(
            (x, y),
            (x + width, y + height),
            (0.08, 0.09, 0.11, 0.92),
            rounding=4.0 * style_scale,
            smoothing=self._frame.corner_smoothing,
        )
        text_x = x + pad
        if semantic_color is not None:
            overlay.circle_filled(
                (text_x + dot_radius, y + height * 0.5),
                dot_radius,
                semantic_color,
                segments=16,
            )
            text_x += prefix_width
        overlay.text((text_x, y + pad), (0.96, 0.96, 0.97, 1.0), self._label)
