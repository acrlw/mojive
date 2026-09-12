"""Gizmo: guides."""

from __future__ import annotations

from typing import Any

import numpy as np

from mojive.drawing.curves import (
    arc_ribbon_mesh,
    arrow_points,
    clip_polygon_rect,
    smooth_capsule_points,
)
from mojive.interaction.gizmo import (
    ACTIVE_HANDLE_COLOR,
    AXIS_END,
    AXIS_HEAD_HALF_PT,
    AXIS_HEAD_LENGTH_PT,
    CENTER_SHELL_RADIUS,
    CONTRAST_EDGE_COLOR,
    CONTRAST_EDGE_PT,
    GUIDE_CORE_COLOR,
    HOVER_COLOR,
    RING_RADIUS,
    RING_WIDTH_PT,
    ROTATE_HANDLES,
    ROTATE_RING_ACTIVE_ALPHA,
    SCREEN_RING_RADIUS,
    SIZE_PT,
    GizmoHandle,
    GizmoMode,
    axis_handle_alpha,
    project,
    rotation_ring_alpha,
    world_scale,
)
from mojive.render.debugdraw import Occlusion
from mojive.types import CameraView
from mojive.ui.camera import ndc_from_viewport, unproject
from mojive.ui.draw2d import Draw2D, draw_drag_link

from .projection import (
    _clip_line_to_rect,
    _project_finite_axis_segment,
    _projected_line_parameters,
    _rotation_dial_segments,
    _rotation_fill_alpha,
    _rotation_sweep,
    _rotation_tick_length_pt,
    _RotationDialProjector,
    _ScreenRotationDialProjector,
    _shortest_rotation_sweep,
    _snap_tick_alpha,
    _split_segment_around_interval,
    _split_segment_around_point,
)
from .state import (
    DRAG_LAYER,
    JOINT_ACTIVE_DARK_COLOR,
    JOINT_CURRENT_TICK_PT,
    JOINT_DRAG_START_TICK_HALF_PT,
    JOINT_RANGE_RADIUS,
    JOINT_RANGE_WIDTH_PT,
    JOINT_ROTATION_AXIS_DASH_PT,
    JOINT_ROTATION_AXIS_GAP_PT,
    ROTATION_TICK_MIN_ALPHA,
    SNAP_TICK_FADE_STEPS,
    TRANSLATION_GUIDE_RADIUS_PT,
    _axis_of,
    _gizmo_geometry_key,
    _HingeAxisProjection,
    _joint_current_tick_color,
    _joint_endpoint_color,
    _joint_value_at_limit,
    _with_alpha,
)


class _Guides:
    """Private guides methods of ObjectGizmo; state belongs to its owner."""

    def _draw_axis_constraint(self, overlay: Draw2D, cam, rect, style_scale: float) -> None:
        axis = _axis_of(self._active)
        if axis < 0:
            return
        origin = np.asarray(self._frame.position, np.float64)
        screen = project(cam, (origin, origin + self._start_basis[:, axis]), rect)
        if np.any(screen[:, 2] <= 0.0):
            return
        segment = _clip_line_to_rect(screen[0, :2], screen[1, :2] - screen[0, :2], rect)
        if segment is None:
            return
        color = (
            ACTIVE_HANDLE_COLOR
            if self._frame.handle_color is not None
            else self._handle_color(axis)
        )
        overlay.line(
            segment[0],
            segment[1],
            (float(color[0]), float(color[1]), float(color[2]), 0.62),
            1.5 * style_scale,
        )

    def _draw_translation_snap_ruler(self, overlay: Draw2D, cam, rect, style_scale: float) -> None:
        axis_index = _axis_of(self._active)
        if axis_index < 0:
            return
        start_axis = self._start_basis[:, axis_index]
        current_position = np.asarray(self._frame.position, np.float64)
        # Anchor the visible ruler to the same current pose and basis used by
        # the arrow.  The drag origin still defines snap values, but must not
        # define a second parallel screen line when the displayed frame trails
        # or is corrected by an adapter.
        # The active arrow is frozen to the drag-start basis. Use that exact
        # axis here as well instead of consulting the live target rotation.
        axis = np.asarray(start_axis, np.float64)
        arrow_scale = world_scale(cam, current_position, rect[3], SIZE_PT * style_scale)
        projected = project(
            cam,
            (current_position, current_position + axis * arrow_scale * AXIS_END),
            rect,
        )
        if np.any(projected[:, 2] <= 0.0):
            return
        origin = projected[0, :2]
        direction = projected[1, :2] - origin
        pixels_per_meter = float(np.linalg.norm(direction))
        if pixels_per_meter < 1e-6:
            return
        direction /= pixels_per_meter
        segment = _clip_line_to_rect(origin, direction, rect)
        if segment is None:
            return

        bounds = _projected_line_parameters(
            cam,
            current_position,
            axis,
            segment,
            rect,
        )
        if bounds is None:
            return

        axis_color = (
            ACTIVE_HANDLE_COLOR
            if self._frame.handle_color is not None
            else self._handle_color(axis_index)
        )
        step = float(self.translation_snap_m)
        current_distance = float(np.dot(current_position - self._start_pos, start_axis))
        bounds = (bounds[0] + current_distance, bounds[1] + current_distance)
        current_step = current_distance / step
        lo = max(
            int(np.ceil(min(bounds) / step)),
            int(np.ceil(current_step - SNAP_TICK_FADE_STEPS)),
        )
        hi = min(
            int(np.floor(max(bounds) / step)),
            int(np.floor(current_step + SNAP_TICK_FADE_STEPS)),
        )
        if lo > hi:
            return

        normal = np.array((-direction[1], direction[0]))
        ticks_visible = pixels_per_meter * step >= 2.0 * style_scale
        ticks: list[tuple[np.ndarray, np.ndarray, float, bool]] = []
        for index in range(lo, hi + 1):
            distance = index * step
            world = current_position + axis * (distance - current_distance)
            point = project(cam, (world,), rect)[0]
            if point[2] <= 0.0:
                continue
            alpha = _snap_tick_alpha(index - current_step) if ticks_visible else 0.0
            if alpha <= 0.01:
                continue
            major = abs(distance - round(distance)) < 1e-6
            half_length = (7.0 if major else 3.5) * style_scale
            a = point[:2] - normal * half_length
            b = point[:2] + normal * half_length
            ticks.append((a, b, alpha, False))

        current = projected[0]
        mask_radius = CENTER_SHELL_RADIUS * SIZE_PT * style_scale
        if current[2] > 0.0 and self._active_joint is None:
            half_length = 14.0 * style_scale
            ticks.append(
                (
                    current[:2] - normal * half_length,
                    current[:2] + normal * half_length,
                    1.0,
                    True,
                )
            )
        # A regular position gizmo lets the rendered 3D arrow own the interval
        # from its center shell through its tip. A slide joint is different:
        # its range and snap ruler are the final, authoritative axis overlay.
        # Keeping the generic arrow-sized hole there can expose a large gap
        # whenever scene depth occludes only part of the underlying 3D arrow.
        arrow_extent = float(np.linalg.norm(projected[1, :2] - current[:2]))
        slide_joint_ruler = getattr(self._active_joint, "type", None) == "slide"
        hidden_interval = None if slide_joint_ruler else (-mask_radius, arrow_extent)
        axis_segments = (
            (segment,)
            if hidden_interval is None
            else _split_segment_around_interval(
                segment[0],
                segment[1],
                current[:2],
                direction,
                hidden_interval[0],
                hidden_interval[1],
            )
        )

        def color(value, alpha: float):
            return (float(value[0]), float(value[1]), float(value[2]), float(value[3]) * alpha)

        for start, end in axis_segments:
            overlay.line(start, end, color((*axis_color, 1.0), 0.92), 1.2 * style_scale)
        for a, b, alpha, is_active in ticks:
            along = float(np.dot((a + b) * 0.5 - current[:2], direction))
            if (
                not is_active
                and hidden_interval is not None
                and hidden_interval[0] <= along <= hidden_interval[1]
            ):
                continue
            tick_color = color(HOVER_COLOR if is_active else (*axis_color, 1.0), alpha)
            for start, end in _split_segment_around_point(a, b, current[:2], mask_radius):
                overlay.line(
                    start,
                    end,
                    tick_color,
                    (2.2 if is_active else 1.2) * style_scale,
                    cap="round",
                    smoothing=self._frame.corner_smoothing,
                )

    def _draw_rotation_snap_ticks(
        self,
        overlay: Draw2D,
        cam,
        rect,
        style_scale: float,
        dial: _RotationDialProjector | _ScreenRotationDialProjector,
    ) -> None:
        ring_radius = (
            SCREEN_RING_RADIUS if self._active is GizmoHandle.ROTATE_SCREEN else RING_RADIUS
        )
        step = float(self.rotation_snap_deg)
        projection_alpha = (
            rotation_ring_alpha(cam, self._start_pos, self._axis)
            if self._frame.active_projection_fade and self._active is not GizmoHandle.ROTATE_SCREEN
            else 1.0
        )
        joint_range = self._joint_range
        core = _with_alpha(
            ACTIVE_HANDLE_COLOR
            if joint_range is not None and joint_range.joint_type == "hinge"
            else GUIDE_CORE_COLOR,
            projection_alpha,
        )

        tick_radius = ring_radius

        def dial_points(angles, radius=tick_radius) -> np.ndarray:
            return dial.points(radius, angles)

        def tick_segment(angle: float, length_pt: float):
            return dial.tick(
                tick_radius,
                angle,
                length_pt * self.rotation_tick_scale * style_scale,
            )

        ticks_visible = (
            self._active is GizmoHandle.ROTATE_SCREEN or projection_alpha >= ROTATION_TICK_MIN_ALPHA
        )
        limited_hinge = (
            self._active is GizmoHandle.ROTATE_Z
            and joint_range is not None
            and joint_range.joint_type == "hinge"
        )
        if limited_hinge:
            # Joint limits, current value, drag sector, and snap ticks all use
            # the joint's absolute angular frame. The cursor-down radial is only
            # a drag integrator reference and must not rotate the visible ruler.
            dial = _RotationDialProjector(
                cam,
                rect,
                self._start_pos,
                self._axis,
                self._start_basis[:, 0],
                SIZE_PT * style_scale,
            )
        ticks: list[tuple[np.ndarray, np.ndarray]] = []
        for degrees in np.arange(0.0, 360.0, step):
            angle = np.radians(degrees)
            if limited_hinge and not joint_range.contains_angle(angle):
                continue
            angle_step = np.radians(step)
            points = dial_points((angle, angle + angle_step))
            if np.any(points[:, 2] <= 0.0):
                continue
            spacing = float(np.linalg.norm(points[1, :2] - points[0, :2]))
            if not ticks_visible or spacing < 2.0 * style_scale:
                continue
            segment = tick_segment(angle, _rotation_tick_length_pt(degrees))
            if segment is not None:
                ticks.append(segment)

        active_tick = None
        has_joint_current_tick = (
            self._joint_range is not None and self._joint_range.joint_type == "hinge"
        )
        if ticks_visible and self._active_joint is None and not has_joint_current_tick:
            angle = self._rotation_angle
            points = dial_points((angle, angle + np.radians(step)))
            spacing = float(np.linalg.norm(points[1, :2] - points[0, :2]))
            if np.all(points[:, 2] > 0.0) and spacing >= 2.0 * style_scale:
                # Highlight the existing tick without extending beyond the
                # tick field when the projected dial becomes narrow.
                active_tick = tick_segment(
                    angle,
                    _rotation_tick_length_pt(np.degrees(angle)),
                )

        for inner, outer in ticks:
            overlay.line(
                inner,
                outer,
                core,
                1.1 * style_scale,
                cap="round_end" if limited_hinge else "round",
                smoothing=self._frame.corner_smoothing,
            )
        if active_tick is not None:
            _fill, pressed, _dark = self._active_rotation_palette()
            overlay.line(
                active_tick[0],
                active_tick[1],
                _with_alpha(pressed, projection_alpha),
                2.2 * style_scale,
                cap="round",
                smoothing=self._frame.corner_smoothing,
            )

    def _hinge_axis_projection(self, cam, rect, style_scale, origin, axis):
        signature = (
            self._frame.corner_smoothing,
            _gizmo_geometry_key(cam, rect, style_scale, None, origin, axis),
        )
        if signature == self._hinge_axis_signature:
            return self._hinge_axis_geometry
        self._hinge_axis_signature = signature
        self._hinge_axis_geometry = None
        alpha = axis_handle_alpha(cam, origin, axis)
        if alpha <= 0:
            return None
        scale = world_scale(cam, origin, rect[3], SIZE_PT * style_scale)
        segment = _project_finite_axis_segment(cam, origin, axis, scale * 1.05, rect)
        center = project(cam, (origin,), rect)[0]
        if segment is None or center[2] <= 0:
            return None
        start, tip = segment
        length = float(np.linalg.norm(tip - start))
        if length < 8 * style_scale:
            return None
        direction = (tip - start) / length
        side = np.array((-direction[1], direction[0]))
        head = min(AXIS_HEAD_LENGTH_PT * style_scale, length * 0.3)
        shaft_width = JOINT_RANGE_WIDTH_PT * style_scale
        body_end = length - head - shaft_width
        shape = arrow_points(
            (0, 0),
            (length, 0),
            shaft_width,
            head_length=head,
            head_width=2 * AXIS_HEAD_HALF_PT * style_scale,
            corner_radius=0.6 * style_scale,
            round_tail=True,
            smoothing=self._frame.corner_smoothing,
        )

        # The rear axis is occluded only until its ray leaves the projected disk.
        # A center ray also handles orthographic and blended camera projections.
        _, ray = unproject(cam, *ndc_from_viewport(*center[:2], rect))
        view = -np.asarray(ray, np.float64)
        facing = float(np.dot(view, axis))
        radial = view - np.asarray(axis) * facing
        radial /= np.linalg.norm(radial)
        rim = project(
            cam,
            (
                origin + radial * scale * JOINT_RANGE_RADIUS,
                origin - radial * scale * JOINT_RANGE_RADIUS,
            ),
            rect,
        )
        center_t = float(np.dot(center[:2] - start, direction))
        rim_t = (rim[:, :2] - start) @ direction
        edge_t = float(min(rim_t) if facing > 0 else max(rim_t))
        lo, hi = np.clip(sorted((center_t, edge_t)), 0, body_end)
        intervals = [(0.0, lo)]
        for value in np.arange(
            lo, hi, (JOINT_ROTATION_AXIS_DASH_PT + JOINT_ROTATION_AXIS_GAP_PT) * style_scale
        ):
            intervals.append((value, min(value + JOINT_ROTATION_AXIS_DASH_PT * style_scale, hi)))
        intervals.append((hi, length))
        merged = []
        for a, b in intervals:
            if b - a <= 1e-5:
                continue
            if merged and a - merged[-1][1] <= 1e-5:
                merged[-1] = (merged[-1][0], b)
            else:
                merged.append((a, b))
        polygons = []
        for a, b in merged:
            half_width = shaft_width * 0.5
            if b < length:
                points = np.asarray(smooth_capsule_points(a, -half_width, b - a, shaft_width, 0))
            else:
                # Join a semicircular tail to the existing arrow outline. Clipping a
                # pre-rounded arrow at its shaft origin would cut that cap in half.
                cut = a + half_width
                points = np.asarray(
                    clip_polygon_rect(shape, (cut, -8 * style_scale, b, 8 * style_scale))
                )
                rounded = []
                for index, point in enumerate(points):
                    following = points[(index + 1) % len(points)]
                    rounded.append(point)
                    if (
                        abs(point[0] - cut) < 1e-6
                        and abs(following[0] - cut) < 1e-6
                        and point[1] * following[1] < 0
                    ):
                        angles = np.linspace(np.pi * 0.5, np.pi * 1.5, 17)
                        if point[1] < 0:
                            angles = angles[::-1]
                        rounded.extend(
                            (cut + half_width * np.cos(t), half_width * np.sin(t))
                            for t in angles[1:-1]
                        )
                points = np.asarray(rounded)
            screen = start + points[:, :1] * direction + points[:, 1:] * side
            polygons.append(tuple(map(tuple, screen.tolist())))
        self._hinge_axis_geometry = _HingeAxisProjection(
            alpha,
            tuple(polygons),
            start + shape[:, :1] * direction + shape[:, 1:] * side,
            (start + lo * direction, start + hi * direction),
        )
        return self._hinge_axis_geometry

    def _draw_rotation_axis_guide(
        self,
        overlay: Draw2D,
        cam: CameraView,
        rect,
        style_scale: float,
    ) -> None:
        """Show the hinge's positive right-hand axis even before a drag begins."""

        short_joint_axis = self._hinge_axis or bool(
            self._joint_range is not None and self._joint_range.joint_type == "hinge"
        )
        axis_index = _axis_of(self._active)
        if axis_index < 0 and not short_joint_axis:
            return
        idle = short_joint_axis and not self._using
        origin = np.asarray(self._frame.position if idle else self._start_pos, np.float64)
        axis = np.asarray(self._frame.rotation[:, 2] if idle else self._axis, np.float64)
        if short_joint_axis:
            geometry = self._hinge_axis_projection(cam, rect, style_scale, origin, axis)
            if geometry is not None:
                color = _with_alpha(self._hinge_range_color(), 0.82 * geometry.alpha)
                for polygon in geometry.polygons:
                    overlay.fringed_concave_fill(polygon, color, origin=(0.0, 0.0))
            return
        scale = world_scale(cam, origin, rect[3], SIZE_PT * style_scale)
        if scale <= 0.0:
            return
        if cam.projection_blend() >= 1.0 - 1e-6:
            extent = max(
                scale * 8.0,
                float(cam.ortho_height) * 0.75 * np.hypot(float(cam.aspect), 1.0),
            )
        else:
            depth = float(np.dot(origin - np.asarray(cam.eye, np.float64), cam.forward()))
            extent = max(scale * 8.0, depth * 2.4)
        segment = _project_finite_axis_segment(
            cam,
            origin,
            axis,
            extent,
            rect,
            inset=min(6.0 * style_scale, 0.1 * min(float(rect[2]), float(rect[3]))),
        )
        if segment is None or float(np.linalg.norm(segment[1] - segment[0])) < 8.0 * style_scale:
            return
        _fill, pressed, _dark = self._active_rotation_palette()
        color = _with_alpha(pressed, 0.82)
        width = 1.6 * style_scale
        overlay.line(
            segment[0],
            segment[1],
            color,
            width,
        )

    def _draw_translation_guide(self, overlay: Draw2D, cam, rect, style_scale: float) -> None:
        screen = project(cam, (self._drag_origin_pos, self._frame.position), rect)
        if np.any(screen[:, 2] <= 0.0):
            return
        draw_drag_link(
            overlay,
            screen[0, :2],
            screen[1, :2],
            GUIDE_CORE_COLOR,
            CONTRAST_EDGE_COLOR,
            2.0 * style_scale,
            TRANSLATION_GUIDE_RADIUS_PT * style_scale,
            CONTRAST_EDGE_PT * style_scale,
            smoothing=self._frame.corner_smoothing,
        )

    def _draw_joint_translation_guide(
        self,
        overlay: Draw2D,
        cam,
        rect,
        style_scale: float,
        *,
        alpha: float = 1.0,
    ) -> None:
        """Draw the slide drag connector and ticks in one pass."""

        screen = project(cam, (self._drag_origin_pos, self._frame.position), rect)
        if np.any(screen[:, 2] <= 0.0):
            return
        start, end = screen[:, :2]
        delta = end - start
        distance = float(np.linalg.norm(delta))
        tangent = delta / distance if distance > 1e-6 else self._axis_screen
        if float(np.linalg.norm(tangent)) < 1e-6:
            return
        normal = np.array((-tangent[1], tangent[0]))
        start_half_tick = JOINT_DRAG_START_TICK_HALF_PT * style_scale
        end_half_tick = JOINT_CURRENT_TICK_PT * 0.5 * style_scale
        core_width = JOINT_RANGE_WIDTH_PT * style_scale
        end_width = core_width
        end_color = ACTIVE_HANDLE_COLOR
        if self._joint_range is not None and self._joint_range.joint_type == "slide":
            end_color = _joint_current_tick_color(end_color)
        color = _with_alpha(JOINT_ACTIVE_DARK_COLOR, alpha)
        if distance > 1e-6:
            overlay.line(start, end, color, core_width)
            overlay.line(
                start - normal * start_half_tick,
                start + normal * start_half_tick,
                color,
                core_width,
                cap="round",
                smoothing=self._frame.corner_smoothing,
            )
        overlay.line(
            end - normal * end_half_tick,
            end + normal * end_half_tick,
            _with_alpha(end_color, alpha),
            end_width,
            cap="round",
            smoothing=self._frame.corner_smoothing,
        )

    def _publish_translation_guide(self, backend: Any, ui_scale: float) -> None:
        dd = getattr(backend, "debug", None)
        # Snap feedback is drawn in the final UI overlay so its connector and
        # endpoints stay above both 2D and 3D gizmo axes.
        active = (
            self._using
            and self._mode is not GizmoMode.DIMENSIONS
            and self._active not in ROTATE_HANDLES
            and not self._snapping
            and not (self._active_joint is not None and self._active_joint.type == "slide")
        )
        if not active or not backend.caps.debug_draw or dd is None:
            self._clear_translation_guide(backend)
            return
        dd.layer(DRAG_LAYER, Occlusion.ALWAYS).drag_link(
            "gizmo.drag",
            self._drag_origin_pos,
            self._frame.position,
            GUIDE_CORE_COLOR,
            CONTRAST_EDGE_COLOR,
            width_px=2.0 * ui_scale,
            radius_px=TRANSLATION_GUIDE_RADIUS_PT * ui_scale,
            smoothing=self._frame.corner_smoothing,
            edge_px=CONTRAST_EDGE_PT * ui_scale,
        )
        self._guide_gpu = True

    def _clear_translation_guide(self, backend: Any) -> None:
        dd = getattr(backend, "debug", None)
        if self._guide_gpu and backend.caps.debug_draw and dd is not None:
            dd.layer(DRAG_LAYER, Occlusion.ALWAYS).clear()
        self._guide_gpu = False

    def _draw_rotation_guide(
        self,
        overlay: Draw2D,
        cam,
        rect,
        style_scale: float,
        dial: _RotationDialProjector | _ScreenRotationDialProjector,
    ) -> None:
        projection_alpha = (
            rotation_ring_alpha(cam, self._start_pos, self._axis)
            if self._frame.active_projection_fade and self._active is not GizmoHandle.ROTATE_SCREEN
            else 1.0
        )
        if projection_alpha <= 0.0:
            return
        joint_range = self._joint_range
        joint_range_ring = joint_range is not None and joint_range.joint_type == "hinge"
        if joint_range_ring:
            dial = _RotationDialProjector(
                cam,
                rect,
                self._start_pos,
                self._axis,
                self._start_basis[:, 0],
                SIZE_PT * style_scale,
            )
            # Limit rebasing updates the numeric drag baseline to discard
            # pointer over-travel. Keep the visible sector anchored at the
            # original mouse-down value until release.
            if len(self._joint_drag_origin_qpos):
                start_angle = float(self._joint_drag_origin_qpos[0])
            elif len(self._start_joint_qpos):
                start_angle = float(self._start_joint_qpos[0])
            else:
                start_angle = float(joint_range.current - self._rotation_angle)
            # The absolute label retains the scalar turn count.  A multi-turn
            # range cannot encode that count on one dial, so show the shortest
            # equivalent sector instead of an almost-full, misleading disk.
            sweep = _rotation_sweep(float(joint_range.current - start_angle))
            if joint_range.has_ambiguous_dial_limits:
                sweep = _shortest_rotation_sweep(sweep)
        else:
            start_angle = 0.0
            sweep = _rotation_sweep(self._rotation_angle)
        ring_radius = (
            SCREEN_RING_RADIUS if self._active is GizmoHandle.ROTATE_SCREEN else RING_RADIUS
        )

        dial_segments = _rotation_dial_segments(cam, self._start_pos, self._axis)
        point_count = max(2, int(np.ceil(dial_segments * abs(sweep) / (2.0 * np.pi))) + 1)
        angles = np.linspace(start_angle, start_angle + sweep, point_count)
        arc = dial.points(ring_radius, angles)
        center = project(cam, (self._start_pos,), rect)[0]
        if center[2] <= 0.0 or np.any(arc[:, 2] <= 0.0):
            return
        center = center[:2]
        arc = arc[:, :2]
        sector = [center, *arc]
        fill_color, pressed_color, dark_color = self._active_rotation_palette()
        border = _with_alpha(dark_color, projection_alpha)

        fill_alpha = _rotation_fill_alpha(sweep) * projection_alpha
        if fill_alpha > 0.0:
            fill = _with_alpha(fill_color, fill_alpha)
            overlay.triangle_fan_fill(sector, fill)
        if not joint_range_ring:
            reference = dial.points(
                ring_radius,
                np.linspace(0.0, 2.0 * np.pi, dial_segments, endpoint=False),
            )
            if np.all(reference[:, 2] > 0.0):
                overlay.polyline(
                    reference[:, :2],
                    _with_alpha(
                        pressed_color,
                        projection_alpha * ROTATE_RING_ACTIVE_ALPHA,
                    ),
                    RING_WIDTH_PT * style_scale,
                    closed=True,
                )
        if abs(sweep) > 1e-6:
            width = RING_WIDTH_PT * style_scale
            start_tick = dial.tick(ring_radius, float(angles[0]), 1.0)
            end_tick = dial.tick(ring_radius, float(angles[-1]), 1.0)
            rounded_start = False
            rounded_end = False
            if joint_range_ring and not joint_range.has_ambiguous_dial_limits:
                rounded_start = _joint_value_at_limit(
                    joint_range, start_angle, lower=True
                ) or _joint_value_at_limit(joint_range, start_angle, lower=False)
                rounded_end = _joint_endpoint_color(joint_range) is not None
            stroke, indices = arc_ribbon_mesh(
                arc,
                None if start_tick is None else start_tick[1] - start_tick[0],
                None if end_tick is None else end_tick[1] - end_tick[0],
                width,
                round_start=rounded_start,
                round_end=rounded_end,
                smoothing=0.0,
            )
            if len(stroke):
                overlay.indexed_fill(stroke, indices, border, outline=stroke)
            else:
                overlay.polyline(arc, border, width)
