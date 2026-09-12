"""Gizmo: joint ranges."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mojive.commands import (
    CommandResult,
    SetQpos,
)
from mojive.drawing.curves import CORNER_SMOOTHING
from mojive.interaction.gizmo import (
    SIZE_PT,
    GizmoHandle,
    axis_active_color,
    axis_handle_alpha,
    axis_hover_color,
    project,
    rotation_ring_alpha,
)
from mojive.ui.draw2d import Draw2D
from mojive.ui.theme import THEME

if TYPE_CHECKING:
    from mojive.session import Session


from .joint_labels import _draw_joint_value_label, _joint_limit_label, _joint_limit_tick_rect
from .projection import _rotation_dial_segments, _RotationDialProjector
from .state import (
    _FULL_TURN,
    JOINT_ACTIVE_DARK_COLOR,
    JOINT_COMPLEMENT_ALPHA,
    JOINT_COMPLEMENT_HOVER_ALPHA,
    JOINT_CURRENT_TICK_PT,
    JOINT_DRAG_START_TICK_HALF_PT,
    JOINT_LIMIT_TICK_PT,
    JOINT_LOWER_LIMIT_COLOR,
    JOINT_PRECISION_MARGIN_PT,
    JOINT_PRECISION_MAX_HINGE_SPAN_DEG,
    JOINT_PRECISION_OFFSET_PT,
    JOINT_PRECISION_PANEL_HEIGHT_PT,
    JOINT_PRECISION_TRACK_PT,
    JOINT_PRECISION_TRIGGER_PT,
    JOINT_RANGE_COLOR,
    JOINT_RANGE_OFFSET_PT,
    JOINT_RANGE_RADIUS,
    JOINT_RANGE_WIDTH_PT,
    JOINT_UPPER_LIMIT_COLOR,
    JointLimitHit,
    _gizmo_geometry_key,
    _HingeRangeProjection,
    _joint_current_tick_color,
    _JointPrecisionProjection,
    _JointRangeState,
    _SlideRangeProjection,
    _with_alpha,
    joint_slide_arrow_polygons,
    joint_slide_arrow_targets,
)


class _JointRanges:
    """Private joint ranges methods of ObjectGizmo; state belongs to its owner."""

    def _draw_joint_range(
        self,
        overlay: Draw2D,
        cam,
        rect,
        style_scale: float,
        *,
        phase: str = "all",
        prepared: _HingeRangeProjection | _SlideRangeProjection | None = None,
    ) -> None:
        state = self._joint_range
        if state is None:
            return
        projection = prepared
        if projection is None:
            projection = self._joint_range_projection(
                cam,
                rect,
                style_scale,
                state,
                self._frame.position,
                self._frame.rotation,
            )
        if projection is None:
            return
        if isinstance(projection, _HingeRangeProjection):
            self._draw_hinge_range(
                overlay,
                cam,
                rect,
                style_scale,
                state,
                phase=phase,
                prepared=projection,
            )
        elif isinstance(projection, _SlideRangeProjection):
            self._draw_slide_range(
                overlay,
                cam,
                rect,
                style_scale,
                state,
                phase=phase,
                prepared=projection,
            )

    def _joint_range_projection(
        self,
        cam,
        rect,
        style_scale: float,
        state: _JointRangeState | None,
        position,
        rotation,
    ) -> _HingeRangeProjection | _SlideRangeProjection | None:
        """Project a scalar joint range once for hit testing and overlay drawing."""

        if state is None:
            self._joint_projection_signature = None
            self._joint_projection = None
            return None
        signature = _gizmo_geometry_key(cam, rect, style_scale, state, position, rotation)
        if signature == self._joint_projection_signature:
            return self._joint_projection
        if state.joint_type == "hinge":
            projection = self._hinge_range_projection(
                cam,
                rect,
                style_scale,
                state,
                position,
                rotation,
            )
        elif state.joint_type == "slide":
            projection = self._slide_range_projection(
                cam,
                rect,
                style_scale,
                state,
                position,
                rotation,
            )
        else:
            projection = None
        self._joint_projection_signature = signature
        self._joint_projection = projection
        return projection

    def _joint_precision_projection(
        self,
        rect,
        style_scale: float,
        state: _JointRangeState | None,
        prepared: _HingeRangeProjection | _SlideRangeProjection | None,
    ) -> _JointPrecisionProjection | None:
        """Expand only scalar limits that collapse below a useful screen distance."""

        if state is None or state.joint_id < 0 or state.qpos_adr < 0:
            return None
        if state.joint_type == "hinge":
            if state.angular_span > np.radians(JOINT_PRECISION_MAX_HINGE_SPAN_DEG):
                return None
            hinge = prepared if isinstance(prepared, _HingeRangeProjection) else None
            if hinge is None or hinge.lower_tick is None or hinge.upper_tick is None:
                return None
            lower = np.asarray(hinge.lower_tick[0], np.float64)
            upper = np.asarray(hinge.upper_tick[0], np.float64)
        elif state.joint_type == "slide":
            slide = prepared if isinstance(prepared, _SlideRangeProjection) else None
            if slide is None:
                return None
            lower = slide.lower
            upper = slide.upper
        else:
            return None
        if float(np.linalg.norm(upper - lower)) >= JOINT_PRECISION_TRIGGER_PT * style_scale:
            return None

        viewport_x, viewport_y, viewport_width, viewport_height = (float(value) for value in rect)
        margin = JOINT_PRECISION_MARGIN_PT * style_scale
        panel_height = JOINT_PRECISION_PANEL_HEIGHT_PT * style_scale
        available_width = viewport_width - 2.0 * margin
        track_width = min(
            JOINT_PRECISION_TRACK_PT * style_scale, available_width - 20.0 * style_scale
        )
        if track_width < 60.0 * style_scale or viewport_height < panel_height + 2.0 * margin:
            return None
        panel_width = track_width + 20.0 * style_scale
        source = (lower + upper) * 0.5
        half_width = panel_width * 0.5
        center_x = float(
            np.clip(
                source[0],
                viewport_x + margin + half_width,
                viewport_x + viewport_width - margin - half_width,
            )
        )
        half_height = panel_height * 0.5
        above = float(source[1] - JOINT_PRECISION_OFFSET_PT * style_scale)
        below = float(source[1] + JOINT_PRECISION_OFFSET_PT * style_scale)
        if above - half_height >= viewport_y + margin:
            center_y = above
        elif below + half_height <= viewport_y + viewport_height - margin:
            center_y = below
        else:
            center_y = float(
                np.clip(
                    above,
                    viewport_y + margin + half_height,
                    viewport_y + viewport_height - margin - half_height,
                )
            )
        start = np.array((center_x - track_width * 0.5, center_y), np.float64)
        end = np.array((center_x + track_width * 0.5, center_y), np.float64)
        normalized = float(
            np.clip(
                (state.current - state.lower) / max(state.upper - state.lower, 1e-12),
                0.0,
                1.0,
            )
        )
        current = start + (end - start) * normalized
        panel_rect = (
            center_x - half_width,
            center_y - half_height,
            center_x + half_width,
            center_y + half_height,
        )
        return _JointPrecisionProjection(
            state.joint_id,
            state.qpos_adr,
            state.joint_type,
            state.lower,
            state.upper,
            start,
            current,
            end,
            panel_rect,
            panel_rect,
            source,
        )

    def _draw_joint_precision(self, overlay: Draw2D, style_scale: float) -> None:
        """Draw a compact-range rail entirely through backend-neutral viewport primitives."""

        rail = self._joint_precision
        if rail is None:
            return
        x0, y0, x1, y1 = rail.panel_rect
        panel_edge = np.array(
            (
                float(np.clip(rail.source[0], x0, x1)),
                y1 if rail.source[1] >= (y0 + y1) * 0.5 else y0,
            ),
            np.float64,
        )
        overlay.line(
            rail.source,
            panel_edge,
            _with_alpha(THEME.text_disabled, 0.60),
            1.0 * style_scale,
        )
        overlay.rect_filled(
            (x0, y0),
            (x1, y1),
            (*THEME.bg_popup[:3], 0.94),
            rounding=5.0 * style_scale,
            smoothing=self._frame.corner_smoothing,
        )
        border = (
            THEME.primary_dim
            if self._joint_precision_hovered or self._joint_precision_active
            else THEME.border
        )
        overlay.rect(
            (x0, y0),
            (x1, y1),
            border,
            1.0 * style_scale,
            rounding=5.0 * style_scale,
            smoothing=self._frame.corner_smoothing,
        )
        core = (
            axis_active_color(JOINT_RANGE_COLOR)
            if self._joint_precision_active
            else axis_hover_color(JOINT_RANGE_COLOR)
            if self._joint_precision_hovered
            else JOINT_RANGE_COLOR
        )
        drag_start = None
        if self._joint_precision_active and len(self._joint_drag_origin_qpos):
            normalized_start = float(
                np.clip(
                    (float(self._joint_drag_origin_qpos[0]) - rail.lower)
                    / max(rail.upper - rail.lower, 1e-12),
                    0.0,
                    1.0,
                )
            )
            drag_start = rail.start + (rail.end - rail.start) * normalized_start
        endpoint_half = 5.5 * style_scale
        start_half = JOINT_DRAG_START_TICK_HALF_PT * style_scale
        current_half = 8.0 * style_scale
        strokes = [
            (rail.start, rail.end, core, JOINT_RANGE_WIDTH_PT * style_scale, None),
        ]
        if drag_start is not None and float(np.linalg.norm(rail.current - drag_start)) > 1e-6:
            strokes.append(
                (
                    drag_start,
                    rail.current,
                    JOINT_ACTIVE_DARK_COLOR,
                    JOINT_RANGE_WIDTH_PT * style_scale,
                    None,
                )
            )
        if drag_start is not None:
            start_a = drag_start - np.array((0.0, start_half))
            start_b = drag_start + np.array((0.0, start_half))
            strokes.append(
                (
                    start_a,
                    start_b,
                    JOINT_ACTIVE_DARK_COLOR,
                    JOINT_RANGE_WIDTH_PT * style_scale,
                    "round",
                )
            )
        current_a = rail.current - np.array((0.0, current_half))
        current_b = rail.current + np.array((0.0, current_half))
        strokes.append((current_a, current_b, core, 4.0 * style_scale, "round"))
        for point, color in (
            (rail.start, JOINT_LOWER_LIMIT_COLOR),
            (rail.end, JOINT_UPPER_LIMIT_COLOR),
        ):
            a = point - np.array((0.0, endpoint_half))
            b = point + np.array((0.0, endpoint_half))
            strokes.append((a, b, color, 3.0 * style_scale, "round"))

        for start, end, color, width, cap in strokes:
            kwargs = {} if cap is None else {"cap": cap}
            overlay.line(start, end, color, width, **kwargs)
        if self._joint_precision_active and self._label:
            _draw_joint_value_label(
                overlay,
                ((x0 + x1) * 0.5, y0),
                core,
                self._label,
                style_scale,
                above=True,
                align_right=False,
                centered=True,
            )

    def _hinge_range_projection(
        self,
        cam,
        rect,
        style_scale: float,
        state: _JointRangeState,
        position=None,
        rotation=None,
    ) -> _HingeRangeProjection | None:
        origin = np.asarray(
            self._frame.position if position is None else position,
            np.float64,
        )
        basis = np.asarray(
            self._frame.rotation if rotation is None else rotation,
            np.float64,
        ).reshape(3, 3)
        alpha = rotation_ring_alpha(cam, origin, basis[:, 2])
        if alpha <= 0.0:
            return None
        dial = _RotationDialProjector(
            cam,
            rect,
            origin,
            basis[:, 2],
            basis[:, 0],
            SIZE_PT * style_scale,
        )
        span = state.angular_span
        full_range = state.covers_full_turn
        allowed = None
        complement = None
        segments = _rotation_dial_segments(cam, origin, basis[:, 2])
        if span > 1e-6:
            point_count = max(2, int(np.ceil(segments * span / _FULL_TURN)) + 1)
            allowed_angles = np.linspace(
                state.lower,
                state.lower + span,
                segments if full_range else point_count,
                endpoint=not full_range,
            )
            candidate = dial.points(JOINT_RANGE_RADIUS, allowed_angles)
            if np.all(candidate[:, 2] > 0.0):
                allowed = candidate[:, :2]
        if not full_range:
            count = max(2, int(np.ceil(segments * (_FULL_TURN - span) / _FULL_TURN)) + 1)
            angles = np.linspace(state.lower + span, state.lower + _FULL_TURN, count)
            candidate = dial.points(JOINT_RANGE_RADIUS, angles)
            if np.all(candidate[:, 2] > 0.0):
                complement = candidate[:, :2]
        return _HingeRangeProjection(
            alpha=alpha,
            allowed=allowed,
            full_range=full_range,
            complement=complement,
            current_tick=dial.tick(
                JOINT_RANGE_RADIUS,
                state.current,
                JOINT_CURRENT_TICK_PT * style_scale,
            ),
            lower_tick=None
            if state.has_ambiguous_dial_limits
            else dial.tick(
                JOINT_RANGE_RADIUS,
                state.lower,
                JOINT_LIMIT_TICK_PT * style_scale,
            ),
            upper_tick=None
            if state.has_ambiguous_dial_limits
            else dial.tick(
                JOINT_RANGE_RADIUS,
                state.upper,
                JOINT_LIMIT_TICK_PT * style_scale,
            ),
        )

    def _draw_hinge_range(
        self,
        overlay: Draw2D,
        cam,
        rect,
        style_scale: float,
        state: _JointRangeState,
        *,
        phase: str = "all",
        prepared: _HingeRangeProjection | None = None,
    ) -> None:
        projection = prepared or self._hinge_range_projection(cam, rect, style_scale, state)
        if projection is None:
            return
        alpha = projection.alpha
        allowed_color = self._hinge_range_color()
        range_color = _with_alpha(allowed_color, alpha)
        range_width = JOINT_RANGE_WIDTH_PT * style_scale
        if phase != "labels" and projection.allowed is not None:
            overlay.polyline(
                projection.allowed,
                range_color,
                range_width,
                closed=projection.full_range,
                cap="butt" if projection.full_range else "round",
                smoothing=0.0,
            )
        if phase != "labels":
            if projection.complement is not None:
                engaged = self._active is GizmoHandle.ROTATE_Z or (
                    self._interactive and self._hovered is GizmoHandle.ROTATE_Z
                )
                opacity = JOINT_COMPLEMENT_HOVER_ALPHA if engaged else JOINT_COMPLEMENT_ALPHA
                overlay.polyline(
                    projection.complement,
                    _with_alpha(allowed_color, alpha * opacity),
                    range_width,
                    closed=False,
                    cap="butt",
                    smoothing=self._frame.corner_smoothing,
                )
            current_tick = projection.current_tick
            if current_tick is not None:
                overlay.line(
                    current_tick[0],
                    current_tick[1],
                    _joint_current_tick_color(range_color),
                    range_width,
                    cap="round",
                    smoothing=self._frame.corner_smoothing,
                )
            if not state.has_ambiguous_dial_limits:
                tick_width = range_width
                lower_tick = projection.lower_tick
                upper_tick = projection.upper_tick
                entries = (
                    (
                        state.lower,
                        _joint_limit_label("MIN", state.lower, "hinge"),
                        lower_tick,
                        lower_tick[1] if lower_tick is not None else None,
                        JOINT_LOWER_LIMIT_COLOR,
                        True,
                        False,
                    ),
                    (
                        state.upper,
                        _joint_limit_label("MAX", state.upper, "hinge"),
                        upper_tick,
                        upper_tick[1] if upper_tick is not None else None,
                        JOINT_UPPER_LIMIT_COLOR,
                        False,
                        True,
                    ),
                )
                for _value, _label, tick, _anchor, color, _above, _align_right in entries:
                    if tick is not None:
                        overlay.line(
                            tick[0],
                            tick[1],
                            _with_alpha(color, alpha),
                            tick_width,
                            cap="round",
                            smoothing=self._frame.corner_smoothing,
                        )
                self._set_joint_limit_hits(
                    state,
                    entries,
                    tick_width=tick_width,
                    tick_cap="round",
                    style_scale=style_scale,
                )

    def _draw_slide_handle(
        self,
        overlay: Draw2D,
        style_scale: float,
        current: np.ndarray,
        tangent: np.ndarray,
        normal: np.ndarray,
        alpha: float,
    ) -> None:
        """Draw opposing external arrows for slide-joint interaction."""

        slide = _SlideRangeProjection(current, current, current, tangent, normal, alpha)
        color = self._flat_color(GizmoHandle.Z, 2, alpha)
        for points in self._slide_arrow_polygons(
            slide, style_scale, smoothing=self._frame.corner_smoothing
        ):
            if len(points):
                overlay.fringed_concave_fill(points, color)

    @staticmethod
    def _slide_arrow_polygons(
        slide: _SlideRangeProjection,
        style_scale: float,
        *,
        for_hit_test: bool = False,
        smoothing: float = CORNER_SMOOTHING,
    ) -> tuple[np.ndarray, np.ndarray]:
        return joint_slide_arrow_polygons(
            slide.current,
            slide.tangent,
            style_scale,
            for_hit_test=for_hit_test,
            smoothing=smoothing,
        )

    @staticmethod
    def _slide_arrow_targets(
        slide: _SlideRangeProjection,
        style_scale: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        return joint_slide_arrow_targets(slide.current, slide.tangent, style_scale)

    @staticmethod
    def _slide_range_projection(
        cam,
        rect,
        style_scale: float,
        state: _JointRangeState,
        position,
        rotation,
    ) -> _SlideRangeProjection | None:
        origin = np.asarray(position, np.float64)
        axis = np.asarray(rotation, np.float64).reshape(3, 3)[:, 2]
        alpha = axis_handle_alpha(cam, origin, axis)
        if alpha <= 0.0:
            return None
        values = np.array((state.lower, state.current, state.upper), np.float64)
        positions = origin + (values - state.current)[:, None] * axis
        projected = project(cam, positions, rect)
        if np.any(projected[:, 2] <= 0.0):
            return None
        lower, current, upper = projected[:, :2]
        direction = upper - lower
        length = float(np.linalg.norm(direction))
        if length < 1e-6:
            return None
        tangent = direction / length
        normal = np.array((-tangent[1], tangent[0]))
        offset = normal * JOINT_RANGE_OFFSET_PT * style_scale
        return _SlideRangeProjection(
            lower + offset,
            current + offset,
            upper + offset,
            tangent,
            normal,
            alpha,
        )

    def _draw_slide_range(
        self,
        overlay: Draw2D,
        cam,
        rect,
        style_scale: float,
        state: _JointRangeState,
        *,
        phase: str = "all",
        prepared: _SlideRangeProjection | None = None,
    ) -> None:
        slide = prepared or self._slide_range_projection(
            cam, rect, style_scale, state, self._frame.position, self._frame.rotation
        )
        if slide is None:
            return
        alpha = slide.alpha
        range_color = self._flat_color(GizmoHandle.Z, 2, alpha)
        lower, current, upper = slide.lower, slide.current, slide.upper
        tangent, normal = slide.tangent, slide.normal
        range_width = JOINT_RANGE_WIDTH_PT * style_scale
        if phase != "labels":
            overlay.line(lower, upper, range_color, range_width)
            self._draw_slide_handle(
                overlay,
                style_scale,
                current,
                tangent,
                normal,
                alpha,
            )

            if self._using and self._active is GizmoHandle.Z and not self._snapping:
                self._draw_joint_translation_guide(
                    overlay,
                    cam,
                    rect,
                    style_scale,
                    alpha=alpha,
                )
            else:
                overlay.line(
                    current - normal * 10.0 * style_scale,
                    current + normal * 10.0 * style_scale,
                    _joint_current_tick_color(range_color),
                    range_width,
                    cap="round",
                    smoothing=self._frame.corner_smoothing,
                )

            half_tick = 6.0 * style_scale
            tick_width = range_width
            entries = (
                (
                    state.lower,
                    _joint_limit_label("MIN", state.lower, "slide"),
                    (lower - normal * half_tick, lower + normal * half_tick),
                    lower,
                    JOINT_LOWER_LIMIT_COLOR,
                    True,
                    True,
                ),
                (
                    state.upper,
                    _joint_limit_label("MAX", state.upper, "slide"),
                    (upper - normal * half_tick, upper + normal * half_tick),
                    upper,
                    JOINT_UPPER_LIMIT_COLOR,
                    False,
                    False,
                ),
            )
            for _value, _label, tick, _anchor, limit_color, _above, _align_right in entries:
                overlay.line(
                    tick[0],
                    tick[1],
                    _with_alpha(limit_color, alpha),
                    tick_width,
                    cap="round",
                    smoothing=self._frame.corner_smoothing,
                )
            self._set_joint_limit_hits(
                state,
                entries,
                tick_width=tick_width,
                tick_cap="round",
                style_scale=style_scale,
            )

    def _set_joint_limit_hits(
        self,
        state: _JointRangeState,
        entries,
        *,
        tick_width: float,
        tick_cap: str,
        style_scale: float,
    ) -> None:
        if state.joint_id < 0 or state.qpos_adr < 0:
            return
        self._joint_limit_hits = tuple(
            JointLimitHit(
                joint_id=state.joint_id,
                qpos_adr=state.qpos_adr,
                value=float(value),
                label=label,
                rect=_joint_limit_tick_rect(tick, tick_width, style_scale),
                semantic_color=semantic_color,
                tick_start=tuple(float(component) for component in tick[0]),
                tick_end=tuple(float(component) for component in tick[1]),
                tick_width=float(tick_width),
                tick_cap=tick_cap,
                label_anchor=tuple(float(component) for component in label_anchor),
                label_above=bool(label_above),
                label_align_right=bool(label_align_right),
            )
            for (
                value,
                label,
                tick,
                label_anchor,
                semantic_color,
                label_above,
                label_align_right,
            ) in entries
            if tick is not None and label_anchor is not None
        )

    def apply_joint_limit(self, session: Session, hit: JointLimitHit) -> CommandResult:
        """Move the selected scalar joint to the endpoint represented by a tick."""

        if not session.paused:
            return CommandResult.bad("Pause the simulation before editing a joint")
        target, reason = self._joint_target(session, session.selected_node)
        if target is None:
            return CommandResult.bad(reason or "The joint target is no longer available")
        joint = target.joint
        if int(joint.joint_id) != hit.joint_id or int(joint.qpos_adr) != hit.qpos_adr:
            return CommandResult.bad("The joint target changed; choose the endpoint again")
        return session.submit(SetQpos(hit.qpos_adr, hit.value))
