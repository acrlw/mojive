"""Interactive position and rotation gizmo behavior."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .. import math3d
from ..adapters.base import IkOptions, NodeKind
from ..commands import SetPose, SolveIk
from ..gizmo import (
    AXIS_COLORS,
    AXIS_END,
    AXIS_HANDLES,
    AXIS_HEAD_HALF_PT,
    AXIS_HEAD_LENGTH_PT,
    AXIS_SHAFT_HALF_PT,
    AXIS_START,
    CENTER_COLOR,
    CENTER_RADIUS,
    CENTER_SHELL_RADIUS,
    CONTRAST_EDGE_COLOR,
    CONTRAST_EDGE_PT,
    GUIDE_CORE_COLOR,
    HOVER_COLOR,
    PLANE_ACTIVE_ALPHA,
    PLANE_ALPHA,
    PLANE_HANDLES,
    RING_RADIUS,
    RING_SEGMENTS,
    RING_WIDTH_PT,
    ROTATE_AXIS_HANDLES,
    ROTATE_HANDLES,
    SCREEN_RING_RADIUS,
    SCREEN_RING_WIDTH_PT,
    SIZE_PT,
    GizmoFrame,
    GizmoHandle,
    GizmoMode,
    GizmoSpace,
    GizmoStyle,
    axis_handle_alpha,
    display_handles,
    hit_test,
    paint_order,
    plane_corners,
    plane_direction,
    plane_handle_alpha,
    project,
    rotation_ring,
    rotation_ring_alpha,
    visibility,
    world_scale,
)
from ..render.debugdraw import Occlusion
from ..types import CameraView
from .camera import ndc_from_viewport, unproject
from .panels.inspector import gizmo_refusal_reason

if TYPE_CHECKING:
    from ..adapters.base import SceneNode
    from ..session import Session

REASON_NO_SELECTION = "nothing selected"
DRAG_LAYER = "ui.gizmo.drag"
_WORLD_BASIS = np.eye(3, dtype=np.float64)
DEFAULT_TRANSLATION_SNAP_M = 0.5
DEFAULT_ROTATION_SNAP_DEG = 5.0
SNAP_TICK_FULL_STEPS = 5.0
SNAP_TICK_FADE_STEPS = 10.0


@dataclass(frozen=True)
class Verdict:
    ok: bool
    reason: str = ""


def verdict(paused: bool, node: SceneNode | None, inverse_kinematics: bool = False) -> Verdict:
    if node is None:
        return Verdict(False, REASON_NO_SELECTION)
    reason = gizmo_refusal_reason(
        paused,
        node.posable,
        inverse_kinematics,
        node.ik_target,
    )
    return Verdict(reason is None, reason or "")


class ObjectGizmo:
    def __init__(self, mode: str = "translate") -> None:
        self._mode = GizmoMode(mode)
        self._style = GizmoStyle.FLAT
        self._space = GizmoSpace.BODY
        self._hovered = GizmoHandle.NONE
        self._active = GizmoHandle.NONE
        self._verdict = Verdict(False, REASON_NO_SELECTION)
        self._using = False
        self._keyboard = False
        self._guide_gpu = False
        self._interactive = True
        self._visible = False
        self._drawn = False
        self._axis_mask = 0b111
        self._plane_mask = 0b111
        self._frame = GizmoFrame()
        self._style_scale = 1.0

        self._start_pos = np.zeros(3, np.float64)
        self._start_mat = np.eye(3, dtype=np.float64)
        self._start_basis = np.eye(3, dtype=np.float64)
        self._current_mat = np.eye(3, dtype=np.float64)
        self._start_cursor = np.zeros(2, np.float64)
        self._axis = np.zeros(3, np.float64)
        self._axis_screen = np.zeros(2, np.float64)
        self._world_per_pt = 0.0
        self._plane_normal = np.zeros(3, np.float64)
        self._plane_start = np.zeros(3, np.float64)
        self._rotation_start_vec = np.zeros(3, np.float64)
        self._last_rot_vec = np.zeros(3, np.float64)
        self._rotation_raw_angle = 0.0
        self._rotation_angle = 0.0
        self._snapping = False
        self._label = ""
        self._edit_started = False
        self.translation_snap_m = DEFAULT_TRANSLATION_SNAP_M
        self.rotation_snap_deg = DEFAULT_ROTATION_SNAP_DEG

    @property
    def mode(self) -> str:
        return self._mode.value

    @property
    def style(self) -> str:
        return self._style.value

    @property
    def space(self) -> str:
        return self._space.value

    @property
    def last_drawn(self) -> bool:
        return self._drawn

    @property
    def interactive(self) -> bool:
        return self._interactive

    @property
    def using(self) -> bool:
        return self._using

    @property
    def keyboard_using(self) -> bool:
        return self._keyboard

    @property
    def last_verdict(self) -> Verdict:
        return self._verdict

    @property
    def hovered(self) -> bool:
        return self._verdict.ok and self._hovered is not GizmoHandle.NONE

    @property
    def hovered_handle(self) -> GizmoHandle:
        return self._hovered

    @property
    def active_handle(self) -> GizmoHandle:
        return self._active

    @property
    def value_label(self) -> str:
        return self._label

    @property
    def snapping(self) -> bool:
        return self._snapping

    def set_mode(self, mode: str) -> None:
        if mode in (GizmoMode.TRANSLATE.value, GizmoMode.ROTATE.value) and not self._using:
            self._mode = GizmoMode(mode)

    def set_style(self, style: str) -> None:
        if style in (GizmoStyle.FLAT.value, GizmoStyle.SOLID.value) and not self._using:
            self._style = GizmoStyle(style)

    def set_space(self, space: str) -> None:
        if space in (GizmoSpace.BODY.value, GizmoSpace.WORLD.value) and not self._using:
            self._space = GizmoSpace(space)

    def toggle_space(self) -> None:
        self.set_space("world" if self._space is GizmoSpace.BODY else "body")

    def cancel(self) -> None:
        self._end()

    def update_hover(
        self,
        session: Session,
        cam: CameraView,
        rect: tuple[float, float, float, float],
        cursor: tuple[float, float],
        *,
        style_scale: float = 1.0,
    ) -> GizmoHandle:
        self._style_scale = float(style_scale)
        node = session.selected_node
        self._verdict = verdict(session.paused, node, session.adapter.caps.inverse_kinematics)
        if not self._verdict.ok:
            self._hovered = GizmoHandle.NONE
            self._axis_mask = self._plane_mask = 0
            return self._hovered
        if self._active is not GizmoHandle.NONE:
            self._hovered = self._active
            return self._hovered
        pos, mat = _node_pose(session, node)
        self._hovered, self._axis_mask, self._plane_mask = hit_test(
            cam, pos, self._basis(mat), rect, cursor, self._mode, self._style_scale
        )
        return self._hovered

    def interact(
        self,
        session: Session,
        cam: CameraView,
        rect: tuple[float, float, float, float],
        cursor: tuple[float, float],
        *,
        claimed: bool,
        left_down: bool,
        released: bool,
        snap: bool = False,
        style_scale: float = 1.0,
    ) -> bool:
        self._style_scale = float(style_scale)
        if not claimed:
            if self._using and not left_down:
                self._end()
            return False
        if released or not left_down:
            self._end()
            return False
        if self._active is GizmoHandle.NONE and not self._begin(session, cam, rect, cursor):
            return False
        self._using = True
        return self._drag(session, cam, rect, cursor, snap=snap)

    def keyboard_interact(
        self,
        session,
        cam,
        rect,
        cursor,
        axis: int,
        *,
        snap: bool = False,
        style_scale: float = 1.0,
    ) -> bool:
        self._style_scale = float(style_scale)
        if axis not in (0, 1, 2):
            if self._keyboard:
                self._end()
            return False
        handle = (
            AXIS_HANDLES[axis] if self._mode is GizmoMode.TRANSLATE else ROTATE_AXIS_HANDLES[axis]
        )
        if self._keyboard and self._active is not handle:
            self._end()
        if not self._keyboard:
            node = session.selected_node
            self._verdict = verdict(session.paused, node, session.adapter.caps.inverse_kinematics)
            if not self._verdict.ok or not self._begin_handle(session, cam, rect, cursor, handle):
                return False
            self._keyboard = self._using = True
            return True
        return self._drag(session, cam, rect, cursor, snap=snap)

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
        self._interactive = bool(interactive)
        node = session.selected_node
        self._verdict = verdict(session.paused, node, session.adapter.caps.inverse_kinematics)
        self._visible = not yielding and self._verdict.ok
        if not self._visible:
            self._clear_translation_guide(backend)
            if backend.caps.gizmo:
                backend.set_gizmo(None)
            self._drawn = False
            return False
        pos, mat = _node_pose(session, node)
        basis = self._basis(mat)
        scale = world_scale(cam, pos, rect[3], SIZE_PT * float(style_scale))
        self._axis_mask, self._plane_mask = visibility(cam, pos, basis, rect, scale)
        frame = self._frame
        frame.mode = self._mode
        frame.style = self._style
        frame.space = self._space
        np.copyto(frame.position, pos, casting="unsafe")
        np.copyto(frame.rotation, basis, casting="unsafe")
        frame.size_px = SIZE_PT * float(ui_scale)
        frame.hovered = self._hovered if interactive else GizmoHandle.NONE
        frame.active = self._active
        frame.axis_mask = self._axis_mask
        frame.plane_mask = self._plane_mask
        self._publish_translation_guide(backend, ui_scale)
        if self._style is GizmoStyle.FLAT:
            if backend.caps.gizmo:
                backend.set_gizmo(None)
            self._drawn = False
            return True
        if not backend.caps.gizmo:
            self._drawn = False
            return False
        self._drawn = bool(backend.set_gizmo(frame))
        return self._drawn

    def draw_overlay(self, cam, rect, *, style_scale: float = 1.0) -> None:
        if not self._visible:
            return
        from imgui_bundle import imgui

        dl = imgui.get_window_draw_list()
        if self._keyboard and not self._snapping:
            self._draw_axis_constraint(imgui, dl, cam, rect, style_scale)
        if self._style is GizmoStyle.FLAT:
            self._draw_flat(imgui, dl, cam, rect, style_scale)
            self._drawn = True
        if self._using and self._snapping and self._active in AXIS_HANDLES:
            self._draw_translation_snap_ruler(imgui, dl, cam, rect, style_scale)
        if self._using and self._snapping and self._active in ROTATE_HANDLES:
            self._draw_rotation_snap_ticks(imgui, dl, cam, rect, style_scale)
        if self._using and self._active not in ROTATE_HANDLES and not self._guide_gpu:
            self._draw_translation_guide(imgui, dl, cam, rect, style_scale)
        if self._using and self._active in ROTATE_HANDLES:
            self._draw_rotation_guide(imgui, dl, cam, rect, style_scale)
        if self._using and self._label:
            self._draw_value_label(imgui, dl, cam, rect, style_scale)

    def _draw_flat(self, imgui, dl, cam, rect, style_scale: float) -> None:
        frame = self._frame
        origin = np.asarray(frame.position, np.float64)
        rotation = np.asarray(frame.rotation, np.float64)
        scale = world_scale(cam, origin, rect[3], SIZE_PT * style_scale)
        visible = display_handles(frame)
        u32 = imgui.color_convert_float4_to_u32

        # The draw list paints in submission order with no depth buffer, so
        # each handle group draws far-to-near (painter's order) to put the
        # nearer handle on top where handles overlap.
        planes = [axis for axis, handle in enumerate(PLANE_HANDLES) if handle in visible]
        for k in paint_order(cam, origin, [plane_direction(rotation, axis) for axis in planes]):
            axis = planes[k]
            handle = PLANE_HANDLES[axis]
            alpha = (
                1.0
                if frame.active is handle
                else plane_handle_alpha(cam, origin, rotation[:, axis])
            )
            if alpha <= 0.0:
                continue
            screen = project(cam, plane_corners(origin, rotation, scale, axis), rect)
            if np.any(screen[:, 2] <= 0.0):
                continue
            opacity = PLANE_ACTIVE_ALPHA if frame.active is handle else PLANE_ALPHA * alpha
            color = self._flat_color(handle, axis, opacity)
            dl.add_convex_poly_filled(
                [imgui.ImVec2(float(p[0]), float(p[1])) for p in screen],
                u32(imgui.ImVec4(*color)),
            )

        axes = [axis for axis, handle in enumerate(AXIS_HANDLES) if handle in visible]
        for k in paint_order(cam, origin, [rotation[:, axis] for axis in axes]):
            axis = axes[k]
            handle = AXIS_HANDLES[axis]
            alpha = (
                1.0 if frame.active is handle else axis_handle_alpha(cam, origin, rotation[:, axis])
            )
            if alpha <= 0.0:
                continue
            screen = project(
                cam,
                (
                    origin + rotation[:, axis] * scale * AXIS_START,
                    origin + rotation[:, axis] * scale * AXIS_END,
                ),
                rect,
            )
            if np.any(screen[:, 2] <= 0.0):
                continue
            start = _masked_axis_start(
                screen[0, :2],
                screen[1, :2],
                CENTER_SHELL_RADIUS * SIZE_PT * style_scale,
            )
            points = _flat_arrow(start, screen[1, :2], style_scale)
            if points:
                dl.add_concave_poly_filled(
                    [imgui.ImVec2(float(p[0]), float(p[1])) for p in points],
                    u32(imgui.ImVec4(*self._flat_color(handle, axis, alpha))),
                )

        if GizmoHandle.SCREEN in visible:
            center = project(cam, (origin,), rect)[0]
            if center[2] > 0.0:
                color = HOVER_COLOR if self._hot(GizmoHandle.SCREEN) else CENTER_COLOR
                position = imgui.ImVec2(float(center[0]), float(center[1]))
                radius = CENTER_RADIUS * SIZE_PT * style_scale
                dl.add_circle_filled(
                    position,
                    radius + CONTRAST_EDGE_PT * style_scale,
                    u32(imgui.ImVec4(*(float(x) for x in CONTRAST_EDGE_COLOR))),
                    24,
                )
                dl.add_circle_filled(
                    position,
                    radius,
                    u32(imgui.ImVec4(*(float(x) for x in color))),
                    24,
                )

        for axis, handle in enumerate(ROTATE_AXIS_HANDLES):
            if handle not in visible:
                continue
            full = frame.active is handle
            alpha = 1.0 if full else rotation_ring_alpha(cam, origin, rotation[:, axis])
            if alpha <= 0.0:
                continue
            ring = rotation_ring(cam, origin, rotation, scale, axis, full=full)
            screen = project(cam, ring, rect)
            if np.any(screen[:, 2] <= 0.0):
                continue
            dl.add_polyline(
                [imgui.ImVec2(float(p[0]), float(p[1])) for p in screen],
                u32(imgui.ImVec4(*self._flat_color(handle, axis, alpha))),
                RING_WIDTH_PT * style_scale,
                (imgui.ImDrawFlags_.closed if full else imgui.ImDrawFlags_.none).value,
            )

        if GizmoHandle.ROTATE_SCREEN in visible:
            center = project(cam, (origin,), rect)[0]
            if center[2] > 0.0:
                color = HOVER_COLOR if self._hot(GizmoHandle.ROTATE_SCREEN) else CENTER_COLOR
                position = imgui.ImVec2(float(center[0]), float(center[1]))
                radius = SCREEN_RING_RADIUS * SIZE_PT * style_scale
                dl.add_circle(
                    position,
                    radius,
                    u32(imgui.ImVec4(*(float(x) for x in CONTRAST_EDGE_COLOR))),
                    RING_SEGMENTS,
                    (SCREEN_RING_WIDTH_PT + 2.0 * CONTRAST_EDGE_PT) * style_scale,
                )
                dl.add_circle(
                    position,
                    radius,
                    u32(imgui.ImVec4(*(float(x) for x in color))),
                    RING_SEGMENTS,
                    SCREEN_RING_WIDTH_PT * style_scale,
                )

    def _flat_color(self, handle: GizmoHandle, axis: int, alpha: float = 1.0):
        color = HOVER_COLOR if self._hot(handle) else AXIS_COLORS[axis]
        return float(color[0]), float(color[1]), float(color[2]), float(alpha)

    def _hot(self, handle: GizmoHandle) -> bool:
        return self._active is handle or (self._interactive and self._hovered is handle)

    def _draw_axis_constraint(self, imgui, dl, cam, rect, style_scale: float) -> None:
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
        color = AXIS_COLORS[axis]
        dl.add_line(
            imgui.ImVec2(*segment[0]),
            imgui.ImVec2(*segment[1]),
            imgui.color_convert_float4_to_u32(
                imgui.ImVec4(float(color[0]), float(color[1]), float(color[2]), 0.62)
            ),
            1.5 * style_scale,
        )

    def _draw_translation_snap_ruler(self, imgui, dl, cam, rect, style_scale: float) -> None:
        axis_index = _axis_of(self._active)
        if axis_index < 0:
            return
        axis = self._start_basis[:, axis_index]
        projected = project(cam, (self._start_pos, self._start_pos + axis), rect)
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
            self._start_pos,
            axis,
            segment,
            rect,
        )
        if bounds is None:
            return

        u32 = imgui.color_convert_float4_to_u32
        axis_color = AXIS_COLORS[axis_index]
        step = float(self.translation_snap_m)
        current_distance = float(np.dot(self._frame.position - self._start_pos, axis))
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
            world = self._start_pos + axis * distance
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

        current = project(cam, (self._frame.position,), rect)[0]
        mask_radius = CENTER_SHELL_RADIUS * SIZE_PT * style_scale
        if current[2] > 0.0:
            half_length = 14.0 * style_scale
            ticks.append(
                (
                    current[:2] - normal * half_length,
                    current[:2] + normal * half_length,
                    1.0,
                    True,
                )
            )
        axis_segments = _split_segment_around_point(
            segment[0], segment[1], current[:2], mask_radius
        )

        def color(value, alpha: float):
            return u32(
                imgui.ImVec4(
                    float(value[0]), float(value[1]), float(value[2]), float(value[3]) * alpha
                )
            )

        for start, end in axis_segments:
            dl.add_line(
                imgui.ImVec2(*start),
                imgui.ImVec2(*end),
                color((*axis_color, 1.0), 0.92),
                1.2 * style_scale,
            )
        for a, b, alpha, is_active in ticks:
            tick_color = color(HOVER_COLOR if is_active else (*axis_color, 1.0), alpha)
            for start, end in _split_segment_around_point(a, b, current[:2], mask_radius):
                dl.add_line(
                    imgui.ImVec2(*start),
                    imgui.ImVec2(*end),
                    tick_color,
                    (2.2 if is_active else 1.2) * style_scale,
                )

    def _draw_rotation_snap_ticks(self, imgui, dl, cam, rect, style_scale: float) -> None:
        ring_radius = (
            SCREEN_RING_RADIUS if self._active is GizmoHandle.ROTATE_SCREEN else RING_RADIUS
        )
        scale = world_scale(cam, self._start_pos, rect[3], SIZE_PT * style_scale)
        if scale <= 0.0:
            return
        tangent = np.cross(self._axis, self._rotation_start_vec)
        step = float(self.rotation_snap_deg)
        u32 = imgui.color_convert_float4_to_u32
        edge = u32(imgui.ImVec4(*CONTRAST_EDGE_COLOR))
        core = u32(imgui.ImVec4(*GUIDE_CORE_COLOR))

        def radial(angle: float) -> np.ndarray:
            cosine = np.cos(angle)
            return (
                cosine * self._rotation_start_vec
                + np.sin(angle) * tangent
                + (1.0 - cosine) * np.dot(self._axis, self._rotation_start_vec) * self._axis
            )

        center = project(cam, (self._start_pos,), rect)[0]
        ring_angles = np.linspace(0.0, 2.0 * np.pi, RING_SEGMENTS, endpoint=False)
        ring = project(
            cam,
            np.asarray(
                [self._start_pos + scale * ring_radius * radial(angle) for angle in ring_angles]
            ),
            rect,
        )
        trace: np.ndarray | None = None
        if np.all(ring[:, 2] > 0.0):
            normals = _outward_normals(ring[:, :2], center[:2])
            trace = ring[:, :2] + normals * (4.0 * style_scale)

        ticks_visible = (
            self._active is GizmoHandle.ROTATE_SCREEN
            or rotation_ring_alpha(cam, self._start_pos, self._axis) > 0.0
        )
        ticks: list[tuple[np.ndarray, np.ndarray]] = []
        for degrees in np.arange(0.0, 360.0, step):
            rounded = round(float(degrees))
            if abs(degrees - rounded) < 1e-6 and rounded % 90 == 0:
                length_pt = 9.0
            elif abs(degrees / 45.0 - round(degrees / 45.0)) < 1e-6:
                length_pt = 7.0
            elif abs(degrees / 15.0 - round(degrees / 15.0)) < 1e-6:
                length_pt = 5.0
            else:
                length_pt = 3.0
            angle = np.radians(degrees)
            direction = radial(angle)
            angle_step = np.radians(step)
            points = project(
                cam,
                (
                    self._start_pos + scale * ring_radius * radial(angle - 1e-3),
                    self._start_pos + scale * ring_radius * direction,
                    self._start_pos + scale * ring_radius * radial(angle + 1e-3),
                    self._start_pos + scale * ring_radius * radial(angle + angle_step),
                ),
                rect,
            )
            if np.any(points[:, 2] <= 0.0):
                continue
            spacing = float(np.linalg.norm(points[3, :2] - points[1, :2]))
            if not ticks_visible or spacing < 2.0 * style_scale:
                continue
            normal = _outward_normals(points[:, :2], center[:2])[1]
            inner = points[1, :2] + normal * (4.0 * style_scale)
            outer = inner + normal * (length_pt * style_scale)
            ticks.append((inner, outer))

        active_tick: tuple[np.ndarray, np.ndarray] | None = None
        direction = radial(self._rotation_angle)
        points = project(
            cam,
            (
                self._start_pos + scale * ring_radius * radial(self._rotation_angle - 1e-3),
                self._start_pos + scale * ring_radius * direction,
                self._start_pos + scale * ring_radius * radial(self._rotation_angle + 1e-3),
            ),
            rect,
        )
        if np.all(points[:, 2] > 0.0) and ticks_visible:
            normal = _outward_normals(points[:, :2], center[:2])[1]
            active_tick = (
                points[1, :2] + normal * (4.0 * style_scale),
                points[1, :2] + normal * (15.0 * style_scale),
            )

        closed = imgui.ImDrawFlags_.closed.value
        if trace is not None:
            ring_points = [imgui.ImVec2(*point) for point in trace]
            dl.add_polyline(ring_points, edge, 2.5 * style_scale, closed)
        for inner, outer in ticks:
            dl.add_line(
                imgui.ImVec2(*inner),
                imgui.ImVec2(*outer),
                core,
                1.1 * style_scale,
            )
        if active_tick is not None:
            active_color = u32(imgui.ImVec4(*HOVER_COLOR))
            dl.add_line(
                imgui.ImVec2(*active_tick[0]),
                imgui.ImVec2(*active_tick[1]),
                active_color,
                2.2 * style_scale,
            )
        if trace is not None:
            dl.add_polyline(ring_points, core, 1.1 * style_scale, closed)

    def _draw_translation_guide(self, imgui, dl, cam, rect, style_scale: float) -> None:
        screen = project(cam, (self._start_pos, self._frame.position), rect)
        if np.any(screen[:, 2] <= 0.0):
            return
        start, end = screen[:, :2]
        delta = end - start
        distance = float(np.linalg.norm(delta))
        edge = imgui.color_convert_float4_to_u32(imgui.ImVec4(*CONTRAST_EDGE_COLOR))
        core = imgui.color_convert_float4_to_u32(imgui.ImVec4(*GUIDE_CORE_COLOR))
        radius = 6.0 * style_scale
        core_width = 2.0 * style_scale
        edge_width = core_width + 2.0 * CONTRAST_EDGE_PT * style_scale
        if distance > 2.0 * radius:
            direction = delta / distance
            a = imgui.ImVec2(*(start + direction * radius))
            b = imgui.ImVec2(*(end - direction * radius))
            dl.add_line(a, b, edge, edge_width)
            dl.add_line(a, b, core, core_width)
        for point in (start, end):
            center = imgui.ImVec2(*point)
            dl.add_circle(center, radius, edge, 24, edge_width)
            dl.add_circle(center, radius, core, 24, core_width)

    def _publish_translation_guide(self, backend: Any, ui_scale: float) -> None:
        dd = getattr(backend, "debug", None)
        active = self._using and self._active not in ROTATE_HANDLES
        if not active or not backend.caps.debug_draw or dd is None:
            self._clear_translation_guide(backend)
            return
        dd.layer(DRAG_LAYER, Occlusion.ALWAYS).drag_link(
            "gizmo.drag",
            self._start_pos,
            self._frame.position,
            GUIDE_CORE_COLOR,
            CONTRAST_EDGE_COLOR,
            width_px=2.0 * ui_scale,
            radius_px=6.0 * ui_scale,
            edge_px=CONTRAST_EDGE_PT * ui_scale,
        )
        self._guide_gpu = True

    def _clear_translation_guide(self, backend: Any) -> None:
        dd = getattr(backend, "debug", None)
        if self._guide_gpu and backend.caps.debug_draw and dd is not None:
            dd.layer(DRAG_LAYER, Occlusion.ALWAYS).clear()
        self._guide_gpu = False

    def _draw_rotation_guide(self, imgui, dl, cam, rect, style_scale: float) -> None:
        sweep = _rotation_sweep(self._rotation_angle)
        tangent = np.cross(self._axis, self._rotation_start_vec)
        ring_radius = (
            SCREEN_RING_RADIUS if self._active is GizmoHandle.ROTATE_SCREEN else RING_RADIUS
        )
        radius = world_scale(cam, self._start_pos, rect[3], SIZE_PT * style_scale) * ring_radius

        def arc_screen(angles):
            cosine = np.cos(angles)[:, None]
            sine = np.sin(angles)[:, None]
            arc = (
                cosine * self._rotation_start_vec
                + sine * tangent
                + (1.0 - cosine) * np.dot(self._axis, self._rotation_start_vec) * self._axis
            )
            return project(cam, self._start_pos + radius * arc, rect)

        point_count = max(2, int(np.ceil(RING_SEGMENTS * abs(sweep) / (2.0 * np.pi))) + 1)
        arc = arc_screen(np.linspace(0.0, sweep, point_count))
        center = project(cam, (self._start_pos,), rect)[0]
        if center[2] <= 0.0 or np.any(arc[:, 2] <= 0.0):
            return
        center = center[:2]
        arc = arc[:, :2]
        sector = [imgui.ImVec2(*center), *(imgui.ImVec2(*p) for p in arc)]
        arc_points = [imgui.ImVec2(*p) for p in arc]
        u32 = imgui.color_convert_float4_to_u32
        border = u32(imgui.ImVec4(1.0, 0.5, 0.06, 1.0))
        turns = int(abs(round(float(np.degrees(self._rotation_angle)), 1)) // 360.0)
        if turns:
            full = arc_screen(
                np.linspace(
                    0.0,
                    np.copysign(2.0 * np.pi, self._rotation_angle),
                    RING_SEGMENTS,
                    endpoint=False,
                )
            )
            if np.all(full[:, 2] > 0.0):
                dl.add_polyline(
                    [imgui.ImVec2(*p) for p in full[:, :2]],
                    u32(imgui.ImVec4(1.0, 0.5, 0.06, 0.42)),
                    1.5 * style_scale,
                    imgui.ImDrawFlags_.closed.value,
                )

        fill_alpha = _rotation_fill_alpha(sweep)
        if fill_alpha > 0.0:
            dl.add_convex_poly_filled(sector, u32(imgui.ImVec4(1.0, 0.5, 0.06, fill_alpha)))
        if abs(sweep) > 1e-6:
            dl.add_polyline(
                arc_points,
                border,
                2.0 * style_scale,
                imgui.ImDrawFlags_.none.value,
            )
        endpoints = (arc[0],) if abs(sweep) < 1e-6 else (arc[0], arc[-1])
        for endpoint in endpoints:
            dl.add_line(
                imgui.ImVec2(*center),
                imgui.ImVec2(*endpoint),
                border,
                2.0 * style_scale,
            )

    def _draw_value_label(self, imgui, dl, cam, rect, style_scale: float) -> None:
        pad = 6.0 * style_scale
        gap = 14.0 * style_scale
        anchor_world = self._frame.position
        if self._active in ROTATE_HANDLES:
            ring_radius = (
                SCREEN_RING_RADIUS if self._active is GizmoHandle.ROTATE_SCREEN else RING_RADIUS
            )
            tangent = np.cross(self._axis, self._rotation_start_vec)
            cosine = np.cos(self._rotation_angle)
            direction = (
                cosine * self._rotation_start_vec
                + np.sin(self._rotation_angle) * tangent
                + (1.0 - cosine) * np.dot(self._axis, self._rotation_start_vec) * self._axis
            )
            radius = world_scale(cam, self._start_pos, rect[3], SIZE_PT * style_scale) * (
                ring_radius + 14.0 / SIZE_PT
            )
            anchor_world = self._start_pos + direction * radius
        anchor = project(cam, (anchor_world,), rect)[0]
        if anchor[2] <= 0.0:
            return
        size = imgui.calc_text_size(self._label)
        width, height = float(size.x) + 2.0 * pad, float(size.y) + 2.0 * pad
        x = float(np.clip(anchor[0] + gap, rect[0] + 4.0, rect[0] + rect[2] - width - 4.0))
        y = float(np.clip(anchor[1] + gap, rect[1] + 4.0, rect[1] + rect[3] - height - 4.0))
        u32 = imgui.color_convert_float4_to_u32
        dl.add_rect_filled(
            imgui.ImVec2(x, y),
            imgui.ImVec2(x + width, y + height),
            u32(imgui.ImVec4(0.08, 0.09, 0.11, 0.92)),
            4.0 * style_scale,
        )
        dl.add_text(
            imgui.ImVec2(x + pad, y + pad),
            u32(imgui.ImVec4(0.96, 0.96, 0.97, 1.0)),
            self._label,
        )

    def _begin(self, session, cam, rect, cursor) -> bool:
        return self._begin_handle(session, cam, rect, cursor, self._hovered)

    def _begin_handle(self, session, cam, rect, cursor, handle: GizmoHandle) -> bool:
        node = session.selected_node
        if node is None or handle is GizmoHandle.NONE:
            return False
        pos, mat = _node_pose(session, node)
        self._active = handle
        np.copyto(self._start_pos, pos)
        np.copyto(self._start_mat, mat)
        np.copyto(self._start_basis, self._basis(mat))
        np.copyto(self._current_mat, mat)
        self._start_cursor[:] = cursor
        self._rotation_raw_angle = 0.0
        self._rotation_angle = 0.0
        self._snapping = False
        self._edit_started = False
        self._label = self._format_value(self._start_pos)

        axis = _axis_of(self._active)
        if axis >= 0:
            self._axis[:] = self._start_basis[:, axis]
        elif self._active is GizmoHandle.ROTATE_SCREEN:
            self._axis[:] = -cam.forward()

        if self._active in (GizmoHandle.X, GizmoHandle.Y, GizmoHandle.Z):
            scale = world_scale(cam, pos, rect[3], SIZE_PT * self._style_scale)
            screen = project(cam, [pos, pos + self._axis * scale * AXIS_END], rect)[:, :2]
            delta = screen[1] - screen[0]
            length = float(np.linalg.norm(delta))
            if length < 1e-6:
                self._end()
                return False
            self._axis_screen[:] = delta / length
            self._world_per_pt = scale / length
            return True

        if self._active in (GizmoHandle.SCREEN, GizmoHandle.ROTATE_SCREEN):
            self._plane_normal[:] = cam.forward()
        else:
            self._plane_normal[:] = self._axis

        hit = _cursor_plane(cam, rect, cursor, pos, self._plane_normal)
        if hit is None:
            self._end()
            return False
        if self._active in ROTATE_HANDLES:
            v = hit - pos
            n = float(np.linalg.norm(v))
            if n < 1e-9:
                self._end()
                return False
            self._rotation_start_vec[:] = self._last_rot_vec[:] = v / n
        else:
            self._plane_start[:] = hit
        return True

    def _drag(self, session, cam, rect, cursor, *, snap: bool) -> bool:
        handle = self._active
        self._snapping = bool(snap)
        pos = self._start_pos.copy()
        mat = self._start_mat
        if handle in (GizmoHandle.X, GizmoHandle.Y, GizmoHandle.Z):
            travel = float(np.dot(np.asarray(cursor) - self._start_cursor, self._axis_screen))
            pos += self._axis * (travel * self._world_per_pt)
        elif handle in (GizmoHandle.SCREEN, GizmoHandle.YZ, GizmoHandle.ZX, GizmoHandle.XY):
            hit = _cursor_plane(cam, rect, cursor, self._start_pos, self._plane_normal)
            if hit is None:
                return False
            pos += hit - self._plane_start
        else:
            hit = _cursor_plane(cam, rect, cursor, self._start_pos, self._plane_normal)
            if hit is None:
                return False
            v = hit - self._start_pos
            n = float(np.linalg.norm(v))
            if n < 1e-9:
                return False
            v /= n
            angle = float(
                np.arctan2(
                    np.dot(self._axis, np.cross(self._last_rot_vec, v)),
                    np.dot(self._last_rot_vec, v),
                )
            )
            if abs(angle) >= 1e-9:
                self._last_rot_vec[:] = v
                self._rotation_raw_angle += angle
            self._rotation_angle = (
                _snap_value(self._rotation_raw_angle, np.radians(self.rotation_snap_deg))
                if snap
                else self._rotation_raw_angle
            )
            delta = math3d.rotvec_to_mat3(self._axis * self._rotation_angle)
            self._current_mat[:] = delta @ self._start_mat
            mat = self._current_mat

        if snap and handle not in ROTATE_HANDLES:
            delta = self._start_basis.T @ (pos - self._start_pos)
            delta = _snap_translation(delta, handle, self.translation_snap_m)
            pos = self._start_pos + self._start_basis @ delta

        node = session.selected_node
        if node is None:
            self._end()
            return False
        if node.posable:
            command = SetPose(
                node_id=node.node_id,
                position=np.asarray(pos, np.float32),
                rotation=np.asarray(mat, np.float32),
            )
        else:
            command = SolveIk(
                node_id=node.node_id,
                target_position=np.asarray(pos, np.float32),
                target_rotation=np.asarray(mat, np.float32),
                options=IkOptions(
                    position=handle not in ROTATE_HANDLES,
                    rotation=handle in ROTATE_HANDLES,
                ),
                record_undo=not self._edit_started,
            )
        result = session.submit(command)
        if not result.ok:
            self._verdict = Verdict(False, result.message)
            self._end()
            return False
        self._edit_started = True
        self._label = self._format_value(pos)
        return True

    def _format_value(self, position) -> str:
        axis = _axis_of(self._active)
        name = (
            "Screen"
            if self._active is GizmoHandle.ROTATE_SCREEN
            else ("XYZ"[axis] if axis >= 0 else "")
        )
        if self._active in ROTATE_HANDLES:
            degrees = round(float(np.degrees(self._rotation_angle)), 1)
            turns = int(abs(degrees) // 360.0)
            suffix = f" · {turns}×360°" if turns else ""
            snap = f" · SNAP {_format_step(self.rotation_snap_deg)}°" if self._snapping else ""
            return f"{name} {degrees:+.1f}°{suffix}{snap}"
        delta = np.asarray(position, np.float64) - self._start_pos
        local = self._start_basis.T @ delta
        if self._active in AXIS_HANDLES:
            value = f"{name} {local[axis]:+.3f} m"
            return self._with_translation_snap(value)
        plane_axes = {
            GizmoHandle.YZ: (1, 2),
            GizmoHandle.ZX: (2, 0),
            GizmoHandle.XY: (0, 1),
        }.get(self._active)
        if plane_axes is not None:
            a, b = plane_axes
            value = f"{'XYZ'[a]} {local[a]:+.3f}  {'XYZ'[b]} {local[b]:+.3f} m"
            return self._with_translation_snap(value)
        value = f"X {local[0]:+.3f}  Y {local[1]:+.3f}  Z {local[2]:+.3f} m"
        return self._with_translation_snap(value)

    def _with_translation_snap(self, value: str) -> str:
        if not self._snapping:
            return value
        return f"{value} · SNAP {_format_step(self.translation_snap_m)} m"

    def _basis(self, rotation) -> np.ndarray:
        if self._space is GizmoSpace.BODY:
            return np.asarray(rotation, np.float64).reshape(3, 3)
        return _WORLD_BASIS

    def _end(self) -> None:
        self._using = False
        self._keyboard = False
        self._snapping = False
        self._active = GizmoHandle.NONE
        self._label = ""
        self._edit_started = False


def _axis_of(handle: GizmoHandle) -> int:
    return {
        GizmoHandle.X: 0,
        GizmoHandle.Y: 1,
        GizmoHandle.Z: 2,
        GizmoHandle.YZ: 0,
        GizmoHandle.ZX: 1,
        GizmoHandle.XY: 2,
        GizmoHandle.ROTATE_X: 0,
        GizmoHandle.ROTATE_Y: 1,
        GizmoHandle.ROTATE_Z: 2,
    }.get(handle, -1)


def _snap_value(value: float, step: float) -> float:
    return float(np.round(float(value) / float(step)) * float(step))


def _snap_translation(delta: np.ndarray, handle: GizmoHandle, step: float) -> np.ndarray:
    snapped = np.asarray(delta, np.float64).copy()
    axes = {
        GizmoHandle.X: (0,),
        GizmoHandle.Y: (1,),
        GizmoHandle.Z: (2,),
        GizmoHandle.YZ: (1, 2),
        GizmoHandle.ZX: (2, 0),
        GizmoHandle.XY: (0, 1),
        GizmoHandle.SCREEN: (0, 1, 2),
    }[handle]
    snapped[list(axes)] = np.round(snapped[list(axes)] / step) * step
    return snapped


def _format_step(value: float) -> str:
    return f"{float(value):g}"


def _rotation_sweep(angle: float) -> float:
    shown = np.radians(round(float(np.degrees(angle)), 1))
    return float(np.copysign(np.fmod(abs(shown), 2.0 * np.pi), shown))


def _rotation_fill_alpha(sweep: float) -> float:
    fade = (np.pi - abs(sweep)) / np.radians(60.0)
    return 0.28 * float(np.clip(fade, 0.0, 1.0))


def _clip_line_to_rect(origin, direction, rect) -> tuple[np.ndarray, np.ndarray] | None:
    origin = np.asarray(origin, np.float64)
    direction = np.asarray(direction, np.float64)
    if float(np.linalg.norm(direction)) < 1e-6:
        return None
    x, y, w, h = rect
    limits = ((x, x + w), (y, y + h))
    lo, hi = -np.inf, np.inf
    for axis in range(2):
        if abs(direction[axis]) < 1e-9:
            if not limits[axis][0] <= origin[axis] <= limits[axis][1]:
                return None
            continue
        t0 = (limits[axis][0] - origin[axis]) / direction[axis]
        t1 = (limits[axis][1] - origin[axis]) / direction[axis]
        lo, hi = max(lo, min(t0, t1)), min(hi, max(t0, t1))
    if lo > hi:
        return None
    return origin + lo * direction, origin + hi * direction


def _projected_line_parameters(cam, origin, axis, segment, rect) -> tuple[float, float] | None:
    mvp = np.asarray(cam.proj_matrix(), np.float64) @ np.asarray(cam.view_matrix(), np.float64)
    clip_origin = mvp @ np.append(np.asarray(origin, np.float64), 1.0)
    clip_axis = mvp @ np.append(np.asarray(axis, np.float64), 0.0)
    x, y, width, height = rect
    values = []
    for point in segment:
        ndc = np.array(
            (
                2.0 * (float(point[0]) - x) / width - 1.0,
                1.0 - 2.0 * (float(point[1]) - y) / height,
            )
        )
        denominator = clip_axis[:2] - ndc * clip_axis[3]
        component = int(np.argmax(np.abs(denominator)))
        if abs(denominator[component]) < 1e-10:
            return None
        numerator = ndc[component] * clip_origin[3] - clip_origin[component]
        values.append(float(numerator / denominator[component]))
    return values[0], values[1]


def _snap_tick_alpha(offset_steps: float) -> float:
    distance = abs(float(offset_steps))
    if distance <= SNAP_TICK_FULL_STEPS:
        return 1.0
    if distance >= SNAP_TICK_FADE_STEPS:
        return 0.0
    t = (distance - SNAP_TICK_FULL_STEPS) / (SNAP_TICK_FADE_STEPS - SNAP_TICK_FULL_STEPS)
    return float(1.0 - t * t * (3.0 - 2.0 * t))


def _split_segment_around_point(
    start, end, center, radius: float
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    start = np.asarray(start, np.float64)
    end = np.asarray(end, np.float64)
    center = np.asarray(center, np.float64)
    delta = end - start
    length = float(np.linalg.norm(delta))
    if length < 1e-9 or radius <= 0.0:
        return ((start, end),)
    direction = delta / length
    along = float(np.dot(center - start, direction))
    perpendicular = float(np.linalg.norm(center - (start + along * direction)))
    if perpendicular >= radius:
        return ((start, end),)
    half_gap = float(np.sqrt(radius * radius - perpendicular * perpendicular))
    gap_start = max(0.0, along - half_gap)
    gap_end = min(length, along + half_gap)
    if gap_start >= gap_end:
        return ((start, end),)
    segments = []
    if gap_start > 1e-6:
        segments.append((start, start + direction * gap_start))
    if gap_end < length - 1e-6:
        segments.append((start + direction * gap_end, end))
    return tuple(segments)


def _outward_normals(points: np.ndarray, center: np.ndarray) -> np.ndarray:
    points = np.asarray(points, np.float64)
    tangents = np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0)
    normals = np.column_stack((-tangents[:, 1], tangents[:, 0]))
    lengths = np.linalg.norm(normals, axis=1)
    radial = points - np.asarray(center, np.float64)
    degenerate = lengths < 1e-9
    normals[degenerate] = radial[degenerate]
    lengths[degenerate] = np.linalg.norm(normals[degenerate], axis=1)
    normals /= np.maximum(lengths[:, None], 1e-9)
    inward = np.sum(normals * radial, axis=1) < 0.0
    normals[inward] *= -1.0
    return normals


def _flat_arrow(start: np.ndarray, end: np.ndarray, style_scale: float) -> list[np.ndarray]:
    direction = np.asarray(end, np.float64) - np.asarray(start, np.float64)
    length = float(np.linalg.norm(direction))
    if length < 1e-6:
        return []
    direction /= length
    side = np.array((-direction[1], direction[0]))
    shaft = AXIS_SHAFT_HALF_PT * style_scale
    head = min(AXIS_HEAD_LENGTH_PT * style_scale, length * 0.42)
    wing = AXIS_HEAD_HALF_PT * style_scale
    neck = np.asarray(end) - direction * head
    return [
        np.asarray(start) - side * shaft,
        neck - side * shaft,
        neck - side * wing,
        np.asarray(end),
        neck + side * wing,
        neck + side * shaft,
        np.asarray(start) + side * shaft,
    ]


def _masked_axis_start(origin: np.ndarray, end: np.ndarray, radius: float) -> np.ndarray:
    direction = np.asarray(end, np.float64) - np.asarray(origin, np.float64)
    length = float(np.linalg.norm(direction))
    if length <= radius or length < 1e-9:
        return np.asarray(end, np.float64)
    return np.asarray(origin, np.float64) + direction * (float(radius) / length)


def _cursor_plane(cam, rect, cursor, point, normal) -> np.ndarray | None:
    ndc = ndc_from_viewport(cursor[0], cursor[1], rect)
    origin, direction = unproject(cam, *ndc)
    den = float(np.dot(direction, normal))
    if abs(den) < 1e-8:
        return None
    t = float(np.dot(np.asarray(point) - origin, normal) / den)
    return np.asarray(origin, np.float64) + np.asarray(direction, np.float64) * t


def _node_pose(session: Session, node: SceneNode) -> tuple[np.ndarray, np.ndarray]:
    frame = session.frame
    if node.kind is NodeKind.SITE:
        i = int(node.site_index)
        pos = np.zeros(3, np.float64)
        mat = np.eye(3, dtype=np.float64)
        if frame.site_xpos is not None and 0 <= i < len(frame.site_xpos):
            pos = np.asarray(frame.site_xpos[i], np.float64).reshape(3)
        if frame.site_xmat is not None and 0 <= i < len(frame.site_xmat):
            mat = np.asarray(frame.site_xmat[i], np.float64).reshape(3, 3)
        return pos, mat
    i = int(node.body_index)
    pos = np.zeros(3, np.float64)
    mat = np.eye(3, dtype=np.float64)
    if frame.body_xpos is not None and 0 <= i < len(frame.body_xpos):
        pos = np.asarray(frame.body_xpos[i], np.float64).reshape(3)
    if frame.body_xmat is not None and 0 <= i < len(frame.body_xmat):
        mat = np.asarray(frame.body_xmat[i], np.float64).reshape(3, 3)
    return pos, mat
