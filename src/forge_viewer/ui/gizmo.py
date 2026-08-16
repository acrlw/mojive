from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .. import math3d
from ..commands import SetPose
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
    plane_corners,
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


@dataclass(frozen=True)
class Verdict:
    ok: bool
    reason: str = ""


def verdict(paused: bool, node: SceneNode | None) -> Verdict:
    if node is None:
        return Verdict(False, REASON_NO_SELECTION)
    reason = gizmo_refusal_reason(paused, node.posable)
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
        self._rotation_angle = 0.0
        self._label = ""

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

    def update_hover(
        self,
        session: Session,
        cam: CameraView,
        rect: tuple[float, float, float, float],
        cursor: tuple[float, float],
    ) -> GizmoHandle:
        node = session.selected_node
        self._verdict = verdict(session.paused, node)
        if not self._verdict.ok:
            self._hovered = GizmoHandle.NONE
            self._axis_mask = self._plane_mask = 0
            return self._hovered
        if self._active is not GizmoHandle.NONE:
            self._hovered = self._active
            return self._hovered
        pos, mat = _node_pose(session, node)
        self._hovered, self._axis_mask, self._plane_mask = hit_test(
            cam, pos, self._basis(mat), rect, cursor, self._mode
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
    ) -> bool:

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
        return self._drag(session, cam, rect, cursor)

    def keyboard_interact(self, session, cam, rect, cursor, axis: int) -> bool:

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
            self._verdict = verdict(session.paused, node)
            if not self._verdict.ok or not self._begin_handle(session, cam, rect, cursor, handle):
                return False
            self._keyboard = self._using = True
            return True
        return self._drag(session, cam, rect, cursor)

    def publish(
        self,
        backend: Any,
        session: Session,
        cam: CameraView,
        rect: tuple[float, float, float, float],
        *,
        pixel_scale: float,
        yielding: bool,
        interactive: bool,
    ) -> bool:

        self._interactive = bool(interactive)
        node = session.selected_node
        self._verdict = verdict(session.paused, node)
        self._visible = not yielding and self._verdict.ok
        if not self._visible:
            self._clear_translation_guide(backend)
            if backend.caps.gizmo:
                backend.set_gizmo(None)
            self._drawn = False
            return False
        pos, mat = _node_pose(session, node)
        basis = self._basis(mat)
        scale = world_scale(cam, pos, rect[3])
        self._axis_mask, self._plane_mask = visibility(cam, pos, basis, rect, scale)
        frame = self._frame
        frame.mode = self._mode
        frame.style = self._style
        frame.space = self._space
        np.copyto(frame.position, pos, casting="unsafe")
        np.copyto(frame.rotation, basis, casting="unsafe")
        frame.size_px = SIZE_PT * float(pixel_scale)
        frame.hovered = self._hovered if interactive else GizmoHandle.NONE
        frame.active = self._active
        frame.axis_mask = self._axis_mask
        frame.plane_mask = self._plane_mask
        self._publish_translation_guide(backend, pixel_scale)
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
        if self._keyboard:
            self._draw_axis_constraint(imgui, dl, cam, rect, style_scale)
        if self._style is GizmoStyle.FLAT:
            self._draw_flat(imgui, dl, cam, rect, style_scale)
            self._drawn = True
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
        scale = world_scale(cam, origin, rect[3])
        visible = display_handles(frame)
        u32 = imgui.color_convert_float4_to_u32

        for axis, handle in enumerate(PLANE_HANDLES):
            if handle not in visible:
                continue
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

        for axis, handle in enumerate(AXIS_HANDLES):
            if handle not in visible:
                continue
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
                radius = SCREEN_RING_RADIUS * SIZE_PT
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

    def _publish_translation_guide(self, backend: Any, pixel_scale: float) -> None:
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
            width_px=2.0 * pixel_scale,
            radius_px=6.0 * pixel_scale,
            edge_px=CONTRAST_EDGE_PT * pixel_scale,
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
        radius = world_scale(cam, self._start_pos, rect[3]) * ring_radius

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
            radius = world_scale(cam, self._start_pos, rect[3]) * ring_radius
            anchor_world = self._start_pos + self._rotation_start_vec * radius
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
        self._rotation_angle = 0.0
        self._label = self._format_value(self._start_pos)

        axis = _axis_of(self._active)
        if axis >= 0:
            self._axis[:] = self._start_basis[:, axis]
        elif self._active is GizmoHandle.ROTATE_SCREEN:
            self._axis[:] = -cam.forward()

        if self._active in (GizmoHandle.X, GizmoHandle.Y, GizmoHandle.Z):
            scale = world_scale(cam, pos, rect[3])
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

    def _drag(self, session, cam, rect, cursor) -> bool:
        handle = self._active
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
            if abs(angle) < 1e-9:
                return False
            delta = math3d.rotvec_to_mat3(self._axis * angle)
            self._current_mat[:] = delta @ self._current_mat
            self._last_rot_vec[:] = v
            self._rotation_angle += angle
            mat = self._current_mat

        node = session.selected_node
        if node is None:
            self._end()
            return False
        result = session.submit(
            SetPose(
                node_id=node.node_id,
                position=np.asarray(pos, np.float32),
                rotation=np.asarray(mat, np.float32),
            )
        )
        if not result.ok:
            self._verdict = Verdict(False, result.message)
            self._end()
            return False
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
            return f"{name} {degrees:+.1f}°{suffix}"
        delta = np.asarray(position, np.float64) - self._start_pos
        local = self._start_basis.T @ delta
        if self._active in AXIS_HANDLES:
            return f"{name} {local[axis]:+.3f} m"
        plane_axes = {
            GizmoHandle.YZ: (1, 2),
            GizmoHandle.ZX: (2, 0),
            GizmoHandle.XY: (0, 1),
        }.get(self._active)
        if plane_axes is not None:
            a, b = plane_axes
            return f"{'XYZ'[a]} {local[a]:+.3f}  {'XYZ'[b]} {local[b]:+.3f} m"
        return f"X {local[0]:+.3f}  Y {local[1]:+.3f}  Z {local[2]:+.3f} m"

    def _basis(self, rotation) -> np.ndarray:
        if self._space is GizmoSpace.BODY:
            return np.asarray(rotation, np.float64).reshape(3, 3)
        return _WORLD_BASIS

    def _end(self) -> None:
        self._using = False
        self._keyboard = False
        self._active = GizmoHandle.NONE
        self._label = ""


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
    i = int(node.body_index)
    pos = np.zeros(3, np.float64)
    mat = np.eye(3, dtype=np.float64)
    if frame.body_xpos is not None and 0 <= i < len(frame.body_xpos):
        pos = np.asarray(frame.body_xpos[i], np.float64).reshape(3)
    if frame.body_xmat is not None and 0 <= i < len(frame.body_xmat):
        mat = np.asarray(frame.body_xmat[i], np.float64).reshape(3, 3)
    return pos, mat
