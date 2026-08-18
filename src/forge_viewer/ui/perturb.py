"""Translation and rotation perturbation geometry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .. import math3d
from ..commands import ClearPerturb, Perturb
from ..gizmo import (
    AXIS_COLORS,
    AXIS_SHAFT_HALF_PT,
    CENTER_COLOR,
    CENTER_RADIUS,
    CENTER_SHELL_RADIUS,
    CONTRAST_EDGE_COLOR,
    CONTRAST_EDGE_PT,
    GUIDE_CORE_COLOR,
    SIZE_PT,
    axis_handle_alpha,
    world_scale,
)
from ..log import get_logger
from ..types import CameraView
from .camera import camera_basis

log = get_logger("perturb")

if TYPE_CHECKING:
    from ..adapters.base import SceneNode
    from ..session import PerturbState, Session

try:
    from ..render.debugdraw import Occlusion  # type: ignore[attr-defined]
except Exception:
    Occlusion = None  # type: ignore[assignment]


DEG_PER_PIXEL = 0.4


AXIS_OVERSHOOT = 2.0


DRAG_WIDTH_PX = 2.0


GRAB_RADIUS_PX = 6.0
TIP_RADIUS_PX = 6.0


OUTLINE_BORDER_RGBA = tuple(float(v) for v in CONTRAST_EDGE_COLOR)
OUTLINE_RGBA = (0.98, 0.98, 0.98, 1.0)
OUTLINE_WIDTH_PT = 2.0
OUTLINE_BORDER_WIDTH_PT = OUTLINE_WIDTH_PT + 2.0 * CONTRAST_EDGE_PT
OUTLINE_CORNER_RADIUS_PT = 4.0
OUTLINE_CORNER_SEGMENTS = 5

DRAG_EDGE_RGBA = tuple(float(v) for v in CONTRAST_EDGE_COLOR)
DRAG_RGBA = tuple(float(v) for v in GUIDE_CORE_COLOR)

MARK_LAYER = "ui.perturb.mark"

DRAG_LAYER = "ui.perturb.drag"


def freeze_plane_depth(cam: CameraView, grab_point) -> float:
    _, _, forward = camera_basis(cam)
    return float(
        np.dot(np.asarray(grab_point, np.float64) - np.asarray(cam.eye, np.float64), forward)
    )


def point_on_frozen_plane(cam: CameraView, origin, direction, plane_depth: float) -> np.ndarray:
    _, _, forward = camera_basis(cam)
    o = np.asarray(origin, np.float64)
    d = np.asarray(direction, np.float64)
    denom = float(np.dot(d, forward))
    if abs(denom) < 1e-9:
        denom = 1e-9
    depth_o = float(np.dot(o - np.asarray(cam.eye, np.float64), forward))
    t = (plane_depth - depth_o) / denom
    return (o + d * t).astype(np.float64)


def delta_rotvec(cam: CameraView, dx_px: float, dy_px: float) -> np.ndarray:
    right, up, _ = camera_basis(cam)
    k = float(np.deg2rad(DEG_PER_PIXEL))
    return (up * (float(dx_px) * k) + right * (float(dy_px) * k)).astype(np.float64)


_CORNER_SIGNS = np.array(
    [[sx, sy, sz] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)], np.float64
)


def box_corners(center, mat3, half) -> np.ndarray:
    c = np.asarray(center, np.float64).reshape(3)
    r = np.asarray(mat3, np.float64).reshape(3, 3)
    h = np.broadcast_to(np.asarray(half, np.float64).reshape(-1), (3,))
    return c + (_CORNER_SIGNS * h) @ r.T


def silhouette_edges(center, mat3, half, eye) -> list[tuple[np.ndarray, np.ndarray]]:
    c = np.asarray(center, np.float64).reshape(3)
    r = np.asarray(mat3, np.float64).reshape(3, 3)
    h = np.broadcast_to(np.asarray(half, np.float64).reshape(-1), (3,)).astype(np.float64)
    e = np.asarray(eye, np.float64).reshape(3)

    facing: dict[tuple[int, float], bool] = {}
    for axis in range(3):
        normal = r[:, axis]
        for sign in (-1.0, 1.0):
            face_center = c + normal * (sign * h[axis])
            facing[(axis, sign)] = bool(np.dot(normal * sign, e - face_center) > 0.0)

    corners = box_corners(c, r, h)
    index = {tuple(s): i for i, s in enumerate(_CORNER_SIGNS.astype(int).tolist())}

    out: list[tuple[np.ndarray, np.ndarray]] = []
    for along in range(3):
        u, v = [a for a in range(3) if a != along]
        for su in (-1.0, 1.0):
            for sv in (-1.0, 1.0):
                if facing[(u, su)] == facing[(v, sv)]:
                    continue
                a = [0, 0, 0]
                b = [0, 0, 0]
                a[u] = b[u] = int(su)
                a[v] = b[v] = int(sv)
                a[along], b[along] = -1, 1
                out.append((corners[index[tuple(a)]], corners[index[tuple(b)]]))
    return out


def silhouette_loop(edges: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    if not edges:
        return np.zeros((0, 3), np.float64)
    unused = list(edges)
    a, b = unused.pop(0)
    loop = [a, b]
    while unused:
        end = loop[-1]
        for index, (u, v) in enumerate(unused):
            if np.array_equal(u, end):
                loop.append(v)
                unused.pop(index)
                break
            if np.array_equal(v, end):
                loop.append(u)
                unused.pop(index)
                break
        else:
            return np.zeros((0, 3), np.float64)
    if np.array_equal(loop[-1], loop[0]):
        loop.pop()
    return np.asarray(loop, np.float64)


def project(cam: CameraView, points, rect: tuple[float, float, float, float]) -> np.ndarray:
    pts = np.asarray(points, np.float64).reshape(-1, 3)
    mvp = np.asarray(cam.proj_matrix(), np.float64) @ np.asarray(cam.view_matrix(), np.float64)
    h = np.concatenate([pts, np.ones((len(pts), 1))], axis=1) @ mvp.T
    w = h[:, 3:4].copy()
    safe = np.where(np.abs(w) < 1e-9, 1e-9, w)
    ndc = h[:, :3] / safe
    rx, ry, rw, rh = rect
    sx = rx + (ndc[:, 0] * 0.5 + 0.5) * rw
    sy = ry + (0.5 - ndc[:, 1] * 0.5) * rh
    return np.stack([sx, sy, w[:, 0]], axis=1)


def rounded_loop(
    points,
    cam: CameraView,
    rect: tuple[float, float, float, float],
    radius_px: float,
    segments: int = OUTLINE_CORNER_SEGMENTS,
) -> np.ndarray:
    loop = np.asarray(points, np.float64).reshape(-1, 3)
    screen = project(cam, loop, rect)[:, :2]
    curves: list[np.ndarray] = []
    for index, corner in enumerate(loop):
        previous = loop[index - 1]
        following = loop[(index + 1) % len(loop)]
        incoming_px = float(np.linalg.norm(screen[index] - screen[index - 1]))
        outgoing_px = float(np.linalg.norm(screen[(index + 1) % len(loop)] - screen[index]))
        if min(incoming_px, outgoing_px) < 1e-5:
            curves.append(corner[None, :])
            continue
        trim_px = min(float(radius_px), 0.4 * incoming_px, 0.4 * outgoing_px)
        start = corner + (previous - corner) * (trim_px / incoming_px)
        end = corner + (following - corner) * (trim_px / outgoing_px)
        t = np.linspace(0.0, 1.0, segments + 1, dtype=np.float64)[:, None]
        curves.append((1.0 - t) ** 2 * start + 2.0 * (1.0 - t) * t * corner + t**2 * end)
    return np.concatenate(curves, axis=0)


def axis_draw_order(cam: CameraView, rotation) -> tuple[int, int, int]:
    forward = camera_basis(cam)[2]
    r = np.asarray(rotation, np.float64).reshape(3, 3)
    order = sorted(range(3), key=lambda axis: float(np.dot(r[:, axis], forward)), reverse=True)
    return order[0], order[1], order[2]


@dataclass
class MarkBudget:
    primitives: int = 0
    dropped: int = 0
    note: str = ""


class PerturbController:
    def __init__(self) -> None:
        self.budget = MarkBudget()
        self.outline_corner_radius_pt = OUTLINE_CORNER_RADIUS_PT
        self._last_note = ""
        self._published = ""

    def begin(
        self,
        session: Session,
        cam: CameraView,
        node: SceneNode,
        grab_point,
        mode: str,
        *,
        body_radius: float = 0.1,
    ) -> PerturbState:
        st = session.perturb
        pos, mat = current_pose(session, node)
        st.active = True
        st.node_id = int(node.node_id)
        st.object_id = int(node.object_id)
        st.mode = mode
        st.grab_point = np.asarray(grab_point, np.float32).copy()
        st.start_pos = pos.astype(np.float32)
        st.start_mat = mat.astype(np.float32)
        st.target_pos = pos.astype(np.float32)
        st.target_mat = mat.astype(np.float32)
        st.plane_depth = freeze_plane_depth(cam, grab_point)
        st.body_radius = float(body_radius)
        return st

    def drag_translate(self, session: Session, cam: CameraView, origin, direction) -> np.ndarray:
        st = session.perturb
        p = point_on_frozen_plane(cam, origin, direction, st.plane_depth)
        st.target_pos = (
            np.asarray(st.start_pos, np.float64) + (p - np.asarray(st.grab_point, np.float64))
        ).astype(np.float32)
        return p

    def drag_rotate(
        self, session: Session, cam: CameraView, dx_px: float, dy_px: float
    ) -> np.ndarray:
        st = session.perturb
        delta = math3d.rotvec_to_mat3(delta_rotvec(cam, dx_px, dy_px))
        st.target_mat = (
            np.asarray(delta, np.float64) @ np.asarray(st.target_mat, np.float64)
        ).astype(np.float32)
        return st.target_mat

    def end(self, session: Session) -> None:
        session.submit(ClearPerturb())

    def apply(self, session: Session) -> None:
        st = session.perturb
        if not st.active or st.node_id < 0:
            return
        node = session.node(st.node_id)
        if node is None:
            return
        result = session.submit(
            Perturb(
                node_id=st.node_id,
                target_position=np.asarray(st.target_pos, np.float32),
                target_rotation=np.asarray(st.target_mat, np.float32),
                mode=st.mode,
            )
        )
        if not result.ok:
            self._note(result.message)

    def publish_marks(
        self,
        backend: Any,
        session: Session,
        cam: CameraView,
        *,
        rect: tuple[float, float, float, float],
        ui_scale: float = 1.0,
        style_scale: float = 1.0,
    ) -> MarkBudget:
        budget = self.budget
        budget.primitives = 0
        budget.dropped = 0
        budget.note = ""
        st = session.perturb
        if not st.active:
            if self._published:
                self._clear(backend)
                self._published = ""
            return budget

        dd = backend.debug
        caps = backend.caps
        if not caps.debug_draw or dd is None or Occlusion is None:
            budget.dropped = 1

            why = "has no debug draw" if Occlusion is not None else "is missing render.debugdraw"
            budget.note = f"{caps.name} {why}; perturbation feedback uses the 2D fallback"
            self._note(budget.note)
            return budget

        node = session.node(st.node_id)
        pose = current_pose(session, node) if node is not None else (None, None)
        if st.mode == "translate":
            layer_name = DRAG_LAYER
            self._publish_drag(dd, st, pose, budget, ui_scale)
        else:
            layer_name = MARK_LAYER
            self._publish_mark(dd, st, pose, cam, rect, budget, ui_scale, style_scale)

        if self._published and self._published != layer_name:
            self._clear(backend, self._published)
        self._published = layer_name
        return budget

    def _publish_drag(
        self,
        dd: Any,
        st: PerturbState,
        pose: tuple[np.ndarray | None, np.ndarray | None],
        budget: MarkBudget,
        ui_scale: float,
    ) -> None:
        layer = dd.layer(DRAG_LAYER, Occlusion.ALWAYS)
        grab = grab_point_now(st, *pose).astype(np.float32)
        tip = grab_point_target(st).astype(np.float32)
        layer.drag_link(
            "perturb.drag",
            grab,
            tip,
            DRAG_RGBA,
            DRAG_EDGE_RGBA,
            width_px=DRAG_WIDTH_PX * ui_scale,
            radius_px=GRAB_RADIUS_PX * ui_scale,
            edge_px=CONTRAST_EDGE_PT * ui_scale,
        )
        budget.primitives = 1

    def _publish_mark(
        self,
        dd: Any,
        st: PerturbState,
        pose: tuple[np.ndarray | None, np.ndarray | None],
        cam: CameraView,
        rect: tuple[float, float, float, float],
        budget: MarkBudget,
        ui_scale: float,
        style_scale: float = 1.0,
    ) -> None:
        layer = dd.layer(MARK_LAYER, Occlusion.ALWAYS)
        center = (
            np.asarray(pose[0], np.float64)
            if pose[0] is not None
            else np.asarray(st.target_pos, np.float64)
        )
        axis_len = world_scale(cam, center, rect[3], SIZE_PT * style_scale)
        size = axis_len / AXIS_OVERSHOOT
        half = np.full(3, size, np.float64)

        outline = silhouette_edges(center, st.target_mat, half, cam.eye)
        loop = silhouette_loop(outline)
        if len(loop):
            loop = rounded_loop(
                loop,
                cam,
                rect,
                self.outline_corner_radius_pt * style_scale,
            )
            layer.polyline(
                "perturb.outline.border",
                loop,
                OUTLINE_BORDER_RGBA,
                OUTLINE_BORDER_WIDTH_PT * ui_scale,
                closed=True,
            )
            layer.polyline(
                "perturb.outline",
                loop,
                OUTLINE_RGBA,
                OUTLINE_WIDTH_PT * ui_scale,
                closed=True,
            )

        axis_width = 2.0 * AXIS_SHAFT_HALF_PT * ui_scale
        shell_radius = CENTER_SHELL_RADIUS * SIZE_PT * ui_scale
        rotation = np.asarray(st.target_mat, np.float64)
        order = axis_draw_order(cam, rotation)
        directions = rotation[:, order].T
        colors = np.asarray(AXIS_COLORS)[list(order)].copy()
        for row, direction in zip(colors, directions, strict=True):
            row[3] = axis_handle_alpha(cam, center, direction)
        layer.arrows(
            "perturb.axes",
            np.broadcast_to(center, (3, 3)),
            center + directions * axis_len,
            colors,
            axis_width,
            start_mask_px=shell_radius,
        )
        layer.point(
            "perturb.center.edge",
            center,
            CONTRAST_EDGE_COLOR,
            (CENTER_RADIUS * SIZE_PT + CONTRAST_EDGE_PT) * ui_scale,
        )
        layer.point(
            "perturb.center",
            center,
            CENTER_COLOR,
            CENTER_RADIUS * SIZE_PT * ui_scale,
        )
        budget.primitives = 2 * len(loop) + 5 if len(loop) else 5

    def _clear(self, backend: Any, only: str = "") -> None:
        dd = backend.debug
        if dd is None or not backend.caps.debug_draw or Occlusion is None:
            return
        if only in ("", MARK_LAYER):
            dd.layer(MARK_LAYER, Occlusion.ALWAYS).clear()
        if only in ("", DRAG_LAYER):
            dd.layer(DRAG_LAYER, Occlusion.ALWAYS).clear()

    def _note(self, message: str) -> None:
        if message and message != self._last_note:
            self._last_note = message
            log.warning("{}", message)


def fallback_segments(
    cam: CameraView,
    st: PerturbState,
    rect: tuple[float, float, float, float],
    center,
    style_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    center = np.asarray(center, np.float64)
    size = world_scale(cam, center, rect[3], SIZE_PT * style_scale) / AXIS_OVERSHOOT
    edges = silhouette_edges(center, st.target_mat, np.full(3, size), cam.eye)
    points = silhouette_loop(edges)
    scr = project(cam, points, rect)
    return scr, np.roll(scr, -1, axis=0)


def draw_fallback(
    cam: CameraView,
    st: PerturbState,
    rect: tuple[float, float, float, float],
    cursor: tuple[float, float],
    center,
    style_scale: float = 1.0,
) -> None:
    from imgui_bundle import imgui

    dl = imgui.get_window_draw_list()
    border = imgui.color_convert_float4_to_u32(imgui.ImVec4(*OUTLINE_BORDER_RGBA))
    color = imgui.color_convert_float4_to_u32(imgui.ImVec4(*OUTLINE_RGBA))
    a, _b = fallback_segments(cam, st, rect, center, style_scale)
    if len(a) and np.all(a[:, 2] > 0.0):
        points = [imgui.ImVec2(*p[:2]) for p in a]
        closed = imgui.ImDrawFlags_.closed.value
        dl.add_polyline(points, border, OUTLINE_BORDER_WIDTH_PT * style_scale, closed)
        dl.add_polyline(points, color, OUTLINE_WIDTH_PT * style_scale, closed)
    dl.add_circle(
        imgui.ImVec2(*cursor),
        10.0 * style_scale,
        border,
        24,
        (1.5 + 2.0 * CONTRAST_EDGE_PT) * style_scale,
    )
    dl.add_circle(imgui.ImVec2(*cursor), 10.0 * style_scale, color, 24, 1.5 * style_scale)


def current_pose(session: Session, node: SceneNode) -> tuple[np.ndarray, np.ndarray]:
    frame = session.frame
    i = int(node.body_index)
    pos = np.zeros(3, np.float64)
    mat = np.eye(3, dtype=np.float64)
    if frame.body_xpos is not None and 0 <= i < len(frame.body_xpos):
        pos = np.asarray(frame.body_xpos[i], np.float64).reshape(3)
    if frame.body_xmat is not None and 0 <= i < len(frame.body_xmat):
        mat = np.asarray(frame.body_xmat[i], np.float64).reshape(3, 3)
    return pos, mat


def grab_point_local(st: PerturbState) -> np.ndarray:
    r = np.asarray(st.start_mat, np.float64).reshape(3, 3)
    return r.T @ (np.asarray(st.grab_point, np.float64) - np.asarray(st.start_pos, np.float64))


def grab_point_now(st: PerturbState, pos, mat) -> np.ndarray:
    if pos is None or mat is None:
        return np.asarray(st.grab_point, np.float64)
    return np.asarray(pos, np.float64) + np.asarray(mat, np.float64) @ grab_point_local(st)


def grab_point_target(st: PerturbState) -> np.ndarray:
    return np.asarray(st.target_pos, np.float64) + np.asarray(
        st.target_mat, np.float64
    ) @ grab_point_local(st)


def rotvec_of(mat3) -> np.ndarray:
    m = np.asarray(mat3, np.float64).reshape(3, 3)
    cos = (np.trace(m) - 1.0) * 0.5
    angle = float(np.arccos(np.clip(cos, -1.0, 1.0)))
    if angle < 1e-9:
        return np.zeros(3)
    axis = np.array([m[2, 1] - m[1, 2], m[0, 2] - m[2, 0], m[1, 0] - m[0, 1]])
    n = np.linalg.norm(axis)
    if n < 1e-9:
        d = np.clip(np.diag(m), -1.0, 1.0)
        axis = np.sqrt(np.maximum((d + 1.0) * 0.5, 0.0))
        n = np.linalg.norm(axis)
        return axis / max(n, 1e-9) * angle
    return axis / n * angle
