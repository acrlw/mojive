"""Orientation view gizmo layout and interaction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..types import CameraView
from .camera import PITCH_LIMIT, OrbitCamera, camera_basis
from .theme import THEME

RADIUS_PT = 34.0

BALL_PT = 9.5
MARGIN_PT = 10.0
LINE_PT = 2.0


AXIS_NAMES = ("X", "Y", "Z")

TOP_YAW = -90.0


LABEL_FILL = (0.97, 0.97, 0.98, 1.0)

DARK_MIX = 0.72

DARK_BASE = (0.10, 0.11, 0.13)

HOVER_GAIN = 1.35
HOVER_LIFT = 0.10

BACK_FADE_START = 0.72
BACK_FADE_END = 0.96


@dataclass(frozen=True)
class Ball:
    axis: int
    sign: float
    screen: tuple[float, float]
    radius: float
    depth: float

    alpha: float = 1.0

    @property
    def positive(self) -> bool:
        return self.sign > 0.0

    @property
    def label(self) -> str:
        return AXIS_NAMES[self.axis] if self.positive else f"-{AXIS_NAMES[self.axis]}"


def layout(
    cam: CameraView, center: tuple[float, float], radius_pt: float, ball_pt: float
) -> list[Ball]:
    right, up, forward = camera_basis(cam)
    out: list[Ball] = []
    for axis in range(3):
        world = np.zeros(3)
        for sign in (1.0, -1.0):
            world[:] = 0.0
            world[axis] = sign
            depth = float(np.dot(world, forward))
            sx = center[0] + float(np.dot(world, right)) * radius_pt

            sy = center[1] - float(np.dot(world, up)) * radius_pt
            out.append(
                Ball(
                    axis=axis,
                    sign=sign,
                    screen=(sx, sy),
                    radius=ball_pt,
                    depth=depth,
                    alpha=_back_alpha(depth),
                )
            )
    out.sort(key=lambda b: -b.depth)
    return out


def hit_test(balls: list[Ball], cursor: tuple[float, float]) -> Ball | None:
    best: Ball | None = None
    for b in balls:
        if b.alpha <= 0.1:
            continue
        dx = cursor[0] - b.screen[0]
        dy = cursor[1] - b.screen[1]
        if dx * dx + dy * dy <= b.radius * b.radius and (best is None or b.depth < best.depth):
            best = b
    return best


def widget_center(
    rect: tuple[float, float, float, float], style_scale: float
) -> tuple[float, float]:
    r = (RADIUS_PT + BALL_PT) * style_scale
    m = MARGIN_PT * style_scale
    return (rect[0] + rect[2] - m - r, rect[1] + m + r)


def yaw_pitch_for(axis: int, sign: float, current_yaw: float) -> tuple[float, float]:
    if axis == 2:
        return TOP_YAW, PITCH_LIMIT * (1.0 if sign > 0 else -1.0)
    yaw = 0.0 if axis == 0 else 90.0
    if sign < 0:
        yaw += 180.0
    return yaw, 0.0


class ViewCube:
    def __init__(self) -> None:
        self._balls: list[Ball] = []
        self._hover: Ball | None = None
        self._center: tuple[float, float] = (0.0, 0.0)

    @property
    def balls(self) -> list[Ball]:
        return self._balls

    def update(
        self, cam: CameraView, rect: tuple[float, float, float, float], cursor, style_scale: float
    ) -> Ball | None:
        self._center = widget_center(rect, style_scale)
        self._balls = layout(cam, self._center, RADIUS_PT * style_scale, BALL_PT * style_scale)
        self._hover = hit_test(self._balls, cursor)
        return self._hover

    @property
    def hovered(self) -> Ball | None:
        return self._hover

    def drag(self, camera: OrbitCamera, dx: float, dy: float) -> None:
        camera.orbit(dx, dy)

    def click(self, camera: OrbitCamera, ball: Ball, sink) -> None:
        yaw, pitch = yaw_pitch_for(ball.axis, ball.sign, camera.yaw)
        camera.look_from(yaw, pitch, sink)

    def draw(self, style_scale: float = 1.0) -> None:
        from imgui_bundle import imgui

        dl = imgui.get_window_draw_list()
        if not self._balls:
            return
        center = imgui.ImVec2(*self._center)
        u32 = imgui.color_convert_float4_to_u32

        if self._hover is not None:
            dl.add_circle_filled(
                center,
                (RADIUS_PT + BALL_PT + 2.0) * style_scale,
                u32(imgui.ImVec4(0.0, 0.0, 0.0, 0.28)),
                32,
            )

        for b in self._balls:
            if b.alpha <= 0.0:
                continue
            hovered = b is self._hover
            rgb = _axis_rgb(b.axis)
            pos = imgui.ImVec2(*b.screen)

            face = (
                (_lift(rgb) if hovered else rgb) if b.positive else (rgb if hovered else _dark(rgb))
            )
            color = u32(imgui.ImVec4(*face, b.alpha))
            if b.positive:
                outline = _lollipop_outline(self._center, b.screen, b.radius, LINE_PT * style_scale)
                _draw_lollipop(imgui, dl, outline, color)
            else:
                dl.add_circle_filled(pos, b.radius, color, 24)

                dl.add_circle(
                    pos,
                    b.radius,
                    u32(imgui.ImVec4(*rgb, b.alpha)),
                    24,
                    1.6 * style_scale,
                )

            label_alpha = _label_alpha(b, hovered)
            if label_alpha > 0.0:
                white = u32(imgui.ImVec4(*LABEL_FILL[:3], label_alpha))
                _centered_label(imgui, dl, b.label, pos, b.radius, white)


def _back_alpha(depth: float) -> float:
    return float(np.clip((BACK_FADE_END - depth) / (BACK_FADE_END - BACK_FADE_START), 0.0, 1.0))


def _lollipop_outline(
    center: tuple[float, float],
    ball: tuple[float, float],
    radius: float,
    line_width: float,
    segments: int = 24,
) -> list[tuple[float, float]]:
    center_v = np.asarray(center, np.float64)
    ball_v = np.asarray(ball, np.float64)
    direction = ball_v - center_v
    distance = float(np.linalg.norm(direction))
    if distance <= radius:
        angles = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
        return [tuple(ball_v + radius * np.array((np.cos(a), np.sin(a)))) for a in angles]

    direction /= distance
    side = np.array((-direction[1], direction[0]))
    half = min(line_width * 0.5, radius * 0.5)
    angle = float(np.arcsin(half / radius))
    arc = np.linspace(-np.pi + angle, np.pi - angle, segments)
    points = [center_v - side * half]
    points.extend(ball_v + radius * (np.cos(a) * direction + np.sin(a) * side) for a in arc)
    points.append(center_v + side * half)
    return [tuple(point) for point in points]


def _draw_lollipop(imgui, dl, outline: list[tuple[float, float]], color: int) -> None:
    points = [imgui.ImVec2(*point) for point in outline]
    aa_flag = imgui.ImDrawListFlags_.anti_aliased_fill.value
    flags = dl.flags
    dl.flags = flags & ~aa_flag
    dl.add_concave_poly_filled(points, color)
    dl.flags = flags
    if not (flags & aa_flag):
        return

    vertices = np.asarray(outline, np.float64)
    edges = np.roll(vertices, -1, axis=0) - vertices
    lengths = np.linalg.norm(edges, axis=1)
    normals = np.column_stack((edges[:, 1], -edges[:, 0])) / lengths[:, None]
    miters = (np.roll(normals, 1, axis=0) + normals) * 0.5
    scale = np.minimum(1.0 / np.maximum(np.sum(miters * miters, axis=1), 1e-4), 100.0)
    outer = vertices + miters * scale[:, None]

    count = len(vertices)
    base = dl._vtx_current_idx
    transparent = color & 0x00FFFFFF
    uv = imgui.get_io().fonts.tex_uv_white_pixel
    dl.prim_reserve(count * 6, count * 2)
    for inner, fringe in zip(vertices, outer, strict=True):
        dl.prim_write_vtx(imgui.ImVec2(*inner), uv, color)
        dl.prim_write_vtx(imgui.ImVec2(*fringe), uv, transparent)
    for i in range(count):
        j = (i + 1) % count
        dl.prim_write_idx(base + i * 2)
        dl.prim_write_idx(base + j * 2)
        dl.prim_write_idx(base + j * 2 + 1)
        dl.prim_write_idx(base + i * 2)
        dl.prim_write_idx(base + j * 2 + 1)
        dl.prim_write_idx(base + i * 2 + 1)


def _label_alpha(ball: Ball, hovered: bool) -> float:
    if ball.positive or hovered:
        return ball.alpha
    return 1.0 - _back_alpha(-ball.depth)


def _dark(rgb: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(  # type: ignore[return-value]
        c * (1.0 - DARK_MIX) + base * DARK_MIX for c, base in zip(rgb, DARK_BASE, strict=True)
    )


def _lift(rgb: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(min(1.0, c * HOVER_GAIN + HOVER_LIFT) for c in rgb)  # type: ignore[return-value]


def _ink_box(font, size: float, text: str):
    baked = font.get_font_baked(size)
    pen = 0.0
    x0 = y0 = 1e9
    x1 = y1 = -1e9
    for ch in text:
        g = baked.find_glyph(ord(ch))
        if g is not None and g.x1 > g.x0 and g.y1 > g.y0:
            x0 = min(x0, pen + g.x0)
            x1 = max(x1, pen + g.x1)
            y0 = min(y0, g.y0)
            y1 = max(y1, g.y1)
            pen += g.advance_x
        elif g is not None:
            pen += g.advance_x
    return None if x1 < x0 else (x0, y0, x1, y1)


def _centered_label(imgui, dl, text: str, pos, radius: float, color: int) -> None:
    font = imgui.get_font()
    size = imgui.get_font_size()
    box = _ink_box(font, size, text)
    if box is None:
        return
    limit = radius * 1.5
    width = box[2] - box[0]
    if width > limit > 0.0:
        size *= limit / width
        box = _ink_box(font, size, text) or box

    pen_x = round(pos.x - (box[0] + box[2]) * 0.5)
    pen_y = round(pos.y - (box[1] + box[3]) * 0.5)
    baked = font.get_font_baked(size)
    tex = imgui.get_io().fonts.tex_data.get_tex_ref()
    for ch in text:
        g = baked.find_glyph(ord(ch))
        if g is None:
            continue
        if g.x1 > g.x0 and g.y1 > g.y0:
            dl.add_image(
                tex,
                imgui.ImVec2(pen_x + g.x0, pen_y + g.y0),
                imgui.ImVec2(pen_x + g.x1, pen_y + g.y1),
                imgui.ImVec2(g.u0, g.v0),
                imgui.ImVec2(g.u1, g.v1),
                color,
            )
        pen_x += g.advance_x


def _axis_rgb(axis: int) -> tuple[float, float, float]:
    return tuple(float(c) for c in THEME.axis_color(axis)[:3])  # type: ignore[return-value]
