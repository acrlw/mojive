"""Overlay features draw through the Draw2D protocol, so a recording fake can
verify what would be painted — and in which order — without a window."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from mojive import commands as cmd
from mojive.adapters.static import StaticSceneAdapter
from mojive.gizmo import AXIS_COLORS, paint_order, plane_direction
from mojive.render.backend import BackendCaps
from mojive.scene import Scene
from mojive.session import Session
from mojive.types import CameraView
from mojive.ui import viewcube as vc
from mojive.ui.draw2d import (
    Draw2D,
    ImguiDraw2D,
    _anti_alias_fringe_outer,
    _capped_polyline_outline,
    _round_cap_polyline_outline,
)
from mojive.ui.gizmo import ObjectGizmo

RECT = (0.0, 0.0, 800.0, 600.0)


class RecordingDraw2D:
    """Draw2D fake that records every call as a (name, args) tuple."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple]] = []

    def __getattr__(self, name: str):
        if name.startswith("__"):
            raise AttributeError(name)

        def record(*args, **kwargs):
            self.calls.append((name, args))

        return record

    def text_size(self, text: str) -> tuple[float, float]:
        return (6.0 * len(text), 12.0)


class CaptureBackend:
    caps = BackendCaps(name="capture", gizmo=True)

    def __init__(self) -> None:
        self.frame = None

    def set_gizmo(self, frame) -> bool:
        self.frame = frame
        return frame is not None


def camera() -> CameraView:
    return CameraView(
        eye=np.array((4.0, -6.0, 3.0), np.float32),
        target=np.zeros(3, np.float32),
        up=np.array((0.0, 0.0, 1.0), np.float32),
        aspect=RECT[2] / RECT[3],
    )


def axis_of(color) -> int:
    return int(np.argmin(np.linalg.norm(AXIS_COLORS[:, :3] - color[:3], axis=1)))


def test_imgui_adapter_covers_the_protocol_surface() -> None:
    missing = [
        name
        for name, _member in inspect.getmembers(Draw2D, inspect.isfunction)
        if not callable(getattr(ImguiDraw2D, name, None))
    ]
    assert not missing


@pytest.fixture
def native_draw():
    """Exercise native ImGui tessellation without opening a GPU window."""
    from imgui_bundle import imgui

    context = imgui.create_context()
    draw_list = imgui.ImDrawList(imgui.get_draw_list_shared_data())
    draw_list._reset_for_new_frame()
    draw_list.flags = (
        imgui.ImDrawListFlags_.anti_aliased_lines.value
        | imgui.ImDrawListFlags_.anti_aliased_fill.value
    )
    draw_list.push_clip_rect((-1000.0, -1000.0), (1000.0, 1000.0))
    draw = ImguiDraw2D(draw_list)
    yield draw
    draw._dl = None
    del draw_list
    imgui.destroy_context(context)


@pytest.mark.parametrize("cap", ("butt", "round", "round_start", "round_end"))
def test_native_line_and_polyline_share_exact_vertices(native_draw, cap):
    draw = native_draw
    start, end = (100.25, 120.75), (84.5, 63.125)
    color = (1.0, 1.0, 1.0, 1.0)
    draw.line(start, end, color, 3.5, cap=cap)
    count = len(draw._dl.vtx_buffer)
    line = np.array([(v.pos.x, v.pos.y) for v in draw._dl.vtx_buffer])
    draw.polyline((start, end), color, 3.5, cap=cap)
    polyline = np.array([(v.pos.x, v.pos.y) for v in list(draw._dl.vtx_buffer)[count:]])

    assert line == pytest.approx(polyline)
    assert {v.col >> 24 for v in draw._dl.vtx_buffer} == {0, 255}


@pytest.mark.parametrize("scale", (1.0, 2.25, 4.0))
@pytest.mark.parametrize("direction", ((0.0, -1.0), (-0.8660254, 0.5), (0.8660254, 0.5)))
def test_native_frame_arrow_head_and_shaft_remain_coaxial(
    native_draw, monkeypatch, scale, direction
):
    from mojive.ui.viewport_widgets import _draw_axis_arrow_glyph

    draw = native_draw
    parts = {}

    def record(name, method):
        def submit(*args, **kwargs):
            start = len(draw._dl.vtx_buffer)
            method(*args, **kwargs)
            parts[name] = np.array([(v.pos.x, v.pos.y) for v in list(draw._dl.vtx_buffer)[start:]])

        return submit

    monkeypatch.setattr(draw, "line", record("shaft", draw.line))
    monkeypatch.setattr(draw, "fringed_concave_fill", record("head", draw.fringed_concave_fill))
    center = np.array((100.25, 100.75))
    _draw_axis_arrow_glyph(
        draw,
        center,
        direction,
        (1.0,) * 4,
        1.18 * scale,
        1.46 * scale,
        clear_radius=3.0 * scale,
        base=7.6,
        tip=10.0,
        wing=1.8,
        corner_radius=0.25 * scale,
    )
    normal = np.array((-direction[1], direction[0]))
    normal /= np.linalg.norm(normal)
    for points in parts.values():
        across = (points - center) @ normal
        assert (across.min() + across.max()) * 0.5 == pytest.approx(0.0, abs=1e-5)
    assert set(parts) == {"shaft", "head"}


def test_fill_fringe_expands_outward_for_both_polygon_windings() -> None:
    square = np.array(((0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)))
    expected = np.array(((-1.0, -1.0), (3.0, -1.0), (3.0, 3.0), (-1.0, 3.0)))

    assert _anti_alias_fringe_outer(square) == pytest.approx(expected)
    assert _anti_alias_fringe_outer(square[::-1]) == pytest.approx(expected[::-1])


def test_round_cap_polyline_is_one_capsule_silhouette() -> None:
    outline = np.asarray(_round_cap_polyline_outline(((0.0, 0.0), (10.0, 0.0)), 4.0))

    assert outline[:, 0].min() == pytest.approx(-2.0)
    assert outline[:, 0].max() == pytest.approx(12.0)
    assert outline[:, 1].min() == pytest.approx(-2.0)
    assert outline[:, 1].max() == pytest.approx(2.0)
    assert len(outline) > 4


def test_asymmetric_cap_polyline_is_flat_at_start_and_round_at_end() -> None:
    outline = np.asarray(
        _capped_polyline_outline(
            ((0.0, 0.0), (10.0, 0.0)),
            4.0,
            round_start=False,
            round_end=True,
        )
    )

    assert outline[:, 0].min() == pytest.approx(0.0)
    assert outline[:, 0].max() == pytest.approx(12.0)
    assert outline[:, 1].min() == pytest.approx(-2.0)
    assert outline[:, 1].max() == pytest.approx(2.0)


def test_round_cap_polyline_keeps_clockwise_screen_winding_for_a_reflex_arc() -> None:
    angles = np.linspace(0.0, np.radians(240.0), 80)
    path = np.column_stack((100.0 * np.cos(angles), 100.0 * np.sin(angles)))

    outline = np.asarray(_round_cap_polyline_outline(path, 4.0))
    signed_area = 0.5 * np.sum(
        outline[:, 0] * np.roll(outline[:, 1], -1) - outline[:, 1] * np.roll(outline[:, 0], -1)
    )

    assert signed_area > 0.0
    assert signed_area == pytest.approx(np.radians(240.0) * 100.0 * 4.0, rel=0.02)


def test_flat_gizmo_submits_handles_in_painter_order() -> None:
    gizmo = ObjectGizmo()
    cam = camera()
    scene = Scene()
    obj = scene.box(name="editable")
    session = Session(StaticSceneAdapter(scene))
    session.submit(cmd.Select(obj.object_id))
    assert gizmo.publish(
        CaptureBackend(),
        session,
        cam,
        RECT,
        ui_scale=1.0,
        style_scale=1.0,
        yielding=False,
        interactive=False,
    )
    overlay = RecordingDraw2D()
    gizmo.draw_overlay(cam, RECT, overlay, style_scale=1.0)

    names = [name for name, _args in overlay.calls]
    assert names == ["convex_fill"] * 3 + ["concave_fill"] * 3 + ["circle_filled"] * 2

    origin = np.zeros(3)
    rotation = np.eye(3)
    # Camera eye (4, -6, 3): the Y handle is farthest, the X handle nearest.
    planes = [args[1] for name, args in overlay.calls if name == "convex_fill"]
    expected_planes = paint_order(
        cam, origin, [plane_direction(rotation, axis) for axis in range(3)]
    )
    assert [axis_of(color) for color in planes] == list(expected_planes) == [0, 2, 1]

    arrows = [args[1] for name, args in overlay.calls if name == "concave_fill"]
    expected_arrows = paint_order(cam, origin, [rotation[:, axis] for axis in range(3)])
    assert [axis_of(color) for color in arrows] == list(expected_arrows) == [1, 2, 0]


def test_viewcube_submits_balls_back_to_front() -> None:
    cube = vc.ViewCube()
    cam = CameraView(
        eye=np.array((4.0, -4.0, 3.0), np.float32),
        target=np.zeros(3, np.float32),
        up=np.array((0.0, 0.0, 1.0), np.float32),
    )
    cube.update(cam, RECT, cursor=(-1000.0, -1000.0), style_scale=1.0)
    overlay = RecordingDraw2D()
    cube.draw(overlay, style_scale=1.0)

    expected: list[str] = []
    for ball in cube.balls:  # layout() is already sorted far-to-near
        if ball.alpha <= 0.0:
            continue
        expected.append("fringed_concave_fill" if ball.positive else "circle_filled")
        if not ball.positive:
            expected.append("circle")
        if vc._label_alpha(ball, False) > 0.0:
            expected.append("centered_label")
    assert [name for name, _args in overlay.calls] == expected

    labels = [args[0] for name, args in overlay.calls if name == "centered_label"]
    assert labels == [
        ball.label for ball in cube.balls if ball.alpha > 0.0 and vc._label_alpha(ball, False) > 0.0
    ]
