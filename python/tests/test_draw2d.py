"""Overlay features draw through the Draw2D protocol, so a recording fake can
verify what would be painted — and in which order — without a window."""

from __future__ import annotations

import inspect
import subprocess
import sys

import numpy as np
import pytest

from mojive import commands as cmd
from mojive.adapters.static import StaticSceneAdapter
from mojive.curves2d import capped_polyline_points
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


def test_geometry_and_draw_protocol_import_without_loading_graphics_libraries():
    script = """
import importlib.abc
import sys

class NoGraphics(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'imgui_bundle', 'moderngl', 'glfw', 'wgpu', 'mujoco'}:
            raise AssertionError('Unexpected graphics import: ' + fullname)

sys.meta_path.insert(0, NoGraphics())
from mojive.curves2d import arrow_points
from mojive.draglink2d import smooth_drag_link_mesh
from mojive.ui.draw2d import Draw2D, ImguiDraw2D
assert len(arrow_points((0, 0), (20, 0))) > 3
assert len(smooth_drag_link_mesh(20, 5, 2)[1]) > 0
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


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


@pytest.mark.parametrize("smoothing", (0.0, 0.6, 1.0))
@pytest.mark.parametrize("round_tail", (False, True))
@pytest.mark.parametrize("direction", ((0.6, 0.8), (-0.8, 0.6)))
def test_cached_arrow_preserves_reference_boundary_fringe_and_fill(
    native_draw, smoothing, round_tail, direction
):
    from mojive.curves2d import arrow_points

    draw = native_draw
    start = np.array((100.25, 120.75))
    end = start + np.asarray(direction) * 100.0
    options = {"smoothing": smoothing, "round_tail": round_tail}
    color = (0.2, 0.5, 0.8, 0.7)
    outline = arrow_points(start, end, 2.0, **options)
    draw.fringed_concave_fill(outline, color)
    reference = np.array([(v.pos.x, v.pos.y) for v in draw._dl.vtx_buffer])
    colors = [v.col for v in draw._dl.vtx_buffer]
    index_start = len(draw._dl.idx_buffer)
    draw.arrow(start, end, color, 2.0, **options)
    actual = np.array([(v.pos.x, v.pos.y) for v in list(draw._dl.vtx_buffer)[len(reference) :]])
    # The shared mesh adds one interior fan vertex; its boundary and external
    # fringe retain the existing coverage and color, including translucent fills.
    np.testing.assert_allclose(actual[1:], reference, atol=2e-5, rtol=0)
    assert [v.col for v in list(draw._dl.vtx_buffer)[len(reference) + 1 :]] == colors
    indices = np.asarray(list(draw._dl.idx_buffer)[index_start:]) - len(reference)
    triangles = actual[indices[: len(outline) * 3].reshape(-1, 3)]
    a, b = triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    areas = (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]) * 0.5
    expected = (
        np.sum(
            outline[:, 0] * np.roll(outline[:, 1], -1) - outline[:, 1] * np.roll(outline[:, 0], -1)
        )
        * 0.5
    )
    assert areas.min() >= -1e-4
    assert areas.sum() == pytest.approx(expected, abs=0.002)


def test_arrow_motion_reuses_local_mesh_and_antialias_preparation(native_draw):
    from mojive.curves2d import arrow_mesh
    from mojive.ui.draw2d import _cached_fringe_points

    arrow_mesh.cache_clear()
    _cached_fringe_points.cache_clear()
    for i in range(24):
        start = (100 + i * 0.125, 120 + i * 0.0625)
        dx, dy = ((60, 80), (-60, 80), (-80, -60))[i % 3]
        native_draw.arrow(start, (start[0] + dx, start[1] + dy), (1.0,) * 4)
    assert arrow_mesh.cache_info().misses == 1
    assert _cached_fringe_points.cache_info().misses == 1


@pytest.mark.parametrize("method", ("convex_fill", "concave_fill", "fringed_concave_fill"))
def test_native_fill_antialiasing_is_identical_for_reversed_winding(native_draw, method):
    path = ((10.25, 20.5), (28.75, 23.25), (15.25, 40.5))
    draw = native_draw
    fill = getattr(draw, method)
    fill(path, (1.0,) * 4)
    count = len(draw._dl.vtx_buffer)
    first = [(v.pos.x, v.pos.y, v.col) for v in draw._dl.vtx_buffer]
    fill(path[::-1], (1.0,) * 4)
    second = [(v.pos.x, v.pos.y, v.col) for v in list(draw._dl.vtx_buffer)[count:]]
    assert first == second
    assert {color >> 24 for _, _, color in first} == {0, 255}


def test_triangle_fan_fallback_reads_the_base_after_a_reservation_rollover(native_draw):
    class RollingDrawList:
        # ImGui builds using 16-bit indices may roll over inside PrimReserve.
        flags = 0
        _vtx_current_idx = 65534

        def __init__(self):
            self.indices = []

        def prim_reserve(self, index_count, vertex_count):
            self._vtx_current_idx = 0

        def prim_write_vtx(self, *args):
            self._vtx_current_idx += 1

        def prim_write_idx(self, index):
            self.indices.append(index)

    original = native_draw._dl
    rolling = RollingDrawList()
    try:
        native_draw._dl = rolling
        native_draw.triangle_fan_fill(((10, 10), (30, 10), (30, 30), (10, 30)), (1.0,) * 4)
        assert rolling.indices == [0, 1, 2, 0, 2, 3]
    finally:
        native_draw._dl = original


def test_indexed_fill_accepts_numpy_contours(native_draw):
    points = np.array(((10.0, 10.0), (30.0, 10.0), (20.0, 30.0)))
    native_draw.indexed_fill(points, (0, 1, 2), (1.0,) * 4, outline=points)
    assert {v.col >> 24 for v in native_draw._dl.vtx_buffer} == {0, 255}


@pytest.mark.parametrize("direction", ((1.0, 0.0), (0.6, 0.8), (-0.8, -0.6)))
def test_local_mesh_placement_preserves_vertices_and_reuses_fringe(native_draw, direction):
    from mojive.draglink2d import smooth_drag_link_mesh
    from mojive.ui.draw2d import _cached_fringe_points

    points, indices, outline, hole = smooth_drag_link_mesh(8.0, 5.0, 2.0)
    ux, uy = direction
    origin = (125.25, -300.5)

    def placed(path):
        return tuple((origin[0] + x * ux - y * uy, origin[1] + x * uy + y * ux) for x, y in path)

    draw = native_draw
    draw.indexed_fill(
        placed(points), indices, (0.4, 0.7, 1.0, 0.5), outline=placed(outline), hole=placed(hole)
    )
    reference = [(v.pos.x, v.pos.y, v.col) for v in draw._dl.vtx_buffer]
    reference_indices = list(draw._dl.idx_buffer)
    draw.indexed_fill(
        points,
        indices,
        (0.4, 0.7, 1.0, 0.5),
        outline=outline,
        hole=hole,
        origin=origin,
        direction=direction,
    )
    placed_vertices = [
        (v.pos.x, v.pos.y, v.col) for v in list(draw._dl.vtx_buffer)[len(reference) :]
    ]
    assert np.array(placed_vertices)[:, :2] == pytest.approx(np.array(reference)[:, :2], abs=4e-5)
    assert [v[2] for v in placed_vertices] == [v[2] for v in reference]
    assert [
        i - len(reference) for i in list(draw._dl.idx_buffer)[len(reference_indices) :]
    ] == reference_indices
    # Transform only the newly appended range and reuse local AA for a new placement.
    assert [
        (v.pos.x, v.pos.y, v.col) for v in list(draw._dl.vtx_buffer)[: len(reference)]
    ] == reference
    misses = _cached_fringe_points.cache_info().misses
    draw.indexed_fill(
        points,
        indices,
        (0.4, 0.7, 1.0, 0.5),
        outline=outline,
        hole=hole,
        origin=(200, 300),
        direction=(-0.6, 0.8),
    )
    assert _cached_fringe_points.cache_info().misses == misses


def test_cached_fill_and_color_follow_mutable_input_changes(native_draw):
    draw = native_draw
    points = [[10.0, 10.0], [30.0, 10.0], [20.0, 30.0]]
    color = [1.0, 0.0, 0.0, 1.0]
    draw.convex_fill(tuple(points), color)
    count = len(draw._dl.vtx_buffer)
    points[0][0] = 15.0
    color[:] = [0.0, 1.0, 0.0, 0.5]
    draw.convex_fill(tuple(points), color)
    second = list(draw._dl.vtx_buffer)[count:]
    assert any(v.col == draw._u32(color) for v in second)
    assert min(v.pos.x for v in second) > 13.0


def test_failed_concave_submission_restores_draw_flags(native_draw):
    class FailingDrawList:
        flags = native_draw._dl.flags

        def add_concave_poly_filled(self, *args):
            raise ValueError("rejected geometry")

    original = native_draw._dl
    failing = FailingDrawList()
    try:
        native_draw._dl = failing
        with pytest.raises(ValueError, match="rejected geometry"):
            native_draw.fringed_concave_fill(((0, 0), (1, 0), (1, 1)), (1.0,) * 4)
        assert failing.flags == original.flags
    finally:
        native_draw._dl = original


def test_indexed_hollow_fill_matches_native_and_fallback(native_draw):
    from mojive.draglink2d import smooth_drag_link_mesh

    draw = native_draw
    if not hasattr(draw._dl, "add_indexed_fill"):
        pytest.skip("requires the Mojive native ImGui patch")
    points, indices, outline, hole = smooth_drag_link_mesh(8.0, 5.0, 2.0)
    draw.indexed_fill(points, indices, (1.0,) * 4, outline=outline, hole=hole)
    vertices = [(v.pos.x, v.pos.y, v.uv.x, v.uv.y, v.col) for v in draw._dl.vtx_buffer]
    native_indices = list(draw._dl.idx_buffer)

    class Fallback:
        def __getattr__(self, name):
            if name in {"add_indexed_fill", "add_poly_fringe"}:
                raise AttributeError(name)
            return getattr(native, name)

    native = draw._dl
    try:
        draw._dl = Fallback()
        draw.indexed_fill(points, indices, (1.0,) * 4, outline=outline, hole=hole)
        fallback_vertices = [
            (v.pos.x, v.pos.y, v.uv.x, v.uv.y, v.col)
            for v in list(native.vtx_buffer)[len(vertices) :]
        ]
        fallback_indices = [
            i - len(vertices) for i in list(native.idx_buffer)[len(native_indices) :]
        ]
    finally:
        draw._dl = native
        native = None
    assert fallback_vertices == vertices
    assert fallback_indices == native_indices
    assert len(vertices) == len(points) + 2 * (len(outline) + len(hole))
    assert sum(v[-1] >> 24 == 0 for v in vertices) == len(outline) + len(hole)


@pytest.mark.parametrize("aa", (False, True))
@pytest.mark.parametrize("color", (0xFFFFFFFF, 0x7F80AA33))
def test_native_fringe_matches_fallback_mesh(native_draw, aa, color):
    from mojive.curves2d import smooth_rect_points

    draw = native_draw
    if not hasattr(draw._dl, "add_poly_fringe"):
        pytest.skip("requires the Mojive native ImGui patch")
    path = smooth_rect_points(3.25, 4.75, 27.5, 36.0, 6.0)
    if not aa:
        draw._dl.flags &= ~draw._imgui.ImDrawListFlags_.anti_aliased_fill.value
    draw._write_anti_alias_fringe(path, color)
    vertices = [(v.pos.x, v.pos.y, v.uv.x, v.uv.y, v.col) for v in draw._dl.vtx_buffer]
    indices = list(draw._dl.idx_buffer)

    class Fallback:
        def __getattr__(self, name):
            if name == "add_poly_fringe":
                raise AttributeError(name)
            return getattr(native, name)

    native = draw._dl
    try:
        draw._dl = Fallback()
        draw._write_anti_alias_fringe(path, color)
        fallback_vertices = [
            (v.pos.x, v.pos.y, v.uv.x, v.uv.y, v.col)
            for v in list(native.vtx_buffer)[len(vertices) :]
        ]
        fallback_indices = [i - len(vertices) for i in list(native.idx_buffer)[len(indices) :]]
    finally:
        draw._dl = native
        # Do not retain a native draw list in a failed assertion's traceback.
        native = None
    assert fallback_vertices == vertices
    assert fallback_indices == indices


@pytest.mark.parametrize("scale", (1.0, 2.25, 4.0))
@pytest.mark.parametrize("direction", ((0.0, -1.0), (-0.8660254, 0.5), (0.8660254, 0.5)))
def test_native_frame_arrow_head_and_shaft_remain_coaxial(native_draw, scale, direction):
    from mojive.ui.viewport_widgets import _draw_axis_arrow_glyph

    draw = native_draw
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
    points = np.array([(v.pos.x, v.pos.y) for v in draw._dl.vtx_buffer])
    assert len(points) > 3
    across = (points - center) @ normal
    assert (across.min() + across.max()) * 0.5 == pytest.approx(0.0, abs=1e-5)


def test_fill_fringe_expands_outward_for_both_polygon_windings() -> None:
    square = np.array(((0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)))
    expected = np.array(((-1.0, -1.0), (3.0, -1.0), (3.0, 3.0), (-1.0, 3.0)))

    assert _anti_alias_fringe_outer(square) == pytest.approx(expected)
    assert _anti_alias_fringe_outer(square[::-1]) == pytest.approx(expected[::-1])


def test_round_cap_polyline_is_one_capsule_silhouette() -> None:
    outline = np.asarray(
        capped_polyline_points(((0.0, 0.0), (10.0, 0.0)), 4.0, round_start=True, round_end=True)
    )

    assert outline[:, 0].min() == pytest.approx(-2.0)
    assert outline[:, 0].max() == pytest.approx(12.0)
    assert outline[:, 1].min() == pytest.approx(-2.0)
    assert outline[:, 1].max() == pytest.approx(2.0)
    assert len(outline) > 4


def test_asymmetric_cap_polyline_is_flat_at_start_and_round_at_end() -> None:
    outline = np.asarray(
        capped_polyline_points(
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

    outline = np.asarray(capped_polyline_points(path, 4.0, round_start=True, round_end=True))
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


def test_capped_stroke_cache_tracks_mutable_points_and_cap_style():
    points = np.array(((0.0, 0.0), (20.0, 0.0)))
    options = {"round_start": True, "round_end": True}
    first = capped_polyline_points(points, 4.0, **options)
    assert capped_polyline_points(points.copy(), 4.0, **options) is first
    points[1, 0] = 40.0
    moved = capped_polyline_points(points, 4.0, **options)
    assert max(x for x, y in moved) == pytest.approx(max(x for x, y in first) + 20.0)
    flat = capped_polyline_points(points, 4.0, round_start=True, round_end=False)
    assert max(x for x, y in flat) == pytest.approx(40.0)
    wide = capped_polyline_points(points, 8.0, **options)
    assert max(y for x, y in wide) == pytest.approx(4.0)
    circular = capped_polyline_points(points, 4.0, **options, smoothing=0.0)
    assert circular != moved
    points[1, 0] = 20.0
    assert capped_polyline_points(points, 4.0, **options) is first
