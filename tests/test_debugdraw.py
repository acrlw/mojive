from __future__ import annotations

import gc
import inspect
import json
import os
import shutil
import socket
import tempfile
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pytest

from mojive.bridge import APP, DebugBridge, socket_path
from mojive.render.backend import BackendCaps, NullBackend
from mojive.render.debugdraw import (
    ARROW_CORNER_RADIUS_RATIO,
    ARROW_HEAD_RATIO,
    AXIS_COLORS,
    PRIMITIVE_PATH,
    VERTEX_COUNT,
    DebugDraw,
    DrawPath,
    Occlusion,
    PrimitiveType,
    sector_angle,
    sector_points,
    world_size,
)

RED = (0.9, 0.2, 0.2, 1.0)
A = np.array([0.0, 0.0, 0.0], np.float32)
B = np.array([1.0, 0.0, 0.0], np.float32)
C = np.array([1.0, 1.0, 0.0], np.float32)


def _draw_frames(dd: DebugDraw, n: int, paint) -> list:

    seen = []
    for i in range(n):
        paint(i)
        dd.render_frame(lambda _f: seen.append(dd.stats()), now=float(i))
    return seen


def test_same_id_redrawn_every_frame_does_not_grow():

    dd = DebugDraw()
    layer = dd.layer("drag", Occlusion.GHOST)

    seen = _draw_frames(dd, 100, lambda i: layer.line("drag.line", A, B + i, RED, 2.0))

    assert [s.primitives for s in seen] == [1] * 100
    assert dd.stats().primitives == 1
    assert dd.primitives == dd.stats().primitives

    assert dd.stats().moves == 0

    assert layer.positions_of(PrimitiveType.LINE)[0, 1] == pytest.approx(
        [100.0, 99.0, 99.0], abs=1e-3
    )


def test_batch_entry_stays_one_id_and_still_emits_plain_lines():

    dd = DebugDraw()
    layer = dd.layer("dash", Occlusion.ALWAYS)
    pts_a = np.zeros((6, 3), np.float32)
    pts_b = np.ones((6, 3), np.float32)

    for _ in range(10):
        layer.lines("silhouette", pts_a, pts_b, RED, 1.5)
        dd.render_frame(lambda _f: None)

    assert dd.stats().primitives == 6
    assert layer.count_of(PrimitiveType.LINE) == 6

    layer.lines("silhouette", pts_a[:4], pts_b[:4], RED, 1.5)
    assert dd.stats().primitives == 4


def test_batch_points_and_arrows_keep_one_id_and_per_item_colors():
    dd = DebugDraw()
    layer = dd.layer("physics", Occlusion.ALWAYS)
    colors = np.array([RED, (0.2, 0.8, 0.3, 1.0)], np.float32)
    a = np.zeros((2, 3), np.float32)
    b = np.eye(3, dtype=np.float32)[:2]

    layer.points("contacts", b, colors, 4.0)
    layer.arrows("forces", a, b, colors, 2.0)

    assert layer.count_of(PrimitiveType.POINT) == 2
    assert layer.count_of(PrimitiveType.ARROW) == 2
    frame = dd.build()
    point_stream = frame.stream(DrawPath.POINT)
    arrow_stream = frame.stream(DrawPath.ARROW)
    assert point_stream[:, 3:7] == pytest.approx(colors)
    assert arrow_stream[:, 6:10] == pytest.approx(colors)
    assert arrow_stream[:, 11] == pytest.approx(2.0 * ARROW_HEAD_RATIO)


def test_debug_arrow_uses_the_position_gizmo_head_proportion():
    from mojive.gizmo import (
        ARROW_CORNER_RADIUS_PT,
        AXIS_HEAD_LENGTH_PT,
        AXIS_SHAFT_HALF_PT,
    )

    shaft_width = 2.0 * AXIS_SHAFT_HALF_PT
    assert shaft_width * ARROW_HEAD_RATIO == pytest.approx(AXIS_HEAD_LENGTH_PT)
    assert shaft_width * ARROW_CORNER_RADIUS_RATIO == pytest.approx(ARROW_CORNER_RADIUS_PT)


def test_arrow_start_mask_is_packed_in_screen_pixels():
    dd = DebugDraw()
    dd.layer("gizmo", Occlusion.ALWAYS).arrow("axis", A, B, RED, 4.4, start_mask_px=9.68)
    stream = dd.build().stream(DrawPath.ARROW)
    assert stream.shape == (1, 13)
    assert stream[0, 12] == pytest.approx(9.68)


def test_batched_arrows_keep_per_item_order_and_start_mask():
    dd = DebugDraw()
    starts = np.zeros((3, 3), np.float32)
    ends = np.eye(3, dtype=np.float32)
    colors = AXIS_COLORS.copy()
    dd.layer("gizmo", Occlusion.ALWAYS).arrows(
        "axes", starts, ends, colors, 4.4, start_mask_px=9.68
    )
    stream = dd.build().stream(DrawPath.ARROW)
    assert stream[:, 3:6] == pytest.approx(ends)
    assert stream[:, 6:10] == pytest.approx(colors)
    assert stream[:, 12] == pytest.approx(9.68)


def test_world_text_overwrites_by_id_and_keeps_screen_semantics():
    dd = DebugDraw()
    layer = dd.layer("labels", Occlusion.ALWAYS)
    for value in range(20):
        layer.text(
            "speed",
            (1.0, 2.0, 3.0),
            f"{value} m/s",
            RED,
            offset_px=(8.0, -4.0),
            align=(0.5, 1.0),
        )
    frame = dd.build()
    assert dd.primitives == dd.stats().primitives == 1
    assert frame.text_count == 1
    label = frame.texts[0]
    assert label.text == "19 m/s"
    assert label.anchor == pytest.approx((1.0, 2.0, 3.0))
    assert label.offset_px == pytest.approx((8.0, -4.0))
    assert label.align == pytest.approx((0.5, 1.0))
    assert label.occlusion is Occlusion.ALWAYS


def test_world_text_expires_and_can_replace_a_geometric_id():
    dd = DebugDraw()
    layer = dd.layer("labels")
    layer.line("same", A, B, RED)
    layer.text("same", A, "replacement", duration=0.0)
    assert layer.count_of(PrimitiveType.LINE) == 0
    assert dd.primitives == 1
    dd.render_frame(lambda frame: frame.text_count == 1, now=1.0)
    assert dd.primitives == 0


@pytest.mark.parametrize(
    ("duration", "expect_moves", "expect_expiring"),
    [(-1.0, 0, 0), (0.0, 100, 1)],
    ids=["persistent", "one-frame"],
)
def test_never_expiring_primitives_never_move(duration, expect_moves, expect_expiring):

    dd = DebugDraw()
    layer = dd.layer("mark", Occlusion.ALWAYS)

    seen = _draw_frames(dd, 100, lambda i: layer.point("grab", A, RED, 4.0, duration))

    assert [s.primitives for s in seen] == [1] * 100
    assert dd.stats().moves == expect_moves

    assert [s.expiring for s in seen] == [expect_expiring] * 100


def test_expire_only_walks_the_finite_lifetime_ids():

    dd = DebugDraw()
    layer = dd.layer("bulk", Occlusion.DEPTH)
    layer.lines(
        "bulk", np.zeros((10_000, 3), np.float32), np.ones((10_000, 3), np.float32), RED, 1.0
    )

    t0 = time.perf_counter()
    for i in range(100):
        dd.expire(float(i))
    ms = (time.perf_counter() - t0) * 1000.0

    assert dd.stats().moves == 0
    assert dd.stats().primitives == 10_000
    print(f"\n[metric] 10k persistent lines: 100 expire() calls in {ms:.3f} ms")


def test_expire_runs_after_drawing_not_before():

    dd = DebugDraw()
    layer = dd.layer("once", Occlusion.ALWAYS)
    dd.render_frame(lambda _f: None, now=0.0)
    layer.point("blip", A, RED, 3.0, duration=0.0)

    drawn: list[int] = []
    dd.render_frame(lambda _f: drawn.append(dd.stats().primitives), now=1.0)
    assert drawn == [1]
    assert dd.stats().primitives == 0

    dd2 = DebugDraw()
    layer2 = dd2.layer("once", Occlusion.ALWAYS)
    dd2.now = 0.0
    layer2.point("blip", A, RED, 3.0, duration=0.0)
    dd2.expire(1.0)
    drawn2: list[int] = []
    dd2.render_frame(lambda _f: drawn2.append(dd2.stats().primitives), now=1.0)
    assert drawn2 == [0]


def test_vertex_counts_match_the_spec_table():

    assert VERTEX_COUNT == {
        PrimitiveType.LINE: 2,
        PrimitiveType.ARROW: 2,
        PrimitiveType.POINT: 1,
        PrimitiveType.FRAME: 6,
        PrimitiveType.BOX: 1,
        PrimitiveType.SPHERE: 1,
        PrimitiveType.SECTOR: 3,
        PrimitiveType.STROKE: 3,
        PrimitiveType.DRAG_LINK: 2,
        PrimitiveType.SOLID_ARROW: 1,
        PrimitiveType.SOLID_DOUBLE_ARROW: 1,
        PrimitiveType.CYLINDER: 1,
    }
    assert len(VERTEX_COUNT) == 12


def test_closed_polyline_packs_shared_neighbors_for_continuous_joins():
    dd = DebugDraw()
    points = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]], np.float32)
    dd.layer("outline").polyline("loop", points, RED, 4.0, closed=True)

    frame = dd.build()
    stroke = frame.stream(DrawPath.STROKE)
    assert stroke.shape == (4, 14)
    assert stroke[:, 3:6] == pytest.approx(points)
    assert stroke[:, 6:9] == pytest.approx(np.roll(points, -1, axis=0))
    assert stroke[:, 0:3] == pytest.approx(np.roll(points, 1, axis=0))
    assert np.all(stroke[:, 13] == 4.0)


def test_frame_is_three_independent_segments_and_arrows_have_their_own_path():

    dd = DebugDraw()
    layer = dd.layer("axes", Occlusion.ALWAYS)
    m = np.eye(4, dtype=np.float32)
    m[:3, 3] = (1.0, 2.0, 3.0)
    layer.frame("f", m, axis_len=0.5)

    pos = layer.positions_of(PrimitiveType.FRAME)
    assert pos.shape == (1, 6, 3)
    for k in range(3):
        start, end = pos[0, 2 * k], pos[0, 2 * k + 1]
        assert start == pytest.approx([1.0, 2.0, 3.0])
        assert np.linalg.norm(end - start) == pytest.approx(0.5)

    layer.arrow("a", A, B, RED, 2.0)
    frame = dd.build()
    segs = [b for b in frame.active() if b.path is DrawPath.SEGMENT]
    arrows = [b for b in frame.active() if b.path is DrawPath.ARROW]
    assert PRIMITIVE_PATH[PrimitiveType.FRAME] is DrawPath.SEGMENT
    assert PRIMITIVE_PATH[PrimitiveType.ARROW] is DrawPath.ARROW
    assert len(segs) == 1
    assert segs[0].count == 3
    assert len(arrows) == 1
    assert arrows[0].count == 1

    stream = frame.stream(DrawPath.SEGMENT)
    assert stream.shape[1] == 13
    for k in range(3):
        assert stream[k, 6:10] == pytest.approx(AXIS_COLORS[k])
    assert frame.stream(DrawPath.ARROW)[0, 11] > 0.0


def test_axis_length_ignores_scale_baked_into_the_transform():

    dd = DebugDraw()
    layer = dd.layer("axes", Occlusion.ALWAYS)
    m = np.eye(4, dtype=np.float32) * 5.0
    m[3, 3] = 1.0
    m[:3, 3] = 0.0
    layer.frame("f", m, axis_len=0.25)
    pos = layer.positions_of(PrimitiveType.FRAME)
    for k in range(3):
        assert np.linalg.norm(pos[0, 2 * k + 1] - pos[0, 2 * k]) == pytest.approx(0.25)


def test_batched_frames_preserve_origins_axes_and_one_retained_id():
    dd = DebugDraw()
    layer = dd.layer("axes", Occlusion.ALWAYS)
    positions = np.array([[1, 2, 3], [4, 5, 6]], np.float32)
    rotations = np.array([np.eye(3, dtype=np.float32), np.diag([2.0, 3.0, 4.0])], np.float32)

    layer.frames("frames", positions, rotations, axis_len=0.5)

    packed = layer.positions_of(PrimitiveType.FRAME)
    assert packed.shape == (2, 6, 3)
    assert len(layer._index) == 1
    for frame_index in range(2):
        assert packed[frame_index, 0::2] == pytest.approx(
            np.broadcast_to(positions[frame_index], (3, 3))
        )
        assert np.linalg.norm(
            packed[frame_index, 1::2] - packed[frame_index, 0::2], axis=1
        ) == pytest.approx([0.5, 0.5, 0.5])


def test_solid_primitives_reuse_the_builtin_meshes():

    dd = DebugDraw()
    layer = dd.layer("marks", Occlusion.ALWAYS)
    m = np.eye(4, dtype=np.float32)
    m[:3, 3] = (1.0, 0.0, 0.0)
    layer.box("b", m, RED)
    layer.sphere("s", m, RED)
    layer.solid_arrow("a", m, RED)
    layer.solid_double_arrow("d", m, RED)

    frame = dd.build()
    solids = [b for b in frame.active() if b.path is DrawPath.SOLID]
    assert {str(b.mesh) for b in solids} == {"box", "sphere", "arrow", "double_arrow"}
    assert all(b.count == 1 for b in solids)

    rec = frame.stream(DrawPath.SOLID)[0]
    assert rec[12:15] == pytest.approx([1.0, 0.0, 0.0])
    assert layer.positions_of(PrimitiveType.BOX)[0, 0] == pytest.approx([1.0, 0.0, 0.0])


def test_sector_swept_angle_is_the_rotation_vector_norm():

    center = np.array([1.0, 1.0, 0.0])
    axis = np.array([0.0, 0.0, 1.0])
    angle = 2.0
    dd = DebugDraw()
    layer = dd.layer("joint", Occlusion.DEPTH)
    layer.sector("j0", center, center + axis * angle, center + np.array([1.0, 0.0, 0.0]), RED)

    pos = layer.positions_of(PrimitiveType.SECTOR)
    assert pos.shape == (1, 3, 3)
    assert sector_angle(pos[0, 0], pos[0, 1]) == pytest.approx(angle, abs=1e-6)


def test_sector_is_unambiguous_beyond_180_degrees():

    center = np.zeros(3)
    axis = np.array([0.0, 0.0, 1.0])
    ref = center + np.array([1.0, 0.0, 0.0])
    turn_270 = center + axis * (1.5 * np.pi)
    turn_neg_90 = center - axis * (0.5 * np.pi)

    end_270 = sector_points(center, turn_270, ref, segments=4)[-1]
    end_neg = sector_points(center, turn_neg_90, ref, segments=4)[-1]
    assert end_270 == pytest.approx(end_neg, abs=1e-9)

    assert sector_angle(center, turn_270) == pytest.approx(1.5 * np.pi)
    assert sector_angle(center, turn_neg_90) == pytest.approx(0.5 * np.pi)

    sweep_270 = sector_points(center, turn_270, ref, segments=64)[1:]
    sweep_neg = sector_points(center, turn_neg_90, ref, segments=64)[1:]
    behind = np.array([-1.0, 0.0, 0.0])
    assert np.min(np.linalg.norm(sweep_270 - behind, axis=1)) < 0.1
    assert np.min(np.linalg.norm(sweep_neg - behind, axis=1)) > 0.5


def test_ten_thousand_lines_pack_fast_and_without_allocating():

    n = 10_000
    rng = np.random.default_rng(7)
    pts_a = rng.normal(size=(n, 3)).astype(np.float32)
    pts_b = pts_a + 0.1

    dd = DebugDraw()
    layer = dd.layer("bulk", Occlusion.DEPTH)

    t0 = time.perf_counter()
    layer.lines("bulk", pts_a, pts_b, RED, 1.5)
    submit_ms = (time.perf_counter() - t0) * 1000.0
    assert dd.stats().primitives == n

    dd.build()

    rounds = 20
    t0 = time.perf_counter()
    for _ in range(rounds):
        dd.build()
    pack_ms = (time.perf_counter() - t0) * 1000.0 / rounds

    gc.collect()
    tracemalloc.start()
    before = tracemalloc.get_traced_memory()[0]
    for _ in range(rounds):
        dd.build()
    after = tracemalloc.get_traced_memory()[0]
    tracemalloc.stop()
    grew = after - before

    print(
        f"\n[metric] 10k lines: submit {submit_ms:.3f} ms, "
        f"pack {pack_ms:.3f} ms/frame, allocation growth {grew} bytes over {rounds} packs"
    )
    assert pack_ms < 5.0
    assert grew < 64 * 1024

    stream = dd.build().stream(DrawPath.SEGMENT)
    assert stream.shape == (n, 13)
    assert stream[:, 0:3] == pytest.approx(pts_a)
    assert stream[:, 3:6] == pytest.approx(pts_b)


def test_batches_are_grouped_by_occlusion_then_path():

    dd = DebugDraw()
    for name in ("a", "b", "c"):
        dd.layer(name, Occlusion.GHOST).line(f"{name}.l", A, B, RED, 2.0)
    dd.layer("mark", Occlusion.ALWAYS).point("p", A, RED, 4.0)

    frame = dd.build()
    batch_descriptors = [(b.occlusion, b.path, b.count) for b in frame.active()]
    assert batch_descriptors == [
        (Occlusion.ALWAYS, DrawPath.POINT, 1),
        (Occlusion.GHOST, DrawPath.SEGMENT, 3),
    ]


def test_interaction_axes_and_points_are_drawn_after_the_closed_outline():

    dd = DebugDraw()
    layer = dd.layer("interaction", Occlusion.ALWAYS)
    layer.polyline("outline", (A, B, C), RED, 4.0, closed=True)
    layer.arrow("axis", A, B, RED, 2.0)
    layer.point("grab", A, RED, 4.0)

    frame = dd.build()
    assert [batch.path for batch in frame.active()] == [
        DrawPath.STROKE,
        DrawPath.ARROW,
        DrawPath.POINT,
    ]


def test_layers_share_one_px_scale():

    pytest.importorskip("moderngl")
    from types import SimpleNamespace

    from mojive.math3d import perspective
    from mojive.render.opengl.passes.base import PassContext

    dd = DebugDraw()
    dd.layer("near", Occlusion.DEPTH).line("l", A, B, RED, 4.0)
    dd.layer("far", Occlusion.ALWAYS).line("l", A, B, RED, 4.0)
    frame = dd.build()
    widths = [frame.stream(DrawPath.SEGMENT)[i, 10] for i in range(2)]

    assert set(inspect.signature(DebugDraw.build).parameters) == {"self"}
    assert widths == [4.0, 4.0]

    proj = perspective(np.deg2rad(45.0), 16 / 9, 0.1, 100.0)
    fake = SimpleNamespace(proj=proj, target=SimpleNamespace(height=720))
    px_scale = PassContext.px_scale.fget(fake)
    assert px_scale == pytest.approx(2.0 / (proj[1, 1] * 720))
    assert world_size(widths[0], px_scale, 3.0) == world_size(widths[1], px_scale, 3.0)

    assert world_size(widths[0], px_scale, 6.0) == pytest.approx(
        2.0 * world_size(widths[0], px_scale, 3.0)
    )


def test_entity_sized_things_do_not_go_through_the_pixel_domain():

    dd = DebugDraw()
    layer = dd.layer("m", Occlusion.ALWAYS)
    m = np.eye(4, dtype=np.float32)
    m[:3, :3] *= 0.3
    layer.box("b", m, RED)
    rec = dd.build().stream(DrawPath.SOLID)[0]
    assert rec[0] == pytest.approx(0.3)


def test_over_the_limit_is_counted_not_silently_dropped():

    dd = DebugDraw(limit=10)
    layer = dd.layer("flood", Occlusion.DEPTH)
    for i in range(20):
        layer.line(f"l{i}", A, B, RED, 1.0)
    assert dd.stats().primitives == 10
    assert dd.stats().dropped == 10


def test_layer_occlusion_is_fixed_at_the_layer():

    dd = DebugDraw()
    first = dd.layer("x", Occlusion.GHOST)
    again = dd.layer("x", Occlusion.DEPTH)
    assert again is first
    assert first.occlusion is Occlusion.GHOST


class _Backend:
    def __init__(self) -> None:
        self.caps = BackendCaps(name="fake", debug_draw=True)
        self.debug = DebugDraw()


def test_bridge_counts_dropped_on_a_backend_without_debug_draw():

    br = DebugBridge(NullBackend("test"))
    assert br.available is False

    painted = []
    ok = br.publish("script.ray", Occlusion.GHOST, lambda layer: painted.append(layer))

    assert ok is False and painted == []
    assert br.stats.dropped == 1
    assert br.stats.notes and "debug draw" in br.stats.notes[-1]
    assert br.layer("script.ray", Occlusion.GHOST) is None


def test_bridge_publishes_through_one_entry_on_a_capable_backend():

    backend = _Backend()
    br = DebugBridge(backend)
    assert br.available is True

    ok = br.publish("script.ray", Occlusion.GHOST, lambda layer: layer.arrow("r", A, B, RED, 3.0))

    assert ok is True
    assert br.stats.dropped == 0
    assert backend.debug.stats().primitives == 1
    assert backend.debug.layers()[0].occlusion is Occlusion.GHOST


def test_bridge_rejects_unknown_ops_instead_of_getattr():

    br = DebugBridge(_Backend())
    assert br._apply({"op": "set_overlays", "layer": "x", "id": "a"}) is False
    assert br.stats.invalid == 1
    assert "set_overlays" in br.stats.notes[-1]


def test_bridge_reports_bad_arguments():

    br = DebugBridge(_Backend())
    assert br._apply({"op": "line", "layer": "x", "id": "a", "a": [0, 0, 0]}) is False
    assert br.stats.invalid == 1


def test_bridge_exposes_world_text_without_a_ui_specific_path():
    backend = _Backend()
    br = DebugBridge(backend)
    assert br._apply(
        {
            "op": "text",
            "layer": "script.labels",
            "occlusion": "always",
            "id": "velocity",
            "anchor": [1, 2, 3],
            "text": "3.2 m/s",
            "offset_px": [6, -8],
            "align": [0.5, 1.0],
        }
    )
    frame = backend.debug.build()
    assert frame.text_count == 1
    assert frame.texts[0].text == "3.2 m/s"
    assert frame.texts[0].occlusion is Occlusion.ALWAYS


def test_bridge_exposes_batched_coordinate_frames() -> None:
    backend = _Backend()
    bridge = DebugBridge(backend)

    assert bridge._apply(
        {
            "op": "frames",
            "layer": "policy.frames",
            "id": "bodies",
            "positions": [[0, 0, 0], [1, 2, 3]],
            "rotations": [np.eye(3).tolist(), np.eye(3).tolist()],
            "axis_len": 0.25,
        }
    )

    layer = backend.debug.layer("policy.frames")
    assert layer.count_of(PrimitiveType.FRAME) == 2
    assert len(layer._index) == 1


def test_bridge_reports_commands_dropped_by_the_per_frame_budget() -> None:
    bridge = DebugBridge(_Backend())
    messages = [
        {"op": "point", "layer": "policy", "id": str(index), "p": [index, 0, 0]}
        for index in range(3)
    ]

    assert bridge.apply_batch(messages, budget=2) == 2
    assert bridge.stats.dropped == 1
    assert "Use lines, arrows, points, or frames batches" in bridge.stats.notes[-1]


def test_selection_label_uses_the_structure_object_index() -> None:
    from mojive.adapters.base import NodeType, SceneFrame, SceneNode, SceneSource
    from mojive.render.backend import FrameMode, LabelMode
    from mojive.render.overlay import OverlayPublisher, OverlayState
    from mojive.types import CameraView

    node = SceneNode(1, "selected", NodeType.LINK, object_id=17, body_index=0)
    source = SceneSource(nodes=[node])
    draw = DebugDraw()
    publisher = OverlayPublisher(draw, {})
    publisher.set_scene(source)

    class NoNodeScan(list):
        def __iter__(self):
            raise AssertionError("selection label scanned the full node list")

    source.nodes = NoNodeScan(source.nodes)
    publisher.publish(
        SceneFrame(body_xpos=np.zeros((1, 3), np.float32)),
        OverlayState(CameraView(), 720, 17, LabelMode.SELECTION, FrameMode.NONE, 0),
    )

    labels = draw.layer("scene.labels")._texts
    assert labels["selection"].text == "selected"


def test_tendon_labels_group_segments_in_one_indexed_pass() -> None:
    from mojive.adapters.base import SceneFrame, SceneSource
    from mojive.render.overlay import OverlayPublisher

    source = SceneSource(tendon_names=("first", "second"))
    frame = SceneFrame(
        tendon_segments=np.array(
            [
                [[0, 0, 0], [2, 0, 0]],
                [[0, 2, 0], [2, 2, 0]],
                [[0, 4, 0], [2, 4, 0]],
            ],
            np.float32,
        ),
        tendon_ids=np.array([1, 0, 1], np.int32),
    )
    draw = DebugDraw()
    publisher = OverlayPublisher(draw, {})
    publisher.set_scene(source)
    layer = draw.layer("scene.labels")

    publisher._draw_tendon_labels(layer, frame)

    assert layer._texts["tendon:0"].anchor == pytest.approx([1.0, 2.0, 0.0])
    assert layer._texts["tendon:1"].anchor == pytest.approx([1.0, 2.0, 0.0])


def test_pass_is_registered_and_hands_its_draw_to_the_backend():

    pytest.importorskip("moderngl")
    from mojive.render.opengl import passes
    from mojive.render.opengl.backend import registered

    passes.load_all()
    assert "debug" not in passes.failed(), passes.failed().get("debug")
    factory = registered().get("debug")
    assert factory is not None
    assert isinstance(factory().draw, DebugDraw)


def test_perturbation_feedback_lands_on_the_layers_it_asks_for():

    pytest.importorskip("moderngl")
    from mojive.session import PerturbState
    from mojive.types import CameraView
    from mojive.ui.perturb import MarkBudget, PerturbController

    ctrl = PerturbController()
    dd = DebugDraw()
    st = PerturbState(active=True, node_id=0, mode="translate")
    st.grab_point = np.array([0.1, 0.0, 0.2], np.float32)
    st.target_pos = np.array([0.4, 0.0, 0.2], np.float32)

    ctrl._publish_drag(dd, st, (None, None), MarkBudget(), 1.0)
    drag = dd.layer("ui.perturb.drag")
    assert drag.occlusion is Occlusion.ALWAYS
    assert drag.count_of(PrimitiveType.DRAG_LINK) == 1
    record = dd.build().stream(DrawPath.DRAG_LINK)[0]
    assert record[0:3] == pytest.approx([0.1, 0.0, 0.2])
    assert record[3:6] == pytest.approx([0.5, 0.0, 0.4])
    assert record[14:17] == pytest.approx([2.0, 6.0, 0.75])

    ctrl._publish_mark(
        dd, st, (None, None), CameraView(), (0.0, 0.0, 960.0, 720.0), MarkBudget(), 1.0
    )
    mark = dd.layer("ui.perturb.mark")
    assert mark.occlusion is Occlusion.ALWAYS
    assert mark.count_of(PrimitiveType.BOX) == 0
    assert mark.count_of(PrimitiveType.ARROW) == 3
    assert mark.count_of(PrimitiveType.STROKE) > 0

    before = dd.stats().primitives
    for _ in range(100):
        ctrl._publish_drag(dd, st, (None, None), MarkBudget(), 1.0)
        ctrl._publish_mark(
            dd, st, (None, None), CameraView(), (0.0, 0.0, 960.0, 720.0), MarkBudget(), 1.0
        )
        dd.render_frame(lambda _f: None)
    assert dd.stats().primitives == before


def test_socket_path_is_pid_named_and_falls_back_to_tempdir(monkeypatch):

    monkeypatch.setenv("XDG_RUNTIME_DIR", "/run/user/1000")
    assert socket_path(4321) == Path("/run/user/1000") / APP / "4321.sock"

    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    mine = socket_path()
    assert mine.parent == Path(tempfile.gettempdir()) / APP
    assert mine.name == f"{os.getpid()}.sock"


@pytest.fixture
def short_dir():

    d = Path(tempfile.mkdtemp(prefix="fv"))
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_external_json_lines_are_received_off_thread_and_applied_on_the_main_thread(short_dir):

    backend = _Backend()
    br = DebugBridge(backend)
    path = br.serve(short_dir / "v.sock")
    assert path is not None and path.exists()
    try:
        payload = [
            {
                "op": "arrow",
                "layer": "policy",
                "occlusion": "ghost",
                "id": "a0",
                "a": [0, 0, 0],
                "b": [0, 0, 1],
                "color": [1, 0.4, 0.1, 1],
                "width_px": 3,
            },
            {
                "op": "point",
                "layer": "policy",
                "occlusion": "ghost",
                "id": "p0",
                "p": [1, 2, 3],
                "color": [1, 1, 1, 1],
                "radius_px": 5,
            },
        ]
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(2.0)
            client.connect(str(path))
            client.sendall(b"".join(json.dumps(m).encode() + b"\n" for m in payload))

            deadline = time.time() + 3.0
            applied = 0
            while applied < 2 and time.time() < deadline:
                assert backend.debug.stats().primitives == applied
                applied += br.pump()
                time.sleep(0.01)

        assert applied == 2
        assert backend.debug.stats().primitives == 2
        layer = backend.debug.layers()[0]
        assert layer.name == "policy" and layer.occlusion is Occlusion.GHOST
        assert layer.count_of(PrimitiveType.ARROW) == 1
        assert layer.positions_of(PrimitiveType.POINT)[0, 0] == pytest.approx([1.0, 2.0, 3.0])
    finally:
        br.close()
    assert not path.exists()


def test_external_bad_line_does_not_take_the_connection_down(short_dir):

    backend = _Backend()
    br = DebugBridge(backend)
    path = br.serve(short_dir / "v.sock")
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(2.0)
            client.connect(str(path))
            client.sendall(b"{ this is not json }\n")
            client.sendall(
                json.dumps({"op": "point", "layer": "x", "id": "p", "p": [0, 0, 0]}).encode()
                + b"\n"
            )
            deadline = time.time() + 3.0
            while backend.debug.stats().primitives == 0 and time.time() < deadline:
                br.pump()
                time.sleep(0.01)
        assert backend.debug.stats().primitives == 1
        assert br.stats.invalid == 1
    finally:
        br.close()
