from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from forge_viewer import commands as cmd
from forge_viewer.adapters.base import FrameNeeds, NodeType
from forge_viewer.adapters.static import StaticSceneAdapter
from forge_viewer.adapters.toy import ToyPhysicsAdapter
from forge_viewer.gizmo import (
    AXIS_START,
    CENTER_HIT_PT,
    CENTER_RADIUS,
    CENTER_SHELL_RADIUS,
    GUIDE_CORE_COLOR,
    HOVER_COLOR,
    PLANE_INNER,
    RING_RADIUS,
    RING_SEGMENTS,
    RING_WIDTH_PT,
    SCREEN_RING_RADIUS,
    SIZE_PT,
    GizmoFrame,
    GizmoHandle,
    GizmoMode,
    GizmoSpace,
    axis_handle_alpha,
    axis_rotation,
    display_handles,
    handle_mask,
    hit_test,
    masked_axis_start,
    paint_order,
    plane_corners,
    plane_direction,
    plane_handle_alpha,
    project,
    rotation_dial,
    rotation_ring,
    rotation_ring_alpha,
    rotation_ring_is_full,
    world_scale,
)
from forge_viewer.render.backend import BackendCaps
from forge_viewer.scene import Scene
from forge_viewer.session import Session
from forge_viewer.types import CameraView
from forge_viewer.ui.gizmo import (
    DEFAULT_ROTATION_SNAP_DEG,
    DEFAULT_ROTATION_TICK_SCALE,
    DEFAULT_TRANSLATION_SNAP_M,
    ObjectGizmo,
    RotationDialProjection,
    _clip_line_to_rect,
    _project_rotation_dial,
    _project_rotation_tick,
    _projected_line_parameters,
    _rotation_fill_alpha,
    _rotation_sweep,
    _RotationDialProjector,
    _snap_tick_alpha,
    _split_segment_around_point,
)

RECT = (0.0, 0.0, 800.0, 600.0)


class RecordingDraw2D:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []

    def __getattr__(self, name: str):
        if name.startswith("__"):
            raise AttributeError(name)

        def record(*args, **kwargs):
            self.calls.append((name, args, kwargs))

        return record


@pytest.mark.parametrize(
    ("degrees", "sweep"),
    (
        (181.0, 181.0),
        (359.0, 359.0),
        (359.96, 0.0),
        (360.0, 0.0),
        (405.0, 45.0),
        (720.0, 0.0),
        (-405.0, -45.0),
    ),
)
def test_rotation_guide_wraps_only_after_a_full_turn(degrees: float, sweep: float) -> None:
    assert np.degrees(_rotation_sweep(np.radians(degrees))) == pytest.approx(sweep)


@pytest.mark.parametrize(
    ("degrees", "alpha"),
    ((0.0, 0.0), (90.0, 0.28), (150.0, 0.28), (180.0, 0.28), (247.0, 0.28)),
)
def test_rotation_fill_keeps_constant_opacity(degrees: float, alpha: float) -> None:
    assert _rotation_fill_alpha(np.radians(degrees)) == pytest.approx(alpha)
    assert _rotation_fill_alpha(np.radians(-degrees)) == pytest.approx(alpha)


def test_running_simulation_locks_camera_and_light_gizmos_by_default() -> None:
    session = Session(ToyPhysicsAdapter())
    gizmo = ObjectGizmo()
    camera = next(node for node in session.nodes if node.type is NodeType.CAMERA)
    light = next(node for node in session.nodes if node.type is NodeType.LIGHT)

    for node in (camera, light):
        assert session.entity_gizmo_locked(node)
        assert gizmo.evaluate(session, node).reason == "gizmo is locked while simulation is running"
        session.set_entity_gizmo_lock(node, False)
        assert not session.entity_gizmo_locked(node)
        assert gizmo.evaluate(session, node).ok

    session.submit(cmd.Pause())
    session.set_entity_gizmo_lock(camera, True)
    assert not session.entity_gizmo_locked(camera)
    assert gizmo.evaluate(session, camera).ok


@pytest.mark.parametrize("orthographic", [False, True], ids=("perspective", "orthographic"))
@pytest.mark.parametrize("degrees", [5.0, -30.0], ids=("small", "negative"))
def test_rotation_guide_keeps_the_sector_and_arc_without_center_strokes(
    orthographic: bool,
    degrees: float,
) -> None:
    cam = CameraView(
        eye=np.array((4.0, -6.0, 0.8), np.float32),
        target=np.zeros(3, np.float32),
        up=np.array((0.0, 0.0, 1.0), np.float32),
        aspect=RECT[2] / RECT[3],
        orthographic=orthographic,
        ortho_height=5.0,
    )
    center = np.zeros(3)
    axis = np.array((0.0, 0.0, 1.0))
    start = np.array((1.0, 0.0, 0.0))
    dial = _RotationDialProjector(
        cam,
        RECT,
        center,
        axis,
        start,
        SIZE_PT,
        RotationDialProjection.ORTHOGRAPHIC,
    )
    gizmo = ObjectGizmo("rotate")
    gizmo._active = GizmoHandle.ROTATE_Z
    gizmo._axis[:] = axis
    gizmo._rotation_angle = np.radians(degrees)
    overlay = RecordingDraw2D()

    gizmo._draw_rotation_guide(overlay, cam, RECT, 1.0, dial)

    fills = [args[0] for name, args, _kwargs in overlay.calls if name == "triangle_fan_fill"]
    polylines = [kwargs for name, _args, kwargs in overlay.calls if name == "polyline"]
    strokes = [args[0] for name, args, _kwargs in overlay.calls if name == "fringed_concave_fill"]
    dots = [args[0] for name, args, _kwargs in overlay.calls if name == "circle_filled"]
    projected_center = project(cam, (center,), RECT)[0, :2]
    assert len(fills) == 1
    assert fills[0][0] == pytest.approx(projected_center)
    assert sum(bool(kwargs.get("closed")) for kwargs in polylines) == 1
    assert sum(not bool(kwargs.get("closed")) for kwargs in polylines) == 0
    assert len(strokes) == 1
    stroke = strokes[0]
    point_count = len(stroke) // 2
    caps = (stroke[0] - stroke[-1], stroke[point_count] - stroke[point_count - 1])
    for cap, angle in zip(caps, (0.0, gizmo._rotation_angle), strict=True):
        tick = dial.tick(RING_RADIUS, angle, 1.0)
        assert tick is not None
        radial = tick[1] - tick[0]
        cross = cap[0] * radial[1] - cap[1] * radial[0]
        assert abs(cross) <= 1e-6 * np.linalg.norm(cap) * np.linalg.norm(radial)
        assert np.linalg.norm(cap) == pytest.approx(RING_WIDTH_PT)
    assert not dots
    assert all(name != "line" for name, _args, _kwargs in overlay.calls)


@pytest.mark.parametrize("degrees", [285.0, -285.0])
def test_rotation_guide_uses_an_oriented_triangle_fan_for_reflex_sectors(
    degrees: float,
) -> None:
    cam = camera()
    center = np.zeros(3)
    axis = np.array((0.0, 0.0, 1.0))
    start = np.array((1.0, 0.0, 0.0))
    dial = _RotationDialProjector(
        cam,
        RECT,
        center,
        axis,
        start,
        SIZE_PT,
        RotationDialProjection.ORTHOGRAPHIC,
    )
    gizmo = ObjectGizmo("rotate")
    gizmo._active = GizmoHandle.ROTATE_Z
    gizmo._axis[:] = axis
    gizmo._rotation_angle = np.radians(degrees)
    overlay = RecordingDraw2D()

    gizmo._draw_rotation_guide(overlay, cam, RECT, 1.0, dial)

    fills = [args[0] for name, args, _kwargs in overlay.calls if name == "triangle_fan_fill"]
    assert len(fills) == 1
    sector = np.asarray(fills[0])
    assert sector[0] == pytest.approx(project(cam, (center,), RECT)[0, :2])
    assert sector[1] == pytest.approx(dial.points(RING_RADIUS, (0.0,))[0, :2])
    assert sector[-1] == pytest.approx(dial.points(RING_RADIUS, (np.radians(degrees),))[0, :2])
    assert all(name != "convex_fill" for name, _args, _kwargs in overlay.calls)


def test_rotation_snap_highlight_matches_the_corresponding_tick_length() -> None:
    cam = camera()
    axis = np.array((0.0, 0.0, 1.0))
    dial = _RotationDialProjector(
        cam,
        RECT,
        np.zeros(3),
        axis,
        np.array((1.0, 0.0, 0.0)),
        SIZE_PT,
        RotationDialProjection.ORTHOGRAPHIC,
    )
    gizmo = ObjectGizmo("rotate")
    gizmo._active = GizmoHandle.ROTATE_Z
    gizmo._axis[:] = axis
    gizmo._rotation_angle = np.radians(5.0)
    overlay = RecordingDraw2D()

    gizmo._draw_rotation_snap_ticks(overlay, cam, RECT, 1.0, dial)

    lines = [args for name, args, _kwargs in overlay.calls if name == "line"]
    core = [args for args in lines if np.allclose(args[2], GUIDE_CORE_COLOR)]
    highlighted = [args for args in lines if np.allclose(args[2], HOVER_COLOR)]
    assert core
    assert len(highlighted) == 1
    start, end = highlighted[0][:2]
    assert np.linalg.norm(end - start) == pytest.approx(4.0 * DEFAULT_ROTATION_TICK_SCALE)


def camera(*, orthographic: bool = False) -> CameraView:
    return CameraView(
        eye=np.array((4.0, -6.0, 3.0), np.float32),
        target=np.zeros(3, np.float32),
        up=np.array((0.0, 0.0, 1.0), np.float32),
        aspect=RECT[2] / RECT[3],
        orthographic=orthographic,
        ortho_height=5.0,
    )


def session_at(position=(0.0, 0.0, 0.0), rotation=None) -> tuple[Session, object]:
    scene = Scene()
    obj = scene.box(name="editable", position=position, rotation=rotation)
    session = Session(StaticSceneAdapter(scene))
    session.submit(cmd.Select(obj.object_id))
    return session, session.selected_node


@pytest.mark.parametrize("orthographic", [False, True], ids=("perspective", "orthographic"))
def test_gizmo_keeps_the_same_screen_size_at_different_depths(orthographic: bool) -> None:

    cam = CameraView(
        eye=np.array((0.0, -5.0, 0.0), np.float32),
        target=np.zeros(3, np.float32),
        up=np.array((0.0, 0.0, 1.0), np.float32),
        aspect=RECT[2] / RECT[3],
        orthographic=orthographic,
        ortho_height=5.0,
    )
    for origin in (np.zeros(3), np.array((0.0, 5.0, 0.0))):
        scale = world_scale(cam, origin, RECT[3])
        screen = project(cam, (origin, origin + np.array((0.0, 0.0, scale))), RECT)
        assert np.linalg.norm(screen[1, :2] - screen[0, :2]) == pytest.approx(SIZE_PT)


@pytest.mark.parametrize("orthographic", [False, True], ids=("perspective", "orthographic"))
def test_center_and_axis_hit_use_the_viewport_projection(orthographic: bool) -> None:
    cam = camera(orthographic=orthographic)
    origin = np.zeros(3)
    rotation = np.eye(3)
    center = project(cam, (origin,), RECT)[0, :2]
    assert hit_test(cam, origin, rotation, RECT, center, GizmoMode.TRANSLATE)[0] is (
        GizmoHandle.SCREEN
    )

    scale = world_scale(cam, origin, RECT[3])
    x_axis = project(cam, (origin + np.array((0.58 * scale, 0.0, 0.0)),), RECT)[0, :2]
    assert hit_test(cam, origin, rotation, RECT, x_axis, GizmoMode.TRANSLATE)[0] is GizmoHandle.X


def test_hover_clears_when_the_viewport_does_not_own_input() -> None:
    session, _ = session_at()
    gizmo = ObjectGizmo()
    cam = camera()
    center = tuple(project(cam, (np.zeros(3),), RECT)[0, :2])

    assert gizmo.update_hover(session, cam, RECT, center) is GizmoHandle.SCREEN
    assert gizmo.update_hover(session, cam, RECT, center, enabled=False) is GizmoHandle.NONE


def test_axis_hit_respects_the_center_mask_and_visible_shaft_width() -> None:
    cam = CameraView(
        eye=np.array((0.0, -5.0, 0.0)),
        target=np.zeros(3),
        up=np.array((0.0, 0.0, 1.0)),
        aspect=RECT[2] / RECT[3],
    )
    origin = np.zeros(3)
    center = project(cam, (origin,), RECT)[0, :2]
    masked = center + np.array((0.5 * (CENTER_HIT_PT + CENTER_SHELL_RADIUS * SIZE_PT), 0.0))
    assert hit_test(cam, origin, np.eye(3), RECT, masked, GizmoMode.TRANSLATE)[0] is (
        GizmoHandle.NONE
    )

    scale = world_scale(cam, origin, RECT[3])
    shaft = project(cam, (origin + np.array((0.5 * scale, 0.0, 0.0)),), RECT)[0, :2]
    assert (
        hit_test(
            cam,
            origin,
            np.eye(3),
            RECT,
            shaft + np.array((0.0, 6.5)),
            GizmoMode.TRANSLATE,
        )[0]
        is GizmoHandle.NONE
    )


def test_overlapping_axis_hit_matches_the_topmost_drawn_handle() -> None:
    cam = CameraView(
        eye=np.array((5.0, -5.0, 0.0)),
        target=np.zeros(3),
        up=np.array((0.0, 0.0, 1.0)),
        aspect=RECT[2] / RECT[3],
    )
    origin = np.zeros(3)
    scale = world_scale(cam, origin, RECT[3])
    cursor = project(cam, (origin + np.array((0.5 * scale, 0.0, 0.0)),), RECT)[0, :2]

    assert paint_order(cam, origin, np.eye(3).T) == (1, 2, 0)
    assert hit_test(cam, origin, np.eye(3), RECT, cursor, GizmoMode.TRANSLATE)[0] is (GizmoHandle.X)


def test_translation_center_shell_masks_continuous_axes_but_not_planes() -> None:
    from forge_viewer.gizmo import CONTRAST_EDGE_PT, SIZE_PT

    visible_radius = CENTER_RADIUS + CONTRAST_EDGE_PT / SIZE_PT
    assert AXIS_START < CENTER_RADIUS < visible_radius < CENTER_SHELL_RADIUS < PLANE_INNER
    assert np.allclose(masked_axis_start(np.zeros(2), np.array((20.0, 0.0)), 7.0), (7, 0))


def test_active_3d_translation_axis_keeps_the_center_shell_mask() -> None:
    from types import SimpleNamespace

    from forge_viewer.render.forge.passes.gizmo import GizmoPass

    frame = GizmoFrame()
    frame.active = GizmoHandle.X
    context = SimpleNamespace(
        camera=camera(),
        ctx=SimpleNamespace(enable=lambda *_: None, disable=lambda *_: None),
        target=SimpleNamespace(fbo=SimpleNamespace(depth_mask=True)),
    )
    draws = []
    gizmo_pass = GizmoPass()
    gizmo_pass._draw = lambda *args, **kwargs: draws.append((args, kwargs))

    gizmo_pass._draw_translate(context, frame, 1.0)

    assert len(draws) == 1
    assert draws[0][1]["mask_radius"] == CENTER_SHELL_RADIUS


def test_snap_tick_fade_has_a_five_step_core_and_ten_step_cutoff() -> None:
    assert _snap_tick_alpha(0.0) == 1.0
    assert _snap_tick_alpha(-5.0) == 1.0
    assert _snap_tick_alpha(7.5) == pytest.approx(0.5)
    assert _snap_tick_alpha(10.0) == 0.0


def test_snap_ruler_segments_are_clipped_by_the_center_shell() -> None:
    segments = _split_segment_around_point((0.0, 0.0), (20.0, 0.0), (10.0, 0.0), 3.0)
    assert len(segments) == 2
    assert segments[0][0] == pytest.approx((0.0, 0.0))
    assert segments[0][1] == pytest.approx((7.0, 0.0))
    assert segments[1][0] == pytest.approx((13.0, 0.0))
    assert segments[1][1] == pytest.approx((20.0, 0.0))


def test_rotation_idle_uses_front_half_rings_and_an_interactive_outer_ring() -> None:
    cam = camera()
    origin = np.zeros(3)
    rotation = np.eye(3)
    scale = world_scale(cam, origin, RECT[3])
    ring = rotation_ring(cam, origin, rotation, scale, 2, full=False)
    to_eye = cam.eye - origin
    assert np.min((ring - origin) @ to_eye) >= -1e-8
    full_ring = rotation_ring(cam, origin, rotation, scale, 2, full=True)
    assert len(full_ring) == RING_SEGMENTS
    assert not np.allclose(full_ring[0], full_ring[-1])

    center = project(cam, (origin,), RECT)[0, :2]
    outer = center + np.array((SCREEN_RING_RADIUS * SIZE_PT, 0.0))
    assert hit_test(cam, origin, rotation, RECT, outer, GizmoMode.ROTATE)[0] is (
        GizmoHandle.ROTATE_SCREEN
    )

    edge_cam = CameraView(
        eye=np.array((4.0, -6.0, 0.0)),
        target=origin,
        up=np.array((0.0, 0.0, 1.0)),
        aspect=RECT[2] / RECT[3],
    )
    assert rotation_ring_alpha(edge_cam, origin, rotation[:, 2]) == 0.0


def test_single_axis_rotation_uses_and_hits_the_full_ring() -> None:
    cam = camera()
    origin = np.zeros(3)
    rotation = np.eye(3)
    frame = GizmoFrame(
        mode=GizmoMode.ROTATE,
        handle_mask=handle_mask(GizmoHandle.ROTATE_Z),
    )
    assert rotation_ring_is_full(frame, GizmoHandle.ROTATE_Z)

    scale = world_scale(cam, origin, RECT[3])
    front = rotation_ring(cam, origin, rotation, scale, 2, full=False)
    full = rotation_ring(cam, origin, rotation, scale, 2, full=True)
    front_screen = project(cam, front, RECT)[:, :2]
    full_screen = project(cam, full, RECT)[:, :2]
    distances = np.array(
        [
            min(np.linalg.norm(point - front_screen[index]) for index in range(len(front_screen)))
            for point in full_screen
        ]
    )
    back_point = full_screen[int(np.argmax(distances))]
    assert distances.max() > 20.0
    assert (
        hit_test(
            cam,
            origin,
            rotation,
            RECT,
            back_point,
            GizmoMode.ROTATE,
            allowed_handles=frame.handle_mask,
        )[0]
        is GizmoHandle.ROTATE_Z
    )

    frame.handle_mask = handle_mask(GizmoHandle.ROTATE_X, GizmoHandle.ROTATE_Z)
    assert not rotation_ring_is_full(frame, GizmoHandle.ROTATE_Z)


def test_translation_axes_and_planes_fade_when_their_projection_degenerates() -> None:
    cam = CameraView(
        eye=np.array((5.0, 0.0, 0.0)),
        target=np.zeros(3),
        up=np.array((0.0, 0.0, 1.0)),
        aspect=RECT[2] / RECT[3],
    )
    assert axis_handle_alpha(cam, np.zeros(3), np.array((1.0, 0.0, 0.0))) == 0.0
    assert plane_handle_alpha(cam, np.zeros(3), np.array((0.0, 1.0, 0.0))) == 0.0


def test_axis_rotation_maps_mesh_z_to_each_object_axis_without_mirroring() -> None:
    rotation = np.array(((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)))
    for axis in range(3):
        mapped = axis_rotation(rotation, axis)
        assert mapped[:, 2] == pytest.approx(rotation[:, axis])
        assert np.linalg.det(mapped) == pytest.approx(1.0)


def test_paint_order_sorts_handles_far_to_near_along_the_view() -> None:
    eye = np.array((3.0, -4.0, 2.2), np.float32)
    origin = np.zeros(3, np.float32)
    up = np.array((0.0, 0.0, 1.0), np.float32)
    # View direction ~= (-0.55, 0.73, -0.40): Y is farthest, X nearest.
    for cam in (
        CameraView(eye=eye, target=origin.copy(), up=up),
        CameraView(eye=eye, target=origin.copy(), up=up, orthographic=True),
    ):
        assert paint_order(cam, origin, [np.eye(3)[:, i] for i in range(3)]) == (1, 2, 0)
    yaw90 = np.array(((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)))
    cam = CameraView(eye=eye, target=origin.copy(), up=up)
    assert paint_order(cam, origin, [yaw90[:, i] for i in range(3)]) == (0, 1, 2)


def test_plane_direction_points_toward_the_quad_center() -> None:
    origin = np.zeros(3)
    rotation = np.array(((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)))
    for axis in range(3):
        direction = plane_direction(rotation, axis)
        center = plane_corners(origin, rotation, 1.0, axis).mean(axis=0)
        assert direction == pytest.approx(
            center * np.linalg.norm(direction) / np.linalg.norm(center)
        )


class CaptureBackend:
    caps = BackendCaps(name="capture", gizmo=True)

    def __init__(self) -> None:
        self.frame = None

    def set_gizmo(self, frame) -> bool:
        self.frame = frame
        return frame is not None


def test_hidpi_scales_solid_gizmo_and_hit_geometry() -> None:
    session, _ = session_at()
    gizmo = ObjectGizmo()
    gizmo.set_style("3d")
    backend = CaptureBackend()
    cam = camera()
    center = tuple(project(cam, (np.zeros(3),), RECT)[0, :2])

    assert gizmo.update_hover(session, cam, RECT, center, style_scale=2.0) is GizmoHandle.SCREEN
    assert gizmo.publish(
        backend,
        session,
        cam,
        RECT,
        ui_scale=2.0,
        style_scale=2.0,
        yielding=False,
        interactive=True,
    )
    assert backend.frame.size_px == 2.0 * SIZE_PT

    gizmo.set_mode("rotate")
    outer = (center[0] + SCREEN_RING_RADIUS * SIZE_PT * 2.0, center[1])
    assert (
        gizmo.update_hover(session, cam, RECT, outer, style_scale=2.0) is GizmoHandle.ROTATE_SCREEN
    )


def test_body_and_world_space_publish_the_same_basis_used_for_interaction() -> None:
    rotation = np.array(((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)))
    session, _ = session_at(rotation=rotation)
    gizmo = ObjectGizmo()
    gizmo.set_style("3d")
    backend = CaptureBackend()
    cam = camera()

    gizmo.publish(
        backend,
        session,
        cam,
        RECT,
        ui_scale=1.0,
        style_scale=1.0,
        yielding=False,
        interactive=True,
    )
    assert backend.frame.space is GizmoSpace.BODY
    assert backend.frame.rotation == pytest.approx(rotation)

    gizmo.set_space("world")
    gizmo.publish(
        backend,
        session,
        cam,
        RECT,
        ui_scale=1.0,
        style_scale=1.0,
        yielding=False,
        interactive=True,
    )
    assert backend.frame.space is GizmoSpace.WORLD
    assert backend.frame.rotation == pytest.approx(np.eye(3))


def test_flat_is_default_and_active_handle_hides_the_rest() -> None:
    gizmo = ObjectGizmo()
    assert gizmo.style == "2d"
    gizmo.set_style("3d")
    assert gizmo.style == "3d"

    frame = GizmoFrame()
    assert set(display_handles(frame)) == {
        GizmoHandle.X,
        GizmoHandle.Y,
        GizmoHandle.Z,
        GizmoHandle.YZ,
        GizmoHandle.ZX,
        GizmoHandle.XY,
        GizmoHandle.SCREEN,
    }
    frame.active = GizmoHandle.X
    assert display_handles(frame) == (GizmoHandle.X,)
    frame.mode = GizmoMode.ROTATE
    frame.active = GizmoHandle.NONE
    assert set(display_handles(frame)) == {
        GizmoHandle.ROTATE_X,
        GizmoHandle.ROTATE_Y,
        GizmoHandle.ROTATE_Z,
        GizmoHandle.ROTATE_SCREEN,
    }
    frame.active = GizmoHandle.ROTATE_Z
    assert display_handles(frame) == (GizmoHandle.ROTATE_Z,)

    frame.active = GizmoHandle.NONE
    frame.handle_mask = handle_mask(GizmoHandle.ROTATE_Z)
    assert display_handles(frame) == (GizmoHandle.ROTATE_Z,)


def test_multiple_direct_joints_require_an_explicit_gizmo_target() -> None:
    from forge_viewer.adapters.base import JointInfo, SceneNode

    joints = (
        JointInfo(3, "yaw", "hinge", False, (0.0, 0.0), 5, 4, 1, body=2),
        JointInfo(4, "lift", "slide", True, (-0.2, 0.2), 6, 5, 1, body=2),
    )

    class JointSession:
        structure_generation = 1

        @staticmethod
        def joints_for_body(body_index: int):
            return joints if body_index == 2 else ()

    session = JointSession()
    node = SceneNode(2, "compound", NodeType.LINK, body_index=2)
    session.selected_node = node
    gizmo = ObjectGizmo()

    assert gizmo.joint_choices(session) == joints
    target, reason = gizmo._joint_target(session, node)
    assert target is None
    assert "viewport picker" in reason

    gizmo.select_joint(node.body_index, joints[1].joint_id)
    target, reason = gizmo._joint_target(session, node)
    assert not reason
    assert target is not None and target.joint is joints[1]
    assert target.mode is GizmoMode.TRANSLATE
    assert target.handles == handle_mask(GizmoHandle.Z)


def test_axis_drag_moves_the_body_but_not_across_other_local_axes() -> None:
    session, node = session_at()
    gizmo = ObjectGizmo()
    cam = camera()
    origin = np.zeros(3)
    scale = world_scale(cam, origin, RECT[3])
    start = project(cam, (origin + np.array((0.55 * scale, 0.0, 0.0)),), RECT)[0, :2]

    assert gizmo.update_hover(session, cam, RECT, tuple(start)) is GizmoHandle.X
    gizmo.interact(session, cam, RECT, tuple(start), claimed=True, left_down=True, released=False)
    assert gizmo.using
    axis_screen = project(cam, (origin, origin + np.array((scale, 0.0, 0.0))), RECT)[:, :2]
    direction = axis_screen[1] - axis_screen[0]
    direction /= np.linalg.norm(direction)
    end = start + direction * 36.0
    assert gizmo.interact(
        session, cam, RECT, tuple(end), claimed=True, left_down=True, released=False
    )

    position = session.frame.body_xpos[node.body_index]
    assert position[0] > 0.1
    assert position[1:] == pytest.approx((0.0, 0.0), abs=1e-7)
    assert gizmo.value_label.startswith("X +") and gizmo.value_label.endswith(" m")


def test_shift_snaps_axis_translation_to_the_configured_increment() -> None:
    session, node = session_at()
    gizmo = ObjectGizmo()
    gizmo.translation_snap_m = 0.25
    cam = camera()
    origin = np.zeros(3)
    scale = world_scale(cam, origin, RECT[3])
    screen_axis = project(cam, (origin, origin + np.array((scale, 0.0, 0.0))), RECT)[:, :2]
    direction = screen_axis[1] - screen_axis[0]
    direction /= np.linalg.norm(direction)
    start = project(cam, (origin + np.array((0.55 * scale, 0.0, 0.0)),), RECT)[0, :2]

    assert gizmo.update_hover(session, cam, RECT, tuple(start)) is GizmoHandle.X
    gizmo.interact(
        session, cam, RECT, tuple(start), claimed=True, left_down=True, released=False, snap=True
    )
    gizmo.interact(
        session,
        cam,
        RECT,
        tuple(start + direction * 57.0),
        claimed=True,
        left_down=True,
        released=False,
        snap=True,
    )

    position = np.asarray(session.frame.body_xpos[node.body_index])
    assert position[0] / 0.25 == pytest.approx(round(position[0] / 0.25), abs=1e-6)
    assert position[1:] == pytest.approx((0.0, 0.0), abs=1e-7)
    assert gizmo.snapping
    assert gizmo.value_label.endswith("· SNAP 0.25 m")


def test_shift_snaps_rotation_from_the_drag_origin() -> None:
    session, node = session_at()
    gizmo = ObjectGizmo("rotate")
    gizmo.rotation_snap_deg = 15.0
    cam = camera()
    origin = np.zeros(3)
    scale = world_scale(cam, origin, RECT[3])
    start_angle = next(
        angle
        for angle in np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False)
        if hit_test(
            cam,
            origin,
            np.eye(3),
            RECT,
            tuple(
                project(
                    cam,
                    (RING_RADIUS * scale * np.array((np.cos(angle), np.sin(angle), 0.0)),),
                    RECT,
                )[0, :2]
            ),
            GizmoMode.ROTATE,
        )[0]
        is GizmoHandle.ROTATE_Z
    )
    points = [
        RING_RADIUS
        * scale
        * np.array((np.cos(start_angle + angle), np.sin(start_angle + angle), 0.0))
        for angle in (0.0, np.radians(22.0))
    ]
    start, end = project(cam, points, RECT)[:, :2]

    gizmo.update_hover(session, cam, RECT, tuple(start))
    gizmo.interact(session, cam, RECT, tuple(start), claimed=True, left_down=True, released=False)
    assert gizmo.interact(
        session,
        cam,
        RECT,
        tuple(end),
        claimed=True,
        left_down=True,
        released=False,
    )
    assert gizmo.interact(
        session,
        cam,
        RECT,
        tuple(end),
        claimed=True,
        left_down=True,
        released=False,
        snap=True,
    )

    rotation = np.asarray(session.frame.body_xmat[node.body_index]).reshape(3, 3)
    expected = np.array(
        (
            (np.cos(np.radians(15.0)), -np.sin(np.radians(15.0)), 0.0),
            (np.sin(np.radians(15.0)), np.cos(np.radians(15.0)), 0.0),
            (0.0, 0.0, 1.0),
        )
    )
    assert rotation == pytest.approx(expected, abs=1e-6)
    assert gizmo.value_label.endswith("· SNAP 15°")


def test_gizmo_snap_defaults_match_the_settings_resets() -> None:
    gizmo = ObjectGizmo()
    assert gizmo.translation_snap_m == DEFAULT_TRANSLATION_SNAP_M == 0.5
    assert gizmo.rotation_snap_deg == DEFAULT_ROTATION_SNAP_DEG == 5.0
    assert gizmo.rotation_tick_scale == DEFAULT_ROTATION_TICK_SCALE == 1.25
    assert gizmo.rotation_dial_projection is RotationDialProjection.ORTHOGRAPHIC


@pytest.mark.parametrize("orthographic", [False, True], ids=("perspective", "orthographic"))
def test_rotation_dial_layers_use_independent_world_radii(orthographic: bool) -> None:
    cam = camera(orthographic=orthographic)
    center = np.array((0.35, -0.2, 0.4))
    axis = np.array((0.31, -0.22, 0.925))
    axis /= np.linalg.norm(axis)
    start = np.cross(axis, np.array((0.0, 0.0, 1.0)))
    start /= np.linalg.norm(start)
    angles = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
    size_px = SIZE_PT
    scale = world_scale(cam, center, RECT[3], size_px)
    spacing = 0.1
    radii = (RING_RADIUS - spacing, RING_RADIUS, RING_RADIUS + spacing)
    layers = [
        _project_rotation_dial(
            cam,
            RECT,
            center,
            axis,
            start,
            size_px,
            radius,
            angles,
            RotationDialProjection.CLASSIC,
        )
        for radius in radii
    ]
    for layer, radius in zip(layers, radii, strict=True):
        expected = project(
            cam,
            rotation_dial(center, axis, start, scale, radius, angles),
            RECT,
        )
        assert layer == pytest.approx(expected, abs=1e-6)

    coincident = [
        _project_rotation_dial(
            cam,
            RECT,
            center,
            axis,
            start,
            size_px,
            RING_RADIUS,
            angles,
            RotationDialProjection.CLASSIC,
        )
        for _ in range(3)
    ]
    assert coincident[0] == pytest.approx(coincident[1], abs=1e-6)
    assert coincident[1] == pytest.approx(coincident[2], abs=1e-6)


@pytest.mark.parametrize("camera_projection", [False, True], ids=("perspective", "orthographic"))
def test_orthographic_rotation_dial_layers_are_concentric_and_homothetic(
    camera_projection: bool,
) -> None:
    cam = camera(orthographic=camera_projection)
    center = np.array((0.35, -0.2, 0.4))
    axis = np.array((0.31, -0.22, 0.925))
    axis /= np.linalg.norm(axis)
    start = np.cross(axis, np.array((0.0, 0.0, 1.0)))
    start /= np.linalg.norm(start)
    angles = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    radii = np.array((RING_RADIUS - 0.1, RING_RADIUS, RING_RADIUS + 0.1))
    projected_center = project(cam, (center,), RECT)[0, :2]
    dial = _RotationDialProjector(
        cam,
        RECT,
        center,
        axis,
        start,
        SIZE_PT,
        RotationDialProjection.ORTHOGRAPHIC,
    )
    layers = [dial.points(radius, angles)[:, :2] for radius in radii]

    normalized = [
        (layer - projected_center) / radius for layer, radius in zip(layers, radii, strict=True)
    ]
    assert normalized[0] == pytest.approx(normalized[1], abs=1e-10)
    assert normalized[1] == pytest.approx(normalized[2], abs=1e-10)
    for layer in layers:
        assert np.mean(layer, axis=0) == pytest.approx(projected_center, abs=1e-10)


def test_orthographic_rotation_dial_is_independent_of_scene_focal_length() -> None:
    center = np.array((0.35, -0.2, 0.4))
    axis = np.array((0.31, -0.22, 0.925))
    axis /= np.linalg.norm(axis)
    start = np.cross(axis, np.array((0.0, 0.0, 1.0)))
    start /= np.linalg.norm(start)
    angles = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    normalized_layers = []
    for fov_y in np.radians((5.0, 45.0, 140.0)):
        cam = replace(camera(), fov_y=fov_y)
        projected_center = project(cam, (center,), RECT)[0, :2]
        dial = _RotationDialProjector(
            cam,
            RECT,
            center,
            axis,
            start,
            SIZE_PT,
            RotationDialProjection.ORTHOGRAPHIC,
        )
        layer = dial.points(RING_RADIUS, angles)[:, :2]
        normalized_layers.append((layer - projected_center) / RING_RADIUS)

    assert normalized_layers[0] == pytest.approx(normalized_layers[1], abs=1e-10)
    assert normalized_layers[1] == pytest.approx(normalized_layers[2], abs=1e-10)


@pytest.mark.parametrize("distance", [4.0, 12.0], ids=("near", "far"))
@pytest.mark.parametrize("projection", list(RotationDialProjection))
def test_rotation_tick_keeps_its_screen_length_across_camera_distance(
    distance: float,
    projection: RotationDialProjection,
) -> None:
    cam = CameraView(
        eye=np.array((distance, -1.5 * distance, 0.75 * distance)),
        target=np.zeros(3),
        up=np.array((0.0, 0.0, 1.0)),
        aspect=RECT[2] / RECT[3],
    )
    center = np.zeros(3)
    tick = _project_rotation_tick(
        cam,
        RECT,
        center,
        np.array((0.0, 0.0, 1.0)),
        np.array((1.0, 0.0, 0.0)),
        SIZE_PT,
        RING_RADIUS,
        0.8,
        9.0,
        projection,
    )
    assert tick is not None
    assert np.linalg.norm(tick[1] - tick[0]) == pytest.approx(9.0)


@pytest.mark.parametrize("orthographic", [False, True], ids=("perspective", "orthographic"))
def test_screen_rotation_dial_keeps_a_constant_screen_radius(orthographic: bool) -> None:
    cam = camera(orthographic=orthographic)
    center = np.array((0.35, -0.2, 0.4))
    view_basis = np.asarray(cam.view_matrix())[:3, :3].T
    angles = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
    scale = world_scale(cam, center, RECT[3], SIZE_PT * 1.5)
    world = rotation_dial(
        center,
        -cam.forward(),
        view_basis[:, 0],
        scale,
        SCREEN_RING_RADIUS,
        angles,
    )
    points = project(cam, world, RECT)
    projected_center = project(cam, (center,), RECT)[0, :2]
    radii = np.linalg.norm(points[:, :2] - projected_center, axis=1)
    expected = SCREEN_RING_RADIUS * SIZE_PT * 1.5
    assert radii == pytest.approx(np.full(len(angles), expected), abs=1e-5)


def test_screen_translation_reports_all_xyz_components() -> None:
    session, _node = session_at()
    gizmo = ObjectGizmo()
    cam = camera()
    center = project(cam, (np.zeros(3),), RECT)[0, :2]

    assert gizmo.update_hover(session, cam, RECT, tuple(center)) is GizmoHandle.SCREEN
    gizmo.interact(session, cam, RECT, tuple(center), claimed=True, left_down=True, released=False)
    assert gizmo.interact(
        session,
        cam,
        RECT,
        tuple(center + np.array((32.0, -18.0))),
        claimed=True,
        left_down=True,
        released=False,
    )

    label = gizmo.value_label
    assert label.startswith("X ") and "  Y " in label and "  Z " in label
    assert "delta" not in label.lower() and "distance" not in label.lower()


def test_keyboard_axis_moves_without_hitting_or_clicking_the_arrow() -> None:
    session, node = session_at()
    gizmo = ObjectGizmo()
    cam = camera()
    start = np.array((143.0, 411.0))
    axis = project(cam, (np.zeros(3), np.array((1.0, 0.0, 0.0))), RECT)[:, :2]
    direction = axis[1] - axis[0]
    direction /= np.linalg.norm(direction)

    assert gizmo.keyboard_interact(session, cam, RECT, tuple(start), 0)
    assert gizmo.keyboard_using and gizmo.active_handle is GizmoHandle.X
    assert gizmo.keyboard_interact(session, cam, RECT, tuple(start + direction * 36.0), 0)
    position = np.asarray(session.frame.body_xpos[node.body_index])
    assert position[0] > 0.1
    assert position[1:] == pytest.approx((0.0, 0.0), abs=1e-7)

    gizmo.keyboard_interact(session, cam, RECT, tuple(start), -1)
    assert not gizmo.using and not gizmo.keyboard_using


def test_keyboard_axis_rotates_without_hitting_or_clicking_the_ring() -> None:
    session, node = session_at()
    gizmo = ObjectGizmo("rotate")
    cam = camera()
    world = np.array(((0.48, 0.37, 0.0), (0.21, 0.56, 0.0)))
    start, end = project(cam, world, RECT)[:, :2]

    assert gizmo.keyboard_interact(session, cam, RECT, tuple(start), 2)
    assert gizmo.keyboard_using and gizmo.active_handle is GizmoHandle.ROTATE_Z
    assert gizmo.keyboard_interact(session, cam, RECT, tuple(end), 2)
    rotation = np.asarray(session.frame.body_xmat[node.body_index]).reshape(3, 3)
    assert not np.allclose(rotation, np.eye(3))
    assert gizmo.value_label.startswith("Z ") and gizmo.value_label.endswith("°")


def test_keyboard_constraint_line_keeps_the_exact_projected_axis_direction() -> None:
    segment = _clip_line_to_rect((50.0, 50.0), (2.0, 1.0), (0.0, 0.0, 100.0, 100.0))
    assert segment is not None
    start, end = segment
    assert start == pytest.approx((0.0, 25.0))
    assert end == pytest.approx((100.0, 75.0))
    delta = end - start
    assert delta[0] - 2.0 * delta[1] == pytest.approx(0.0)


@pytest.mark.parametrize("orthographic", [False, True])
def test_snap_ruler_recovers_world_distance_from_projected_points(orthographic: bool) -> None:
    cam = camera(orthographic=orthographic)
    origin = np.array((0.2, -0.1, 0.3))
    axis = np.array((0.6, 0.3, 0.74161985))
    expected = (-0.75, 1.25)
    segment = project(cam, tuple(origin + axis * value for value in expected), RECT)[:, :2]

    parameters = _projected_line_parameters(cam, origin, axis, segment, RECT)

    assert parameters == pytest.approx(expected)


@pytest.mark.parametrize(
    ("space", "style", "axis"),
    (
        ("body", "2d", np.array((0.0, 1.0, 0.0))),
        ("world", "3d", np.array((1.0, 0.0, 0.0))),
    ),
)
def test_axis_drag_uses_the_selected_body_or_world_frame(space, style, axis) -> None:
    rotation = np.array(((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)))
    session, node = session_at(rotation=rotation)
    gizmo = ObjectGizmo()
    gizmo.set_space(space)
    gizmo.set_style(style)
    cam = camera()
    scale = world_scale(cam, np.zeros(3), RECT[3])
    start = project(cam, (axis * scale * 0.55,), RECT)[0, :2]

    assert gizmo.update_hover(session, cam, RECT, tuple(start)) is GizmoHandle.X
    gizmo.interact(session, cam, RECT, tuple(start), claimed=True, left_down=True, released=False)
    screen_axis = project(cam, (np.zeros(3), axis * scale), RECT)[:, :2]
    direction = screen_axis[1] - screen_axis[0]
    direction /= np.linalg.norm(direction)
    end = start + direction * 36.0
    assert gizmo.interact(
        session, cam, RECT, tuple(end), claimed=True, left_down=True, released=False
    )

    position = np.asarray(session.frame.body_xpos[node.body_index])
    assert np.dot(position, axis) > 0.1
    assert np.linalg.norm(position - axis * np.dot(position, axis)) < 1e-6


def test_plane_drag_stays_in_the_selected_local_plane() -> None:
    session, node = session_at()
    gizmo = ObjectGizmo()
    cam = camera()
    origin = np.zeros(3)
    scale = world_scale(cam, origin, RECT[3])
    world_start = scale * np.array((0.31, 0.31, 0.0))
    world_end = world_start + np.array((0.24, -0.13, 0.0))
    start, end = project(cam, (world_start, world_end), RECT)[:, :2]

    assert gizmo.update_hover(session, cam, RECT, tuple(start)) is GizmoHandle.XY
    gizmo.interact(session, cam, RECT, tuple(start), claimed=True, left_down=True, released=False)
    assert gizmo.interact(
        session, cam, RECT, tuple(end), claimed=True, left_down=True, released=False
    )

    assert session.frame.body_xpos[node.body_index] == pytest.approx((0.24, -0.13, 0.0), abs=1e-6)


def test_rotation_drag_preserves_position_and_a_rigid_rotation() -> None:
    session, node = session_at()
    gizmo = ObjectGizmo("rotate")
    gizmo.set_style("3d")
    cam = camera()
    origin = np.zeros(3)
    scale = world_scale(cam, origin, RECT[3])

    samples = np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False)
    start_angle = next(
        angle
        for angle in samples
        if hit_test(
            cam,
            origin,
            np.eye(3),
            RECT,
            tuple(
                project(
                    cam,
                    ((RING_RADIUS * scale * np.array((np.cos(angle), np.sin(angle), 0.0))),),
                    RECT,
                )[0, :2]
            ),
            GizmoMode.ROTATE,
        )[0]
        is GizmoHandle.ROTATE_Z
    )
    world_start = RING_RADIUS * scale * np.array((np.cos(start_angle), np.sin(start_angle), 0.0))
    world_end = (
        RING_RADIUS
        * scale
        * np.array((np.cos(start_angle + 0.35), np.sin(start_angle + 0.35), 0.0))
    )
    start, end = project(cam, (world_start, world_end), RECT)[:, :2]

    assert gizmo.update_hover(session, cam, RECT, tuple(start)) is GizmoHandle.ROTATE_Z
    gizmo.interact(session, cam, RECT, tuple(start), claimed=True, left_down=True, released=False)
    assert gizmo.using
    assert gizmo.interact(
        session, cam, RECT, tuple(end), claimed=True, left_down=True, released=False
    )

    position = session.frame.body_xpos[node.body_index]
    rotation = session.frame.body_xmat[node.body_index]
    assert position == pytest.approx((0.0, 0.0, 0.0), abs=1e-7)
    assert rotation @ rotation.T == pytest.approx(np.eye(3), abs=1e-6)
    assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1e-6)
    assert not np.allclose(rotation, np.eye(3))
    assert gizmo.value_label.startswith("Z +") and gizmo.value_label.endswith("°")

    backend = CaptureBackend()
    assert gizmo.publish(
        backend,
        session,
        cam,
        RECT,
        ui_scale=1.0,
        style_scale=1.0,
        yielding=False,
        interactive=True,
    )
    assert backend.frame.rotation == pytest.approx(np.eye(3))
    assert backend.frame.position == pytest.approx(origin)

    for angle in np.linspace(0.35, 0.35 - 10.5 * np.pi, 190)[1:]:
        world = (
            RING_RADIUS
            * scale
            * np.array((np.cos(start_angle + angle), np.sin(start_angle + angle), 0.0))
        )
        cursor = project(cam, (world,), RECT)[0, :2]
        assert gizmo.interact(
            session, cam, RECT, tuple(cursor), claimed=True, left_down=True, released=False
        )
    shown_degrees = float(gizmo.value_label.split()[1][:-1])
    assert shown_degrees == pytest.approx(np.degrees(0.35 - 10.5 * np.pi), abs=1.0)
    assert gizmo.value_label.endswith("· 5×360°")


def test_outer_ring_rotates_around_the_camera_axis() -> None:
    session, node = session_at()
    gizmo = ObjectGizmo("rotate")
    cam = camera()
    center = project(cam, (np.zeros(3),), RECT)[0, :2]
    radius = SCREEN_RING_RADIUS * SIZE_PT
    start = center + np.array((radius, 0.0))
    end = center + np.array((0.0, radius))

    assert gizmo.update_hover(session, cam, RECT, tuple(start)) is GizmoHandle.ROTATE_SCREEN
    gizmo.interact(session, cam, RECT, tuple(start), claimed=True, left_down=True, released=False)
    assert gizmo.interact(
        session, cam, RECT, tuple(end), claimed=True, left_down=True, released=False
    )

    rotation = np.asarray(session.frame.body_xmat[node.body_index]).reshape(3, 3)
    view_axis = -cam.forward()
    assert rotation @ view_axis == pytest.approx(view_axis, abs=1e-6)
    assert not np.allclose(rotation, np.eye(3))
    assert gizmo.value_label.startswith("Screen ")


@pytest.mark.physics
@pytest.mark.parametrize(
    ("body_name", "handle", "amount"),
    (
        ("hinge_body", GizmoHandle.ROTATE_Z, 0.3),
        ("slide_body", GizmoHandle.Z, 0.1),
        ("ball_body", GizmoHandle.ROTATE_Z, 0.3),
    ),
)
def test_joint_gizmo_edits_only_the_selected_joint_dof(
    body_name: str, handle: GizmoHandle, amount: float
) -> None:
    from forge_viewer.adapters.mujoco_adapter import MuJoCoAdapter
    from forge_viewer.assets import resolve

    adapter = MuJoCoAdapter(resolve("joint_types"))
    session = Session(adapter)
    assert session.submit(cmd.Pause())
    node = next(item for item in session.nodes if item.name == body_name)
    assert session.submit(cmd.Select(node.object_id))
    gizmo = ObjectGizmo()
    needs = gizmo.frame_needs(session)
    assert needs.qpos and needs.diagnostics
    session.tick(FrameNeeds(poses=True, qpos=True, diagnostics=True), wall_dt=0.0)
    assert gizmo.evaluate(session, node).ok

    target, reason = gizmo._joint_target(session, node)
    assert target is not None, reason
    pose = gizmo._target_pose(session, node, target)
    assert pose is not None
    position, basis = pose
    cam = CameraView(eye=np.array((2.0, -4.0, 2.0)), target=position.copy())

    if target.mode is GizmoMode.ROTATE:
        scale = world_scale(cam, position, RECT[3], SIZE_PT)
        start_world = position + basis[:, 0] * scale * RING_RADIUS
        end_world = (
            position
            + (basis[:, 0] * np.cos(amount) + basis[:, 1] * np.sin(amount)) * scale * RING_RADIUS
        )
        start = project(cam, (start_world,), RECT)[0, :2]
        end = project(cam, (end_world,), RECT)[0, :2]
    else:
        start = project(cam, (position + basis[:, 2] * 0.1,), RECT)[0, :2]
        end = None

    assert gizmo._begin_handle(session, cam, RECT, start, handle)
    if end is None:
        end = start + gizmo._axis_screen * (amount / gizmo._world_per_pt)
    assert gizmo._drag(session, cam, RECT, end, snap=False)

    joint = target.joint
    if joint.type == "ball":
        expected = np.array((np.cos(amount / 2.0), 0.0, 0.0, np.sin(amount / 2.0)))
        assert adapter.data.qpos[joint.qpos_adr : joint.qpos_adr + 4] == pytest.approx(
            expected, abs=1e-6
        )
    else:
        assert adapter.data.qpos[joint.qpos_adr] == pytest.approx(amount)


@pytest.mark.physics
def test_inspector_reports_the_actual_hinge_joint_gizmo(monkeypatch) -> None:
    from types import SimpleNamespace

    from forge_viewer.adapters.mujoco_adapter import MuJoCoAdapter
    from forge_viewer.assets import resolve
    from forge_viewer.ui.panels import PanelContext
    from forge_viewer.ui.panels import inspector as inspector_module
    from forge_viewer.ui.panels.inspector import InspectorPanel

    session = Session(MuJoCoAdapter(resolve("joint_types")))
    assert session.submit(cmd.Pause())
    node = next(item for item in session.nodes if item.name == "hinge_body")
    assert session.submit(cmd.Select(node.object_id))
    gizmo = ObjectGizmo()
    session.tick(gizmo.frame_needs(session))

    lines: list[str] = []
    fake_imgui = SimpleNamespace(
        ImVec4=lambda *values: values,
        separator=lambda: None,
        text_colored=lambda _color, value: lines.append(value),
        text_wrapped=lambda value: lines.append(value),
    )
    monkeypatch.setattr(inspector_module, "imgui", fake_imgui)

    InspectorPanel()._gizmo_reason(PanelContext(session, None, gizmo=gizmo), node)

    assert lines == ["gizmo: active (hinge / revolute joint; rotate about its axis)"]
