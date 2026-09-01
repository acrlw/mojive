from __future__ import annotations

import numpy as np
import pytest

from forge_viewer import commands as cmd
from forge_viewer import math3d
from forge_viewer.adapters.base import FrameNeeds, NodeType
from forge_viewer.adapters.static import StaticSceneAdapter
from forge_viewer.adapters.toy import ToyPhysicsAdapter
from forge_viewer.gizmo import (
    ACTIVE_HANDLE_COLOR,
    AXIS_COLORS,
    AXIS_END,
    AXIS_START,
    CENTER_COLOR,
    CENTER_HIT_PT,
    CENTER_RADIUS,
    CENTER_SHELL_RADIUS,
    GUIDE_CORE_COLOR,
    HOVER_COLOR,
    JOINT_HANDLE_COLOR,
    PLANE_INNER,
    RING_RADIUS,
    RING_SEGMENTS,
    RING_WIDTH_PT,
    ROTATE_AXIS_HANDLES,
    ROTATE_RING_ACTIVE_ALPHA,
    ROTATE_RING_ALPHA,
    ROTATE_RING_HOVER_ALPHA,
    SCREEN_RING_RADIUS,
    SIZE_PT,
    TRACKBALL_ACTIVE_ALPHA,
    TRACKBALL_ALPHA,
    TRACKBALL_COLOR,
    TRACKBALL_HOVER_ALPHA,
    TRACKBALL_RADIUS,
    GizmoFrame,
    GizmoHandle,
    GizmoMode,
    GizmoSpace,
    axis_active_color,
    axis_handle_alpha,
    axis_hover_color,
    axis_rotation,
    display_handles,
    handle_mask,
    handle_projection_alpha,
    hit_test,
    masked_axis_start,
    paint_order,
    plane_corners,
    plane_direction,
    plane_handle_alpha,
    project,
    rotation_dial,
    rotation_handle_color,
    rotation_ring,
    rotation_ring_alpha,
    rotation_ring_is_full,
    trackball_color,
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
    JOINT_ACTIVE_DARK_COLOR,
    JOINT_CURRENT_COLOR,
    JOINT_CURRENT_TICK_PT,
    JOINT_DRAG_START_TICK_HALF_PT,
    JOINT_LIMIT_TICK_PT,
    JOINT_LOWER_LIMIT_COLOR,
    JOINT_RANGE_COLOR,
    JOINT_RANGE_RADIUS,
    JOINT_RANGE_WIDTH_PT,
    JOINT_SLIDE_ARROW_OFFSET_PT,
    JOINT_UPPER_LIMIT_COLOR,
    TRACKBALL_RAD_PER_PT,
    TRANSLATION_GUIDE_RADIUS_PT,
    ObjectGizmo,
    _clip_line_to_rect,
    _clip_segment_to_rect,
    _joint_drag_label_color,
    _JointRangeState,
    _project_finite_axis_segment,
    _project_rotation_dial,
    _project_rotation_tick,
    _projected_line_parameters,
    _rotation_fill_alpha,
    _rotation_sweep,
    _RotationDialProjector,
    _ScreenRotationDialProjector,
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

    def text_size(self, text: str) -> tuple[float, float]:
        return (8.0 * len(text), 14.0)


@pytest.mark.parametrize(
    ("state", "expected_color", "expected_alpha"),
    (
        ("idle", AXIS_COLORS[0], ROTATE_RING_ALPHA),
        ("hovered", axis_hover_color(AXIS_COLORS[0]), ROTATE_RING_HOVER_ALPHA),
        ("active", axis_active_color(AXIS_COLORS[0]), ROTATE_RING_ACTIVE_ALPHA),
    ),
)
def test_rotation_ring_interaction_preserves_axis_hue_and_uses_transparency(
    state: str,
    expected_color,
    expected_alpha: float,
) -> None:
    frame = GizmoFrame(mode=GizmoMode.ROTATE)
    if state == "hovered":
        frame.hovered = GizmoHandle.ROTATE_X
    elif state == "active":
        frame.active = GizmoHandle.ROTATE_X

    color = rotation_handle_color(frame, GizmoHandle.ROTATE_X, 0)

    assert np.allclose(color[:3], expected_color[:3])
    assert color[0] > color[1] and color[0] > color[2]
    assert color[3] == pytest.approx(expected_alpha)
    assert color[3] < 1.0


def test_rotation_ring_color_respects_scalar_joint_override() -> None:
    frame = GizmoFrame(
        mode=GizmoMode.ROTATE,
        hovered=GizmoHandle.ROTATE_Z,
        handle_color=JOINT_HANDLE_COLOR,
    )

    color = rotation_handle_color(frame, GizmoHandle.ROTATE_Z, 2, 0.5)

    assert np.allclose(color[:3], axis_hover_color(JOINT_HANDLE_COLOR)[:3])
    assert color[3] == pytest.approx(0.5 * ROTATE_RING_HOVER_ALPHA)


def test_pressed_scalar_joint_rotation_ring_keeps_its_semantic_purple() -> None:
    frame = GizmoFrame(
        mode=GizmoMode.ROTATE,
        active=GizmoHandle.ROTATE_Z,
        handle_color=JOINT_HANDLE_COLOR,
    )

    color = rotation_handle_color(frame, GizmoHandle.ROTATE_Z, 2)

    assert np.allclose(color[:3], ACTIVE_HANDLE_COLOR[:3])
    assert color[0] > color[1] and color[2] > color[1]
    assert color[3] == pytest.approx(ROTATE_RING_ACTIVE_ALPHA)


@pytest.mark.parametrize(
    ("interaction", "alpha"),
    (
        ("idle", TRACKBALL_ALPHA),
        ("hovered", TRACKBALL_HOVER_ALPHA),
        ("active", TRACKBALL_ACTIVE_ALPHA),
    ),
)
def test_trackball_feedback_uses_neutral_gray_for_every_interaction(interaction, alpha) -> None:
    frame = GizmoFrame(
        mode=GizmoMode.ROTATE,
        hovered=(GizmoHandle.ROTATE_TRACKBALL if interaction == "hovered" else GizmoHandle.NONE),
        active=GizmoHandle.ROTATE_TRACKBALL if interaction == "active" else GizmoHandle.NONE,
    )

    color = trackball_color(frame)

    assert np.allclose(color[:3], TRACKBALL_COLOR[:3])
    assert np.all(color[:3] < CENTER_COLOR[:3])
    assert color[3] == pytest.approx(alpha)


def test_ball_rotation_value_label_uses_the_active_axis_color() -> None:
    cam = camera()
    gizmo = ObjectGizmo("rotate")
    gizmo._active = GizmoHandle.ROTATE_Z
    gizmo._active_joint = object()
    gizmo._axis[:] = (0.0, 0.0, 1.0)
    gizmo._rotation_angle = np.radians(15.0)
    gizmo._label = "Z +15.0°"
    dial = _RotationDialProjector(
        cam,
        RECT,
        np.zeros(3),
        gizmo._axis,
        np.array((1.0, 0.0, 0.0)),
        SIZE_PT,
    )
    overlay = RecordingDraw2D()

    gizmo._draw_value_label(overlay, cam, RECT, 1.0, dial)

    dot_color = next(args[2] for name, args, _kwargs in overlay.calls if name == "circle_filled")
    assert np.allclose(dot_color, axis_active_color(AXIS_COLORS[2]))


def test_trackball_value_label_has_no_joint_semantic_dot() -> None:
    gizmo = ObjectGizmo("rotate")
    gizmo._active = GizmoHandle.ROTATE_TRACKBALL
    gizmo._active_joint = object()
    gizmo._label = "Trackball +12.0° -4.0°"
    overlay = RecordingDraw2D()

    gizmo._draw_value_label(overlay, camera(), RECT, 1.0, None)

    assert not any(name == "circle_filled" for name, _args, _kwargs in overlay.calls)


@pytest.mark.parametrize(
    ("current", "expected"),
    (
        (-1.0, JOINT_LOWER_LIMIT_COLOR),
        (0.25, JOINT_CURRENT_COLOR),
        (1.0, JOINT_UPPER_LIMIT_COLOR),
    ),
)
def test_joint_drag_label_dot_uses_endpoint_semantics(current, expected) -> None:
    state = _JointRangeState("hinge", current, -1.0, 1.0)
    assert np.allclose(_joint_drag_label_color(state), expected)


@pytest.mark.parametrize("joint_type", ("hinge", "slide"))
@pytest.mark.parametrize(
    ("current", "expected"),
    ((-1.0, JOINT_LOWER_LIMIT_COLOR), (1.0, JOINT_UPPER_LIMIT_COLOR)),
)
def test_joint_current_tick_changes_to_the_reached_endpoint_color(
    joint_type: str,
    current: float,
    expected,
) -> None:
    gizmo = ObjectGizmo("rotate" if joint_type == "hinge" else "translate")
    gizmo._joint_range = _JointRangeState(joint_type, current, -1.0, 1.0)
    overlay = RecordingDraw2D()

    gizmo._draw_joint_range(overlay, camera(), RECT, 1.0)

    current_tick = next(
        args
        for name, args, _kwargs in overlay.calls
        if name == "line" and args[3] == pytest.approx(4.0)
    )
    assert np.allclose(current_tick[2], expected)


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
    ((0.0, 0.0), (90.0, 0.24), (150.0, 0.24), (180.0, 0.24), (247.0, 0.24)),
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
    )
    gizmo = ObjectGizmo("rotate")
    gizmo._active = GizmoHandle.ROTATE_Z
    gizmo._axis[:] = axis
    gizmo._rotation_angle = np.radians(5.0)
    overlay = RecordingDraw2D()

    gizmo._draw_rotation_snap_ticks(overlay, cam, RECT, 1.0, dial)

    lines = [args for name, args, _kwargs in overlay.calls if name == "line"]
    core = [args for args in lines if np.allclose(args[2], GUIDE_CORE_COLOR)]
    highlighted = [
        args for args in lines if np.allclose(args[2], axis_active_color(AXIS_COLORS[2]))
    ]
    assert core
    assert len(highlighted) == 1
    start, end = highlighted[0][:2]
    assert np.linalg.norm(end - start) == pytest.approx(4.0 * DEFAULT_ROTATION_TICK_SCALE)


def test_rotation_snap_does_not_duplicate_the_limited_hinge_current_tick() -> None:
    cam = camera()
    axis = np.array((0.0, 0.0, 1.0))
    dial = _RotationDialProjector(
        cam,
        RECT,
        np.zeros(3),
        axis,
        np.array((1.0, 0.0, 0.0)),
        SIZE_PT,
    )
    gizmo = ObjectGizmo("rotate")
    gizmo._active = GizmoHandle.ROTATE_Z
    gizmo._axis[:] = axis
    gizmo._rotation_angle = np.radians(5.0)
    gizmo._joint_range = _JointRangeState(
        "hinge",
        np.radians(20.0),
        np.radians(-60.0),
        np.radians(60.0),
    )
    overlay = RecordingDraw2D()

    gizmo._draw_joint_range(overlay, cam, RECT, 1.0)
    gizmo._draw_rotation_snap_ticks(overlay, cam, RECT, 1.0, dial)

    lines = [args for name, args, _kwargs in overlay.calls if name == "line"]
    snap_ticks = [
        args
        for args in lines
        if np.allclose(args[2], ACTIVE_HANDLE_COLOR) and args[3] == pytest.approx(1.1)
    ]
    current_ticks = [
        args
        for args in lines
        if np.allclose(args[2], ACTIVE_HANDLE_COLOR) and args[3] == pytest.approx(4.0)
    ]
    assert snap_ticks
    assert len(current_ticks) == 1


def test_rotation_snap_does_not_add_primary_current_tick_for_a_joint() -> None:
    cam = camera()
    axis = np.array((0.0, 0.0, 1.0))
    dial = _RotationDialProjector(
        cam,
        RECT,
        np.zeros(3),
        axis,
        np.array((1.0, 0.0, 0.0)),
        SIZE_PT,
    )
    gizmo = ObjectGizmo("rotate")
    gizmo._active = GizmoHandle.ROTATE_Z
    gizmo._active_joint = object()
    gizmo._axis[:] = axis
    overlay = RecordingDraw2D()

    gizmo._draw_rotation_snap_ticks(overlay, cam, RECT, 1.0, dial)

    assert not any(
        name == "line" and args[3] == pytest.approx(2.2) for name, args, _kwargs in overlay.calls
    )


@pytest.mark.parametrize(
    ("degrees", "reachable"),
    ((0.0, True), (240.0, True), (260.0, True), (270.0, False), (330.0, True)),
)
def test_joint_range_membership_wraps_across_zero(
    degrees: float,
    reachable: bool,
) -> None:
    joint_range = _JointRangeState(
        "hinge",
        np.radians(70.0),
        np.radians(-30.0),
        np.radians(260.0),
    )
    assert joint_range.contains_angle(np.radians(degrees)) is reachable


def test_rotation_snap_ticks_are_clipped_to_the_reachable_hinge_arc() -> None:
    cam = camera()
    axis = np.array((0.0, 0.0, 1.0))
    dial = _RotationDialProjector(
        cam,
        RECT,
        np.zeros(3),
        axis,
        np.array((0.0, 1.0, 0.0)),
        SIZE_PT,
    )
    gizmo = ObjectGizmo("rotate")
    gizmo._active = GizmoHandle.ROTATE_Z
    gizmo._axis[:] = axis
    gizmo.rotation_snap_deg = 30.0
    gizmo._joint_range = _JointRangeState(
        "hinge",
        np.radians(70.0),
        np.radians(-30.0),
        np.radians(260.0),
    )
    overlay = RecordingDraw2D()

    gizmo._draw_rotation_snap_ticks(overlay, cam, RECT, 1.0, dial)

    snap_ticks = [
        args
        for name, args, _kwargs in overlay.calls
        if name == "line" and np.allclose(args[2], ACTIVE_HANDLE_COLOR)
    ]
    assert len(snap_ticks) == 10
    expected_angles = np.radians((*range(0, 270, 30), 330))
    stable_dial = _RotationDialProjector(
        cam,
        RECT,
        np.zeros(3),
        axis,
        np.array((1.0, 0.0, 0.0)),
        SIZE_PT,
    )
    assert np.asarray([tick[0] for tick in snap_ticks]) == pytest.approx(
        stable_dial.points(RING_RADIUS, expected_angles)[:, :2]
    )
    for ring, outside, *_rest in snap_ticks:
        center = stable_dial.points(0.0, (0.0,))[0, :2]
        assert np.linalg.norm(outside - center) > np.linalg.norm(ring - center)


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


def test_precise_axis_input_applies_a_body_frame_delta_and_one_undo() -> None:
    rotation = math3d.rotvec_to_mat3(np.array((0.0, 0.0, np.pi / 2.0)))
    session, node = session_at(rotation=rotation)
    gizmo = ObjectGizmo()
    gizmo._hovered = GizmoHandle.X

    edit = gizmo.precise_input(session)
    assert edit is not None
    assert (edit.action, edit.label, edit.unit) == ("Move", "X", "m")
    assert gizmo.apply_precise_value(session, camera(), edit, 0.125)

    expected = rotation[:, 0] * 0.125
    assert session.frame.body_xpos[node.body_index] == pytest.approx(expected, abs=1e-7)
    assert session.can_undo
    assert session.submit(cmd.Undo())
    assert session.frame.body_xpos[node.body_index] == pytest.approx(np.zeros(3), abs=1e-7)
    assert not session.can_undo


def test_precise_rotation_input_uses_degrees_and_preserves_position() -> None:
    session, node = session_at(position=(0.2, -0.4, 0.6))
    gizmo = ObjectGizmo("rotate")
    gizmo._hovered = GizmoHandle.ROTATE_Z

    edit = gizmo.precise_input(session)
    assert edit is not None
    assert (edit.action, edit.label, edit.unit) == ("Rotate", "Z", "°")
    assert gizmo.apply_precise_value(session, camera(), edit, -12.5)

    expected = math3d.rotvec_to_mat3(np.array((0.0, 0.0, np.radians(-12.5))))
    assert session.frame.body_xpos[node.body_index] == pytest.approx((0.2, -0.4, 0.6))
    assert session.frame.body_xmat[node.body_index] == pytest.approx(expected, abs=1e-6)


def test_precise_input_is_only_available_for_scalar_handles() -> None:
    session, _node = session_at()
    gizmo = ObjectGizmo()

    for handle in (GizmoHandle.SCREEN, GizmoHandle.XY, GizmoHandle.YZ, GizmoHandle.ZX):
        gizmo._hovered = handle
        assert gizmo.precise_input(session) is None


def test_precise_world_translation_accepts_an_absolute_axis_value() -> None:
    session, node = session_at(position=(0.2, -0.4, 0.6))
    gizmo = ObjectGizmo()
    gizmo.set_space("world")
    gizmo._hovered = GizmoHandle.X

    edit = gizmo.precise_input(session)

    assert edit is not None
    assert edit.absolute_value == pytest.approx(0.2)
    assert edit.absolute_label == "target world X position"
    assert gizmo.apply_precise_value(session, camera(), edit, 1.25, absolute=True)
    assert session.frame.body_xpos[node.body_index] == pytest.approx((1.25, -0.4, 0.6))


def test_precise_world_rotation_accepts_an_absolute_euler_component() -> None:
    initial = np.radians((10.0, 20.0, 30.0))
    session, node = session_at(rotation=math3d.euler_xyz_to_mat3(initial))
    gizmo = ObjectGizmo("rotate")
    gizmo.set_space("world")
    gizmo._hovered = GizmoHandle.ROTATE_Y

    edit = gizmo.precise_input(session)

    assert edit is not None
    assert edit.absolute_value == pytest.approx(20.0, abs=1e-5)
    assert edit.absolute_label == "target world Y rotation"
    assert gizmo.apply_precise_value(session, camera(), edit, -40.0, absolute=True)
    expected = math3d.euler_xyz_to_mat3(np.radians((10.0, -40.0, 30.0)))
    assert session.frame.body_xmat[node.body_index] == pytest.approx(expected, abs=1e-6)


def test_precise_body_frame_input_does_not_claim_an_ambiguous_absolute_value() -> None:
    session, _node = session_at(rotation=math3d.euler_xyz_to_mat3(np.radians((20.0, 0.0, 0.0))))
    gizmo = ObjectGizmo("rotate")
    gizmo._hovered = GizmoHandle.ROTATE_X

    edit = gizmo.precise_input(session)

    assert edit is not None
    assert edit.absolute_value is None
    result = gizmo.apply_precise_value(session, camera(), edit, 45.0, absolute=True)
    assert not result.ok
    assert "unavailable" in result.message


def test_clicking_a_gizmo_without_motion_does_not_create_an_undo_record() -> None:
    session, _node = session_at()
    gizmo = ObjectGizmo()
    cam = camera()
    scale = world_scale(cam, np.zeros(3), RECT[3])
    cursor = project(cam, (np.array((0.55 * scale, 0.0, 0.0)),), RECT)[0, :2]

    assert gizmo._begin_handle(session, cam, RECT, cursor, GizmoHandle.X)
    assert gizmo._drag(session, cam, RECT, cursor, snap=False)
    gizmo.cancel()

    assert not session.can_undo


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


def test_rotation_inner_disk_is_a_background_trackball_hit() -> None:
    cam = camera()
    origin = np.zeros(3)
    rotation = np.eye(3)
    center = project(cam, (origin,), RECT)[0, :2]

    assert hit_test(cam, origin, rotation, RECT, center, GizmoMode.ROTATE)[0] is (
        GizmoHandle.ROTATE_TRACKBALL
    )

    scale = world_scale(cam, origin, RECT[3])
    ring_points = project(
        cam,
        rotation_ring(cam, origin, rotation, scale, 2, full=False),
        RECT,
    )[:, :2]
    axis_point = next(
        point
        for point in ring_points
        if np.linalg.norm(point - center) <= TRACKBALL_RADIUS * SIZE_PT
        and hit_test(cam, origin, rotation, RECT, point, GizmoMode.ROTATE)[0] in ROTATE_AXIS_HANDLES
    )
    axis_hit = hit_test(cam, origin, rotation, RECT, axis_point, GizmoMode.ROTATE)[0]
    assert axis_hit in ROTATE_AXIS_HANDLES


def test_trackball_drag_matches_blender_view_axis_mapping() -> None:
    session, node = session_at()
    gizmo = ObjectGizmo("rotate")
    cam = camera()
    start = project(cam, (np.zeros(3),), RECT)[0, :2]
    screen_delta = np.array((24.0, -13.0))
    end = start + screen_delta

    assert gizmo.update_hover(session, cam, RECT, tuple(start)) is (GizmoHandle.ROTATE_TRACKBALL)
    assert gizmo.interact(
        session,
        cam,
        RECT,
        tuple(start),
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
    )

    angles = np.array((screen_delta[1], screen_delta[0])) * TRACKBALL_RAD_PER_PT
    view_basis = np.asarray(cam.view_matrix(), np.float64)[:3, :3].T
    rotvec = view_basis[:, 0] * angles[0] + view_basis[:, 1] * angles[1]
    expected = math3d.rotvec_to_mat3(rotvec)
    assert session.frame.body_xmat[node.body_index] == pytest.approx(expected, abs=1e-6)
    assert gizmo.value_label.startswith("Trackball ")


def test_trackball_does_not_offer_scalar_precise_input() -> None:
    session, _node = session_at()
    gizmo = ObjectGizmo("rotate")
    gizmo._hovered = GizmoHandle.ROTATE_TRACKBALL

    assert gizmo.precise_input(session) is None


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


@pytest.mark.parametrize(
    ("mode", "handle", "eye"),
    (
        (GizmoMode.TRANSLATE, GizmoHandle.Z, (0.0, 0.0, 5.0)),
        (GizmoMode.ROTATE, GizmoHandle.ROTATE_Z, (5.0, 0.0, 0.0)),
    ),
)
def test_joint_active_handle_keeps_projection_degeneracy_fade(
    mode: GizmoMode,
    handle: GizmoHandle,
    eye: tuple[float, float, float],
) -> None:
    up = (0.0, 1.0, 0.0) if mode is GizmoMode.TRANSLATE else (0.0, 0.0, 1.0)
    cam = CameraView(
        eye=np.asarray(eye),
        target=np.zeros(3),
        up=np.asarray(up),
        aspect=RECT[2] / RECT[3],
    )
    frame = GizmoFrame(mode=mode, active=handle)
    direction = frame.rotation[:, 2]

    assert handle_projection_alpha(frame, handle, cam, frame.position, direction) == 1.0
    frame.active_projection_fade = True
    assert handle_projection_alpha(frame, handle, cam, frame.position, direction) == 0.0


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
        GizmoHandle.ROTATE_TRACKBALL,
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

    joint_node = SceneNode(
        3,
        "yaw",
        NodeType.JOINT,
        body_index=2,
        joint_index=joints[0].joint_id,
    )
    session.selected_node = joint_node
    target, reason = gizmo._joint_target(session, joint_node)
    assert not reason
    assert target is not None and target.joint is joints[0]
    assert gizmo.selected_joint_id(joint_node.body_index) == joints[0].joint_id


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


def test_translation_snap_ruler_is_centered_on_the_current_arrow_axis() -> None:
    cam = camera()
    gizmo = ObjectGizmo()
    gizmo._active = GizmoHandle.Z
    gizmo._using = True
    gizmo._snapping = True
    gizmo.translation_snap_m = 0.25
    gizmo._start_pos[:] = 0.0
    gizmo._start_basis[:] = np.eye(3)
    gizmo._frame.position[:] = (0.08, -0.05, 0.5)
    gizmo._frame.rotation[:] = np.eye(3)
    overlay = RecordingDraw2D()

    gizmo._draw_translation_snap_ruler(overlay, cam, RECT, 1.0)

    axis_lines = [
        args
        for name, args, _kwargs in overlay.calls
        if name == "line" and args[3] == pytest.approx(1.2) and args[2][3] == pytest.approx(0.92)
    ]
    ruler = max(axis_lines, key=lambda args: np.linalg.norm(args[1] - args[0]))
    arrow_axis = project(
        cam,
        (
            gizmo._frame.position,
            gizmo._frame.position + gizmo._frame.rotation[:, 2],
        ),
        RECT,
    )[:, :2]
    direction = arrow_axis[1] - arrow_axis[0]
    direction /= np.linalg.norm(direction)
    for point in ruler[:2]:
        offset = np.asarray(point) - arrow_axis[0]
        assert abs(direction[0] * offset[1] - direction[1] * offset[0]) < 1e-6


def test_translation_snap_ruler_does_not_add_primary_current_tick_for_a_joint() -> None:
    gizmo = ObjectGizmo()
    gizmo._active = GizmoHandle.Z
    gizmo._active_joint = object()
    gizmo.translation_snap_m = 0.25
    gizmo._start_basis[:] = np.eye(3)
    gizmo._frame.position[:] = (0.0, 0.0, 0.5)
    overlay = RecordingDraw2D()

    gizmo._draw_translation_snap_ruler(overlay, camera(), RECT, 1.0)

    assert not any(
        name == "line" and np.allclose(args[2], HOVER_COLOR)
        for name, args, _kwargs in overlay.calls
    )


@pytest.mark.parametrize("style_scale", (1.0, 2.0, 4.0))
def test_slide_joint_snap_ruler_is_one_continuous_final_axis_overlay(style_scale: float) -> None:
    from types import SimpleNamespace

    gizmo = ObjectGizmo()
    gizmo._active = GizmoHandle.Z
    gizmo._active_joint = SimpleNamespace(type="slide")
    gizmo.translation_snap_m = 0.25
    gizmo._start_basis[:] = np.eye(3)
    gizmo._frame.position[:] = (0.0, 0.0, 0.5)
    overlay = RecordingDraw2D()

    cam = camera()
    gizmo._draw_translation_snap_ruler(overlay, cam, RECT, style_scale)

    axis_lines = [
        args
        for name, args, _kwargs in overlay.calls
        if name == "line"
        and args[3] == pytest.approx(1.2 * style_scale)
        and args[2][3] == pytest.approx(0.92)
    ]
    assert len(axis_lines) == 1
    center = project(cam, (gizmo._frame.position,), RECT)[0, :2]
    direction = project(
        cam,
        (gizmo._frame.position, gizmo._frame.position + gizmo._start_basis[:, 2]),
        RECT,
    )[:, :2]
    direction = direction[1] - direction[0]
    direction /= np.linalg.norm(direction)
    along = sorted(
        float(np.dot(np.asarray(point) - center, direction)) for point in axis_lines[0][:2]
    )
    assert along[0] < 0.0 < along[1]
    arrow_scale = world_scale(cam, gizmo._frame.position, RECT[3], SIZE_PT * style_scale)
    arrow_tip = project(
        cam,
        (gizmo._frame.position + gizmo._start_basis[:, 2] * arrow_scale * AXIS_END,),
        RECT,
    )[0, :2]
    arrow_extent = float(np.linalg.norm(arrow_tip - center))
    positive_tick_offsets = []
    for name, args, _kwargs in overlay.calls:
        if name != "line" or args[3] != pytest.approx(1.2 * style_scale):
            continue
        tick_direction = np.asarray(args[1]) - np.asarray(args[0])
        if abs(float(np.dot(tick_direction, direction))) > 1e-6:
            continue
        midpoint = (np.asarray(args[0]) + np.asarray(args[1])) * 0.5
        positive_tick_offsets.append(float(np.dot(midpoint - center, direction)))
    assert any(0.0 < offset < arrow_extent for offset in positive_tick_offsets)


def test_position_snap_ruler_keeps_the_3d_arrow_owned_interval() -> None:
    gizmo = ObjectGizmo()
    gizmo._active = GizmoHandle.Z
    gizmo.translation_snap_m = 0.25
    gizmo._start_basis[:] = np.eye(3)
    gizmo._frame.position[:] = (0.0, 0.0, 0.5)
    overlay = RecordingDraw2D()

    gizmo._draw_translation_snap_ruler(overlay, camera(), RECT, 1.0)

    axis_lines = [
        args
        for name, args, _kwargs in overlay.calls
        if name == "line" and args[3] == pytest.approx(1.2) and args[2][3] == pytest.approx(0.92)
    ]
    assert len(axis_lines) == 2


def test_translation_snap_guide_draws_smaller_endpoints_above_its_connector() -> None:
    gizmo = ObjectGizmo()
    gizmo._drag_origin_pos[:] = (0.0, 0.0, 0.0)
    gizmo._frame.position[:] = (0.0, 0.0, 0.5)
    overlay = RecordingDraw2D()

    gizmo._draw_translation_guide(overlay, camera(), RECT, 1.0)

    rings = [call for call in overlay.calls if call[0] == "circle"]
    dots = [call for call in overlay.calls if call[0] == "circle_filled"]
    assert [call[0] for call in overlay.calls] == [
        "line",
        "circle",
        "circle_filled",
        "line",
        "circle",
        "circle_filled",
    ]
    assert len(rings) == 2
    assert len(dots) == 2
    assert all(call[1][1] == pytest.approx(TRANSLATION_GUIDE_RADIUS_PT) for call in rings)
    assert dots[-1][1][1] == pytest.approx(TRANSLATION_GUIDE_RADIUS_PT)


def test_joint_translation_guide_is_a_dark_purple_segment_with_asymmetric_ticks() -> None:
    gizmo = ObjectGizmo()
    gizmo._drag_origin_pos[:] = (0.0, 0.0, 0.0)
    gizmo._frame.position[:] = (0.0, 0.0, 0.5)
    overlay = RecordingDraw2D()

    gizmo._draw_joint_translation_guide(overlay, camera(), RECT, 1.0)

    assert [name for name, _args, _kwargs in overlay.calls] == ["line"] * 3
    connector, start_tick, end_tick = (args for _name, args, _kwargs in overlay.calls)
    assert np.allclose(connector[2], JOINT_ACTIVE_DARK_COLOR)
    assert np.allclose(start_tick[2], JOINT_ACTIVE_DARK_COLOR)
    assert np.allclose(end_tick[2], ACTIVE_HANDLE_COLOR)
    assert connector[3] == pytest.approx(JOINT_RANGE_WIDTH_PT)
    assert np.linalg.norm(start_tick[1] - start_tick[0]) == pytest.approx(
        2.0 * JOINT_DRAG_START_TICK_HALF_PT
    )
    assert np.linalg.norm(end_tick[1] - end_tick[0]) == pytest.approx(JOINT_CURRENT_TICK_PT)
    assert end_tick[3] == pytest.approx(4.0)


def test_joint_translation_guide_stays_in_the_final_overlay_without_dots() -> None:
    from types import SimpleNamespace

    gizmo = ObjectGizmo()
    gizmo._visible = True
    gizmo._using = True
    gizmo._active = GizmoHandle.Z
    gizmo._active_joint = SimpleNamespace(type="slide")
    gizmo._frame.active = GizmoHandle.Z
    gizmo._frame.position[:] = (0.0, 0.0, 0.5)
    gizmo._drag_origin_pos[:] = 0.0
    overlay = RecordingDraw2D()

    gizmo.draw_overlay(camera(), RECT, overlay, style_scale=1.0)

    assert not any(name in {"circle", "circle_filled"} for name, _args, _kwargs in overlay.calls)
    assert [name for name, _args, _kwargs in overlay.calls[-3:]] == ["line"] * 3


def test_translation_snap_guide_is_the_last_axis_overlay_group() -> None:
    gizmo = ObjectGizmo()
    gizmo._visible = True
    gizmo._using = True
    gizmo._snapping = True
    gizmo._active = GizmoHandle.Z
    gizmo._frame.active = GizmoHandle.Z
    gizmo._frame.position[:] = (0.0, 0.0, 0.5)
    gizmo._start_basis[:] = np.eye(3)
    gizmo._drag_origin_pos[:] = 0.0
    overlay = RecordingDraw2D()

    gizmo.draw_overlay(camera(), RECT, overlay, style_scale=1.0)

    assert [call[0] for call in overlay.calls[-6:]] == [
        "line",
        "circle",
        "circle_filled",
        "line",
        "circle",
        "circle_filled",
    ]


def test_translation_snap_guide_does_not_stay_in_the_gpu_layer() -> None:
    class Layer:
        def __init__(self) -> None:
            self.cleared = False

        def clear(self) -> None:
            self.cleared = True

    class Debug:
        def __init__(self) -> None:
            self.value = Layer()

        def layer(self, *_args, **_kwargs):
            return self.value

    class Backend:
        caps = BackendCaps(name="test", debug_draw=True)
        debug = Debug()

    gizmo = ObjectGizmo()
    gizmo._using = True
    gizmo._snapping = True
    gizmo._active = GizmoHandle.Z
    gizmo._guide_gpu = True

    gizmo._publish_translation_guide(Backend(), 1.0)

    assert not gizmo._guide_gpu
    assert Backend.debug.value.cleared


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
    assert gizmo.translation_snap_m == DEFAULT_TRANSLATION_SNAP_M == 0.1
    assert gizmo.rotation_snap_deg == DEFAULT_ROTATION_SNAP_DEG == 5.0
    assert gizmo.rotation_tick_scale == DEFAULT_ROTATION_TICK_SCALE == 1.25
    assert gizmo.remember_precise_input_choices


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
        )
        for _ in range(3)
    ]
    assert coincident[0] == pytest.approx(coincident[1], abs=1e-6)
    assert coincident[1] == pytest.approx(coincident[2], abs=1e-6)


@pytest.mark.parametrize("orthographic", [False, True], ids=("perspective", "orthographic"))
@pytest.mark.parametrize("axis_index", range(3), ids=("x", "y", "z"))
def test_active_axis_dial_matches_the_idle_full_ring(
    orthographic: bool,
    axis_index: int,
) -> None:
    cam = camera(orthographic=orthographic)
    center = np.array((0.35, -0.2, 0.4))
    rotation = math3d.rotvec_to_mat3(np.array((0.3, -0.2, 0.4)))
    basis = axis_rotation(rotation, axis_index)
    scale = world_scale(cam, center, RECT[3], SIZE_PT)
    angles = np.linspace(0.0, 2.0 * np.pi, RING_SEGMENTS, endpoint=False)
    idle = project(
        cam,
        rotation_ring(cam, center, rotation, scale, axis_index, full=True),
        RECT,
    )
    active = _RotationDialProjector(
        cam,
        RECT,
        center,
        basis[:, 2],
        basis[:, 0],
        SIZE_PT,
    ).points(RING_RADIUS, angles)

    assert active == pytest.approx(idle, abs=1e-6)


@pytest.mark.parametrize("distance", [4.0, 12.0], ids=("near", "far"))
def test_rotation_tick_keeps_its_screen_length_across_camera_distance(
    distance: float,
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
    active = _RotationDialProjector(
        cam,
        RECT,
        center,
        -cam.forward(),
        view_basis[:, 0],
        SIZE_PT * 1.5,
    ).points(SCREEN_RING_RADIUS, angles)
    projected_center = project(cam, (center,), RECT)[0, :2]
    radii = np.linalg.norm(points[:, :2] - projected_center, axis=1)
    expected = SCREEN_RING_RADIUS * SIZE_PT * 1.5
    assert radii == pytest.approx(np.full(len(angles), expected), abs=1e-5)
    assert active == pytest.approx(points, abs=1e-6)


@pytest.mark.parametrize("orthographic", [False, True], ids=("perspective", "orthographic"))
def test_active_screen_rotation_matches_the_idle_pixel_ring_with_stale_aspect(
    orthographic: bool,
) -> None:
    cam = camera(orthographic=orthographic).with_aspect(0.75)
    center = np.array((0.35, -0.2, 0.4))
    view_basis = np.asarray(cam.view_matrix())[:3, :3].T
    angles = np.linspace(0.0, 2.0 * np.pi, RING_SEGMENTS, endpoint=False)
    dial = _ScreenRotationDialProjector(
        cam,
        RECT,
        center,
        -cam.forward(),
        view_basis[:, 0],
        SIZE_PT * 1.5,
    )

    points = dial.points(SCREEN_RING_RADIUS, angles)
    projected_center = project(cam, (center,), RECT)[0, :2]
    radii = np.linalg.norm(points[:, :2] - projected_center, axis=1)

    assert radii == pytest.approx(
        np.full(len(angles), SCREEN_RING_RADIUS * SIZE_PT * 1.5),
        abs=1e-6,
    )


@pytest.mark.parametrize("orthographic", [False, True], ids=("perspective", "orthographic"))
def test_screen_rotation_overlay_reference_ring_does_not_change_on_press(
    orthographic: bool,
) -> None:
    cam = camera(orthographic=orthographic).with_aspect(0.75)
    center = np.array((0.35, -0.2, 0.4))
    view_basis = np.asarray(cam.view_matrix())[:3, :3].T
    gizmo = ObjectGizmo("rotate")
    gizmo.set_style("3d")
    gizmo._visible = True
    gizmo._using = True
    gizmo._active = GizmoHandle.ROTATE_SCREEN
    gizmo._start_pos[:] = center
    gizmo._axis[:] = -cam.forward()
    gizmo._rotation_start_vec[:] = view_basis[:, 0]
    overlay = RecordingDraw2D()

    gizmo.draw_overlay(cam, RECT, overlay, style_scale=1.0)

    reference = next(
        args[0]
        for name, args, kwargs in overlay.calls
        if name == "polyline" and kwargs.get("closed") is True
    )
    projected_center = project(cam, (center,), RECT)[0, :2]
    radii = np.linalg.norm(np.asarray(reference)[:, :2] - projected_center, axis=1)
    assert radii == pytest.approx(
        np.full(len(radii), SCREEN_RING_RADIUS * SIZE_PT),
        abs=1e-6,
    )


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


def test_finite_segment_clip_does_not_extend_past_its_world_endpoints() -> None:
    segment = _clip_segment_to_rect((20.0, 50.0), (80.0, 50.0), RECT)
    assert segment is not None
    assert segment[0] == pytest.approx((20.0, 50.0))
    assert segment[1] == pytest.approx((80.0, 50.0))

    clipped = _clip_segment_to_rect((-20.0, 50.0), (80.0, 50.0), RECT)
    assert clipped is not None
    assert clipped[0] == pytest.approx((0.0, 50.0))
    assert clipped[1] == pytest.approx((80.0, 50.0))


def test_rotation_axis_guide_is_long_for_multi_axis_rotation_and_short_for_hinge() -> None:
    cam = CameraView(
        eye=np.array((0.0, -5.0, 0.0)),
        target=np.zeros(3),
        up=np.array((0.0, 0.0, 1.0)),
        aspect=RECT[2] / RECT[3],
    )
    gizmo = ObjectGizmo("rotate")
    gizmo._active = GizmoHandle.ROTATE_Z
    gizmo._axis[:] = (0.0, 0.0, 1.0)

    overlay = RecordingDraw2D()
    gizmo._draw_rotation_axis_guide(overlay, cam, RECT, 1.0)
    long_line = next(args for name, args, _kwargs in overlay.calls if name == "line")
    long_length = float(np.linalg.norm(long_line[1] - long_line[0]))
    assert long_length > RECT[3] * 0.8
    assert np.allclose(long_line[2][:3], axis_active_color(AXIS_COLORS[2])[:3])

    gizmo._joint_range = _JointRangeState("hinge", 0.0, -1.0, 1.0)
    overlay = RecordingDraw2D()
    gizmo._draw_rotation_axis_guide(overlay, cam, RECT, 1.0)
    short_line = next(args for name, args, _kwargs in overlay.calls if name == "line")
    short_length = float(np.linalg.norm(short_line[1] - short_line[0]))
    assert short_length < long_length * 0.5
    assert np.allclose(short_line[2][:3], ACTIVE_HANDLE_COLOR[:3])


def test_rotation_axis_guide_fades_out_at_a_view_aligned_extreme() -> None:
    cam = CameraView(
        eye=np.array((0.0, 0.0, 5.0)),
        target=np.zeros(3),
        up=np.array((0.0, 1.0, 0.0)),
        aspect=RECT[2] / RECT[3],
    )
    segment = _project_finite_axis_segment(
        cam,
        np.zeros(3),
        (0.0, 0.0, 1.0),
        10.0,
        RECT,
        inset=6.0,
    )
    assert segment is not None
    assert np.linalg.norm(segment[1] - segment[0]) < 1e-6

    gizmo = ObjectGizmo("rotate")
    gizmo._active = GizmoHandle.ROTATE_Z
    gizmo._axis[:] = (0.0, 0.0, 1.0)
    overlay = RecordingDraw2D()
    gizmo._draw_rotation_axis_guide(overlay, cam, RECT, 1.0)
    assert overlay.calls == []


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
        assert gizmo.value_label.startswith(f"{joint.name} ")
    if joint.type == "slide":
        session.tick(FrameNeeds(poses=True, qpos=True, diagnostics=True), wall_dt=0.0)
        updated_pose = gizmo._target_pose(session, node, target)
        assert updated_pose is not None
        updated_position, updated_basis = updated_pose
        assert updated_position == pytest.approx(session.frame.body_xpos[node.body_index], abs=1e-7)
        assert updated_position == pytest.approx(position + basis[:, 2] * amount, abs=1e-6)
        assert updated_basis == pytest.approx(basis, abs=1e-6)


@pytest.mark.physics
def test_slide_joint_drag_rebases_at_a_clamped_limit() -> None:
    from forge_viewer.adapters.mujoco_adapter import MuJoCoAdapter
    from forge_viewer.assets import resolve

    adapter = MuJoCoAdapter(resolve("joint_types"))
    session = Session(adapter)
    assert session.submit(cmd.Pause())
    node = next(item for item in session.nodes if item.name == "slide_body")
    assert session.submit(cmd.Select(node.object_id))
    session.tick(FrameNeeds(poses=True, qpos=True, diagnostics=True), wall_dt=0.0)
    gizmo = ObjectGizmo()
    target, reason = gizmo._joint_target(session, node)
    assert target is not None, reason
    pose = gizmo._target_pose(session, node, target)
    assert pose is not None
    position, basis = pose
    cam = CameraView(eye=np.array((2.0, -4.0, 2.0)), target=position.copy())
    start = project(cam, (position + basis[:, 2] * 0.1,), RECT)[0, :2]
    assert gizmo._begin_handle(session, cam, RECT, start, GizmoHandle.Z)
    assert gizmo.interact(
        session,
        cam,
        RECT,
        tuple(start),
        claimed=True,
        left_down=True,
        released=False,
    )
    drag_origin = gizmo._drag_origin_pos.copy()

    upper = target.joint.range[1]
    first = start + gizmo._axis_screen * ((upper + 0.2) / gizmo._world_per_pt)
    assert gizmo.interact(
        session,
        cam,
        RECT,
        tuple(first),
        claimed=True,
        left_down=True,
        released=False,
    )
    assert adapter.data.qpos[target.joint.qpos_adr] == pytest.approx(upper)
    assert gizmo.using
    assert gizmo.active_handle is GizmoHandle.Z
    assert gizmo._drag_origin_pos == pytest.approx(drag_origin)
    assert np.linalg.norm(gizmo._start_pos - gizmo._drag_origin_pos) > 0.1
    farther = first + gizmo._axis_screen * (0.1 / gizmo._world_per_pt)
    assert gizmo.interact(
        session,
        cam,
        RECT,
        tuple(farther),
        claimed=True,
        left_down=True,
        released=False,
    )
    assert adapter.data.qpos[target.joint.qpos_adr] == pytest.approx(upper)
    assert gizmo.using
    assert gizmo._drag_origin_pos == pytest.approx(drag_origin)

    inward = farther - gizmo._axis_screen * (0.02 / gizmo._world_per_pt)
    assert gizmo.interact(
        session,
        cam,
        RECT,
        tuple(inward),
        claimed=True,
        left_down=True,
        released=False,
    )
    assert adapter.data.qpos[target.joint.qpos_adr] == pytest.approx(upper - 0.02)
    assert gizmo.using

    assert not gizmo.interact(
        session,
        cam,
        RECT,
        tuple(inward),
        claimed=True,
        left_down=False,
        released=True,
    )
    assert not gizmo.using


@pytest.mark.physics
def test_hinge_joint_drag_rebases_at_a_clamped_limit() -> None:
    from forge_viewer.adapters.mujoco_adapter import MuJoCoAdapter
    from forge_viewer.assets import resolve

    adapter = MuJoCoAdapter(resolve("joint_types"))
    session = Session(adapter)
    assert session.submit(cmd.Pause())
    node = next(item for item in session.nodes if item.name == "hinge_body")
    assert session.submit(cmd.Select(node.object_id))
    session.tick(FrameNeeds(poses=True, qpos=True, diagnostics=True), wall_dt=0.0)
    gizmo = ObjectGizmo()
    target, reason = gizmo._joint_target(session, node)
    assert target is not None, reason
    pose = gizmo._target_pose(session, node, target)
    assert pose is not None
    position, basis = pose
    cam = CameraView(eye=np.array((2.0, -4.0, 2.0)), target=position.copy())
    scale = world_scale(cam, position, RECT[3], SIZE_PT)

    def cursor(angle: float) -> np.ndarray:
        world = (
            position
            + (basis[:, 0] * np.cos(angle) + basis[:, 1] * np.sin(angle)) * scale * RING_RADIUS
        )
        return project(cam, (world,), RECT)[0, :2]

    assert gizmo._begin_handle(session, cam, RECT, cursor(0.0), GizmoHandle.ROTATE_Z)
    assert gizmo.interact(
        session,
        cam,
        RECT,
        tuple(cursor(0.0)),
        claimed=True,
        left_down=True,
        released=False,
    )
    drag_origin = float(gizmo._joint_drag_origin_qpos[0])
    upper = target.joint.range[1]
    assert gizmo.interact(
        session,
        cam,
        RECT,
        tuple(cursor(upper + 0.2)),
        claimed=True,
        left_down=True,
        released=False,
    )
    assert adapter.data.qpos[target.joint.qpos_adr] == pytest.approx(upper)
    assert gizmo.using
    assert gizmo.active_handle is GizmoHandle.ROTATE_Z
    assert gizmo.interact(
        session,
        cam,
        RECT,
        tuple(cursor(upper + 0.3)),
        claimed=True,
        left_down=True,
        released=False,
    )
    assert adapter.data.qpos[target.joint.qpos_adr] == pytest.approx(upper)
    assert gizmo.using
    assert gizmo._joint_drag_origin_qpos[0] == pytest.approx(drag_origin)

    session.tick(FrameNeeds(poses=True, qpos=True, diagnostics=True), wall_dt=0.0)
    gizmo._joint_range = gizmo._joint_range_state(session, target)
    assert gizmo._joint_range is not None
    dial = _RotationDialProjector(
        cam,
        RECT,
        gizmo._start_pos,
        gizmo._axis,
        gizmo._rotation_start_vec,
        SIZE_PT,
    )
    overlay = RecordingDraw2D()
    gizmo._draw_rotation_guide(overlay, cam, RECT, 1.0, dial)
    assert any(name == "triangle_fan_fill" for name, _args, _kwargs in overlay.calls)

    assert gizmo.interact(
        session,
        cam,
        RECT,
        tuple(cursor(upper + 0.25)),
        claimed=True,
        left_down=True,
        released=False,
    )
    assert adapter.data.qpos[target.joint.qpos_adr] == pytest.approx(upper - 0.05)
    assert gizmo.using

    assert not gizmo.interact(
        session,
        cam,
        RECT,
        tuple(cursor(upper + 0.25)),
        claimed=True,
        left_down=False,
        released=True,
    )
    assert not gizmo.using


@pytest.mark.physics
@pytest.mark.parametrize(
    ("body_name", "handle", "initial", "delta", "expected"),
    (
        ("hinge_body", GizmoHandle.ROTATE_Z, np.radians(-20.0), np.radians(10.0), "-10.0°"),
        ("slide_body", GizmoHandle.Z, -0.1, 0.05, "-0.050 m"),
    ),
)
def test_scalar_joint_drag_label_reports_the_absolute_current_value(
    body_name: str,
    handle: GizmoHandle,
    initial: float,
    delta: float,
    expected: str,
) -> None:
    from forge_viewer.adapters.mujoco_adapter import MuJoCoAdapter
    from forge_viewer.assets import resolve

    adapter = MuJoCoAdapter(resolve("joint_types"))
    session = Session(adapter)
    assert session.submit(cmd.Pause())
    node = next(item for item in session.nodes if item.name == body_name)
    assert session.submit(cmd.Select(node.object_id))
    gizmo = ObjectGizmo()
    target, reason = gizmo._joint_target(session, node)
    assert target is not None, reason
    assert session.submit(cmd.SetQpos(target.joint.qpos_adr, initial))
    session.tick(FrameNeeds(poses=True, qpos=True, diagnostics=True), wall_dt=0.0)
    pose = gizmo._target_pose(session, node, target)
    assert pose is not None
    position, basis = pose
    cam = CameraView(eye=np.array((2.0, -4.0, 2.0)), target=position.copy())

    if handle is GizmoHandle.ROTATE_Z:
        scale = world_scale(cam, position, RECT[3], SIZE_PT)

        def ring_cursor(angle: float) -> np.ndarray:
            world = (
                position
                + (basis[:, 0] * np.cos(angle) + basis[:, 1] * np.sin(angle)) * scale * RING_RADIUS
            )
            return project(cam, (world,), RECT)[0, :2]

        start = ring_cursor(0.0)
        end = ring_cursor(delta)
    else:
        start = project(cam, (position + basis[:, 2] * 0.1,), RECT)[0, :2]
        assert gizmo._begin_handle(session, cam, RECT, start, handle)
        end = start + gizmo._axis_screen * (delta / gizmo._world_per_pt)
        assert gizmo._drag(session, cam, RECT, end, snap=False)
        assert expected in gizmo.value_label
        return

    assert gizmo._begin_handle(session, cam, RECT, start, handle)
    assert gizmo._drag(session, cam, RECT, end, snap=False)
    assert expected in gizmo.value_label


@pytest.mark.physics
@pytest.mark.parametrize(
    ("body_name", "handle", "amount", "unit"),
    (
        ("hinge_body", GizmoHandle.ROTATE_Z, 1000.0, "°"),
        ("slide_body", GizmoHandle.Z, -1000.0, "m"),
    ),
)
def test_precise_joint_input_uses_display_units_and_clamps_to_the_range(
    body_name: str,
    handle: GizmoHandle,
    amount: float,
    unit: str,
) -> None:
    from forge_viewer.adapters.mujoco_adapter import MuJoCoAdapter
    from forge_viewer.assets import resolve

    adapter = MuJoCoAdapter(resolve("joint_types"))
    session = Session(adapter)
    assert session.submit(cmd.Pause())
    node = next(item for item in session.nodes if item.name == body_name)
    assert session.submit(cmd.Select(node.object_id))
    session.tick(FrameNeeds(poses=True, qpos=True, diagnostics=True), wall_dt=0.0)
    gizmo = ObjectGizmo()
    gizmo._hovered = handle

    edit = gizmo.precise_input(session)
    assert edit is not None
    assert edit.unit == unit
    assert edit.label
    target, reason = gizmo._joint_target(session, node)
    assert target is not None, reason
    assert gizmo.apply_precise_value(session, camera(), edit, amount)

    expected = target.joint.range[1] if amount > 0.0 else target.joint.range[0]
    assert adapter.data.qpos[target.joint.qpos_adr] == pytest.approx(expected)


@pytest.mark.physics
@pytest.mark.parametrize(
    ("body_name", "handle", "current", "requested", "expected_display"),
    (
        ("hinge_body", GizmoHandle.ROTATE_Z, np.radians(-20.0), 30.0, -20.0),
        ("slide_body", GizmoHandle.Z, -0.1, 0.2, -0.1),
    ),
)
def test_precise_joint_input_accepts_an_absolute_qpos(
    body_name: str,
    handle: GizmoHandle,
    current: float,
    requested: float,
    expected_display: float,
) -> None:
    from forge_viewer.adapters.mujoco_adapter import MuJoCoAdapter
    from forge_viewer.assets import resolve

    adapter = MuJoCoAdapter(resolve("joint_types"))
    session = Session(adapter)
    assert session.submit(cmd.Pause())
    node = next(item for item in session.nodes if item.name == body_name)
    assert session.submit(cmd.Select(node.object_id))
    target, reason = ObjectGizmo()._joint_target(session, node)
    assert target is not None, reason
    assert session.submit(cmd.SetQpos(target.joint.qpos_adr, current))
    session.tick(FrameNeeds(poses=True, qpos=True, diagnostics=True), wall_dt=0.0)
    gizmo = ObjectGizmo()
    gizmo._hovered = handle

    edit = gizmo.precise_input(session)

    assert edit is not None
    assert edit.absolute_value == pytest.approx(expected_display)
    assert "joint position" in edit.absolute_label
    assert gizmo.apply_precise_value(
        session,
        camera(),
        edit,
        requested,
        absolute=True,
    )
    expected = np.radians(requested) if target.joint.type == "hinge" else requested
    assert adapter.data.qpos[target.joint.qpos_adr] == pytest.approx(expected)


@pytest.mark.physics
@pytest.mark.parametrize(
    ("body_name", "handle", "current", "delta", "expected"),
    (
        (
            "hinge_body",
            GizmoHandle.ROTATE_Z,
            np.radians(-20.0),
            10.0,
            np.radians(-10.0),
        ),
        ("slide_body", GizmoHandle.Z, -0.1, 0.05, -0.05),
    ),
)
def test_precise_joint_relative_input_keeps_display_units(
    body_name: str,
    handle: GizmoHandle,
    current: float,
    delta: float,
    expected: float,
) -> None:
    from forge_viewer.adapters.mujoco_adapter import MuJoCoAdapter
    from forge_viewer.assets import resolve

    adapter = MuJoCoAdapter(resolve("joint_types"))
    session = Session(adapter)
    assert session.submit(cmd.Pause())
    node = next(item for item in session.nodes if item.name == body_name)
    assert session.submit(cmd.Select(node.object_id))
    gizmo = ObjectGizmo()
    target, reason = gizmo._joint_target(session, node)
    assert target is not None, reason
    assert session.submit(cmd.SetQpos(target.joint.qpos_adr, current))
    session.tick(FrameNeeds(poses=True, qpos=True, diagnostics=True), wall_dt=0.0)
    gizmo._hovered = handle
    edit = gizmo.precise_input(session)

    assert edit is not None
    assert gizmo.apply_precise_value(session, camera(), edit, delta)
    assert adapter.data.qpos[target.joint.qpos_adr] == pytest.approx(expected)


@pytest.mark.physics
@pytest.mark.parametrize(
    ("body_name", "range_call", "labels", "current_value"),
    (
        (
            "hinge_body",
            "polyline",
            {"MIN -60.0°", "MAX +60.0°"},
            np.radians(20.0),
        ),
        ("slide_body", "line", {"MIN -0.350 m", "MAX +0.350 m"}, 0.1),
    ),
)
def test_limited_joint_gizmo_draws_the_converted_range_and_colored_limits(
    body_name: str,
    range_call: str,
    labels: set[str],
    current_value: float,
) -> None:
    from forge_viewer.adapters.mujoco_adapter import MuJoCoAdapter
    from forge_viewer.assets import resolve

    session = Session(MuJoCoAdapter(resolve("joint_types")))
    assert session.submit(cmd.Pause())
    node = next(item for item in session.nodes if item.name == body_name)
    assert session.submit(cmd.Select(node.object_id))
    gizmo = ObjectGizmo()
    target, reason = gizmo._joint_target(session, node)
    assert target is not None, reason
    assert session.submit(cmd.SetQpos(target.joint.qpos_adr, current_value))
    session.tick(FrameNeeds(poses=True, qpos=True, diagnostics=True), wall_dt=0.0)
    pose = gizmo._target_pose(session, node, target)
    assert pose is not None
    cam = CameraView(eye=np.array((2.0, -4.0, 2.0)), target=pose[0].copy())
    assert gizmo.publish(
        CaptureBackend(),
        session,
        cam,
        RECT,
        ui_scale=1.0,
        style_scale=1.0,
        yielding=False,
        interactive=True,
    )

    overlay = RecordingDraw2D()
    gizmo.draw_overlay(cam, RECT, overlay, style_scale=1.0)

    if body_name == "slide_body":
        arrows = [
            args
            for name, args, _kwargs in overlay.calls
            if name == "fringed_concave_fill" and np.allclose(args[1][:3], JOINT_HANDLE_COLOR[:3])
        ]
        assert len(arrows) == 2
        slide = gizmo._slide_range_projection(
            cam,
            RECT,
            1.0,
            gizmo._joint_range,
            gizmo._frame.position,
            gizmo._frame.rotation,
        )
        assert slide is not None
        arrow_polygons = gizmo._slide_arrow_polygons(slide, 1.0)
        signed_offsets = [
            float(np.dot(np.mean(points, axis=0) - slide.current, slide.normal))
            for points in arrow_polygons
        ]
        assert signed_offsets[0] > 0.0
        assert signed_offsets[1] > 0.0
        assert signed_offsets == pytest.approx([JOINT_SLIDE_ARROW_OFFSET_PT] * 2)
        assert min(signed_offsets) > JOINT_CURRENT_TICK_PT * 0.5 + 4.0
        longitudinal_offsets = [
            float(np.dot(np.mean(points, axis=0) - slide.current, slide.tangent))
            for points in arrow_polygons
        ]
        assert longitudinal_offsets[0] > 0.0
        assert longitudinal_offsets[1] < 0.0
        arrow_cursor = np.mean(arrow_polygons[0], axis=0)
        assert gizmo.update_hover(session, cam, RECT, tuple(arrow_cursor)) is GizmoHandle.Z
        assert gizmo.update_hover(session, cam, RECT, tuple(slide.current)) is GizmoHandle.Z
        highlighted = RecordingDraw2D()
        gizmo.draw_overlay(cam, RECT, highlighted, style_scale=1.0)
        assert any(
            name == "line"
            and np.allclose(args[0], slide.lower)
            and np.allclose(args[1], slide.upper)
            and np.allclose(args[2], axis_hover_color(JOINT_RANGE_COLOR))
            for name, args, _kwargs in highlighted.calls
        )
        assert {hit.label for hit in gizmo.joint_limit_hits} == labels

    range_args = [
        args
        for name, args, kwargs in overlay.calls
        if name == range_call
        and np.allclose(args[2 if name == "line" else 1], JOINT_RANGE_COLOR)
        and (name == "line" or kwargs.get("closed") is False)
    ][-1]
    lower, upper = target.joint.range
    if target.joint.type == "hinge":
        dial = _RotationDialProjector(
            cam,
            RECT,
            pose[0],
            pose[1][:, 2],
            pose[1][:, 0],
            SIZE_PT,
        )
        expected_limits = dial.points(
            JOINT_RANGE_RADIUS,
            (lower, upper),
        )[:, :2]
        assert range_args[0][[0, -1]] == pytest.approx(expected_limits, abs=1e-6)
    else:
        axis = pose[1][:, 2]
        expected_limits = project(
            cam,
            (
                pose[0] + axis * (lower - current_value),
                pose[0] + axis * (upper - current_value),
            ),
            RECT,
        )[:, :2]
        assert range_args[1] - range_args[0] == pytest.approx(
            expected_limits[1] - expected_limits[0], abs=1e-6
        )
    texts = {
        args[2]: args[1]
        for name, args, _kwargs in overlay.calls
        if name == "text" and args[2] in labels
    }
    assert set(texts) == labels
    assert all(
        np.allclose(color, (220 / 255, 223 / 255, 227 / 255, 1.0)) for color in texts.values()
    )
    label_indices = [
        index
        for index, (name, args, _kwargs) in enumerate(overlay.calls)
        if name == "text" and args[2] in labels
    ]
    geometry_indices = [
        index
        for index, (name, _args, _kwargs) in enumerate(overlay.calls)
        if name in {"line", "polyline", "fringed_concave_fill", "triangle_fan_fill"}
    ]
    assert min(label_indices) > max(geometry_indices)


@pytest.mark.physics
@pytest.mark.parametrize("body_name", ("hinge_body", "slide_body"))
def test_joint_limit_labels_write_the_selected_endpoint(body_name: str) -> None:
    from forge_viewer.adapters.mujoco_adapter import MuJoCoAdapter
    from forge_viewer.assets import resolve

    adapter = MuJoCoAdapter(resolve("joint_types"))
    session = Session(adapter)
    assert session.submit(cmd.Pause())
    node = next(item for item in session.nodes if item.name == body_name)
    assert session.submit(cmd.Select(node.object_id))
    gizmo = ObjectGizmo()
    session.tick(FrameNeeds(poses=True, qpos=True, diagnostics=True), wall_dt=0.0)
    target, reason = gizmo._joint_target(session, node)
    assert target is not None, reason
    pose = gizmo._target_pose(session, node, target)
    assert pose is not None
    cam = CameraView(eye=np.array((2.0, -4.0, 2.0)), target=pose[0].copy())
    assert gizmo.publish(
        CaptureBackend(),
        session,
        cam,
        RECT,
        ui_scale=1.0,
        style_scale=1.0,
        yielding=False,
        interactive=True,
    )
    gizmo.draw_overlay(cam, RECT, RecordingDraw2D(), style_scale=1.0)

    lower_hit, upper_hit = gizmo.joint_limit_hits
    assert lower_hit.label.startswith("MIN")
    assert upper_hit.label.startswith("MAX")
    assert gizmo.apply_joint_limit(session, upper_hit)
    assert adapter.data.qpos[target.joint.qpos_adr] == pytest.approx(target.joint.range[1])
    assert gizmo.apply_joint_limit(session, lower_hit)
    assert adapter.data.qpos[target.joint.qpos_adr] == pytest.approx(target.joint.range[0])


def test_hinge_joint_range_continuously_fades_before_its_projection_degenerates() -> None:
    facing = 0.18
    view = np.array((np.sqrt(1.0 - facing * facing), 0.0, facing))
    cam = CameraView(
        eye=-view * 5.0,
        target=np.zeros(3),
        up=np.array((0.0, 1.0, 0.0)),
        aspect=RECT[2] / RECT[3],
    )
    gizmo = ObjectGizmo("rotate")
    gizmo._joint_range = _JointRangeState("hinge", 0.0, -1.0, 1.0)
    overlay = RecordingDraw2D()

    gizmo._draw_joint_range(overlay, cam, RECT, 1.0)

    expected_alpha = rotation_ring_alpha(cam, np.zeros(3), np.array((0.0, 0.0, 1.0)))
    range_color = next(
        args[1]
        for name, args, _kwargs in overlay.calls
        if name == "polyline" and np.allclose(args[1][:3], JOINT_RANGE_COLOR[:3])
    )
    limit_colors = [
        args[2]
        for name, args, _kwargs in overlay.calls
        if name == "circle_filled"
        and (
            np.allclose(args[2][:3], JOINT_LOWER_LIMIT_COLOR[:3])
            or np.allclose(args[2][:3], JOINT_UPPER_LIMIT_COLOR[:3])
        )
    ]
    assert 0.0 < expected_alpha < 1.0
    assert range_color[3] == pytest.approx(JOINT_RANGE_COLOR[3] * expected_alpha)
    assert [color[3] for color in limit_colors] == pytest.approx([expected_alpha, expected_alpha])


def test_hinge_joint_range_draws_only_the_allowed_arc_across_180_degrees() -> None:
    cam = CameraView(
        eye=np.array((0.0, 0.0, 5.0)),
        target=np.zeros(3),
        up=np.array((0.0, 1.0, 0.0)),
        aspect=RECT[2] / RECT[3],
    )
    lower, upper = np.radians((-170.0, 170.0))
    gizmo = ObjectGizmo("rotate")
    gizmo._joint_range = _JointRangeState("hinge", 0.0, lower, upper)
    overlay = RecordingDraw2D()

    gizmo._draw_joint_range(overlay, cam, RECT, 1.0)

    allowed = next(
        (args, kwargs)
        for name, args, kwargs in overlay.calls
        if name == "polyline" and np.allclose(args[1], JOINT_RANGE_COLOR)
    )
    assert JOINT_RANGE_RADIUS == RING_RADIUS
    assert np.allclose(JOINT_RANGE_COLOR, (173 / 255, 150 / 255, 184 / 255, 1.0))
    assert allowed[1]["closed"] is False

    dial = _RotationDialProjector(
        cam,
        RECT,
        np.zeros(3),
        np.array((0.0, 0.0, 1.0)),
        np.array((1.0, 0.0, 0.0)),
        SIZE_PT,
    )
    expected = dial.points(JOINT_RANGE_RADIUS, (lower, upper))[:, :2]
    assert allowed[0][0][[0, -1]] == pytest.approx(expected, abs=1e-6)
    range_strokes = [
        args
        for name, args, _kwargs in overlay.calls
        if name == "polyline" and args[2] == pytest.approx(RING_WIDTH_PT)
    ]
    assert len(range_strokes) == 1

    ticks = {
        "current": next(
            args
            for name, args, _kwargs in overlay.calls
            if name == "line" and np.allclose(args[2], JOINT_CURRENT_COLOR)
        ),
        "lower": next(
            args
            for name, args, _kwargs in overlay.calls
            if name == "line" and np.allclose(args[2], JOINT_LOWER_LIMIT_COLOR)
        ),
        "upper": next(
            args
            for name, args, _kwargs in overlay.calls
            if name == "line" and np.allclose(args[2], JOINT_UPPER_LIMIT_COLOR)
        ),
    }
    assert np.linalg.norm(ticks["current"][1] - ticks["current"][0]) == pytest.approx(
        JOINT_CURRENT_TICK_PT
    )
    for limit in ("lower", "upper"):
        assert np.linalg.norm(ticks[limit][1] - ticks[limit][0]) == pytest.approx(
            JOINT_LIMIT_TICK_PT
        )


@pytest.mark.parametrize("interaction", ("hovered", "active"))
def test_hinge_joint_range_keeps_semantic_purple_for_hover_and_press(interaction: str) -> None:
    cam = CameraView(
        eye=np.array((0.0, 0.0, 5.0)),
        target=np.zeros(3),
        up=np.array((0.0, 1.0, 0.0)),
        aspect=RECT[2] / RECT[3],
    )
    gizmo = ObjectGizmo("rotate")
    gizmo._joint_range = _JointRangeState("hinge", 0.0, -1.0, 1.0)
    if interaction == "hovered":
        gizmo._interactive = True
        gizmo._hovered = GizmoHandle.ROTATE_Z
    else:
        gizmo._active = GizmoHandle.ROTATE_Z
    overlay = RecordingDraw2D()

    gizmo._draw_joint_range(overlay, cam, RECT, 1.0)

    expected = (
        axis_hover_color(JOINT_RANGE_COLOR) if interaction == "hovered" else ACTIVE_HANDLE_COLOR
    )
    allowed = next(
        args
        for name, args, kwargs in overlay.calls
        if name == "polyline" and kwargs.get("closed") is False and np.allclose(args[1], expected)
    )
    assert np.allclose(allowed[1], expected)


def test_full_hinge_range_has_no_unavailable_overlay() -> None:
    cam = CameraView(
        eye=np.array((0.0, 0.0, 5.0)),
        target=np.zeros(3),
        up=np.array((0.0, 1.0, 0.0)),
        aspect=RECT[2] / RECT[3],
    )
    gizmo = ObjectGizmo("rotate")
    gizmo._joint_range = _JointRangeState("hinge", 0.0, -np.pi, np.pi)
    overlay = RecordingDraw2D()

    gizmo._draw_joint_range(overlay, cam, RECT, 1.0)

    range_calls = [
        (args, kwargs)
        for name, args, kwargs in overlay.calls
        if name == "polyline" and np.allclose(args[1], JOINT_RANGE_COLOR)
    ]
    assert len(range_calls) == 1
    assert np.allclose(range_calls[0][0][1], JOINT_RANGE_COLOR)
    assert range_calls[0][1]["closed"] is True


def test_active_hinge_guide_keeps_one_allowed_range_arc() -> None:
    cam = CameraView(
        eye=np.array((0.0, 0.0, 5.0)),
        target=np.zeros(3),
        up=np.array((0.0, 1.0, 0.0)),
        aspect=RECT[2] / RECT[3],
    )
    gizmo = ObjectGizmo("rotate")
    gizmo._active = GizmoHandle.ROTATE_Z
    gizmo._axis[:] = (0.0, 0.0, 1.0)
    gizmo._start_basis[:] = np.eye(3)
    gizmo._rotation_start_vec[:] = (0.0, 1.0, 0.0)
    gizmo._start_joint_qpos = np.array((np.radians(-17.0),))
    gizmo._rotation_angle = np.radians(40.0)
    gizmo._joint_range = _JointRangeState(
        "hinge", np.radians(23.0), np.radians(-90.0), np.radians(200.0)
    )
    dial = _RotationDialProjector(
        cam,
        RECT,
        gizmo._start_pos,
        gizmo._axis,
        np.array((1.0, 0.0, 0.0)),
        SIZE_PT,
    )
    overlay = RecordingDraw2D()

    gizmo._draw_joint_range(overlay, cam, RECT, 1.0)
    gizmo._draw_rotation_guide(overlay, cam, RECT, 1.0, dial)

    range_strokes = [
        args
        for name, args, _kwargs in overlay.calls
        if name == "polyline" and args[2] == pytest.approx(RING_WIDTH_PT)
    ]
    assert len(range_strokes) == 1
    assert not any(
        name == "polyline" and kwargs.get("closed") for name, _args, kwargs in overlay.calls
    )
    assert not any(
        name == "polyline" and kwargs.get("closed") and np.allclose(args[1], HOVER_COLOR)
        for name, args, kwargs in overlay.calls
    )
    assert any(name == "fringed_concave_fill" for name, _args, _kwargs in overlay.calls)
    sector_args = next(args for name, args, _kwargs in overlay.calls if name == "triangle_fan_fill")
    sector = np.asarray(sector_args[0])
    assert np.allclose(sector_args[1][:3], ACTIVE_HANDLE_COLOR[:3])
    arc_color = next(
        args[1] for name, args, _kwargs in overlay.calls if name == "fringed_concave_fill"
    )
    assert np.allclose(arc_color, JOINT_ACTIVE_DARK_COLOR)
    stable_dial = _RotationDialProjector(
        cam,
        RECT,
        gizmo._start_pos,
        gizmo._axis,
        gizmo._start_basis[:, 0],
        SIZE_PT,
    )
    assert sector[1] == pytest.approx(stable_dial.points(RING_RADIUS, (np.radians(-17.0),))[0, :2])
    assert sector[-1] == pytest.approx(stable_dial.points(RING_RADIUS, (np.radians(23.0),))[0, :2])


def test_multi_turn_hinge_guide_wraps_like_the_rotation_gizmo() -> None:
    cam = CameraView(
        eye=np.array((0.0, 0.0, 5.0)),
        target=np.zeros(3),
        up=np.array((0.0, 1.0, 0.0)),
        aspect=RECT[2] / RECT[3],
    )
    gizmo = ObjectGizmo("rotate")
    gizmo._active = GizmoHandle.ROTATE_Z
    gizmo._axis[:] = (0.0, 0.0, 1.0)
    gizmo._start_basis[:] = np.eye(3)
    gizmo._rotation_start_vec[:] = (1.0, 0.0, 0.0)
    start = np.radians(-540.0)
    current = np.radians(128.3)
    gizmo._start_joint_qpos = np.array((current,))
    gizmo._joint_drag_origin_qpos = np.array((start,))
    gizmo._joint_range = _JointRangeState(
        "hinge",
        current,
        np.radians(-540.0),
        np.radians(540.0),
    )
    dial = _RotationDialProjector(
        cam,
        RECT,
        gizmo._start_pos,
        gizmo._axis,
        gizmo._rotation_start_vec,
        SIZE_PT,
    )
    overlay = RecordingDraw2D()

    gizmo._draw_rotation_guide(overlay, cam, RECT, 1.0, dial)

    sector_args = next(args for name, args, _kwargs in overlay.calls if name == "triangle_fan_fill")
    sector = np.asarray(sector_args[0])
    wrapped = _rotation_sweep(current - start)
    if abs(wrapped) > np.pi:
        wrapped -= np.copysign(2.0 * np.pi, wrapped)
    segments = 64
    assert len(sector) <= segments + 2
    stable_dial = _RotationDialProjector(
        cam,
        RECT,
        gizmo._start_pos,
        gizmo._axis,
        gizmo._start_basis[:, 0],
        SIZE_PT,
    )
    assert sector[1] == pytest.approx(stable_dial.points(RING_RADIUS, (start,))[0, :2])
    assert sector[-1] == pytest.approx(stable_dial.points(RING_RADIUS, (start + wrapped,))[0, :2])
    assert sector[-1] == pytest.approx(stable_dial.points(RING_RADIUS, (current,))[0, :2])


def test_multi_turn_hinge_range_uses_numeric_badge_without_false_endpoint_ticks() -> None:
    cam = CameraView(
        eye=np.array((0.0, 0.0, 5.0)),
        target=np.zeros(3),
        up=np.array((0.0, 1.0, 0.0)),
        aspect=RECT[2] / RECT[3],
    )
    gizmo = ObjectGizmo("rotate")
    gizmo._frame.position[:] = 0.0
    gizmo._frame.rotation[:] = np.eye(3)
    state = _JointRangeState(
        "hinge",
        0.0,
        np.radians(-540.0),
        np.radians(405.0),
    )
    overlay = RecordingDraw2D()

    gizmo._draw_hinge_range(overlay, cam, RECT, 1.0, state)

    endpoint_lines = (
        args
        for name, args, _kwargs in overlay.calls
        if name == "line"
        and (
            np.allclose(args[2][:3], JOINT_LOWER_LIMIT_COLOR[:3])
            or np.allclose(args[2][:3], JOINT_UPPER_LIMIT_COLOR[:3])
        )
    )
    assert not tuple(endpoint_lines)
    labels = {args[2] for name, args, _kwargs in overlay.calls if name == "text"}
    assert "MIN -540.0°" in labels
    assert "MAX +405.0°" in labels


def test_multi_turn_hinge_hides_static_limit_badge_during_drag() -> None:
    cam = CameraView(
        eye=np.array((0.0, 0.0, 5.0)),
        target=np.zeros(3),
        up=np.array((0.0, 1.0, 0.0)),
        aspect=RECT[2] / RECT[3],
    )
    gizmo = ObjectGizmo("rotate")
    gizmo._frame.position[:] = 0.0
    gizmo._frame.rotation[:] = np.eye(3)
    gizmo._using = True
    state = _JointRangeState("hinge", 0.0, np.radians(-540.0), np.radians(540.0))
    overlay = RecordingDraw2D()

    gizmo._draw_hinge_range(overlay, cam, RECT, 1.0, state)

    labels = {args[2] for name, args, _kwargs in overlay.calls if name == "text"}
    assert not any(label.startswith(("MIN ", "MAX ")) for label in labels)
    assert gizmo.joint_limit_hits == ()


@pytest.mark.parametrize(
    ("joint_type", "eye", "up"),
    (
        ("hinge", (5.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        ("slide", (0.0, 0.0, 5.0), (0.0, 1.0, 0.0)),
    ),
)
def test_joint_range_is_hidden_when_its_projection_degenerates(
    joint_type: str,
    eye: tuple[float, float, float],
    up: tuple[float, float, float],
) -> None:
    cam = CameraView(
        eye=np.asarray(eye),
        target=np.zeros(3),
        up=np.asarray(up),
        aspect=RECT[2] / RECT[3],
    )
    gizmo = ObjectGizmo("rotate" if joint_type == "hinge" else "translate")
    gizmo._joint_range = _JointRangeState(joint_type, 0.0, -1.0, 1.0)
    overlay = RecordingDraw2D()

    gizmo._draw_joint_range(overlay, cam, RECT, 1.0)

    assert overlay.calls == []


def test_active_joint_rotation_guide_is_hidden_when_the_ring_is_edge_on() -> None:
    cam = CameraView(
        eye=np.array((5.0, 0.0, 0.0)),
        target=np.zeros(3),
        up=np.array((0.0, 0.0, 1.0)),
        aspect=RECT[2] / RECT[3],
    )
    gizmo = ObjectGizmo("rotate")
    gizmo._active = GizmoHandle.ROTATE_Z
    gizmo._axis[:] = (0.0, 0.0, 1.0)
    gizmo._rotation_start_vec[:] = (1.0, 0.0, 0.0)
    gizmo._frame.active_projection_fade = True
    dial = _RotationDialProjector(
        cam,
        RECT,
        gizmo._start_pos,
        gizmo._axis,
        gizmo._rotation_start_vec,
        SIZE_PT,
    )
    overlay = RecordingDraw2D()

    gizmo._draw_rotation_guide(overlay, cam, RECT, 1.0, dial)

    assert overlay.calls == []


@pytest.mark.physics
def test_hinge_joint_range_stays_fixed_while_the_current_marker_moves() -> None:
    from forge_viewer.adapters.mujoco_adapter import MuJoCoAdapter
    from forge_viewer.assets import resolve

    session = Session(MuJoCoAdapter(resolve("joint_types")))
    assert session.submit(cmd.Pause())
    node = next(item for item in session.nodes if item.name == "hinge_body")
    assert session.submit(cmd.Select(node.object_id))
    gizmo = ObjectGizmo()
    target, reason = gizmo._joint_target(session, node)
    assert target is not None, reason
    session.tick(FrameNeeds(poses=True, qpos=True, diagnostics=True), wall_dt=0.0)
    pose = gizmo._target_pose(session, node, target)
    assert pose is not None
    cam = CameraView(eye=np.array((2.0, -4.0, 2.0)), target=pose[0].copy())

    samples = []
    for current in np.radians((-35.0, 40.0)):
        assert session.submit(cmd.SetQpos(target.joint.qpos_adr, current))
        session.tick(FrameNeeds(poses=True, qpos=True, diagnostics=True), wall_dt=0.0)
        assert gizmo.publish(
            CaptureBackend(),
            session,
            cam,
            RECT,
            ui_scale=1.0,
            style_scale=1.0,
            yielding=False,
            interactive=True,
        )
        overlay = RecordingDraw2D()
        gizmo.draw_overlay(cam, RECT, overlay, style_scale=1.0)
        allowed = next(
            args[0]
            for name, args, kwargs in overlay.calls
            if name == "polyline"
            and np.allclose(args[1], JOINT_RANGE_COLOR)
            and kwargs.get("closed") is False
        )
        current_tick = next(
            args[:2]
            for name, args, _kwargs in overlay.calls
            if name == "line" and np.allclose(args[2], JOINT_CURRENT_COLOR)
        )
        samples.append((allowed[[0, -1]].copy(), np.asarray(current_tick).copy()))

    assert samples[1][0] == pytest.approx(samples[0][0], abs=1e-6)
    assert not np.allclose(samples[1][1], samples[0][1])


@pytest.mark.physics
@pytest.mark.parametrize("body_name", ("hinge_body", "slide_body"))
def test_scalar_joint_gizmo_uses_a_joint_color_instead_of_xyz(body_name: str) -> None:
    from forge_viewer.adapters.mujoco_adapter import MuJoCoAdapter
    from forge_viewer.assets import resolve

    session = Session(MuJoCoAdapter(resolve("joint_types")))
    assert session.submit(cmd.Pause())
    node = next(item for item in session.nodes if item.name == body_name)
    assert session.submit(cmd.Select(node.object_id))
    gizmo = ObjectGizmo()
    gizmo.set_style("3d")
    session.tick(gizmo.frame_needs(session), wall_dt=0.0)
    pose = gizmo._target_pose(session, node, gizmo._joint_target(session, node)[0])
    assert pose is not None
    backend = CaptureBackend()

    assert gizmo.publish(
        backend,
        session,
        CameraView(eye=np.array((2.0, -4.0, 2.0)), target=pose[0].copy()),
        RECT,
        ui_scale=1.0,
        style_scale=1.0,
        yielding=False,
        interactive=True,
    )
    assert backend.frame.handle_color == pytest.approx(JOINT_HANDLE_COLOR)


@pytest.mark.physics
def test_unlimited_joint_gizmo_does_not_invent_limits() -> None:
    from forge_viewer.adapters.mujoco_adapter import MuJoCoAdapter
    from forge_viewer.assets import resolve

    session = Session(MuJoCoAdapter(resolve("joint_types")))
    assert session.submit(cmd.Pause())
    node = next(item for item in session.nodes if item.name == "hinge_free_body")
    assert session.submit(cmd.Select(node.object_id))
    gizmo = ObjectGizmo()
    session.tick(gizmo.frame_needs(session), wall_dt=0.0)
    target, reason = gizmo._joint_target(session, node)
    assert target is not None, reason
    pose = gizmo._target_pose(session, node, target)
    assert pose is not None
    cam = CameraView(eye=np.array((2.0, -4.0, 2.0)), target=pose[0].copy())
    assert gizmo.publish(
        CaptureBackend(),
        session,
        cam,
        RECT,
        ui_scale=1.0,
        style_scale=1.0,
        yielding=False,
        interactive=True,
    )

    overlay = RecordingDraw2D()
    gizmo.draw_overlay(cam, RECT, overlay, style_scale=1.0)
    assert not any(
        name == "text" and args[2].startswith(("MIN ", "MAX "))
        for name, args, _kwargs in overlay.calls
    )


@pytest.mark.physics
def test_inspector_omits_redundant_active_gizmo_status(monkeypatch) -> None:
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

    assert lines == []
