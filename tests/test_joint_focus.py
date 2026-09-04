from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
import pytest

from mojive.adapters.base import (
    CameraInfo,
    DiagnosticFrame,
    JointInfo,
    NodeType,
    SceneFrame,
    SceneNode,
)
from mojive.session import Session
from mojive.types import CameraView
from mojive.ui.app import ViewerApp
from mojive.ui.camera import FOCUS_DURATION, OrbitCamera
from mojive.ui.gestures import InputState


class _CameraSink:
    def __init__(self) -> None:
        self.views = []

    def set_camera(self, camera) -> None:
        self.views.append(camera)


class _FocusSession:
    def __init__(self, joint: JointInfo, node: SceneNode, frame: SceneFrame) -> None:
        self.joints = [joint]
        self.nodes = [node]
        self.frame = frame

    @staticmethod
    def node_local_bounds(_node_id):
        return np.zeros(3), np.array((0.04, 0.05, 0.3))

    @staticmethod
    def node_world_bounds(_node_id):
        return None

    @staticmethod
    def bounds():
        return np.full(3, -1.0), np.full(3, 1.0)


def _focus_app(joint: JointInfo, frame: SceneFrame) -> ViewerApp:
    node = SceneNode(
        node_id=7,
        name=joint.name,
        type=NodeType.JOINT,
        body_index=1,
        joint_index=joint.joint_id,
    )
    app = object.__new__(ViewerApp)
    app.session = _FocusSession(joint, node, frame)
    app.camera = OrbitCamera(pivot=np.zeros(3), distance=3.0, yaw=0.0)
    app.camera_out = _CameraSink()
    app._viewport_rect = (0.0, 0.0, 1200.0, 800.0)
    app._model_camera_id = -1
    app._model_camera_view = None
    app._model_camera_projection_target = None
    app._pending_joint_focus_id = joint.joint_id
    return app


def test_hinge_focus_targets_the_anchor_from_the_nearest_readable_oblique_view() -> None:
    joint = JointInfo(0, "elbow", "hinge", True, (-1.0, 1.0), 0, 0, 1)
    frame = SceneFrame(
        body_xpos=np.array(((0.0, 0.0, 0.0), (1.0, 2.0, 3.0)), np.float32),
        body_xmat=np.repeat(np.eye(3, dtype=np.float32)[None], 2, axis=0),
        qpos=np.array((0.0,), np.float32),
        diagnostics=DiagnosticFrame(
            joint_xpos=np.array(((1.5, 2.0, 3.0),), np.float32),
            joint_xaxis=np.array(((1.0, 0.0, 0.0),), np.float32),
        ),
    )
    app = _focus_app(joint, frame)
    app.camera = OrbitCamera(pivot=np.array((1.5, 2.0, 3.0)), distance=3.0, yaw=0.0, pitch=0.0)

    app._apply_pending_joint_focus()
    app.camera.advance(FOCUS_DURATION, app.camera_out)

    assert app.camera.pivot == pytest.approx((1.5, 2.0, 3.0))
    direction = app.camera.direction()
    assert abs(np.dot(direction, (1.0, 0.0, 0.0))) == pytest.approx(
        np.cos(np.deg2rad(35.0)), abs=1e-6
    )
    assert app.camera.pitch == pytest.approx(35.0)
    assert direction[1] == pytest.approx(0.0, abs=1e-6)
    assert app._pending_joint_focus_id is None


def test_joint_focus_can_avoid_an_occluder_with_a_small_nearby_turn() -> None:
    target = SceneNode(7, "elbow", NodeType.JOINT, body_index=1, joint_index=0)
    blocker = SceneNode(8, "torso", NodeType.LINK, object_id=2, body_index=2)
    target_link = SceneNode(9, "forearm", NodeType.LINK, object_id=1, body_index=1)
    app = object.__new__(ViewerApp)
    app.camera = OrbitCamera(pivot=np.zeros(3), distance=3.0, yaw=0.0)
    nearby = np.array((np.cos(np.deg2rad(15.0)), np.sin(np.deg2rad(15.0)), 0.0))

    def query(pick):
        if pick.origin[1] < 0.1:
            return blocker.object_id, 1.0
        return target_link.object_id, 2.0

    app.session = SimpleNamespace(
        adapter=SimpleNamespace(caps=SimpleNamespace(raycast=True)),
        bounds=lambda: (np.full(3, -2.0), np.full(3, 2.0)),
        query=query,
        node_by_object_id=lambda object_id: {
            blocker.object_id: blocker,
            target_link.object_id: target_link,
        }.get(object_id),
    )

    direction = app._least_occluded_joint_direction(
        np.zeros(3),
        0.3,
        (np.array((1.0, 0.0, 0.0)), nearby, np.array((-1.0, 0.0, 0.0))),
        target,
        np.array((3.0, 0.0, 0.0)),
    )

    assert direction == pytest.approx(nearby)


def test_joint_focus_does_not_cross_the_model_only_to_avoid_an_occluder() -> None:
    target = SceneNode(7, "elbow", NodeType.JOINT, body_index=1, joint_index=0)
    blocker = SceneNode(8, "torso", NodeType.LINK, object_id=2, body_index=2)
    target_link = SceneNode(9, "forearm", NodeType.LINK, object_id=1, body_index=1)
    app = object.__new__(ViewerApp)
    app.camera = OrbitCamera(pivot=np.zeros(3), distance=3.0, yaw=0.0)

    def query(pick):
        if pick.origin[0] > 0.0:
            return blocker.object_id, 1.0
        return target_link.object_id, 2.0

    app.session = SimpleNamespace(
        adapter=SimpleNamespace(caps=SimpleNamespace(raycast=True)),
        bounds=lambda: (np.full(3, -2.0), np.full(3, 2.0)),
        query=query,
        node_by_object_id=lambda object_id: {
            blocker.object_id: blocker,
            target_link.object_id: target_link,
        }.get(object_id),
    )

    direction = app._least_occluded_joint_direction(
        np.zeros(3),
        0.3,
        (np.array((1.0, 0.0, 0.0)), np.array((-1.0, 0.0, 0.0))),
        target,
        np.array((3.0, 0.0, 0.0)),
    )

    assert direction == pytest.approx((1.0, 0.0, 0.0))


def test_hinge_focus_prefers_an_unblocked_view_from_above() -> None:
    target = SceneNode(7, "elbow", NodeType.JOINT, body_index=1, joint_index=0)
    target_link = SceneNode(9, "forearm", NodeType.LINK, object_id=1, body_index=1)
    app = object.__new__(ViewerApp)
    app.camera = OrbitCamera(pivot=np.zeros(3), distance=3.0, yaw=0.0)
    app.session = SimpleNamespace(
        adapter=SimpleNamespace(caps=SimpleNamespace(raycast=True)),
        bounds=lambda: (np.full(3, -2.0), np.full(3, 2.0)),
        query=lambda _pick: (target_link.object_id, 2.0),
        node_by_object_id=lambda object_id: target_link if object_id == 1 else None,
    )

    direction = app._least_occluded_joint_direction(
        np.zeros(3),
        0.3,
        (np.array((0.8, 0.0, -0.6)), np.array((0.8, 0.0, 0.6))),
        target,
        np.array((3.0, 0.0, -2.0)),
        preferred_up=(0.0, 0.0, 1.0),
    )

    assert direction == pytest.approx((0.8, 0.0, 0.6))


def test_hinge_focus_keeps_the_upper_view_when_only_the_lower_view_is_clear() -> None:
    target = SceneNode(7, "elbow", NodeType.JOINT, body_index=1, joint_index=0)
    blocker = SceneNode(8, "torso", NodeType.LINK, object_id=2, body_index=2)
    target_link = SceneNode(9, "forearm", NodeType.LINK, object_id=1, body_index=1)
    app = object.__new__(ViewerApp)
    app.camera = OrbitCamera(pivot=np.zeros(3), distance=3.0, yaw=0.0)

    def query(pick):
        return (blocker.object_id, 1.0) if pick.origin[2] > 0.0 else (target_link.object_id, 2.0)

    app.session = SimpleNamespace(
        adapter=SimpleNamespace(caps=SimpleNamespace(raycast=True)),
        bounds=lambda: (np.full(3, -2.0), np.full(3, 2.0)),
        query=query,
        node_by_object_id=lambda object_id: {
            blocker.object_id: blocker,
            target_link.object_id: target_link,
        }.get(object_id),
    )

    direction = app._least_occluded_joint_direction(
        np.zeros(3),
        0.3,
        (np.array((0.8, 0.0, 0.6)), np.array((0.8, 0.0, -0.6))),
        target,
        np.array((3.0, 0.0, 2.0)),
        preferred_up=(0.0, 0.0, 1.0),
    )

    assert direction == pytest.approx((0.8, 0.0, 0.6))


def test_slide_focus_looks_perpendicular_at_the_range_midpoint() -> None:
    joint = JointInfo(0, "rail", "slide", True, (-0.5, 0.5), 0, 0, 1, axis=(0.0, 1.0, 0.0))
    frame = SceneFrame(
        body_xpos=np.array(((0.0, 0.0, 0.0), (0.0, 0.5, 0.0)), np.float32),
        body_xmat=np.repeat(np.eye(3, dtype=np.float32)[None], 2, axis=0),
        qpos=np.array((0.5,), np.float32),
        diagnostics=DiagnosticFrame(
            joint_xpos=np.array(((0.0, 0.0, 0.0),), np.float32),
            joint_xaxis=np.array(((0.0, 1.0, 0.0),), np.float32),
        ),
    )
    app = _focus_app(joint, frame)

    app._apply_pending_joint_focus()
    app.camera.advance(FOCUS_DURATION, app.camera_out)

    assert app.camera.pivot == pytest.approx((0.0, 0.0, 0.0))
    assert np.dot(app.camera.direction(), (0.0, 1.0, 0.0)) == pytest.approx(0.0, abs=1e-9)
    # The complete limited travel remains inside the focus sphere.
    expected_distance, _height = app.camera._framing_distance(0.8, 1.5)
    assert app.camera.distance == pytest.approx(expected_distance)


def test_vertical_slide_focus_uses_iso_elevation_instead_of_a_level_view() -> None:
    joint = JointInfo(0, "lift", "slide", True, (-0.5, 0.5), 0, 0, 1, axis=(0.0, 0.0, 1.0))
    frame = SceneFrame(
        body_xpos=np.array(((0.0, 0.0, 0.0), (0.0, 0.0, 0.5)), np.float32),
        body_xmat=np.repeat(np.eye(3, dtype=np.float32)[None], 2, axis=0),
        qpos=np.array((0.5,), np.float32),
        diagnostics=DiagnosticFrame(
            joint_xpos=np.array(((0.0, 0.0, 0.0),), np.float32),
            joint_xaxis=np.array(((0.0, 0.0, 1.0),), np.float32),
        ),
    )
    app = _focus_app(joint, frame)

    app._apply_pending_joint_focus()
    app.camera.advance(FOCUS_DURATION, app.camera_out)

    assert app.camera.pitch == pytest.approx(30.0)


def test_hierarchy_node_focus_preserves_azimuth_at_iso_elevation() -> None:
    node = SceneNode(
        node_id=12,
        name="forearm",
        type=NodeType.GEOM,
        body_index=1,
        geom_index=0,
    )
    center = np.array((1.0, 2.0, 0.0))
    half = np.array((0.1, 0.2, 0.3))
    session = SimpleNamespace(
        node=lambda node_id: node if node_id == node.node_id else None,
        node_world_bounds=lambda _node_id: (center, half),
        bounds=lambda: (np.full(3, -2.0), np.full(3, 2.0)),
    )
    app = object.__new__(ViewerApp)
    app.session = session
    app.camera = OrbitCamera(pivot=np.zeros(3), distance=3.0, yaw=0.0)
    app.camera_out = _CameraSink()
    app._viewport_rect = (0.0, 0.0, 1200.0, 800.0)
    app._model_camera_id = -1
    app._model_camera_view = None
    app._model_camera_projection_target = None
    app._pending_joint_focus_id = None
    app._pending_node_focus_id = None

    assert app.request_node_focus(node.node_id)
    app._apply_pending_node_focus()
    app.camera.advance(FOCUS_DURATION, app.camera_out)

    horizontal = np.array((2.0, -2.0, 0.0)) / np.sqrt(8.0)
    expected_direction = horizontal * np.cos(np.deg2rad(30.0)) + np.array((0.0, 0.0, 0.5))
    assert app.camera.pivot == pytest.approx(center)
    assert app.camera.direction() == pytest.approx(expected_direction)
    expected_distance, _height = app.camera._framing_distance(np.linalg.norm(half), 1.15)
    assert app.camera.distance == pytest.approx(expected_distance)


def test_hierarchy_camera_focus_uses_the_camera_world_position() -> None:
    node = SceneNode(
        node_id=13,
        name="inspection_camera",
        type=NodeType.CAMERA,
        body_index=0,
        camera_index=0,
    )
    authored_view = CameraView(
        eye=np.array((0.0, -4.1, 2.55)),
        target=np.array((0.0, -0.94, 0.58)),
    )
    session = SimpleNamespace(
        node=lambda node_id: node if node_id == node.node_id else None,
        node_world_bounds=lambda _node_id: None,
        bounds=lambda: (np.full(3, -2.0), np.full(3, 2.0)),
        frame=SceneFrame(
            body_xpos=np.zeros((1, 3), np.float32),
            body_xmat=np.eye(3, dtype=np.float32)[None],
        ),
        cameras=(CameraInfo(7, node.name),),
        camera_view=lambda camera_id: authored_view if camera_id == 7 else None,
        scene_models=(),
        source=None,
    )
    app = object.__new__(ViewerApp)
    app.session = session
    app.camera = OrbitCamera(pivot=np.zeros(3), distance=3.0, yaw=0.0)
    app.camera_out = _CameraSink()
    app._viewport_rect = (0.0, 0.0, 1200.0, 800.0)
    app._model_camera_id = -1
    app._model_camera_view = None
    app._model_camera_projection_target = None
    app._pending_joint_focus_id = None
    app._pending_node_focus_id = None

    assert app.request_node_focus(node.node_id)
    app._apply_pending_node_focus()
    app.camera.advance(FOCUS_DURATION, app.camera_out)

    assert app.camera.pivot == pytest.approx(authored_view.eye)


def test_joint_selection_highlights_its_renderable_parent_without_changing_target() -> None:
    parent = SceneNode(4, "forearm", NodeType.LINK, object_id=17, body_index=3)
    joint = SceneNode(
        5,
        "elbow",
        NodeType.JOINT,
        parent=parent.node_id,
        body_index=parent.body_index,
        joint_index=2,
    )
    session = object.__new__(Session)
    session._selected = 0
    session._selected_node_id = joint.node_id
    session._by_node_id = {parent.node_id: parent, joint.node_id: joint}

    assert session.selected_node is joint
    assert session.selection_highlight_object_id == parent.object_id


def test_viewport_double_click_requires_two_complete_short_left_clicks() -> None:
    app = object.__new__(ViewerApp)
    app.router = SimpleNamespace(
        wants_camera=lambda: True,
        released=False,
        travel=0.0,
        started_with_left=True,
    )
    app.window = SimpleNamespace(style_scale=1.0)
    node = SceneNode(3, "elbow", NodeType.LINK, body_index=1)
    app.session = SimpleNamespace(
        node_by_object_id=lambda _object_id: node,
        submit=lambda _command: None,
    )
    app._pick_at = lambda _cursor: 9
    focused = []
    app._request_node_joint_focus = lambda candidate: focused.append(candidate) or True
    app._last_viewport_click = None

    down = InputState(left=True, over_viewport=True, cursor=(100.0, 80.0))
    up = InputState(over_viewport=True, cursor=(100.0, 80.0))
    app._poll_pick(down)
    app.router.released = True
    app._poll_pick(up)
    app.router.released = False
    app._poll_pick(down)
    app.router.released = True
    app._poll_pick(up)

    assert focused == [node]


def test_viewport_double_click_falls_back_to_generic_node_focus() -> None:
    app = object.__new__(ViewerApp)
    app.router = SimpleNamespace(
        wants_camera=lambda: True,
        released=True,
        travel=0.0,
        started_with_left=True,
    )
    app.window = SimpleNamespace(style_scale=1.0)
    node = SceneNode(3, "camera", NodeType.CAMERA, object_id=9)
    submitted = []
    app.session = SimpleNamespace(
        node_by_object_id=lambda _object_id: node,
        submit=submitted.append,
    )
    app._pick_at = lambda _cursor: node.object_id
    app._request_node_joint_focus = lambda _candidate: False
    focused = []
    app.request_node_focus = lambda node_id: focused.append(node_id) or True
    app._last_viewport_click = (time.monotonic(), (100.0, 80.0), node.object_id)

    app._poll_pick(InputState(over_viewport=True, cursor=(100.0, 80.0)))

    assert focused == [node.node_id]
    assert len(submitted) == 1


def test_viewport_camera_drag_breaks_a_pending_double_click() -> None:
    app = object.__new__(ViewerApp)
    app.router = SimpleNamespace(
        wants_camera=lambda: True,
        released=False,
        travel=0.0,
        started_with_left=True,
    )
    app.window = SimpleNamespace(style_scale=1.0)
    node = SceneNode(3, "elbow", NodeType.LINK, body_index=1)
    app.session = SimpleNamespace(
        node_by_object_id=lambda _object_id: node,
        submit=lambda _command: None,
    )
    app._pick_at = lambda _cursor: 9
    focused = []
    app._request_node_joint_focus = lambda candidate: focused.append(candidate) or True
    app._last_viewport_click = None
    down = InputState(left=True, over_viewport=True, cursor=(100.0, 80.0))
    up = InputState(over_viewport=True, cursor=(100.0, 80.0))

    app._poll_pick(down)
    app.router.released = True
    app._poll_pick(up)
    app.router.released = False
    app.router.travel = 20.0
    app._poll_pick(down)
    app.router.released = True
    app._poll_pick(up)
    app.router.released = False
    app.router.travel = 0.0
    app._poll_pick(down)
    app.router.released = True
    app._poll_pick(up)

    assert not focused
