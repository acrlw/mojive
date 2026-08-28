from __future__ import annotations

import ast
import importlib
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from forge_viewer import math3d
from forge_viewer.adapters.base import (
    AdapterCaps,
    FrameNeeds,
    NodeType,
    SceneFrame,
    SceneNode,
    SceneSource,
)
from forge_viewer.commands import ClearPerturb, Select
from forge_viewer.log import configure
from forge_viewer.session import Session, _node_local_bounds, _node_world_bounds
from forge_viewer.types import CameraView, InstancePoseSource, MeshData, MeshKey, MeshShape
from forge_viewer.ui import perturb as P

APP_PATH = Path(__file__).resolve().parents[1] / "src" / "forge_viewer" / "ui" / "app.py"


class FakeAdapter:
    def __init__(self) -> None:
        self.caps = AdapterCaps(
            name="fake", simulation=True, write_pose=True, perturb=True, raycast=False
        )
        self.body_xpos = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], np.float32)
        self.body_xmat = np.array([np.eye(3), np.eye(3)], np.float32)
        self.perturbs: list[tuple] = []

    @property
    def structure_revision(self) -> int:
        return 0

    def load(self, path): ...
    def reload(self): ...
    def reset(self): ...
    def step(self, count: int = 1): ...

    def frame(self, needs: FrameNeeds) -> SceneFrame:
        return SceneFrame(body_xpos=self.body_xpos.copy(), body_xmat=self.body_xmat.copy())

    def scene_source(self) -> SceneSource:
        return SceneSource()

    def nodes(self) -> list[SceneNode]:
        return [
            SceneNode(node_id=0, name="world", type=NodeType.WORLD, parent=-1, body_index=0),
            SceneNode(
                node_id=1,
                name="cube",
                type=NodeType.LINK,
                parent=0,
                object_id=1,
                posable=True,
                body_index=1,
            ),
        ]

    def joints(self):
        return []

    def actuators(self):
        return []

    def set_qpos(self, index, value):
        return True

    def set_ctrl(self, index, value):
        return True

    def set_pose(self, node_id, position, rotation):
        return True

    def apply_perturb(self, node_id, target_position, target_rotation, mode):
        self.perturbs.append((node_id, np.array(target_position), np.array(target_rotation), mode))
        return True

    def clear_perturb(self):
        self.perturbs.clear()

    def raycast(self, origin, direction):
        return (0, float("inf"))

    def camera_hint(self):
        return None

    def timestep(self):
        return 0.002

    def release(self): ...


RECT = (0.0, 0.0, 960.0, 720.0)


def make_session() -> tuple[Session, FakeAdapter]:
    adapter = FakeAdapter()
    session = Session(adapter)
    session.submit(Select(1))
    session.tick(FrameNeeds())
    return session, adapter


def side_camera(eye=(10.0, 0.0, 0.0)) -> CameraView:

    return CameraView(
        eye=np.array(eye, np.float32),
        target=np.zeros(3, np.float32),
        up=np.array([0.0, 0.0, 1.0], np.float32),
        aspect=1.0,
    )


def cursor_ray(cam: CameraView, ndc=(0.3, 0.15)):
    return math3d.unproject_ray(ndc[0], ndc[1], cam.view_matrix(), cam.proj_matrix())


def view_depth(cam: CameraView, point) -> float:
    _, _, forward = P.camera_basis(cam)
    return float(np.dot(np.asarray(point, np.float64) - np.asarray(cam.eye, np.float64), forward))


def spring_drag(
    session, adapter, cam, controller, origin, direction, *, live_plane: bool, steps=40
):

    st = session.perturb
    depths = []
    for _ in range(steps):
        if live_plane:
            st.plane_depth = P.freeze_plane_depth(cam, adapter.body_xpos[1])
        controller.drag_translate(session, cam, origin, direction)

        adapter.body_xpos[1] += 0.5 * (np.asarray(st.target_pos) - adapter.body_xpos[1])
        session.tick(FrameNeeds())
        depths.append(view_depth(cam, adapter.body_xpos[1]))
    return depths


def test_translation_plane_depth_is_frozen_at_press():

    session, adapter = make_session()
    cam = side_camera()
    controller = P.PerturbController()
    node = session.selected_node

    grab = np.array([1.0, 0.0, 0.0], np.float32)
    controller.begin(session, cam, node, grab, "translate")
    assert session.perturb.plane_depth == pytest.approx(9.0)

    origin, direction = cursor_ray(cam)
    depths = spring_drag(session, adapter, cam, controller, origin, direction, live_plane=False)

    assert session.perturb.plane_depth == pytest.approx(9.0)
    assert max(depths) - min(depths) < 1e-3
    assert depths[-1] == pytest.approx(10.0, abs=1e-3)


def test_cursor_grab_point_starts_without_a_screen_space_jump():
    session, _adapter = make_session()
    cam = side_camera()
    controller = P.PerturbController()
    origin, direction = cursor_ray(cam, ndc=(0.3, 0.15))
    body_origin = np.zeros(3, np.float32)
    grab = P.cursor_grab_point(cam, body_origin, origin, direction)

    controller.begin(session, cam, session.selected_node, grab, "translate")
    target_under_cursor = controller.drag_translate(session, cam, origin, direction)

    assert target_under_cursor == pytest.approx(grab)
    assert session.perturb.target_pos == pytest.approx(body_origin, abs=1e-6)


def test_unfrozen_plane_runs_away():

    session, adapter = make_session()
    cam = side_camera()
    controller = P.PerturbController()
    controller.begin(session, cam, session.selected_node, np.array([1.0, 0.0, 0.0]), "translate")
    origin, direction = cursor_ray(cam)
    depths = spring_drag(session, adapter, cam, controller, origin, direction, live_plane=True)
    assert depths[-1] - depths[0] > 10.0


def test_target_point_stays_on_the_frozen_plane_even_after_the_object_moves():

    session, adapter = make_session()
    cam = side_camera()
    controller = P.PerturbController()
    controller.begin(session, cam, session.selected_node, np.zeros(3), "translate")
    origin, direction = cursor_ray(cam)
    first = controller.drag_translate(session, cam, origin, direction).copy()

    adapter.body_xpos[1] = np.array([6.0, 0.0, 0.0], np.float32)
    session.tick(FrameNeeds())
    again = controller.drag_translate(session, cam, origin, direction)

    assert view_depth(cam, first) == pytest.approx(10.0)
    assert np.allclose(first, again)


@pytest.mark.parametrize("pixels", [-137.0, -40.0, 40.0, 137.0])
def test_twist_is_zero_point_four_degrees_per_pixel(pixels):

    cam = side_camera()
    expected = np.deg2rad(0.4 * abs(pixels))
    for dx, dy in ((pixels, 0.0), (0.0, pixels)):
        session, _ = make_session()
        controller = P.PerturbController()
        controller.begin(session, cam, session.selected_node, np.zeros(3), "rotate")
        start = np.array(session.perturb.start_mat, np.float64)
        for _ in range(10):
            controller.drag_rotate(session, cam, dx / 10.0, dy / 10.0)
        delta = np.asarray(session.perturb.target_mat, np.float64) @ start.T
        assert np.linalg.norm(P.rotvec_of(delta)) == pytest.approx(expected, rel=1e-6)


def test_twist_direction_is_read_off_the_screen():

    cam = side_camera()
    rect = (0.0, 0.0, 800.0, 600.0)
    near = np.array([1.0, 0.0, 0.0])
    base = P.project(cam, near[None], rect)[0]

    right = math3d.rotvec_to_mat3(P.delta_rotvec(cam, 50.0, 0.0)) @ near
    moved = P.project(cam, right[None], rect)[0]
    assert moved[0] - base[0] > 1.0
    assert abs(moved[1] - base[1]) < 1e-6

    down = math3d.rotvec_to_mat3(P.delta_rotvec(cam, 0.0, 50.0)) @ near
    moved = P.project(cam, down[None], rect)[0]
    assert moved[1] - base[1] > 1.0
    assert abs(moved[0] - base[0]) < 1e-6


def test_twist_is_a_general_three_d_rotation():

    cam = side_camera()
    session, _ = make_session()
    controller = P.PerturbController()
    controller.begin(session, cam, session.selected_node, np.zeros(3), "rotate")
    controller.drag_rotate(session, cam, 60.0, 0.0)
    controller.drag_rotate(session, cam, 0.0, 60.0)
    axis = P.rotvec_of(np.asarray(session.perturb.target_mat, np.float64))
    axis = axis / np.linalg.norm(axis)
    assert np.count_nonzero(np.abs(axis) > 0.2) >= 2


def test_rotate_mode_submits_a_target_not_a_teleport():

    session, adapter = make_session()
    cam = side_camera()
    controller = P.PerturbController()
    controller.begin(session, cam, session.selected_node, np.zeros(3), "rotate")
    controller.drag_rotate(session, cam, 90.0, 0.0)
    controller.apply(session)
    assert len(adapter.perturbs) == 1
    _node, target_pos, target_rot, mode = adapter.perturbs[0]
    assert mode == "rotate"
    assert np.linalg.norm(target_rot - np.eye(3)) > 0.0
    assert np.allclose(target_pos, 0.0)
    assert np.allclose(adapter.body_xmat[1], np.eye(3))


def test_silhouette_has_six_edges_from_a_general_viewpoint():

    edges = P.silhouette_edges(np.zeros(3), np.eye(3), np.ones(3), np.array([5.0, 6.0, 7.0]))
    assert len(edges) == 6


def test_silhouette_degenerates_to_four_edges_face_on():

    for eye in (np.array([9.0, 0.0, 0.0]), np.array([0.0, -12.0, 0.0]), np.array([0.0, 0.0, 7.0])):
        assert len(P.silhouette_edges(np.zeros(3), np.eye(3), np.ones(3), eye)) == 4


def test_silhouette_is_empty_when_the_viewpoint_is_inside_the_box():

    for eye in (np.zeros(3), np.array([0.5, -0.4, 0.9])):
        assert P.silhouette_edges(np.zeros(3), np.eye(3), np.ones(3), eye) == []


def test_silhouette_uses_the_viewpoint_not_one_forward_direction():

    eye = np.array([1.05, 1.05, 0.0])
    assert len(P.silhouette_edges(np.zeros(3), np.eye(3), np.ones(3), eye)) == 6


def test_silhouette_follows_the_orientation():

    eye = np.array([5.0, 6.0, 7.0])
    a = P.silhouette_edges(np.zeros(3), np.eye(3), np.ones(3), eye)
    r = math3d.axis_angle_to_mat3([0.0, 0.0, 1.0], np.deg2rad(37.0))
    b = P.silhouette_edges(np.zeros(3), r, np.ones(3), eye)
    assert len(a) == len(b) == 6
    assert not np.allclose(
        np.sort(np.ravel([e[0] for e in a])), np.sort(np.ravel([e[0] for e in b]))
    )


def test_mark_size_reuses_the_position_gizmo_screen_scale():

    cam = side_camera()
    near = P.world_scale(cam, np.array([9.0, 0.0, 0.0]), RECT[3])
    far = P.world_scale(cam, np.array([-10.0, 0.0, 0.0]), RECT[3])
    assert far == pytest.approx(near * 20.0, rel=1e-6)


def test_target_local_bounds_combines_body_geometries_and_local_parts():
    body_rotation = math3d.axis_angle_to_mat3((0.0, 0.0, 1.0), np.deg2rad(37.0))
    body_position = np.array((4.0, -3.0, 2.0))
    geom_position = body_position + body_rotation @ np.array((1.0, 0.0, 0.0))
    cap_local = np.eye(4, dtype=np.float32)
    cap_local[2, 3] = 2.0
    source = SceneSource(
        geom_mesh=[
            MeshKey(MeshShape.BOX),
            MeshKey(MeshShape.CAPSULE_CAP),
            MeshKey(MeshShape.BOX),
        ],
        geom_size=np.array(((1.0, 2.0, 0.5), (0.25, 0.25, 0.5), (0.5, 0.5, 0.5))),
        geom_body=np.array((1, 1, 1), np.int32),
        geom_source=np.array((0, 0, 1), np.int32),
        geom_pose_source=np.full(3, int(InstancePoseSource.GEOM), np.uint8),
        geom_local=np.stack((np.eye(4), cap_local, np.eye(4))).astype(np.float32),
        geom_infinite_plane=np.zeros(3, bool),
    )
    frame = SceneFrame(
        body_xpos=np.array(((0.0, 0.0, 0.0), body_position)),
        body_xmat=np.array((np.eye(3), body_rotation)),
        geom_xpos=np.array((geom_position, body_position + body_rotation @ (8.0, 0.0, 0.0))),
        geom_xmat=np.array((body_rotation, body_rotation)),
    )
    geom = SceneNode(10, "part", NodeType.GEOM, body_index=1, geom_index=0)
    body = SceneNode(11, "body", NodeType.LINK, body_index=1)

    geom_center, geom_half = _node_local_bounds(source, frame, geom)
    body_center, body_half = _node_local_bounds(source, frame, body)

    assert geom_center == pytest.approx((1.0, 0.0, 1.0))
    assert geom_half == pytest.approx((1.0, 2.0, 1.5))
    assert body_center == pytest.approx((4.25, 0.0, 1.0))
    assert body_half == pytest.approx((4.25, 2.0, 1.5))


def test_target_local_bounds_uses_asset_vertices_and_rejects_missing_geometry():
    key = MeshKey(MeshShape.ASSET, 7)
    positions = np.array(((-1.0, -2.0, -3.0), (4.0, 5.0, 6.0)), np.float32)
    mesh = MeshData(
        positions=positions,
        normals=np.zeros_like(positions),
        uvs=np.zeros((2, 2), np.float32),
        indices=np.zeros(0, np.uint32),
    )
    source = SceneSource(
        meshes={key: mesh},
        geom_mesh=[key],
        geom_size=np.array(((2.0, 3.0, 4.0),)),
        geom_body=np.array((1,), np.int32),
        geom_source=np.array((0,), np.int32),
        geom_pose_source=np.array((int(InstancePoseSource.GEOM),), np.uint8),
        geom_local=np.array((np.eye(4),), np.float32),
        geom_infinite_plane=np.zeros(1, bool),
    )
    frame = SceneFrame(
        body_xpos=np.zeros((2, 3)),
        body_xmat=np.array((np.eye(3), np.eye(3))),
        geom_xpos=np.array(((1.0, 0.0, 0.0),)),
        geom_xmat=np.array((np.eye(3),)),
    )
    node = SceneNode(1, "mesh", NodeType.LINK, body_index=1)

    center, half = _node_local_bounds(source, frame, node)

    assert center == pytest.approx((4.0, 4.5, 6.0))
    assert half == pytest.approx((5.0, 10.5, 18.0))
    source.meshes.clear()
    assert _node_local_bounds(source, frame, node) is None


def test_model_world_bounds_scans_only_its_geometry_once():
    source = SceneSource(
        geom_mesh=[MeshKey(MeshShape.BOX), MeshKey(MeshShape.BOX)],
        geom_size=np.array(((1.0, 2.0, 0.5), (4.0, 4.0, 4.0))),
        geom_body=np.array((1, 2), np.int32),
        geom_source=np.array((0, 1), np.int32),
        geom_pose_source=np.full(2, int(InstancePoseSource.GEOM), np.uint8),
        geom_local=np.array((np.eye(4), np.eye(4)), np.float32),
        geom_infinite_plane=np.zeros(2, bool),
    )
    frame = SceneFrame(
        geom_xpos=np.array(((5.0, -2.0, 1.0), (-20.0, 0.0, 0.0))),
        geom_xmat=np.array((np.eye(3), np.eye(3))),
    )
    model = SceneNode(1, "selected", NodeType.MODEL, children=[2], model_id=7)
    nodes = [
        model,
        SceneNode(2, "selected geom", NodeType.GEOM, parent=1, geom_index=0, model_id=7),
        SceneNode(3, "other geom", NodeType.GEOM, geom_index=1, model_id=8),
    ]

    center, half = _node_world_bounds(source, frame, model, nodes)

    assert center == pytest.approx((5.0, -2.0, 1.0))
    assert half == pytest.approx((1.0, 2.0, 0.5))


def test_axes_stick_out_of_the_outline():

    assert P.AXIS_OVERSHOOT > 1.0


class FakeCaps:
    def __init__(self, debug_draw: bool) -> None:
        self.name = "fake"
        self.debug_draw = debug_draw


class FakeBackend:
    def __init__(self) -> None:
        self.caps = FakeCaps(False)
        self.debug = None


def test_missing_debug_draw_is_reported_not_swallowed(capsys):

    configure(stream=sys.stderr)
    session, _ = make_session()
    controller = P.PerturbController()
    controller.begin(session, side_camera(), session.selected_node, np.zeros(3), "translate")
    budget = controller.publish_marks(FakeBackend(), session, side_camera(), rect=RECT)
    assert budget.dropped == 1
    assert budget.note
    assert "fake" in capsys.readouterr().err


def test_fallback_draws_a_solid_silhouette_not_a_wire_box():

    session, _ = make_session()
    controller = P.PerturbController()
    cam = side_camera()
    controller.begin(session, cam, session.selected_node, np.zeros(3), "rotate")
    controller.drag_rotate(session, cam, 40.0, 20.0)
    a, b = P.fallback_segments(
        cam, session.perturb, (0.0, 0.0, 800.0, 600.0), session.frame.body_xpos[1]
    )
    assert len(a) == len(b) == 6
    assert np.all(np.linalg.norm(a[:, :2] - b[:, :2], axis=1) > 0.5)


def test_rounded_loop_trims_screen_space_corners():
    cam = side_camera()
    rect = (0.0, 0.0, 800.0, 600.0)
    loop = np.array([[-1.0, 0.0, -1.0], [1.0, 0.0, -1.0], [1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]])

    rounded = P.rounded_loop(loop, cam, rect, 8.0, segments=5)

    assert rounded.shape == (24, 3)
    assert not any(np.allclose(point, corner) for point in rounded for corner in loop)
    assert np.max(np.abs(rounded[:, [0, 2]])) <= 1.0


def test_perturb_corner_radius_is_runtime_configurable():
    controller = P.PerturbController()
    assert controller.outline_corner_radius_pt == P.OUTLINE_CORNER_RADIUS_PT

    controller.outline_corner_radius_pt = 12.5

    assert controller.outline_corner_radius_pt == 12.5


class Layer:
    def __init__(self) -> None:
        self.ids: list[str] = []
        self.calls: dict[str, tuple] = {}

    def _record(self, i, *a):
        self.ids.append(i)
        self.calls[i] = a

    line = lines = point = points = box = frame = _record

    def drag_link(self, ident, *args, **kwargs):
        self._record(ident, *args, kwargs)

    def arrow(self, ident, *args, start_mask_px=0.0):
        self._record(ident, *args, start_mask_px)

    def arrows(self, ident, *args, start_mask_px=0.0):
        self._record(ident, *args, start_mask_px)

    def polyline(
        self,
        ident,
        points,
        color,
        width,
        *,
        closed=False,
        duration=-1.0,
    ):
        self._record(ident, points, color, width, closed)

    def clear(self):
        self.ids.clear()


class DebugDraw:
    def __init__(self) -> None:
        self.layers: dict[str, Layer] = {}
        self.tiers: dict[str, str] = {}

    def layer(self, name, occlusion):
        self.tiers[name] = occlusion
        return self.layers.setdefault(name, Layer())


class DebugBackend:
    def __init__(self) -> None:
        self.caps = FakeCaps(True)
        self.debug = DebugDraw()


class FakeOcclusion:
    DEPTH = "depth"
    ALWAYS = "always"
    GHOST = "ghost"


OCCLUSION = P.Occlusion if P.Occlusion is not None else FakeOcclusion


def test_marks_use_stable_ids_so_dragging_does_not_grow(monkeypatch):

    monkeypatch.setattr(P, "Occlusion", OCCLUSION)
    session, _ = make_session()
    cam = side_camera()
    controller = P.PerturbController()
    backend = DebugBackend()
    controller.begin(session, cam, session.selected_node, np.zeros(3), "translate")
    origin, direction = cursor_ray(cam)
    seen: set[str] = set()
    for _ in range(100):
        controller.drag_translate(session, cam, origin, direction)
        controller.publish_marks(backend, session, cam, rect=RECT)
        layer = backend.debug.layers["ui.perturb.drag"]
        seen.update(layer.ids)
        assert len(layer.ids) == 1
        layer.ids.clear()
    assert seen == {"perturb.drag"}


def test_layers_and_occlusion_tiers_match_the_spec(monkeypatch):

    monkeypatch.setattr(P, "Occlusion", OCCLUSION)
    session, _ = make_session()
    cam = side_camera()
    controller = P.PerturbController()
    backend = DebugBackend()

    controller.begin(session, cam, session.selected_node, np.zeros(3), "translate")
    controller.publish_marks(backend, session, cam, rect=RECT)
    assert backend.debug.tiers["ui.perturb.drag"] == OCCLUSION.ALWAYS

    controller.begin(session, cam, session.selected_node, np.zeros(3), "rotate")
    controller.drag_rotate(session, cam, 30.0, 10.0)
    budget = controller.publish_marks(backend, session, cam, rect=RECT)
    assert backend.debug.tiers["ui.perturb.mark"] == OCCLUSION.ALWAYS

    ids = backend.debug.layers["ui.perturb.mark"].ids
    assert ids == [
        "perturb.outline.border",
        "perturb.outline",
        "perturb.axes",
        "perturb.center.edge",
        "perturb.center",
    ]
    layer = backend.debug.layers["ui.perturb.mark"]
    border = layer.calls["perturb.outline.border"]
    core = layer.calls["perturb.outline"]
    assert border[1] == P.OUTLINE_BORDER_RGBA
    assert core[1] == P.OUTLINE_RGBA
    assert border[2] > core[2]
    assert border[3] is core[3] is True
    expected_points = 6 * (P.OUTLINE_CORNER_SEGMENTS + 1)
    assert len(border[0]) == len(core[0]) == expected_points
    shell = P.CENTER_SHELL_RADIUS * P.SIZE_PT
    axes = layer.calls["perturb.axes"]
    assert axes[4] == pytest.approx(shell)
    center = layer.calls["perturb.center"]
    edge = layer.calls["perturb.center.edge"]
    assert edge[1] == pytest.approx(P.CONTRAST_EDGE_COLOR)
    assert edge[2] - center[2] == pytest.approx(P.CONTRAST_EDGE_PT)
    assert center[1] == pytest.approx(P.CENTER_COLOR)
    assert center[2] == pytest.approx(P.CENTER_RADIUS * P.SIZE_PT)
    assert budget.dropped == 0


def test_twist_gizmo_scales_world_length_and_pixel_geometry(monkeypatch):
    monkeypatch.setattr(P, "Occlusion", OCCLUSION)
    session, _ = make_session()
    cam = side_camera()
    controller = P.PerturbController()
    backend = DebugBackend()
    controller.begin(session, cam, session.selected_node, np.zeros(3), "rotate")

    controller.publish_marks(backend, session, cam, rect=RECT)
    base = backend.debug.layers[P.MARK_LAYER].calls
    base_axes = base["perturb.axes"]
    base_length = np.linalg.norm(base_axes[1] - base_axes[0], axis=1)
    base_width = base_axes[3]
    base_center_radius = base["perturb.center"][2]

    controller.publish_marks(
        backend,
        session,
        cam,
        rect=RECT,
        ui_scale=2.0,
        style_scale=2.0,
    )
    scaled = backend.debug.layers[P.MARK_LAYER].calls
    scaled_axes = scaled["perturb.axes"]
    scaled_length = np.linalg.norm(scaled_axes[1] - scaled_axes[0], axis=1)

    assert scaled_length == pytest.approx(base_length * 2.0)
    assert scaled_axes[3] == pytest.approx(base_width * 2.0)
    assert scaled["perturb.center"][2] == pytest.approx(base_center_radius * 2.0)


def test_twist_marks_follow_the_body_position(monkeypatch):

    monkeypatch.setattr(P, "Occlusion", OCCLUSION)
    session, adapter = make_session()
    cam = side_camera()
    controller = P.PerturbController()
    backend = DebugBackend()
    controller.begin(session, cam, session.selected_node, np.zeros(3), "rotate")

    center = np.array([2.0, 3.0, 4.0], np.float32)
    adapter.body_xpos[1] = center
    session.tick(FrameNeeds())
    seen: list[tuple[np.ndarray, np.ndarray]] = []
    silhouette = P.silhouette_edges

    def record_silhouette(mark_center, rotation, *args):
        seen.append((np.asarray(mark_center).copy(), np.asarray(rotation).copy()))
        return silhouette(mark_center, rotation, *args)

    monkeypatch.setattr(P, "silhouette_edges", record_silhouette)
    session.perturb.start_mat = math3d.axis_angle_to_mat3([1.0, 0.0, 0.0], 0.7)
    controller.drag_rotate(session, cam, 40.0, 20.0)
    controller.publish_marks(backend, session, cam, rect=RECT)

    layer = backend.debug.layers[P.MARK_LAYER]
    assert seen[-1][0] == pytest.approx(center)
    assert seen[-1][1] == pytest.approx(session.perturb.target_mat)
    assert not np.allclose(seen[-1][1], session.perturb.start_mat)
    assert layer.calls["perturb.axes"][0] == pytest.approx(np.broadcast_to(center, (3, 3)))
    assert layer.calls["perturb.center"][0] == pytest.approx(center)


def test_twist_outline_tracks_cached_bounds_while_axes_stay_on_body_origin(monkeypatch):
    monkeypatch.setattr(P, "Occlusion", OCCLUSION)
    session, adapter = make_session()
    cam = side_camera((10.0, 7.0, 5.0))
    controller = P.PerturbController()
    local_center = np.array((0.8, -0.4, 0.3), np.float32)
    local_half = np.array((2.0, 1.0, 0.5), np.float32)
    controller.begin(
        session,
        cam,
        session.selected_node,
        np.zeros(3),
        "rotate",
        local_bounds=(local_center, local_half),
    )
    target_rotation = math3d.axis_angle_to_mat3((0.0, 0.0, 1.0), np.deg2rad(70.0))
    session.perturb.target_mat = target_rotation.astype(np.float32)
    body_center = np.array((2.0, 3.0, 4.0), np.float32)
    adapter.body_xpos[1] = body_center
    session.tick(FrameNeeds())

    seen: list[tuple[np.ndarray, np.ndarray]] = []
    silhouette = P.silhouette_edges

    def record_silhouette(center, rotation, half, eye):
        seen.append((np.asarray(center).copy(), np.asarray(half).copy()))
        return silhouette(center, rotation, half, eye)

    monkeypatch.setattr(P, "silhouette_edges", record_silhouette)
    backend = DebugBackend()
    controller.publish_marks(backend, session, cam, rect=RECT)

    expected_center = body_center + target_rotation @ local_center
    assert seen[-1][0] == pytest.approx(expected_center)
    assert np.all(seen[-1][1] > local_half)
    axes = backend.debug.layers[P.MARK_LAYER].calls["perturb.axes"]
    assert axes[0] == pytest.approx(np.broadcast_to(body_center, (3, 3)))


def test_twist_axes_reuse_position_gizmo_view_angle_fade(monkeypatch):

    monkeypatch.setattr(P, "Occlusion", OCCLUSION)
    session, _ = make_session()
    cam = side_camera()
    controller = P.PerturbController()
    backend = DebugBackend()
    controller.begin(session, cam, session.selected_node, np.zeros(3), "rotate")
    controller.publish_marks(backend, session, cam, rect=RECT)

    layer = backend.debug.layers[P.MARK_LAYER]
    colors = layer.calls["perturb.axes"][2]
    alpha_by_axis = {
        int(np.argmin(np.linalg.norm(P.AXIS_COLORS[:, :3] - color[:3], axis=1))): float(color[3])
        for color in colors
    }
    assert [alpha_by_axis[axis] for axis in range(3)] == pytest.approx([0.0, 1.0, 1.0])


def test_twist_axes_use_the_same_far_to_near_order_as_view_gizmo(monkeypatch):
    monkeypatch.setattr(P, "Occlusion", OCCLUSION)
    session, _ = make_session()
    cam = side_camera((4.0, 3.0, 2.0))
    controller = P.PerturbController()
    backend = DebugBackend()
    controller.begin(session, cam, session.selected_node, np.zeros(3), "rotate")
    controller.publish_marks(backend, session, cam, rect=RECT)

    colors = backend.debug.layers[P.MARK_LAYER].calls["perturb.axes"][2]
    expected = P.axis_draw_order(cam, session.perturb.target_mat)
    assert colors[:, :3] == pytest.approx(P.AXIS_COLORS[list(expected), :3])


def test_switching_mode_wipes_the_other_layer(monkeypatch):

    monkeypatch.setattr(P, "Occlusion", OCCLUSION)
    session, _ = make_session()
    cam = side_camera()
    controller = P.PerturbController()
    backend = DebugBackend()

    controller.begin(session, cam, session.selected_node, np.zeros(3), "rotate")
    controller.drag_rotate(session, cam, 30.0, 10.0)
    controller.publish_marks(backend, session, cam, rect=RECT)
    assert backend.debug.layers["ui.perturb.mark"].ids

    controller.begin(session, cam, session.selected_node, np.zeros(3), "translate")
    controller.publish_marks(backend, session, cam, rect=RECT)
    assert backend.debug.layers["ui.perturb.mark"].ids == []
    assert backend.debug.layers["ui.perturb.drag"].ids

    session.submit(ClearPerturb())
    controller.publish_marks(backend, session, cam, rect=RECT)
    assert backend.debug.layers["ui.perturb.drag"].ids == []


def _stub(name: str, **attrs) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def load_app_module():

    saved = {}
    for name, attrs in (
        (
            "forge_viewer.ui.window",
            {
                "Window": type("Window", (), {}),
                "WindowConfig": type("WindowConfig", (), {}),
            },
        ),
        (
            "forge_viewer.ui.theme",
            {
                "AXIS_COLORS": {"x": (1, 0, 0, 1), "y": (0, 1, 0, 1), "z": (0, 0, 1, 1)},
                "THEME": object(),
                "Theme": type("Theme", (), {}),
            },
        ),
        (
            "forge_viewer.ui.panels",
            {"PanelContext": type("PanelContext", (), {}), "PanelSet": type("PanelSet", (), {})},
        ),
    ):
        try:
            module = importlib.import_module(name)
            if all(hasattr(module, k) for k in attrs):
                continue
        except ImportError:
            pass
        saved[name] = sys.modules.get(name)
        sys.modules[name] = _stub(name, **attrs)
    try:
        return importlib.import_module("forge_viewer.ui.app")
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


def _calls_in_methods() -> dict[str, set[str]]:

    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ViewerApp")
    out: dict[str, set[str]] = {}
    for node in cls.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        names = set()
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call):
                func = sub.func
                names.add(func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", ""))
        out[node.name] = names
    return out


def test_startup_is_the_only_place_that_shows_the_window():

    calls = _calls_in_methods()
    showing = [name for name, names in calls.items() if "show" in names]
    assert showing == ["_startup"]

    for entry in ("run", "sync"):
        assert "_startup" in calls[entry]
        assert not ({"show", "set_scene", "_frame_scene"} & calls[entry])


def test_run_and_sync_go_through_the_same_startup(monkeypatch):

    app_mod = load_app_module()
    ViewerApp = app_mod.ViewerApp
    seen: list[str] = []

    monkeypatch.setattr(ViewerApp, "_startup", lambda self: seen.append("startup"))
    monkeypatch.setattr(ViewerApp, "frame", lambda self: seen.append("frame"))
    monkeypatch.setattr(ViewerApp, "release", lambda self: seen.append("release"))
    monkeypatch.setattr(ViewerApp, "_should_close", lambda self: True)

    app = ViewerApp.__new__(ViewerApp)
    app._frame_index = 0
    app.run(max_frames=0)
    app.sync()
    assert seen.count("startup") == 2
    assert "release" not in seen


def test_app_release_is_idempotent():
    app_mod = load_app_module()
    ViewerApp = app_mod.ViewerApp

    class Resource:
        def __init__(self) -> None:
            self.calls = 0

        def close(self) -> None:
            self.calls += 1

        def release(self) -> None:
            self.calls += 1

    app = ViewerApp.__new__(ViewerApp)
    app._released = False
    app._model_dialog = None
    app._scene_dialog = None
    app._resource_dialog = None
    app._resource_repair_dialog = None
    bridge = Resource()
    app.debug_bridge = bridge
    app.camera_preview = Resource()
    app.backend = Resource()
    app.session = Resource()

    app.release()
    app.release()

    assert app.debug_bridge is None
    assert bridge.calls == 1
    assert app.camera_preview.calls == 1
    assert app.backend.calls == 1
    assert app.session.calls == 1


def test_app_release_continues_after_one_resource_fails():
    app_mod = load_app_module()
    ViewerApp = app_mod.ViewerApp

    class FailingBridge:
        def close(self) -> None:
            raise RuntimeError("bridge failed")

    class Resource:
        def __init__(self) -> None:
            self.calls = 0

        def release(self) -> None:
            self.calls += 1

    app = ViewerApp.__new__(ViewerApp)
    app._released = False
    app._model_dialog = None
    app._scene_dialog = None
    app._resource_dialog = None
    app._resource_repair_dialog = None
    app.debug_bridge = FailingBridge()
    app.camera_preview = Resource()
    app.backend = Resource()
    app.session = Resource()

    app.release()

    assert app.camera_preview.calls == app.backend.calls == app.session.calls == 1


def test_frame_publishes_marks_between_tick_and_render():

    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ViewerApp")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "frame")
    watched = (
        "_claim_gesture",
        "_poll_camera",
        "_poll_perturb",
        "_poll_pick",
        "tick",
        "_publish_perturb_marks",
        "render",
    )
    order = []
    for sub in ast.walk(fn):
        if (
            isinstance(sub, ast.Call)
            and isinstance(sub.func, ast.Attribute)
            and sub.func.attr in watched
        ):
            order.append((sub.lineno, sub.func.attr))
    seq = [name for _, name in sorted(order)]

    for consumer in ("_poll_camera", "_poll_perturb", "_poll_pick"):
        assert seq.index("_claim_gesture") < seq.index(consumer)

    assert seq.index("tick") < seq.index("_publish_perturb_marks") < seq.index("render")
