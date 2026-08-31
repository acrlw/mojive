from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from forge_viewer import commands as cmd
from forge_viewer import math3d
from forge_viewer.adapters.base import FrameNeeds, NodeType, SceneFrame, SceneNode, SceneSource
from forge_viewer.adapters.static import StaticSceneAdapter
from forge_viewer.gizmo import camera_icon_segments, project
from forge_viewer.render.backend import BackendCaps
from forge_viewer.render.debugdraw import DebugDraw, Occlusion, PrimitiveType
from forge_viewer.scene import Scene
from forge_viewer.session import Session
from forge_viewer.types import CameraView, Light, LightSet, LightType
from forge_viewer.ui.gizmo import (
    ObjectGizmo,
    _node_pose,
    _set_camera_from_world,
    _set_light_from_world,
)
from forge_viewer.ui.scene_entities import (
    CAMERA_HELPER_SIZE_PT,
    HELPER_ICON_LAYER,
    HELPER_LAYER,
    LIGHT_HELPER_SCALE_PT,
    SceneEntityHelpers,
    camera_frustum_segments,
    camera_rotation,
    light_icon_segments,
    spot_cone_segments,
    spot_helper_length,
)


def test_perspective_frustum_uses_camera_projection_planes() -> None:
    view = CameraView(
        eye=np.zeros(3, np.float32),
        target=np.array((0.0, 0.0, -1.0), np.float32),
        up=np.array((0.0, 1.0, 0.0), np.float32),
        fov_y=np.deg2rad(90.0),
        aspect=2.0,
        near=1.0,
        far=3.0,
    )
    starts, ends = camera_frustum_segments(view)
    points = np.unique(np.concatenate((starts, ends)), axis=0)
    near = points[np.isclose(points[:, 2], -1.0)]
    far = points[np.isclose(points[:, 2], -3.0)]
    assert np.unique(np.abs(near[:, 0])) == pytest.approx([2.0])
    assert np.unique(np.abs(near[:, 1])) == pytest.approx([1.0])
    assert np.unique(np.abs(far[:, 0])) == pytest.approx([6.0])
    assert np.unique(np.abs(far[:, 1])) == pytest.approx([3.0])


def test_camera_helper_is_a_screen_facing_camera_glyph() -> None:
    editor = CameraView(
        eye=np.array((0.0, -5.0, 2.0), np.float32),
        target=np.array((0.0, 0.0, 1.0), np.float32),
        aspect=1.5,
    )
    eye = np.array((0.0, 0.0, 1.0), np.float32)
    views = (
        CameraView(eye=eye, target=np.array((1.0, 0.0, 1.0), np.float32)),
        CameraView(eye=eye, target=np.array((0.0, 1.0, 1.0), np.float32)),
    )

    starts, ends = camera_icon_segments(views, editor, 800.0, 26.0)

    assert starts.shape == ends.shape == (48, 3)
    # The editor glyph does not collapse or turn into a projection triangle
    # when the represented camera points in a different direction.
    assert starts[:24] == pytest.approx(starts[24:])
    assert ends[:24] == pytest.approx(ends[24:])


def test_spot_influence_uses_range_and_cutoff() -> None:
    light = Light(
        type=LightType.SPOT,
        position=np.zeros(3, np.float32),
        direction=np.array((0.0, 0.0, -1.0), np.float32),
        range=2.0,
        cutoff=45.0,
    )
    starts, ends = spot_cone_segments(light)
    points = np.concatenate((starts, ends))
    rim = points[np.isclose(points[:, 2], -2.0)]
    assert np.max(np.linalg.norm(rim[:, :2], axis=1)) == pytest.approx(2.0)


def test_spot_helper_is_screen_bounded_and_hidden_behind_camera() -> None:
    light = Light(
        type=LightType.SPOT,
        position=np.zeros(3, np.float32),
        direction=np.array((0.0, 0.0, -1.0), np.float32),
        range=10.0,
        cutoff=5.1,
    )
    visible = CameraView(
        eye=np.array((0.0, -3.0, 1.0), np.float32),
        target=np.zeros(3, np.float32),
        up=np.array((0.0, 0.0, 1.0), np.float32),
    )
    behind = CameraView(
        eye=np.zeros(3, np.float32),
        target=np.array((0.0, -1.0, 0.0), np.float32),
        up=np.array((0.0, 0.0, 1.0), np.float32),
    )

    assert 0.0 < spot_helper_length(light, visible, 800.0) < light.range
    assert (
        spot_helper_length(replace(light, position=np.array((0.0, 1.0, 0.0))), behind, 800.0) == 0.0
    )


def test_small_cutoff_spot_helper_stays_inside_its_screen_budget() -> None:
    camera = CameraView(
        eye=np.array((0.0, -3.0, 1.0), np.float32),
        target=np.zeros(3, np.float32),
        up=np.array((0.0, 0.0, 1.0), np.float32),
        aspect=1.8,
    )
    light = Light(
        type=LightType.SPOT,
        position=np.zeros(3, np.float32),
        direction=np.array((1.0, 0.0, 0.0), np.float32),
        range=10.0,
        cutoff=5.1,
    )
    rect = (0.0, 0.0, 1800.0, 1000.0)

    length = spot_helper_length(light, camera, rect[3])
    starts, ends = spot_cone_segments(light, length)
    screen = project(camera, np.concatenate((starts, ends)), rect)
    anchor = project(camera, [light.position], rect)[0]

    assert np.all(screen[:, 2] > 0.0)
    assert np.max(np.linalg.norm(screen[:, :2] - anchor[:2], axis=1)) <= 220.0


def test_near_camera_spot_helper_uses_its_projected_screen_extent() -> None:
    camera = CameraView(
        eye=np.array((-3.66184, -3.66184, 2.68583), np.float32),
        target=np.array((0.0, 0.0, 0.271), np.float32),
        up=np.array((0.0, 0.0, 1.0), np.float32),
        aspect=1.5,
        near=0.0143,
        far=244.0,
    )
    light = Light(
        type=LightType.SPOT,
        position=np.array((0.0, -6.0, 4.0), np.float32),
        direction=np.array((0.0023, 0.8854, -0.4649), np.float32),
        range=10.0,
        cutoff=5.1,
    )
    rect = (0.0, 0.0, 1200.0, 800.0)

    length = spot_helper_length(light, camera, rect[3])
    starts, ends = spot_cone_segments(light, length)
    screen = project(camera, np.concatenate((starts, ends)), rect)
    anchor = project(camera, [light.position], rect)[0]

    assert length < 0.02
    assert np.all(screen[:, 2] > camera.near)
    assert np.max(np.linalg.norm(screen[:, :2] - anchor[:2], axis=1)) <= 220.0


def test_orthographic_frustum_preserves_parallel_planes() -> None:
    view = CameraView(
        eye=np.zeros(3, np.float32),
        target=np.array((0.0, 0.0, -1.0), np.float32),
        up=np.array((0.0, 1.0, 0.0), np.float32),
        near=1.0,
        far=3.0,
        aspect=2.0,
        orthographic=True,
        ortho_height=4.0,
    )
    starts, ends = camera_frustum_segments(view)
    points = np.unique(np.concatenate((starts, ends)), axis=0)
    near = points[np.isclose(points[:, 2], -1.0)]
    far = points[np.isclose(points[:, 2], -3.0)]
    assert np.max(np.abs(near[:, :2]), axis=0) == pytest.approx((4.0, 2.0))
    assert np.max(np.abs(far[:, :2]), axis=0) == pytest.approx((4.0, 2.0))


def test_camera_rotation_handles_collinear_up_vector() -> None:
    basis = camera_rotation(
        CameraView(
            eye=np.zeros(3, np.float32),
            target=np.array((1.0, 0.0, 0.0), np.float32),
            up=np.array((1.0, 0.0, 0.0), np.float32),
        )
    )
    assert basis.T @ basis == pytest.approx(np.eye(3), abs=1e-6)
    assert np.linalg.det(basis) == pytest.approx(1.0)


def test_helpers_publish_selected_frustum_and_pick_camera_anchor() -> None:
    scene = Scene()
    camera_id = scene.add_camera(
        "shot",
        CameraView(
            eye=np.array((0.0, 0.0, 1.0), np.float32),
            target=np.array((0.0, 1.0, 1.0), np.float32),
            near=0.1,
            far=2.0,
        ),
    )
    session = Session(StaticSceneAdapter(scene))
    session.tick(FrameNeeds())
    node = next(node for node in session.nodes if node.type is NodeType.CAMERA)
    assert session.submit(cmd.Select(node.object_id))

    editor_camera = CameraView(
        eye=np.array((0.0, -5.0, 2.0), np.float32),
        target=np.array((0.0, 0.0, 1.0), np.float32),
        aspect=1.25,
    )
    backend = SimpleNamespace(debug=DebugDraw())
    helpers = SceneEntityHelpers()
    helpers.publish(backend, session, editor_camera, 800.0, 1.0)
    layer = backend.debug.layer(HELPER_LAYER)
    icon_layer = backend.debug.layer(HELPER_ICON_LAYER)
    assert layer.occlusion is Occlusion.GHOST
    assert icon_layer.occlusion is Occlusion.ALWAYS
    assert layer.count_of(PrimitiveType.LINE) == 12
    assert icon_layer.count_of(PrimitiveType.STROKE) == 24
    assert layer.count_of(PrimitiveType.POINT) == 0

    hit = helpers.pick(session, editor_camera, (0.0, 0.0, 1000.0, 800.0), (500.0, 400.0), 1.0)
    assert hit == node.object_id
    assert session.camera_view(camera_id) is not None


def test_selected_camera_frustum_uses_preview_aspect() -> None:
    scene = Scene()
    scene.add_camera(
        "shot",
        CameraView(
            eye=np.zeros(3, np.float32),
            target=np.array((0.0, 0.0, -1.0), np.float32),
            up=np.array((0.0, 1.0, 0.0), np.float32),
            fov_y=np.deg2rad(90.0),
            near=1.0,
            far=3.0,
        ),
    )
    session = Session(StaticSceneAdapter(scene))
    session.tick(FrameNeeds())
    node = next(node for node in session.nodes if node.type is NodeType.CAMERA)
    assert session.submit(cmd.Select(node.object_id))

    backend = SimpleNamespace(debug=DebugDraw())
    helpers = SceneEntityHelpers()
    helpers.publish(
        backend,
        session,
        CameraView(),
        800.0,
        1.0,
        selected_camera_aspect=16.0 / 9.0,
    )

    store = backend.debug.layer(HELPER_LAYER)._stores[PrimitiveType.LINE]
    frustum = store.positions[store.count - 12 : store.count]
    near = np.unique(frustum.reshape(-1, 3), axis=0)
    near = near[np.isclose(near[:, 2], -1.0)]
    assert np.unique(np.abs(near[:, 0])) == pytest.approx([16.0 / 9.0])
    assert np.unique(np.abs(near[:, 1])) == pytest.approx([1.0])


def test_view_through_camera_hides_editor_helpers() -> None:
    scene = Scene()
    camera_id = scene.add_camera(
        "moving shot",
        CameraView(
            eye=np.array((0.0, 0.0, 1.0), np.float32),
            target=np.array((0.0, 1.0, 1.0), np.float32),
            near=0.1,
            far=4.0,
        ),
    )
    scene.add_light(
        "moving key",
        Light(
            type=LightType.SPOT,
            position=np.array((1.0, -2.0, 3.0), np.float32),
            direction=np.array((0.0, 1.0, -1.0), np.float32),
        ),
    )
    session = Session(StaticSceneAdapter(scene))
    session.tick(FrameNeeds())
    node = next(node for node in session.nodes if node.type is NodeType.CAMERA)
    assert session.submit(cmd.Select(node.object_id))
    view = session.camera_view(camera_id)
    backend = SimpleNamespace(debug=DebugDraw())
    helpers = SceneEntityHelpers()

    helpers.publish(backend, session, view, 800.0, 1.0, True)

    layer = backend.debug.layer(HELPER_LAYER)
    icon_layer = backend.debug.layer(HELPER_ICON_LAYER)
    assert layer.count_of(PrimitiveType.LINE) == 0
    assert layer.count_of(PrimitiveType.POINT) == 0
    assert icon_layer.count_of(PrimitiveType.LINE) == 0
    assert icon_layer.count_of(PrimitiveType.STROKE) == 0
    assert helpers.pick(session, view, (0.0, 0.0, 1000.0, 800.0), (500.0, 400.0), 1.0, True) == 0


def test_unselected_light_helpers_use_semantic_icons_without_direction_clutter() -> None:
    scene = Scene()
    for index, light_type in enumerate((LightType.POINT, LightType.DIRECTIONAL, LightType.SPOT)):
        scene.add_light(
            f"light{index}",
            Light(
                type=light_type,
                position=np.array((float(index), 0.0, 1.0), np.float32),
                direction=np.array((0.0, 0.0, -1.0), np.float32),
            ),
        )
    session = Session(StaticSceneAdapter(scene))
    backend = SimpleNamespace(debug=DebugDraw())

    SceneEntityHelpers(show_influence=False).publish(backend, session, CameraView(), 800.0, 1.0)

    layer = backend.debug.layer(HELPER_LAYER)
    icon_layer = backend.debug.layer(HELPER_ICON_LAYER)
    assert layer.count_of(PrimitiveType.POINT) == 0
    assert layer.count_of(PrimitiveType.ARROW) == 0
    assert layer.count_of(PrimitiveType.LINE) == 0
    assert icon_layer.count_of(PrimitiveType.LINE) == 30
    assert icon_layer.count_of(PrimitiveType.STROKE) == 36
    assert set(icon_layer._index) == {
        f"light:{object_id}:{part}"
        for object_id in (node.object_id for node in session.nodes if node.type is NodeType.LIGHT)
        for part in ("ring", "details")
    }


def test_selected_spot_light_publishes_icon_direction_and_influence() -> None:
    scene = Scene()
    scene.add_light(
        "key",
        Light(
            type=LightType.SPOT,
            position=np.array((0.0, 0.0, 1.0), np.float32),
            direction=np.array((0.0, 1.0, -0.25), np.float32),
            range=2.0,
            cutoff=30.0,
        ),
    )
    session = Session(StaticSceneAdapter(scene))
    node = next(node for node in session.nodes if node.type is NodeType.LIGHT)
    assert session.submit(cmd.Select(node.object_id))
    backend = SimpleNamespace(debug=DebugDraw())
    camera = CameraView(
        eye=np.array((0.0, -5.0, 2.0), np.float32),
        target=np.array((0.0, 0.0, 1.0), np.float32),
    )

    SceneEntityHelpers().publish(backend, session, camera, 800.0, 1.0)

    layer = backend.debug.layer(HELPER_LAYER)
    icon_layer = backend.debug.layer(HELPER_ICON_LAYER)
    assert {"lights:directions", f"light:{node.object_id}:range"} <= set(layer._index)
    assert {f"light:{node.object_id}:ring", f"light:{node.object_id}:details"} == set(
        icon_layer._index
    )


def test_light_helper_icon_geometry_is_screen_facing_and_batched() -> None:
    positions = np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)), np.float32)
    starts, ends = light_icon_segments(positions, CameraView(), 800.0)
    assert starts.shape == ends.shape == (44, 3)
    assert np.all(np.isfinite(starts))


def test_scene_entity_icon_geometry_scales_with_its_stroke_width() -> None:
    camera = CameraView(
        eye=np.array((0.0, -5.0, 2.0), np.float32),
        target=np.array((0.0, 0.0, 1.0), np.float32),
        aspect=1.5,
    )
    position = np.array((0.0, 0.0, 1.0), np.float32)
    represented = CameraView(
        eye=position,
        target=np.array((1.0, 0.0, 1.0), np.float32),
    )
    rect = (0.0, 0.0, 1200.0, 800.0)

    camera_sizes = []
    light_sizes = []
    for ui_scale in (1.0, 3.0):
        starts, ends = camera_icon_segments(
            (represented,),
            camera,
            rect[3],
            CAMERA_HELPER_SIZE_PT * ui_scale,
            visible_only=True,
        )
        screen = project(camera, np.concatenate((starts, ends)), rect)
        camera_sizes.append(float(np.ptp(screen[:, :2], axis=0).max()))

        starts, ends = light_icon_segments(
            position[None, :],
            camera,
            rect[3],
            ui_scale,
        )
        screen = project(camera, np.concatenate((starts, ends)), rect)
        light_sizes.append(float(np.ptp(screen[:, :2], axis=0).max()))

    assert camera_sizes[1] / camera_sizes[0] == pytest.approx(3.0)
    assert light_sizes[1] / light_sizes[0] == pytest.approx(3.0)


def test_camera_and_light_helper_sizes_are_visually_balanced() -> None:
    assert pytest.approx(24.0) == CAMERA_HELPER_SIZE_PT
    assert pytest.approx(1.2) == LIGHT_HELPER_SCALE_PT
    # The light's outer ray diameter is 15.6 authored units. Keep it close to
    # the camera icon without making two different helper types identical.
    light_diameter = 15.6 * LIGHT_HELPER_SCALE_PT
    assert 1.15 < CAMERA_HELPER_SIZE_PT / light_diameter < 1.4


def test_scene_entity_helpers_index_large_hierarchies_once_per_structure() -> None:
    class LargeSession:
        structure_generation = 7
        selected = 0
        source = SceneSource(lights=LightSet())
        frame = SceneFrame()

        def __init__(self) -> None:
            self.node_reads = 0
            self._nodes = [SceneNode(i, str(i), NodeType.LINK) for i in range(10_000)]

        @property
        def nodes(self):
            self.node_reads += 1
            return self._nodes

    session = LargeSession()
    backend = SimpleNamespace(debug=DebugDraw())
    helpers = SceneEntityHelpers()

    helpers.publish(backend, session, CameraView(), 800.0, 1.0)
    helpers.publish(backend, session, CameraView(), 800.0, 1.0)
    helpers.pick(session, CameraView(), (0.0, 0.0, 1000.0, 800.0), (0.0, 0.0), 1.0)

    assert session.node_reads == 1


def test_light_and_camera_gizmo_commands_write_entity_transforms() -> None:
    scene = Scene()
    scene.add_light(
        "spot",
        Light(
            type=LightType.SPOT,
            position=np.array((1.0, 2.0, 3.0), np.float32),
            direction=np.array((0.0, 0.0, -1.0), np.float32),
        ),
    )
    scene.add_camera(
        "shot",
        CameraView(
            eye=np.array((3.0, -3.0, 2.0), np.float32),
            target=np.zeros(3, np.float32),
        ),
    )
    session = Session(StaticSceneAdapter(scene))
    light_node = next(node for node in session.nodes if node.type is NodeType.LIGHT)
    camera_node = next(node for node in session.nodes if node.type is NodeType.CAMERA)

    light_rotation = np.array(((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)), np.float32)
    light_command = _set_light_from_world(
        session, light_node, np.array((4.0, 5.0, 6.0)), light_rotation
    )
    assert light_command is not None and session.submit(light_command)
    light_position, light_basis = _node_pose(session, light_node)
    assert light_position == pytest.approx((4.0, 5.0, 6.0))
    assert -light_basis[:, 2] == pytest.approx((0.0, 1.0, 0.0))

    camera_rotation = np.eye(3, dtype=np.float32)
    camera_command = _set_camera_from_world(
        session, camera_node, np.array((2.0, 3.0, 4.0)), camera_rotation
    )
    assert camera_command is not None and session.submit(camera_command)
    camera_position, camera_rotation = _node_pose(session, camera_node)
    assert camera_position == pytest.approx((2.0, 3.0, 4.0))
    assert -camera_rotation[:, 2] == pytest.approx((0.0, 0.0, -1.0))


@pytest.mark.parametrize("light_type", (LightType.POINT, LightType.SPOT))
def test_light_rotation_gizmo_keeps_its_drag_frame_after_write_back(
    light_type: LightType,
) -> None:
    scene = Scene()
    scene.add_light(
        "light",
        Light(
            type=light_type,
            position=np.zeros(3, np.float32),
            direction=np.array((0.0, 0.0, -1.0), np.float32),
        ),
    )
    session = Session(StaticSceneAdapter(scene))
    node = next(node for node in session.nodes if node.type is NodeType.LIGHT)
    assert session.submit(cmd.Select(node.object_id))
    editor_camera = CameraView(
        eye=np.array((4.0, -6.0, 3.0), np.float32),
        target=np.zeros(3, np.float32),
        up=np.array((0.0, 0.0, 1.0), np.float32),
        aspect=4.0 / 3.0,
    )
    rect = (0.0, 0.0, 800.0, 600.0)
    gizmo = ObjectGizmo("rotate")
    gizmo.set_style("3d")
    start_position, start_basis = _node_pose(session, node)
    scale = 0.5
    start = start_position + start_basis[:, 1] * scale
    end = start_position + start_basis[:, 2] * scale
    start_cursor, end_cursor = project(editor_camera, (start, end), rect)[:, :2]
    assert gizmo.keyboard_interact(session, editor_camera, rect, start_cursor, 0)
    assert gizmo.keyboard_interact(session, editor_camera, rect, end_cursor, 0)
    assert not np.allclose(_node_pose(session, node)[1], start_basis)

    backend = SimpleNamespace(
        caps=BackendCaps(name="capture", gizmo=True),
        set_gizmo=lambda _frame: True,
    )
    assert gizmo.publish(
        backend,
        session,
        editor_camera,
        rect,
        ui_scale=1.0,
        style_scale=1.0,
        yielding=False,
        interactive=True,
    )
    assert gizmo._frame.position == pytest.approx(start_position)
    assert gizmo._frame.rotation == pytest.approx(start_basis)


def test_light_gizmo_converts_world_pose_to_parent_body_frame() -> None:
    body_rotation = math3d.euler_xyz_to_mat3(np.array((0.0, 0.0, np.pi * 0.5)))
    local_light = Light(
        type=LightType.SPOT,
        position=np.zeros(3, np.float32),
        direction=np.array((0.0, 0.0, -1.0), np.float32),
    )
    source = SceneSource(lights=LightSet(lights=(local_light,)))
    frame = SceneFrame(
        body_xpos=np.array(((0.0, 0.0, 0.0), (2.0, 3.0, 4.0)), np.float32),
        body_xmat=np.array((np.eye(3), body_rotation), np.float32),
    )
    session = SimpleNamespace(source=source, frame=frame)
    node = SceneNode(1, "spot", NodeType.LIGHT, body_index=1, light_index=0)
    world_position = np.array((2.0, 4.0, 4.0), np.float32)
    world_rotation = np.eye(3, dtype=np.float32)

    command = _set_light_from_world(session, node, world_position, world_rotation)
    assert command.light.position == pytest.approx((1.0, 0.0, 0.0), abs=1e-6)
    assert command.light.direction == pytest.approx((0.0, 0.0, -1.0), abs=1e-6)
