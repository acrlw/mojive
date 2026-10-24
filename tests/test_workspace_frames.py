"""Composite frames reuse owned storage without retaining stale primary data."""

from dataclasses import replace
from weakref import ref

import numpy as np

from mojive.adapters.base import DiagnosticFrame, FrameNeeds, SceneFrame
from mojive.adapters.static import StaticSceneAdapter
from mojive.adapters.workspace import WorkspaceAdapter
from mojive.scene import Scene
from mojive.types import CameraView, Light, LightSet


class PublishedPrimary(StaticSceneAdapter):
    def __init__(self, scene):
        super().__init__(scene)
        self.published = SceneFrame()
        self.request = None

    def frame(self, needs):
        self.request = needs
        return self.published


def test_workspace_frame_reuse_clears_optional_data_from_previous_request():
    primary_scene, authored = Scene(), Scene()
    primary_scene.box()
    authored.box()
    primary = PublishedPrimary(primary_scene)
    diagnostics = DiagnosticFrame()
    primary.published = replace(
        primary_scene.frame,
        time=1.25,
        step=20,
        paused=True,
        physics_hz=240.0,
        qpos=np.ones(1),
        qvel=np.ones(1),
        ctrl=np.ones(1),
        actuator_activation=np.ones(1),
        site_xpos=np.zeros((1, 3)),
        site_xmat=np.zeros((1, 3, 3)),
        contacts=np.zeros((1, 7)),
        contact_forces=np.zeros((1, 2, 3)),
        contact_island_rgba=np.ones((1, 4)),
        tendon_segments=np.zeros((1, 2, 3)),
        tendon_ids=np.zeros(1, int),
        tendon_widths=np.ones(1),
        tendon_island_rgba=np.ones((1, 4)),
        sensors=np.ones(1),
        equality_enabled=np.ones(1, bool),
        flex_vertices=np.zeros((1, 3)),
        flex_island_rgba=np.ones((1, 4)),
        island_rgba=np.ones((1, 4)),
        diagnostics=diagnostics,
        mesh_updates={},
        debug_commands=({"op": "clear"},),
    )
    workspace = WorkspaceAdapter(primary, authored)
    needs = FrameNeeds(
        qpos=True,
        qvel=True,
        actuator=True,
        contacts=True,
        sensors=True,
        diagnostics=True,
        deformables=True,
        tendons=True,
        islands=True,
    )
    frame = workspace.frame(needs)
    assert primary.request is needs
    assert frame is not primary.published
    assert frame.qpos is primary.published.qpos
    assert frame.diagnostics is diagnostics
    assert frame.geom_xpos.shape == (2, 3)
    assert frame.step == 20 and frame.time == 1.25 and frame.paused

    primary.published = SceneFrame(time=2.0, step=21)
    needs = FrameNeeds.none()
    assert workspace.frame(needs) is frame
    assert primary.request is needs
    assert frame.step == 21 and frame.time == 2.0 and not frame.paused
    assert frame.physics_hz is None
    for name in (
        "geom_xpos",
        "geom_xmat",
        "body_xpos",
        "body_xmat",
        "site_xpos",
        "site_xmat",
        "qpos",
        "qvel",
        "ctrl",
        "actuator_activation",
        "contacts",
        "contact_forces",
        "sensors",
        "equality_enabled",
        "contact_island_rgba",
        "tendon_segments",
        "tendon_ids",
        "tendon_widths",
        "tendon_island_rgba",
        "flex_vertices",
        "flex_island_rgba",
        "island_rgba",
        "diagnostics",
        "mesh_updates",
        "debug_commands",
    ):
        assert getattr(frame, name) is None, name


def test_workspace_reused_frame_tracks_pose_and_structure_changes():
    primary_scene, authored = Scene(), Scene()
    primary_object = primary_scene.box(position=(1, 2, 3))
    authored_object = authored.box(position=(4, 5, 6))
    workspace = WorkspaceAdapter(StaticSceneAdapter(primary_scene), authored)
    frame = workspace.frame(FrameNeeds())
    positions = frame.geom_xpos
    np.testing.assert_array_equal(positions, ((1, 2, 3), (4, 5, 6)))

    primary_object.set_pose((2, 3, 4))
    authored_object.set_pose((5, 6, 7))
    assert workspace.frame(FrameNeeds()) is frame
    assert frame.geom_xpos is positions
    np.testing.assert_array_equal(positions, ((2, 3, 4), (5, 6, 7)))

    added = authored.box(position=(8, 9, 10))
    assert workspace.frame(FrameNeeds()) is frame
    assert workspace.scene_source().instance_count == 3
    assert frame.body_xpos.shape == (4, 3)
    np.testing.assert_array_equal(frame.geom_xpos, ((2, 3, 4), (5, 6, 7), (8, 9, 10)))

    primary_object.remove()
    added.remove()
    assert workspace.frame(FrameNeeds()) is frame
    assert workspace.scene_source().instance_count == 1
    np.testing.assert_array_equal(frame.geom_xpos, ((5, 6, 7),))
    assert frame.body_xpos.shape == (2, 3)


def test_workspace_reused_frame_refreshes_cameras_and_dynamic_lights():
    primary_scene, authored = Scene(camera=CameraView()), Scene()
    authored_camera = authored.add_camera("authored", CameraView())
    authored_light = authored.add_light("authored", Light())
    primary = PublishedPrimary(primary_scene)
    dynamic_camera, dynamic_light = CameraView(), Light()
    primary.published = SceneFrame(
        cameras=(dynamic_camera,), lights=LightSet(lights=(dynamic_light,))
    )
    workspace = WorkspaceAdapter(primary, authored)
    frame = workspace.frame(FrameNeeds())
    assert frame.cameras[0] is dynamic_camera
    assert frame.lights.lights[0] is dynamic_light
    assert frame.lights.lights[1] is authored_light.value

    revision = workspace.structure_revision
    updated_camera = replace(dynamic_camera, eye=np.array((8, 7, 6), np.float32))
    updated_light = replace(dynamic_light, position=np.array((5, 4, 3), np.float32))
    authored.set_camera(authored_camera, updated_camera)
    authored_light.set(updated_light)
    primary.published = SceneFrame(
        cameras=(updated_camera,), lights=LightSet(lights=(updated_light,))
    )
    assert workspace.structure_revision == revision
    assert workspace.frame(FrameNeeds()) is frame
    assert all(camera is updated_camera for camera in frame.cameras)
    assert all(light is updated_light for light in frame.lights.lights)

    primary.published = SceneFrame()
    assert workspace.frame(FrameNeeds()) is frame
    assert frame.lights is None
    assert frame.cameras[0] is primary_scene.camera
    assert frame.cameras[1] is updated_camera


def test_workspace_frames_and_merged_buffers_belong_to_each_adapter():
    primary_scene, first_scene, second_scene = Scene(), Scene(), Scene()
    primary_object = primary_scene.box(position=(1, 2, 3))
    first_scene.box(position=(4, 5, 6))
    second_scene.box(position=(7, 8, 9))
    primary = StaticSceneAdapter(primary_scene)
    first = WorkspaceAdapter(primary, first_scene)
    second = WorkspaceAdapter(primary, second_scene)
    first_frame = first.frame(FrameNeeds())
    second_frame = second.frame(FrameNeeds())
    assert first_frame is not second_frame
    for name in ("geom_xpos", "geom_xmat", "body_xpos", "body_xmat"):
        assert not np.shares_memory(getattr(first_frame, name), getattr(second_frame, name))

    primary_object.set_pose((10, 11, 12))
    assert second.frame(FrameNeeds()) is second_frame
    np.testing.assert_array_equal(first_frame.geom_xpos, ((1, 2, 3), (4, 5, 6)))
    np.testing.assert_array_equal(second_frame.geom_xpos, ((10, 11, 12), (7, 8, 9)))


def test_release_drops_owned_frame_payloads_without_mutating_borrowers():
    primary_scene, authored = Scene(), Scene()
    primary_scene.box(position=(1, 2, 3))
    authored.box(position=(4, 5, 6))
    primary = PublishedPrimary(primary_scene)
    primary.published = replace(primary_scene.frame, qpos=np.ones(1), diagnostics=DiagnosticFrame())
    workspace = WorkspaceAdapter(primary, authored)
    borrowed = workspace.frame(FrameNeeds())
    buffers = tuple(
        ref(getattr(borrowed, name))
        for name in ("geom_xpos", "geom_xmat", "body_xpos", "body_xmat")
    )

    workspace.release()
    assert workspace._frame is not borrowed
    assert workspace._frame.geom_xpos is None
    assert workspace._frame.qpos is None
    assert workspace._frame.diagnostics is None
    np.testing.assert_array_equal(borrowed.geom_xpos, ((1, 2, 3), (4, 5, 6)))
    assert borrowed.qpos is primary.published.qpos
    assert borrowed.diagnostics is primary.published.diagnostics
    assert all(buffer() is not None for buffer in buffers)
    del borrowed
    assert all(buffer() is None for buffer in buffers)
