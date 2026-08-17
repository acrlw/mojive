from __future__ import annotations

from dataclasses import replace

import numpy as np

from forge_viewer import commands as cmd
from forge_viewer.adapters.base import AdapterCaps, FrameNeeds, NodeKind, SceneAdapterBase
from forge_viewer.adapters.static import StaticSceneAdapter
from forge_viewer.render.builder import SceneSourceBuilder
from forge_viewer.scene import Scene
from forge_viewer.session import Session
from forge_viewer.types import (
    CameraView,
    Environment,
    Light,
    LightKind,
    LightSet,
    Material,
    MeshData,
)


def test_static_scene_builds_without_a_physics_package():
    scene = Scene()
    box = scene.box(name="box", position=(1.0, 2.0, 3.0), size=(0.2, 0.3, 0.4))
    scene.sphere(name="ball", position=(-1.0, 0.0, 0.5))

    source, frame = scene.source, scene.frame
    assert source.instance_count == 2
    assert [n.name for n in source.nodes if n.object_id] == ["box", "ball"]
    assert source.geom_object_id.tolist() == [box.object_id, 2]
    rendered = SceneSourceBuilder().set_source(source)
    rendered.validate()
    assert rendered.count == 2
    assert np.allclose(frame.geom_xpos[0], (1.0, 2.0, 3.0))


def test_pose_updates_do_not_rebuild_structure():
    scene = Scene()
    obj = scene.box(name="moving")
    source = scene.source
    revision = scene.structure_revision

    obj.set_pose((2.0, -1.0, 0.5))

    assert scene.structure_revision == revision
    assert scene.source is source
    assert np.allclose(scene.frame.geom_xpos[0], (2.0, -1.0, 0.5))
    assert np.allclose(scene.frame.body_xpos[1], (2.0, -1.0, 0.5))


def test_session_detects_programmatic_add_and_remove():
    scene = Scene()
    first = scene.box(name="first")
    session = Session(StaticSceneAdapter(scene))
    generation = session.structure_generation

    second = scene.sphere(name="second")
    session.tick(FrameNeeds())
    assert session.structure_generation == generation + 1
    assert session.node_by_object_id(second.object_id).name == "second"

    first.remove()
    session.tick(FrameNeeds())
    assert session.node_by_object_id(first.object_id) is None
    assert session.node_by_object_id(second.object_id).name == "second"


def test_static_session_has_pose_editing_but_no_fake_playback():
    scene = Scene()
    obj = scene.box(name="editable")
    session = Session(StaticSceneAdapter(scene))
    node = session.node_by_object_id(obj.object_id)

    assert session.paused
    assert not session.submit(cmd.Play())
    assert "no simulation" in session.last_message
    assert session.submit(cmd.SetPose(node.node_id, np.ones(3), np.eye(3)))
    assert np.allclose(scene.frame.geom_xpos[0], 1.0)


def test_visibility_edits_reach_the_render_source():
    scene = Scene()
    obj = scene.box(name="visible")
    session = Session(StaticSceneAdapter(scene))
    node = session.node_by_object_id(obj.object_id)

    assert session.submit(cmd.SetVisible(node.node_id, False))
    source_node = next(item for item in session.source.nodes if item.node_id == node.node_id)
    assert not node.visible
    assert not source_node.visible


def test_lights_are_editable_forge_entities_without_physics():
    light = Light(kind=LightKind.POINT, position=np.array([1.0, 2.0, 3.0], np.float32))
    scene = Scene(lights=LightSet(lights=(light,)))
    session = Session(StaticSceneAdapter(scene))

    node = next(node for node in session.nodes if node.light_index == 0)
    assert node.kind.value == "light" and node.object_id

    edited = Light(
        kind=LightKind.POINT,
        position=light.position,
        diffuse=np.array([0.2, 0.4, 0.8], np.float32),
        active=False,
    )
    assert session.submit(cmd.SetLight(0, edited))
    assert scene.lights.lights[0] is edited
    assert session.source.lights.lights[0] is edited
    assert np.allclose(session.frame.lights.lights[0].diffuse, [0.2, 0.4, 0.8])
    assert not node.visible


def test_environment_is_editable_without_a_physics_backend():
    scene = Scene()
    session = Session(StaticSceneAdapter(scene))
    node = next(node for node in session.nodes if node.kind is NodeKind.ENVIRONMENT)

    assert node.object_id
    assert session.submit(cmd.Select(node.object_id))
    assert session.selected_node is node

    environment = Environment(
        headlight=None,
        ambient=np.array([0.1, 0.2, 0.3], np.float32),
        fog_color=np.array([0.3, 0.4, 0.5], np.float32),
        fog_start=2.0,
        fog_end=12.0,
        haze_color=np.array([0.8, 0.7, 0.6], np.float32),
        haze_density=0.04,
    )
    assert session.submit(cmd.SetEnvironment(environment))
    assert np.allclose(scene.lights.ambient, environment.ambient)
    assert np.allclose(session.source.lights.fog_color, environment.fog_color)
    assert session.frame.lights.haze_density == environment.haze_density

    scene.box(name="new body")
    session.tick(FrameNeeds())
    assert session.source.lights.fog_end == environment.fog_end


def test_material_components_support_shared_and_instance_edits():
    shared = Material(name="shared", specular=0.2)
    scene = Scene()
    first = scene.box(name="first", material=shared)
    scene.box(name="second", material=shared)
    session = Session(StaticSceneAdapter(scene))
    first_node = session.node_by_object_id(first.object_id)
    geometry_node = next(
        node
        for node in session.nodes
        if node.parent == first_node.node_id and node.kind is NodeKind.GEOM
    )
    material_id = int(session.source.geom_material[0])

    edited = replace(shared, specular=0.8, shininess=0.9)
    assert session.submit(cmd.SetMaterial(material_id, edited))
    assert session.source.materials[material_id] is edited
    assert scene.source.geom_material == [material_id, material_id]

    rgba = np.array([0.2, 0.4, 0.8, 0.7], np.float32)
    assert session.submit(cmd.SetGeometryColor(geometry_node.node_id, rgba))
    assert np.allclose(session.source.geom_rgba[0], rgba)
    assert not np.allclose(session.source.geom_rgba[1], rgba)

    scene.sphere(name="third")
    session.tick(FrameNeeds())
    assert np.allclose(session.source.geom_rgba[0], rgba)
    assert session.source.materials[session.source.geom_material[0]].specular == 0.8

    object_color = np.array([0.8, 0.3, 0.2, 1.0], np.float32)
    first.set_color(object_color)
    first.set_material(Material(name="unique", emission=0.3))
    session.tick(FrameNeeds())
    assert np.allclose(session.source.geom_rgba[0], object_color)
    assert session.source.materials[session.source.geom_material[0]].emission == 0.3
    assert session.source.materials[session.source.geom_material[1]].specular == 0.8


def test_cameras_are_editable_forge_entities_without_physics():
    initial = CameraView(
        eye=np.array([2.0, -4.0, 3.0], np.float32),
        target=np.array([0.0, 0.0, 0.5], np.float32),
    )
    scene = Scene(camera=initial)
    session = Session(StaticSceneAdapter(scene))

    info = session.cameras[0]
    node = next(node for node in session.nodes if node.camera_index == 0)
    assert node.kind is NodeKind.CAMERA
    assert node.object_id == info.object_id
    assert session.frame.cameras == (initial,)

    edited = CameraView(
        eye=np.array([-3.0, 2.0, 1.5], np.float32),
        target=np.array([0.0, 0.0, 0.0], np.float32),
        fov_y=np.deg2rad(60.0),
    )
    assert session.submit(cmd.SetSceneCamera(info.camera_id, edited))
    assert scene.camera_view(info.camera_id) is edited
    assert session.source.cameras == (edited,)
    assert session.frame.cameras == (edited,)


def test_backend_cannot_veto_a_forge_light_edit():
    class ReadOnlyLights(ToyPhysics):
        def __init__(self):
            super().__init__()
            self.scene.lights = LightSet(lights=(Light(),))

        def set_light(self, light_id, light):
            return False

    session = Session(ReadOnlyLights())
    edited = Light(kind=LightKind.AREA, area_radius=0.5)

    result = session.submit(cmd.SetLight(0, edited))

    assert result.ok and "write-back" in result.message
    assert session.source.lights.lights[0] is edited
    assert session.frame.lights.lights[0].kind is LightKind.AREA

    session.adapter.scene.box(name="new body")
    session.tick(FrameNeeds())
    assert session.source.lights.lights[0].kind is LightKind.AREA


def test_camera_ids_are_resolved_independently_from_source_slots():
    class SparseCameraAdapter(StaticSceneAdapter):
        def cameras(self):
            info = super().cameras()[0]
            return [replace(info, camera_id=42)]

        def camera_view(self, camera_id):
            return self.scene.camera_view(0) if camera_id == 42 else None

        def set_camera_view(self, camera_id, camera):
            return camera_id == 42 and self.scene.set_camera(0, camera)

    scene = Scene(camera=CameraView())
    session = Session(SparseCameraAdapter(scene))
    edited = CameraView(eye=np.array([1.0, 2.0, 3.0], np.float32))

    assert session.submit(cmd.SetSceneCamera(42, edited))
    assert scene.camera_view(0) is edited
    assert session.frame.cameras == (edited,)


def test_custom_mesh_enters_the_same_scene_contract():
    mesh = MeshData(
        positions=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32),
        normals=np.array([[0, 0, 1]] * 3, np.float32),
        uvs=np.array([[0, 0], [1, 0], [0, 1]], np.float32),
        indices=np.array([0, 1, 2], np.uint32),
    )
    scene = Scene()
    scene.mesh(mesh, name="triangle")

    assert next(iter(scene.source.meshes.values())) is mesh
    assert scene.source.instance_count == 1


class ToyPhysics(SceneAdapterBase):
    caps = AdapterCaps(name="toy", simulation=True)

    def __init__(self) -> None:
        self.scene = Scene()
        self.body = self.scene.sphere(name="body")
        self.steps = 0

    @property
    def structure_revision(self) -> int:
        return self.scene.structure_revision

    def scene_source(self):
        return self.scene.source

    def frame(self, needs):
        frame = self.scene.frame
        frame.time = self.steps * 0.01
        return frame

    def step(self, count=1):
        self.steps += count
        self.body.set_pose((self.steps * 0.1, 0.0, 0.0))


def test_custom_physics_only_implements_its_actual_contract():
    adapter = ToyPhysics()
    session = Session(adapter)

    frame = session.tick(FrameNeeds())

    assert adapter.steps == 1
    assert frame.time == 0.01
    assert np.allclose(frame.geom_xpos[0], (0.1, 0.0, 0.0))
    assert session.nodes[1].name == "body"


class ClockedToyPhysics(ToyPhysics):
    def timestep(self):
        return 0.002


def test_simulation_speed_tracks_wall_time_not_render_frames():
    def run(frame_dt, frames, speed=1.0):
        adapter = ClockedToyPhysics()
        session = Session(adapter)
        session.submit(cmd.SetSpeed(speed))
        for _ in range(frames):
            session.tick(FrameNeeds(), wall_dt=frame_dt)
        return adapter.steps

    assert run(1 / 60, 60) == 500
    assert run(1 / 30, 30) == 500
    assert run(1 / 60, 60, speed=0.5) == 250
