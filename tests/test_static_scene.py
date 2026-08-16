from __future__ import annotations

import numpy as np

from forge_viewer import commands as cmd
from forge_viewer.adapters.base import AdapterCaps, FrameNeeds, SceneAdapterBase
from forge_viewer.adapters.static import StaticSceneAdapter
from forge_viewer.render.builder import SceneSourceBuilder
from forge_viewer.scene import Scene
from forge_viewer.session import Session
from forge_viewer.types import Light, LightKind, LightSet, MeshData


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
