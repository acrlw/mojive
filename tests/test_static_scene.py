from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from forge_viewer import commands as cmd
from forge_viewer.adapters.base import (
    CAMERA_OBJECT_BASE,
    LIGHT_OBJECT_BASE,
    AdapterCaps,
    EqualityConstraintInfo,
    FrameNeeds,
    KeyframeInfo,
    NodeType,
    SceneAdapterBase,
    SceneFrame,
    SceneSource,
)
from forge_viewer.adapters.static import StaticSceneAdapter
from forge_viewer.render.builder import SceneSourceBuilder
from forge_viewer.scene import Scene
from forge_viewer.session import Session
from forge_viewer.types import (
    DEFAULT_MATERIAL,
    CameraView,
    Environment,
    Light,
    LightSet,
    LightType,
    Material,
    MeshData,
    MeshShape,
)

pytestmark = pytest.mark.integration


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

    assert session.submit(cmd.Select(first.object_id))
    first.remove()
    session.tick(FrameNeeds())
    assert session.node_by_object_id(first.object_id) is None
    assert session.node_by_object_id(second.object_id).name == "second"
    assert session.selected == 0
    assert session.selected_node is None


def test_scene_authoring_commands_return_stable_entity_ids():
    scene = Scene()
    session = Session(StaticSceneAdapter(scene))

    added = session.submit(
        cmd.AddSceneObject(
            MeshShape.BOX,
            "remote box",
            position=(1.0, 2.0, 3.0),
            color=(0.2, 0.4, 0.8, 1.0),
        )
    )
    assert added.ok and added.entity_id > 0
    assert session.node_by_object_id(added.entity_id).name == "remote box"

    light = session.submit(cmd.AddSceneLight("remote light", Light(type=LightType.POINT)))
    camera = session.submit(cmd.AddSceneCamera("remote camera", CameraView()))
    assert light.ok and camera.ok
    assert session.node_by_object_id(LIGHT_OBJECT_BASE + light.entity_id).name == "remote light"
    assert next(info for info in session.cameras if info.camera_id == camera.entity_id).name == (
        "remote camera"
    )

    assert session.submit(cmd.RemoveSceneObject(added.entity_id))
    assert session.submit(cmd.RemoveSceneLight(light.entity_id))
    assert session.submit(cmd.RemoveSceneCamera(camera.entity_id))
    assert session.node_by_object_id(added.entity_id) is None
    assert session.node_by_object_id(LIGHT_OBJECT_BASE + light.entity_id) is None
    assert all(info.camera_id != camera.entity_id for info in session.cameras)


def test_static_scene_document_commands_track_dirty_state(tmp_path):
    scene = Scene()
    session = Session(StaticSceneAdapter(scene))
    path = tmp_path / "workspace.forge.json"

    added = session.submit(cmd.AddSceneObject(MeshShape.BOX, "box"))
    assert added.ok and session.dirty
    assert session.submit(cmd.SaveScene(path))
    assert path.exists() and session.asset_path == path and not session.dirty

    node = session.node_by_object_id(added.entity_id)
    assert session.submit(cmd.SetPose(node.node_id, np.ones(3), np.eye(3)))
    assert session.dirty
    assert session.submit(cmd.OpenScene(path))
    assert not session.dirty
    assert np.allclose(session.frame.geom_xpos[0], 0.0)

    session._step_counter = 9
    assert session.submit(cmd.NewScene())
    session.tick(FrameNeeds())
    assert session.asset_path is None and not session.dirty
    assert session.source.instance_count == 0
    assert session.frame.step == 0


def test_static_scene_entity_lifecycle_commands_use_selection_identity():
    scene = Scene()
    obj = scene.box(name="box", position=(1.0, 2.0, 3.0))
    session = Session(StaticSceneAdapter(scene))

    assert session.submit(cmd.Select(obj.object_id))
    duplicate = session.submit(cmd.DuplicateSceneEntity(obj.object_id))
    assert duplicate.ok and duplicate.entity_id != obj.object_id
    assert session.selected == duplicate.entity_id
    assert session.selected_node.name == "box Copy"
    assert np.allclose(scene.frame.geom_xpos[1], (1.0, 2.0, 3.0))

    assert session.submit(cmd.RenameSceneEntity(duplicate.entity_id, "copy"))
    assert session.selected_node.name == "copy"
    assert session.submit(cmd.RemoveSceneEntity(duplicate.entity_id))
    assert session.selected == 0
    assert session.node_by_object_id(duplicate.entity_id) is None

    light = scene.add_light("key", Light(type=LightType.DIRECTIONAL))
    camera_id = scene.add_camera("shot", CameraView())
    session.tick(FrameNeeds())
    for object_id, expected_name in (
        (LIGHT_OBJECT_BASE + light.light_id, "key Copy"),
        (CAMERA_OBJECT_BASE + camera_id, "shot Copy"),
    ):
        duplicate = session.submit(cmd.DuplicateSceneEntity(object_id))
        assert duplicate.ok
        assert session.node_by_object_id(duplicate.entity_id).name == expected_name
        assert session.submit(cmd.RemoveSceneEntity(duplicate.entity_id))


def test_static_scene_undo_redo_restores_entities_and_saved_revision(tmp_path):
    session = Session(StaticSceneAdapter(Scene()))
    path = tmp_path / "history.forge.json"

    added = session.submit(cmd.AddSceneObject(MeshShape.BOX, "box"))
    assert added.ok and session.can_undo and session.dirty
    assert session.submit(cmd.SaveScene(path))
    assert not session.dirty

    assert session.submit(cmd.RenameSceneEntity(added.entity_id, "renamed"))
    assert session.node_by_object_id(added.entity_id).name == "renamed"
    assert session.dirty
    assert session.submit(cmd.Undo())
    assert session.node_by_object_id(added.entity_id).name == "box"
    assert not session.dirty and session.can_redo
    assert session.submit(cmd.Redo())
    assert session.node_by_object_id(added.entity_id).name == "renamed"
    assert session.dirty

    assert session.submit(cmd.RemoveSceneEntity(added.entity_id))
    assert session.node_by_object_id(added.entity_id) is None
    assert session.submit(cmd.Undo())
    assert session.node_by_object_id(added.entity_id).name == "renamed"


def test_static_scene_continuous_pose_edit_is_one_history_entry():
    scene = Scene()
    obj = scene.box(name="box")
    session = Session(StaticSceneAdapter(scene))
    node = session.node_by_object_id(obj.object_id)

    assert session.submit(cmd.BeginEditTransaction("Move transform"))
    for x in (0.5, 1.0, 2.0):
        assert session.submit(cmd.SetPose(node.node_id, np.array([x, 0.0, 0.0]), np.eye(3)))
    assert session.submit(cmd.EndEditTransaction())
    assert np.allclose(session.frame.geom_xpos[0], (2.0, 0.0, 0.0))

    assert session.submit(cmd.Undo())
    assert np.allclose(session.frame.geom_xpos[0], 0.0)
    assert not session.can_undo
    assert session.submit(cmd.Redo())
    assert np.allclose(session.frame.geom_xpos[0], (2.0, 0.0, 0.0))


def test_empty_edit_transaction_does_not_create_history():
    session = Session(StaticSceneAdapter(Scene()))

    assert session.submit(cmd.BeginEditTransaction("No-op"))
    assert session.submit(cmd.EndEditTransaction())
    assert not session.can_undo


def test_authored_overlay_survives_adapter_source_rebuilds():
    class RenderOnlyAdapter(StaticSceneAdapter):
        def set_light(self, light_id, light):
            return False

        def set_material(self, material_id, material):
            return False

        def set_geometry_color(self, node_id, rgba):
            return False

    scene = Scene(lights=LightSet(lights=(Light(type=LightType.POINT),)))
    obj = scene.box(name="box")
    session = Session(RenderOnlyAdapter(scene))
    node = session.node_by_object_id(obj.object_id)
    material = replace(DEFAULT_MATERIAL, name="authored", emission=0.4)
    light = replace(
        session.source.lights.lights[0],
        diffuse=np.array([0.2, 0.7, 0.9], np.float32),
    )
    rgba = np.array([0.8, 0.2, 0.4, 1.0], np.float32)

    assert session.submit(cmd.SetLight(0, light))
    assert session.submit(cmd.SetMaterial(0, material))
    assert session.submit(cmd.SetGeometryColor(node.children[0], rgba))
    scene.sphere(name="structure change")
    session.tick(FrameNeeds())

    assert session.source.lights.lights[0] is light
    assert session.source.materials[0] is material
    assert np.allclose(session.source.geom_rgba[0], rgba)


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
    light = Light(type=LightType.POINT, position=np.array([1.0, 2.0, 3.0], np.float32))
    scene = Scene(lights=LightSet(lights=(light,)))
    session = Session(StaticSceneAdapter(scene))

    node = next(node for node in session.nodes if node.light_index == 0)
    assert node.type.value == "light" and node.object_id

    edited = Light(
        type=LightType.POINT,
        position=light.position,
        diffuse=np.array([0.2, 0.4, 0.8], np.float32),
        active=False,
    )
    assert session.submit(cmd.SetLight(0, edited))
    assert scene.lights.lights[0] is edited
    assert session.source.lights.lights[0] is edited
    assert np.allclose(session.frame.lights.lights[0].diffuse, [0.2, 0.4, 0.8])
    assert not node.visible


def test_light_entity_ids_survive_slot_changes():
    scene = Scene()
    first = scene.add_light("key", Light(type=LightType.DIRECTIONAL))
    second = scene.add_light("fill", Light(type=LightType.POINT))
    session = Session(StaticSceneAdapter(scene))
    second_node = session.node_by_object_id(LIGHT_OBJECT_BASE + second.light_id)

    edited = replace(second.value, diffuse=np.array([0.2, 0.5, 0.8], np.float32))
    assert session.submit(cmd.SetLight(1, edited))
    first.remove()
    session.tick(FrameNeeds())

    node = session.node_by_object_id(second_node.object_id)
    assert node.name == "fill"
    assert node.light_index == 0
    assert scene.light("fill").light_id == second.light_id
    assert np.allclose(session.source.lights.lights[0].diffuse, edited.diffuse)

    third = scene.add_light("rim", Light(type=LightType.SPOT))
    assert third.light_id > second.light_id


def test_environment_is_editable_without_a_physics_backend():
    scene = Scene()
    session = Session(StaticSceneAdapter(scene))
    node = next(node for node in session.nodes if node.type is NodeType.ENVIRONMENT)

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
        if node.parent == first_node.node_id and node.type is NodeType.GEOM
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
    assert node.type is NodeType.CAMERA
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


def test_cameras_can_be_added_and_removed_after_session_creation():
    scene = Scene()
    session = Session(StaticSceneAdapter(scene))
    first = scene.add_camera("first", CameraView())
    second = scene.add_camera("second", CameraView(eye=np.array([4.0, 2.0, 3.0], np.float32)))

    session.tick(FrameNeeds())
    second_info = next(camera for camera in session.cameras if camera.camera_id == second)
    scene.remove_camera(first)
    session.tick(FrameNeeds())

    assert [camera.camera_id for camera in session.cameras] == [second]
    assert session.cameras[0].object_id == second_info.object_id
    assert scene.camera is scene.camera_view(second)


def test_backend_cannot_veto_a_forge_light_edit():
    class ReadOnlyLights(ToyPhysics):
        def __init__(self):
            super().__init__()
            self.scene.lights = LightSet(lights=(Light(),))

        def set_light(self, light_id, light):
            return False

    session = Session(ReadOnlyLights())
    edited = Light(type=LightType.AREA, area_radius=0.5)

    result = session.submit(cmd.SetLight(0, edited))

    assert result.ok and "write-back" in result.message
    assert session.source.lights.lights[0] is edited
    assert session.frame.lights.lights[0].type is LightType.AREA

    session.adapter.scene.box(name="new body")
    session.tick(FrameNeeds())
    assert session.source.lights.lights[0].type is LightType.AREA


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


class SparseControlPhysics(SceneAdapterBase):
    caps = AdapterCaps(
        name="sparse-controls",
        simulation=True,
        keyframes=True,
        equality_constraints=True,
    )

    def __init__(self) -> None:
        self.loaded_keyframe = -1
        self.equality_updates: list[tuple[int, bool]] = []

    def scene_source(self):
        return SceneSource()

    def frame(self, needs):
        del needs
        return SceneFrame()

    def keyframes(self):
        return [KeyframeInfo(42, "sparse pose", 1.0)]

    def equality_constraints(self):
        return [EqualityConstraintInfo(7, "sparse weld", "weld", True)]

    def load_keyframe(self, keyframe_id):
        self.loaded_keyframe = int(keyframe_id)
        return keyframe_id == 42

    def set_equality_enabled(self, constraint_id, enabled):
        self.equality_updates.append((int(constraint_id), bool(enabled)))
        return constraint_id == 7


class UnpausableScenePhysics(ToyPhysics):
    caps = replace(ToyPhysics.caps, scene_files=True)

    def new_scene(self):
        self.scene = Scene()
        self.body = self.scene.sphere(name="replacement")

    def set_paused(self, paused):
        return not paused


def test_custom_physics_only_implements_its_actual_contract():
    adapter = ToyPhysics()
    session = Session(adapter)

    frame = session.tick(FrameNeeds())

    assert adapter.steps == 1
    assert frame.time == 0.01
    assert np.allclose(frame.geom_xpos[0], (0.1, 0.0, 0.0))
    assert session.nodes[1].name == "body"


def test_session_routes_sparse_stable_control_ids_without_using_them_as_slots():
    adapter = SparseControlPhysics()
    session = Session(adapter)
    assert session.submit(cmd.Pause())

    keyframe = session.submit(cmd.LoadKeyframe(42))
    equality = session.submit(cmd.SetEqualityEnabled(7, False))

    assert keyframe.ok and session.active_keyframe == 42
    assert adapter.loaded_keyframe == 42
    assert equality.ok and adapter.equality_updates == [(7, False)]
    assert not session.equality_constraints[0].enabled


@pytest.mark.parametrize("count", (0, -1))
def test_session_rejects_non_positive_step_counts(count):
    adapter = ToyPhysics()
    session = Session(adapter)
    assert session.submit(cmd.Pause())

    result = session.submit(cmd.Step(count))
    session.tick(FrameNeeds())

    assert not result.ok
    assert adapter.steps == 0


@pytest.mark.parametrize("factor", (0.0, -1.0, float("nan"), float("inf")))
def test_session_rejects_invalid_simulation_speeds(factor):
    session = Session(ToyPhysics())

    result = session.submit(cmd.SetSpeed(factor))

    assert not result.ok
    assert session.speed == 1.0


def test_scene_replacement_does_not_claim_pause_when_the_adapter_rejects_it():
    session = Session(UnpausableScenePhysics())

    assert session.submit(cmd.NewScene())

    assert not session.paused


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
