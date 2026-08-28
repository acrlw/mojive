from __future__ import annotations

import json
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pytest

from forge_viewer import commands as cmd
from forge_viewer import math3d
from forge_viewer.adapters.base import (
    AdapterCaps,
    FrameNeeds,
    NodeType,
    SceneAdapterBase,
    SceneFrame,
    SceneNode,
    SceneSaveOptions,
    SceneSource,
)
from forge_viewer.adapters.mujoco_adapter import MuJoCoAdapter
from forge_viewer.adapters.workspace import WorkspaceAdapter
from forge_viewer.session import Session
from forge_viewer.types import (
    DEFAULT_MATERIAL,
    CameraView,
    Environment,
    Light,
    LightType,
    MeshShape,
    TextureData,
    TextureType,
)
from forge_viewer.workspace_io import (
    missing_resource_entries,
    missing_resources,
    relocate_workspace_resource,
    repair_workspace_resources,
)

mujoco = pytest.importorskip("mujoco")
pytestmark = [pytest.mark.integration, pytest.mark.physics]

ASSETS = Path(__file__).parents[1] / "assets"


def workspace() -> WorkspaceAdapter:
    primary = MuJoCoAdapter()
    primary.new_scene()
    return WorkspaceAdapter(primary)


def test_empty_workspace_does_not_override_the_editor_default_camera() -> None:
    document = workspace()
    try:
        assert document.camera_hint() is None
    finally:
        document.release()


def test_loaded_workspace_keeps_the_primary_camera_hint() -> None:
    document = WorkspaceAdapter(MuJoCoAdapter(ASSETS / "test_scene.xml"))
    try:
        expected = document.primary.camera_hint()
        actual = document.camera_hint()
        assert expected is not None and actual is not None
        assert actual.eye == pytest.approx(expected.eye)
        assert actual.target == pytest.approx(expected.target)
    finally:
        document.release()


def test_loaded_model_workspace_supports_authored_entities() -> None:
    document = WorkspaceAdapter(MuJoCoAdapter(ASSETS / "test_scene.xml"))
    session = Session(document, ASSETS / "test_scene.xml")

    assert session.adapter.caps.scene_authoring
    result = session.submit(cmd.AddSceneObject(MeshShape.PLANE, "plane"))

    assert result.ok
    assert any(node.name == "plane" for node in session.nodes)


def test_workspace_camera_lookup_uses_stable_scene_metadata(monkeypatch) -> None:
    document = WorkspaceAdapter(MuJoCoAdapter(ASSETS / "test_scene.xml"))
    camera = document.primary.scene_source().cameras[0]

    def reject_camera_enumeration():
        raise AssertionError("camera metadata must not be rebuilt during frame lookup")

    monkeypatch.setattr(document.primary, "cameras", reject_camera_enumeration)

    actual = document.camera_view(0)
    assert actual is not None
    assert actual.eye == pytest.approx(camera.eye)
    assert actual.target == pytest.approx(camera.target)


def test_workspace_authored_light_and_camera_ids_survive_earlier_removals() -> None:
    session = Session(workspace())

    first_light = session.submit(cmd.AddSceneLight("first light", Light())).entity_id
    second_light = session.submit(cmd.AddSceneLight("second light", Light())).entity_id
    assert session.submit(cmd.RemoveSceneLight(first_light))
    assert session.submit(cmd.RemoveSceneLight(second_light))

    first_camera = session.submit(cmd.AddSceneCamera("first camera", CameraView())).entity_id
    second_camera = session.submit(cmd.AddSceneCamera("second camera", CameraView())).entity_id
    assert session.submit(cmd.RemoveSceneCamera(first_camera))
    assert [camera.camera_id for camera in session.cameras] == [second_camera]
    assert session.camera_view(second_camera) is not None
    assert session.submit(cmd.RemoveSceneCamera(second_camera))
    assert session.cameras == []


def test_workspace_remaps_authored_nodes_after_sparse_primary_ids() -> None:
    class SparsePrimary(SceneAdapterBase):
        caps = AdapterCaps(name="sparse")

        def __init__(self) -> None:
            self.source = SceneSource(
                body_names=("world", "primary"),
                nodes=[
                    SceneNode(10, "world", NodeType.WORLD, children=[42], body_index=0),
                    SceneNode(42, "primary", NodeType.LINK, parent=10, body_index=1),
                ],
            )

        def scene_source(self):
            return self.source

        def frame(self, needs):
            del needs
            return SceneFrame()

    document = WorkspaceAdapter(SparsePrimary())
    document.add_scene_object(
        MeshShape.BOX,
        "authored",
        np.ones(3, np.float32),
        np.zeros(3, np.float32),
        np.eye(3, dtype=np.float32),
        np.ones(4, np.float32),
        DEFAULT_MATERIAL,
    )

    nodes = document.scene_source().nodes
    known = {node.node_id for node in nodes}
    authored = next(node for node in nodes if node.name == "authored")

    assert len(known) == len(nodes)
    assert authored.node_id > 42
    assert authored.parent == 10
    assert all(node.parent < 0 or node.parent in known for node in nodes)


def test_workspace_preserves_every_primary_environment_field(tmp_path: Path) -> None:
    model_path = tmp_path / "haze.xml"
    model_path.write_text(
        """<mujoco>
  <visual>
    <headlight ambient=".11 .12 .13" diffuse=".21 .22 .23" specular=".31 .32 .33"/>
    <rgba fog=".1 .2 .3 1" haze=".4 .5 .6 1"/>
    <map fogstart=".25" fogend="4" haze=".7"/>
    <quality numslices="23"/>
  </visual>
  <worldbody>
    <light pos="0 0 2" ambient=".03 .04 .05"/>
    <geom type="plane" size="0 0 0.1"/>
  </worldbody>
</mujoco>
""",
        encoding="utf-8",
    )
    document = workspace()

    document.load(model_path)

    expected = document.primary.scene_source().lights.environment()
    actual = document.scene_source().lights.environment()
    for item in fields(expected):
        left = getattr(expected, item.name)
        right = getattr(actual, item.name)
        if isinstance(left, np.ndarray):
            assert right == pytest.approx(left)
        elif isinstance(left, Light):
            for light_item in fields(left):
                light_left = getattr(left, light_item.name)
                light_right = getattr(right, light_item.name)
                if isinstance(light_left, np.ndarray):
                    assert light_right == pytest.approx(light_left)
                else:
                    assert light_right == light_left
        else:
            assert right == left
    assert document.scene_source().scene_center == pytest.approx(
        document.primary.scene_source().scene_center
    )
    assert document.scene_source().scene_extent == pytest.approx(
        document.primary.scene_source().scene_extent
    )


@pytest.mark.parametrize("operation", ("load", "add"))
def test_workspace_preserves_explicit_world_light_target(tmp_path: Path, operation: str) -> None:
    """Root loading and composition keep the singleton world light target."""

    model_path = tmp_path / "world_target.xml"
    model_path.write_text(
        """<mujoco>
  <worldbody>
    <light name="tracking" mode="targetbodycom" target="world" pos="0 -2 3"/>
    <body name="object"><geom type="sphere" size=".2"/></body>
  </worldbody>
</mujoco>
""",
        encoding="utf-8",
    )
    document = workspace()

    if operation == "load":
        document.load(model_path)
    else:
        document.add_scene_model(model_path, np.zeros(3), np.eye(3))

    assert document.primary.model.nlight == 1
    assert document.primary.model.light_targetbodyid[0] == 0
    assert document.scene_source().instance_count == 1


def test_workspace_composes_flex_with_model_root_transform(tmp_path: Path) -> None:
    """Model composition preserves flex declarations and world-owned vertices."""

    model_path = tmp_path / "cloth.xml"
    model_path.write_text(
        """<mujoco>
  <worldbody>
    <flexcomp name="cloth" type="grid" count="3 3 1" spacing=".2 .2 .2"
              dim="2" radius=".01">
      <pin id="0 2 6 8"/><edge equality="true"/>
    </flexcomp>
  </worldbody>
</mujoco>
""",
        encoding="utf-8",
    )
    direct = MuJoCoAdapter(model_path)
    composed = MuJoCoAdapter()
    composed.new_scene()
    position = np.array((1.0, -2.0, 0.5), np.float32)
    rotation = np.array(((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)), np.float32)

    try:
        model_id = composed.add_scene_model(model_path, position, rotation)

        assert model_id > 0
        assert composed.model.nflex == direct.model.nflex == 1
        expected = np.asarray(direct.data.flexvert_xpos) @ rotation.T + position
        assert composed.data.flexvert_xpos == pytest.approx(expected)
        assert composed.scene_source().instance_count == direct.scene_source().instance_count
    finally:
        direct.release()
        composed.release()


def test_keyless_attached_specs_skip_the_redundant_pre_attach_compile() -> None:
    class KeylessSpec:
        keys = ()

        @staticmethod
        def compile():
            raise AssertionError("keyless child was compiled before the composed model")

    MuJoCoAdapter._resolve_attached_keyframes(KeylessSpec())


def test_attached_model_keyframes_survive_composition(tmp_path: Path) -> None:
    path = tmp_path / "keyed.xml"
    path.write_text(
        """<mujoco model="keyed">
  <worldbody><body name="body"><joint name="joint"/><geom type="sphere" size=".1"/></body></worldbody>
  <keyframe><key name="pose" qpos=".25"/></keyframe>
</mujoco>
""",
        encoding="utf-8",
    )
    adapter = MuJoCoAdapter()
    adapter.new_scene()

    try:
        adapter.add_scene_model(path, np.zeros(3), np.eye(3))

        assert adapter.model.nkey == 1
        assert adapter.keyframes()[0].name.endswith("pose")
        assert float(adapter.model.key_qpos[0, 0]) == pytest.approx(0.25)
    finally:
        adapter.release()


def test_attached_model_resolves_actuators_from_includes_before_copy(tmp_path: Path) -> None:
    included = tmp_path / "included.xml"
    included.write_text(
        """<mujoco>
  <worldbody>
    <body name="body">
      <joint name="first"/><joint name="second"/>
      <geom type="sphere" size=".1"/>
    </body>
  </worldbody>
  <actuator>
    <position name="first_position" joint="first"/>
    <position name="second_position" joint="second"/>
  </actuator>
</mujoco>
""",
        encoding="utf-8",
    )
    scene = tmp_path / "scene.xml"
    scene.write_text(
        """<mujoco>
  <include file="included.xml"/>
  <keyframe><key name="home" qpos=".1 .2" ctrl=".3 .4"/></keyframe>
</mujoco>
""",
        encoding="utf-8",
    )
    assert mujoco.MjModel.from_xml_path(str(scene)).nu == 2
    adapter = MuJoCoAdapter()
    adapter.new_scene()

    try:
        adapter.add_scene_model(scene, np.zeros(3), np.eye(3))

        assert adapter.model.nu == 2
        assert adapter.model.nkey == 1
        assert adapter.model.key_ctrl[0] == pytest.approx((0.3, 0.4))
    finally:
        adapter.release()


def test_attached_nested_model_uses_its_own_mesh_directory(tmp_path: Path) -> None:
    meshes = tmp_path / "meshes"
    meshes.mkdir()
    (meshes / "tetra.obj").write_text(
        """v 0 0 0
v 1 0 0
v 0 1 0
v 0 0 1
f 1 3 2
f 1 2 4
f 1 4 3
f 2 3 4
""",
        encoding="utf-8",
    )
    child = tmp_path / "child.xml"
    child.write_text(
        """<mujoco>
  <compiler meshdir="meshes"/>
  <asset><mesh name="tetra" file="tetra.obj"/></asset>
  <worldbody><body name="attachment"><geom type="mesh" mesh="tetra"/></body></worldbody>
</mujoco>
""",
        encoding="utf-8",
    )
    scene = tmp_path / "scene.xml"
    scene.write_text(
        """<mujoco>
  <asset><model name="payload" file="child.xml"/></asset>
  <worldbody><attach model="payload" body="attachment" prefix="payload_"/></worldbody>
</mujoco>
""",
        encoding="utf-8",
    )
    assert mujoco.MjModel.from_xml_path(str(scene)).nmesh == 1
    adapter = MuJoCoAdapter()
    adapter.new_scene()

    try:
        model_id = adapter.add_scene_model(scene, np.zeros(3), np.eye(3))

        assert adapter.model.nmesh == 1
        assert adapter.model.ngeom == 1
        attachment = next(node for node in adapter.nodes() if node.name.endswith("attachment"))
        assert attachment.model_id == model_id
    finally:
        adapter.release()


def test_model_element_ownership_is_typed_exact_and_preindexed(tmp_path: Path) -> None:
    root = tmp_path / "root.xml"
    root.write_text(
        """<mujoco>
  <asset><material name="forge_1_root_paint"/></asset>
  <worldbody>
    <body name="forge_1_root_body">
      <geom name="forge_1_root_geom" type="box" size=".1 .1 .1"
            material="forge_1_root_paint"/>
    </body>
  </worldbody>
</mujoco>
""",
        encoding="utf-8",
    )
    adapter = MuJoCoAdapter(root)
    attached = adapter.add_scene_model(ASSETS / "test_scene.xml", np.zeros(3), np.eye(3))
    try:
        root_body = next(node for node in adapter.nodes() if node.name == "forge_1_root_body")
        root_geom = next(node for node in adapter.nodes() if node.name == "forge_1_root_geom")
        child_body = next(node for node in adapter.nodes() if node.name == "forge_1_frame")
        assert (root_body.model_id, root_geom.model_id, child_body.model_id) == (0, 0, attached)

        root_material = mujoco.mj_name2id(
            adapter.model, mujoco.mjtObj.mjOBJ_MATERIAL, "forge_1_root_paint"
        )
        assert root_material in adapter.model_material_indices(0)
        assert root_material not in adapter.model_material_indices(attached)

        class NoModelScan(list):
            def __iter__(self):
                raise AssertionError("model ownership lookup rescanned attached models")

        models = adapter._attached_models
        adapter._attached_models = NoModelScan(models)
        try:
            assert adapter._model_element_name("forge_1_frame", mujoco.mjtObj.mjOBJ_BODY) == (
                attached,
                "frame",
            )
        finally:
            adapter._attached_models = models
    finally:
        adapter.release()


def test_workspace_bounds_contain_primary_and_authored_geometry() -> None:
    document = WorkspaceAdapter(MuJoCoAdapter(ASSETS / "test_scene.xml"))
    document.add_scene_object(
        MeshShape.BOX,
        "left",
        np.ones(3, np.float32),
        np.array((-4.0, 0.0, 0.0), np.float32),
        np.eye(3, dtype=np.float32),
        np.ones(4, np.float32),
        DEFAULT_MATERIAL,
    )
    primary = document.primary.scene_source()
    authored = document.scene.source
    merged = document.scene_source()

    for source in (primary, authored):
        distance = float(np.linalg.norm(source.scene_center - merged.scene_center))
        assert distance + source.scene_extent <= merged.scene_extent + 1e-6


@pytest.mark.parametrize(
    "shape",
    (
        MeshShape.BOX,
        MeshShape.SPHERE,
        MeshShape.CYLINDER,
        MeshShape.CONE,
        MeshShape.PLANE,
    ),
)
def test_workspace_authored_hierarchy_has_one_edge_per_node(shape: MeshShape) -> None:
    document = workspace()
    document.add_scene_object(
        shape,
        "primitive",
        np.ones(3, np.float32),
        np.zeros(3, np.float32),
        np.eye(3, dtype=np.float32),
        np.ones(4, np.float32),
        DEFAULT_MATERIAL,
    )

    nodes = document.scene_source().nodes
    assert all(len(node.children) == len(set(node.children)) for node in nodes)
    for node in nodes[1:]:
        assert nodes[node.parent].children.count(node.node_id) == 1
    link = next(node for node in nodes if node.name == "primitive")
    assert [nodes[child].name for child in link.children] == ["primitive.geom"]


def test_workspace_authored_plane_size_is_editable_and_finite() -> None:
    document = workspace()
    object_id = document.add_scene_object(
        MeshShape.PLANE,
        "floor",
        np.array((4.0, 4.0, 0.02), np.float32),
        np.zeros(3, np.float32),
        np.eye(3, dtype=np.float32),
        np.ones(4, np.float32),
        DEFAULT_MATERIAL,
    )
    session = Session(document)
    link = session.node_by_object_id(object_id)
    geometry = session.node(link.children[0])

    assert session.source.geom_infinite_plane.tolist() == [False]
    assert session.submit(
        cmd.SetGeometrySize(geometry.node_id, np.array((3.0, 5.0, 0.02), np.float32))
    )
    assert session.source.geom_size[0] == pytest.approx((3.0, 5.0, 0.02))
    assert session.submit(cmd.Undo())
    assert session.source.geom_size[0] == pytest.approx((4.0, 4.0, 0.02))


def test_editor_plane_is_a_static_mujoco_ground_during_model_composition() -> None:
    from forge_viewer.ui.app import ViewerApp

    document = workspace()
    session = Session(document)
    assert session.submit(cmd.Pause())
    app = ViewerApp.__new__(ViewerApp)
    app.session = session

    app._add_scene_object(MeshShape.PLANE, "plane")

    assert document.primary.model.ngeom == 1
    assert int(document.primary.model.geom_type[0]) == int(mujoco.mjtGeom.mjGEOM_PLANE)
    assert int(document.primary.model.geom_bodyid[0]) == 0
    plane = next(node for node in session.nodes if node.name == "plane")
    assert plane.object_id != 0
    assert session.selected == plane.object_id
    assert session.source.geom_object_id.tolist() == [plane.object_id]
    picked, distance = document.primary.raycast(
        np.array((0.0, 0.0, 3.0)), np.array((0.0, 0.0, -1.0))
    )
    assert picked == plane.object_id
    assert distance == pytest.approx(3.0)

    assert session.submit(cmd.SetPose(plane.node_id, np.array((1.0, 2.0, 0.0)), np.eye(3)))
    assert document.primary.model.geom_pos[0] == pytest.approx((1.0, 2.0, 0.0))
    assert session.submit(cmd.Undo())
    assert document.primary.model.geom_pos[0] == pytest.approx((0.0, 0.0, 0.0))

    plane = session.node_by_object_id(plane.object_id)
    assert plane is not None
    assert session.submit(cmd.SetGeometrySize(plane.node_id, np.array((3.0, 5.0, 1.0), np.float32)))
    assert document.primary.model.geom_size[0] == pytest.approx((3.0, 5.0, 0.02))
    assert session.source.geom_size[0] == pytest.approx((3.0, 5.0, 1.0))
    assert session.submit(cmd.Undo())
    assert document.primary.model.geom_size[0] == pytest.approx((4.0, 4.0, 0.02))
    assert session.submit(cmd.Redo())
    assert document.primary.model.geom_size[0] == pytest.approx((3.0, 5.0, 0.02))
    object_id = session.selected
    plane = session.node_by_object_id(object_id)
    assert plane is not None
    assert session.submit(cmd.RenameModelElement(plane.node_id, "ground plane"))
    assert session.selected == object_id
    assert session.node_by_object_id(object_id).name == "ground plane"
    assert session.submit(cmd.Undo())
    assert session.selected == object_id
    assert session.node_by_object_id(object_id).name == "plane"
    assert session.submit(cmd.Redo())
    assert session.selected == object_id
    assert session.node_by_object_id(object_id).name == "ground plane"

    added = session.submit(cmd.AddSceneModel(ASSETS / "test_scene.urdf", np.zeros(3, np.float32)))
    assert added.ok, added.message
    assert document.primary.model.geom_size[0] == pytest.approx((3.0, 5.0, 0.02))
    assert session.submit(cmd.Play())
    for _ in range(10):
        frame = session.tick(FrameNeeds(poses=True), wall_dt=document.timestep())

    assert frame.geom_xpos[0] == pytest.approx(np.zeros(3))
    assert frame.geom_xmat[0] == pytest.approx(np.eye(3))


def test_workspace_round_trip_preserves_models_resources_and_entities(tmp_path: Path) -> None:
    document = workspace()
    model_id = document.add_scene_model(
        ASSETS / "test_scene.urdf",
        np.array((1.0, 2.0, 3.0), np.float32),
        np.eye(3, dtype=np.float32),
    )
    document.add_scene_object(
        MeshShape.BOX,
        "workcell",
        np.array((0.5, 0.4, 0.3), np.float32),
        np.array((0.0, 0.0, 0.3), np.float32),
        np.eye(3, dtype=np.float32),
        np.array((0.2, 0.4, 0.8, 1.0), np.float32),
        DEFAULT_MATERIAL,
    )
    document.add_scene_light(
        "key",
        Light(type=LightType.SPOT, position=np.array((1.0, -2.0, 3.0), np.float32)),
    )
    document.add_scene_camera(
        "inspection",
        CameraView(
            eye=np.array((2.0, -2.0, 1.5), np.float32),
            target=np.array((0.0, 0.0, 0.5), np.float32),
        ),
    )

    path = tmp_path / "workcell.forge.json"
    document.save_scene(path)
    restored = workspace()
    restored.open_scene(path)

    model = restored.scene_models()[0]
    assert model.model_id == model_id
    assert model.path == (ASSETS / "test_scene.urdf").resolve()
    assert model.position == pytest.approx((1.0, 2.0, 3.0))
    names = {node.name for node in restored.nodes()}
    assert {"workcell", "key", "inspection"} <= names


def test_workspace_exports_formatted_re_loadable_mjcf(tmp_path: Path) -> None:
    document = workspace()
    document.add_scene_model(
        ASSETS / "test_scene.xml",
        np.array((1.0, 2.0, 0.0), np.float32),
        np.eye(3, dtype=np.float32),
    )
    document.add_scene_object(
        MeshShape.CONE,
        "inspection cone",
        np.array((0.2, 0.2, 0.4), np.float32),
        np.array((0.0, 0.0, 0.4), np.float32),
        np.eye(3, dtype=np.float32),
        np.array((0.2, 0.4, 0.8, 1.0), np.float32),
        DEFAULT_MATERIAL,
    )
    document.add_scene_light(
        "inspection light",
        Light(
            type=LightType.SPOT,
            position=np.array((1.0, -2.0, 3.0), np.float32),
            cutoff=30.0,
        ),
    )
    document.add_scene_camera(
        "inspection camera",
        CameraView(
            eye=np.array((2.0, -2.0, 1.5), np.float32),
            target=np.array((0.0, 0.0, 0.5), np.float32),
        ),
    )

    path = tmp_path / "workcell.xml"
    document.save_scene(path)
    xml = path.read_text(encoding="utf-8")
    assert xml.endswith("\n")
    assert '\n    <camera name="forge_camera_0_inspection_camera"' in xml
    assert 'name="forge_light_0_inspection_light"' in xml
    assert 'name="forge_object_0_inspection_cone"' in xml

    model = mujoco.MjModel.from_xml_path(str(path))
    restored = MuJoCoAdapter(path)
    assert model.ncam == restored.model.ncam == 2
    assert model.nlight == restored.model.nlight == 2
    assert model.ngeom == restored.model.ngeom

    document.reload()
    source = document.scene_source()
    assert len(source.cameras) == 2
    assert len(source.lights.lights) == 2
    assert sum(node.type is NodeType.GEOM for node in source.nodes) == model.ngeom


def test_workspace_mjcf_export_stages_file_assets_with_relative_paths(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "meshes").mkdir(parents=True)
    (source / "meshes" / "part.obj").write_text(
        """v 0 0 0
v 1 0 0
v 0 1 0
v 0 0 1
f 1 3 2
f 1 2 4
f 1 4 3
f 2 3 4
""",
        encoding="utf-8",
    )
    model_path = source / "model.xml"
    model_path.write_text(
        """<mujoco model="portable">
  <compiler meshdir="meshes"/>
  <asset><mesh name="part" file="part.obj"/></asset>
  <worldbody><geom type="mesh" mesh="part"/></worldbody>
</mujoco>
""",
        encoding="utf-8",
    )
    document = WorkspaceAdapter(MuJoCoAdapter(model_path))
    export_dir = tmp_path / "export"
    path = export_dir / "portable.xml"

    document.save_scene(path)

    xml = path.read_text(encoding="utf-8")
    assert str(source) not in xml
    assert 'file="portable_assets/mesh_0_part.obj"' in xml
    moved = tmp_path / "moved"
    export_dir.rename(moved)
    assert mujoco.MjModel.from_xml_path(str(moved / "portable.xml")).ngeom == 1


def test_workspace_mjcf_export_roundtrips_forge_render_properties(tmp_path: Path) -> None:
    document = workspace()
    pixels = np.full((6, 4, 4, 3), 96, np.uint8)
    document.scene.add_texture(TextureData("sky", TextureType.CUBE, pixels))
    document.scene.add_texture(TextureData("studio", TextureType.CUBE, pixels + 32))
    assert document.scene.set_skybox("sky")
    document.scene.set_environment(
        Environment(
            ambient=np.array((0.11, 0.12, 0.13), np.float32),
            fog_color=np.array((0.2, 0.3, 0.4), np.float32),
            fog_start=0.75,
            fog_end=8.5,
            haze_color=np.array((0.5, 0.6, 0.7), np.float32),
            haze_density=0.35,
            horizon_haze=False,
            horizon_haze_slices=19,
        )
    )
    document.add_scene_light(
        "panel",
        Light(type=LightType.AREA, area_radius=0.45),
    )
    document.add_scene_light(
        "studio",
        Light(type=LightType.IMAGE, texture="studio", intensity=2.25),
    )

    path = tmp_path / "scene.xml"
    document.save_scene(path)

    restored = MuJoCoAdapter(path).scene_source()
    assert restored.skybox is not None
    assert restored.textures[restored.skybox].type is TextureType.SKYBOX
    assert [light.type for light in restored.lights.lights] == [
        LightType.AREA,
        LightType.IMAGE,
    ]
    assert restored.lights.lights[0].area_radius == pytest.approx(0.45)
    image = restored.lights.lights[1]
    assert image.texture is not None
    assert restored.textures[image.texture].type is TextureType.CUBE
    assert image.intensity == pytest.approx(2.25)
    environment = restored.lights.environment()
    assert environment.ambient == pytest.approx((0.11, 0.12, 0.13))
    assert environment.fog_color == pytest.approx((0.2, 0.3, 0.4))
    assert environment.fog_start == pytest.approx(0.75)
    assert environment.fog_end == pytest.approx(8.5)
    assert environment.haze_color == pytest.approx((0.5, 0.6, 0.7))
    assert environment.haze_density == pytest.approx(0.35)
    assert environment.horizon_haze is False
    assert environment.horizon_haze_slices == 19


def test_workspace_can_switch_from_authored_to_primary_skybox(tmp_path: Path) -> None:
    model = tmp_path / "skybox.xml"
    model.write_text(
        """
        <mujoco>
          <asset>
            <texture name="primary" type="skybox" builtin="gradient"
                     width="8" height="48" rgb1="1 0 0" rgb2="0 0 1"/>
          </asset>
          <worldbody/>
        </mujoco>
        """,
        encoding="utf-8",
    )
    document = WorkspaceAdapter(MuJoCoAdapter(model))
    pixels = np.full((6, 4, 4, 3), 96, np.uint8)
    document.scene.add_texture(TextureData("authored", TextureType.CUBE, pixels))
    session = Session(document, model)

    assert session.submit(cmd.SetSkybox("authored"))
    assert session.source.skybox == "authored"
    assert document.primary.scene_source().skybox is None
    assert session.submit(cmd.SetSkybox("primary"))
    assert session.source.skybox == "primary"
    assert document.scene.skybox is None


def test_workspace_can_export_current_pose_as_key0(tmp_path: Path) -> None:
    model_path = tmp_path / "free-body.xml"
    model_path.write_text(
        """<mujoco model="free-body">
  <worldbody>
    <body name="body"><freejoint/><geom type="box" size="0.1 0.1 0.1"/></body>
  </worldbody>
</mujoco>
""",
        encoding="utf-8",
    )
    primary = MuJoCoAdapter(model_path)
    document = WorkspaceAdapter(primary)
    primary.data.qpos[:3] = (1.0, 2.0, 3.0)
    mujoco.mj_forward(primary.model, primary.data)
    assert document.current_pose_modified()

    path = tmp_path / "posed.xml"
    document.save_scene(path, SceneSaveOptions(current_pose_keyframe="key0"))
    model = mujoco.MjModel.from_xml_path(str(path))
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "key0")
    assert key_id >= 0
    assert model.key_qpos[key_id, :3] == pytest.approx((1.0, 2.0, 3.0))


def test_workspace_resolves_models_from_resource_roots(tmp_path: Path) -> None:
    resources = tmp_path / "shared" / "models"
    resources.mkdir(parents=True)
    model_path = resources / "robot.xml"
    model_path.write_text((ASSETS / "test_scene.xml").read_text(), encoding="utf-8")
    document = workspace()
    document.set_resource_roots((resources,))
    document.add_scene_model(model_path, np.zeros(3), np.eye(3))

    path = tmp_path / "workspace" / "cell.forge.json"
    document.save_scene(path)
    payload = path.read_text(encoding="utf-8")
    assert '"path": "robot.xml"' in payload
    assert missing_resources(path) == ()

    restored = workspace()
    restored.open_scene(path)
    assert restored.scene_models()[0].path == model_path.resolve()


def test_workspace_relocates_one_missing_resource_and_reopens(tmp_path: Path) -> None:
    document = workspace()
    document.add_scene_model(ASSETS / "test_scene.xml", np.zeros(3), np.eye(3))
    path = tmp_path / "workspace" / "cell.forge.json"
    document.save_scene(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["models"][0]["path"] = "missing/robot.xml"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    path.chmod(0o640)
    replacement = tmp_path / "recovered" / "replacement.xml"
    replacement.parent.mkdir()
    replacement.write_text((ASSETS / "test_scene.xml").read_text(), encoding="utf-8")

    missing = missing_resource_entries(path)
    assert len(missing) == 1
    assert missing[0].reference == "missing/robot.xml"
    assert missing[0].expected_path == (path.parent / "missing/robot.xml").resolve()

    result = relocate_workspace_resource(path, missing[0].model_index, replacement)
    assert result.repaired == 1
    assert result.missing == ()
    assert path.stat().st_mode & 0o777 == 0o640
    restored = workspace()
    restored.open_scene(path)
    assert restored.scene_models()[0].path == replacement.resolve()


def test_workspace_repairs_unambiguous_resources_from_directory(tmp_path: Path) -> None:
    document = workspace()
    document.add_scene_model(ASSETS / "test_scene.xml", np.zeros(3), np.eye(3))
    document.add_scene_model(ASSETS / "test_scene.xml", np.ones(3), np.eye(3))
    document.add_scene_model(ASSETS / "test_scene.xml", np.full(3, 2.0), np.eye(3))
    path = tmp_path / "workspace" / "cell.forge.json"
    document.save_scene(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["models"][0]["path"] = "old/robots/arm.xml"
    payload["models"][1]["path"] = "old/tools/gripper.xml"
    payload["models"][2]["path"] = "old/ambiguous.xml"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    root = tmp_path / "new-assets"
    for relative in ("robots/arm.xml", "gripper.xml", "a/ambiguous.xml", "b/ambiguous.xml"):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text((ASSETS / "test_scene.xml").read_text(), encoding="utf-8")

    result = repair_workspace_resources(path, root)
    assert result.repaired == 2
    assert [item.reference for item in result.missing] == ["old/ambiguous.xml"]
    repaired = json.loads(path.read_text(encoding="utf-8"))
    assert (path.parent / repaired["models"][0]["path"]).resolve() == (
        root / "robots/arm.xml"
    ).resolve()
    assert (path.parent / repaired["models"][1]["path"]).resolve() == (
        root / "gripper.xml"
    ).resolve()
    assert repaired["models"][2]["path"] == "old/ambiguous.xml"


def test_mjspec_topology_edits_round_trip_in_workspace(tmp_path: Path) -> None:
    document = workspace()
    model_id = document.add_scene_model(ASSETS / "test_scene.xml", np.zeros(3), np.eye(3))
    model_node = next(
        node
        for node in document.nodes()
        if node.type is NodeType.MODEL and node.model_id == model_id
    )
    body_id = document.add_model_element(model_node.node_id, "body", "fixture")
    geom_id = document.add_model_element(body_id, "geom:box", "fixture_visual")
    assert body_id >= 0 and geom_id >= 0
    assert document.rename_model_element(body_id, "fixture_root")
    geom_id = next(
        node.node_id for node in document.nodes() if node.name == "forge_1_fixture_visual"
    )
    assert document.remove_model_element(geom_id)

    path = tmp_path / "topology.forge.json"
    document.save_scene(path)
    restored = workspace()
    restored.open_scene(path)
    names = {node.name for node in restored.nodes()}
    assert any(name.endswith("fixture_root") for name in names)
    assert not any(name.endswith("fixture_visual") for name in names)


def test_empty_root_body_addition_keeps_diagnostics_finite() -> None:
    document = workspace()
    session = Session(document)
    assert session.submit(cmd.Pause())
    world = next(node for node in session.nodes if node.type is NodeType.WORLD)

    result = session.submit(cmd.AddModelElement(world.node_id, "body", "fixture"))

    assert result.ok, result.message
    assert any(node.name == "fixture" for node in session.nodes)
    diagnostics = session.source.diagnostics
    assert diagnostics is not None
    assert np.isfinite(diagnostics.contact_force_scale)


def test_empty_root_site_creation_is_editable_and_undoable() -> None:
    document = workspace()
    session = Session(document)
    assert session.submit(cmd.Pause())
    world = next(node for node in session.nodes if node.type is NodeType.WORLD)

    result = session.submit(cmd.AddModelElement(world.node_id, "site", "target"))

    assert result.ok, result.message
    site = session.node(result.entity_id)
    assert site is not None
    assert site.type is NodeType.SITE
    assert session.site_properties(site.node_id) is not None
    assert session.submit(cmd.SelectNode(site.node_id))
    assert session.selected_node is not None
    assert session.selected_node.name == "target"

    assert session.submit(cmd.Undo())
    assert all(node.name != "target" for node in session.nodes)
    assert session.submit(cmd.Redo())
    assert any(node.name == "target" and node.type is NodeType.SITE for node in session.nodes)


def test_inline_height_field_lifecycle_edits_resolution_size_and_samples() -> None:
    document = WorkspaceAdapter(MuJoCoAdapter(ASSETS / "test_scene.xml"))
    session = Session(document, ASSETS / "test_scene.xml")
    assert session.submit(cmd.Pause())

    created = session.submit(
        cmd.CreateHeightField(
            0,
            "terrain",
            2,
            3,
            (2.0, 3.0, 4.0, 0.25),
            (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        )
    )

    assert created.ok, created.message
    asset = next(item for item in session.model_assets(0) if item.name == "terrain")
    fields = {field.name: field.value for field in asset.fields}
    assert asset.type == "hfield" and not asset.file
    assert fields["nrow"] == "2"
    assert fields["ncol"] == "3"
    assert len(fields["elevation"].split()) == 6
    assert asset.data_shape == (2, 3)
    assert asset.preview_shape == (2, 3)
    assert asset.preview_range == pytest.approx((0.0, 1.0))
    assert len(asset.preview_values) == 6
    assert document.primary.model.nhfield == 1

    updated = session.submit(
        cmd.SetHeightFieldData(
            0,
            "terrain",
            3,
            2,
            (4.0, 5.0, 6.0, 0.5),
            (0.0, 0.1, 0.2, 0.3, 0.4, 0.5),
        )
    )

    assert updated.ok, updated.message
    model = document.primary.model
    assert tuple(model.hfield_nrow) == (3,)
    assert tuple(model.hfield_ncol) == (2,)
    assert tuple(model.hfield_size[0]) == pytest.approx((4.0, 5.0, 6.0, 0.5))
    assert session.submit(cmd.Undo())
    asset = next(item for item in session.model_assets(0) if item.name == "terrain")
    fields = {field.name: field.value for field in asset.fields}
    assert fields["nrow"] == "2"
    assert fields["ncol"] == "3"


def test_empty_root_model_edit_batch_resolves_new_body_by_key() -> None:
    document = workspace()
    session = Session(document)
    assert session.submit(cmd.Pause())
    world = next(node for node in session.nodes if node.type is NodeType.WORLD)
    root = cmd.ModelElementRef(node_id=world.node_id)
    fixture = cmd.ModelElementRef(batch_key="fixture")

    result = session.submit(
        cmd.ModelEditBatch(
            (
                cmd.AddModelElementEdit(root, "body", "fixture", key="fixture"),
                cmd.AddModelElementEdit(fixture, "geom:box", "fixture_visual"),
                cmd.RenameModelElementEdit(fixture, "fixture_root"),
            )
        )
    )

    assert result.ok, result.message
    names = {node.name for node in session.nodes}
    assert {"fixture_root", "fixture_visual"} <= names


def test_model_edit_batch_restores_selection_by_model_element_identity() -> None:
    document = WorkspaceAdapter(MuJoCoAdapter(ASSETS / "test_scene.xml"))
    session = Session(document)
    assert session.submit(cmd.Pause())
    removed = next(node for node in session.nodes if node.name == "frame")
    selected = next(
        node
        for node in session.nodes
        if node.name == "mark_sphere" and node.type in (NodeType.LINK, NodeType.ROBOT)
    )
    assert session.submit(cmd.SelectNode(selected.node_id))

    result = session.submit(
        cmd.ModelEditBatch(
            (cmd.RemoveModelElementEdit(cmd.ModelElementRef(node_id=removed.node_id)),)
        )
    )

    assert result.ok, result.message
    assert session.selected_node is not None
    assert session.selected_node.name == "mark_sphere"
    assert session.selected == session.selected_node.object_id

    selected = session.selected_node
    result = session.submit(
        cmd.ModelEditBatch(
            (
                cmd.RenameModelElementEdit(
                    cmd.ModelElementRef(node_id=selected.node_id), "renamed_mark_sphere"
                ),
            )
        )
    )
    assert result.ok, result.message
    assert session.selected_node is not None
    assert session.selected_node.name == "renamed_mark_sphere"

    selected = session.selected_node
    result = session.submit(
        cmd.ModelEditBatch(
            (cmd.RemoveModelElementEdit(cmd.ModelElementRef(node_id=selected.node_id)),)
        )
    )
    assert result.ok, result.message
    assert session.selected == 0
    assert session.selected_node is None


def test_model_edit_batch_compiles_once_is_atomic_and_undoes_once(monkeypatch) -> None:
    document = workspace()
    model_id = document.add_scene_model(ASSETS / "test_scene.xml", np.zeros(3), np.eye(3))
    session = Session(document)
    assert session.submit(cmd.Pause())
    model_node = next(
        node for node in session.nodes if node.type is NodeType.MODEL and node.model_id == model_id
    )

    compile_count = 0
    compile_model = document.primary._compile_composed_model

    def counted_compile():
        nonlocal compile_count
        compile_count += 1
        return compile_model()

    monkeypatch.setattr(document.primary, "_compile_composed_model", counted_compile)
    node = cmd.ModelElementRef(node_id=model_node.node_id)
    fixture = cmd.ModelElementRef(batch_key="fixture")
    visual = cmd.ModelElementRef(batch_key="visual")
    result = session.submit(
        cmd.ModelEditBatch(
            (
                cmd.AddModelElementEdit(node, "body", "fixture", key="fixture"),
                cmd.AddModelElementEdit(fixture, "geom:box", "fixture_visual", key="visual"),
                cmd.RenameModelElementEdit(fixture, "fixture_root"),
                cmd.RemoveModelElementEdit(visual),
            )
        )
    )

    assert result.ok, result.message
    assert compile_count == 1
    names = {node.name for node in session.nodes}
    assert any(name.endswith("fixture_root") for name in names)
    assert not any(name.endswith("fixture_visual") for name in names)

    failed = session.submit(
        cmd.ModelEditBatch(
            (
                cmd.AddModelElementEdit(node, "body", "partial", key="partial"),
                cmd.AddModelElementEdit(node, "body", "partial"),
            )
        )
    )
    assert not failed.ok
    assert compile_count == 1
    assert not any(item.name.endswith("partial") for item in session.nodes)

    compile_failed = session.submit(
        cmd.ModelEditBatch((cmd.AddModelElementEdit(node, "joint:hinge", "bad_world_joint"),))
    )
    assert not compile_failed.ok
    assert "joint found in world body" in compile_failed.message
    assert compile_count == 2
    assert "bad_world_joint" not in document.scene_model_source(model_id)

    assert session.submit(cmd.Undo())
    assert compile_count == 3
    assert not any(item.name.endswith("fixture_root") for item in session.nodes)
    assert session.submit(cmd.Redo())
    assert compile_count == 4
    assert any(item.name.endswith("fixture_root") for item in session.nodes)


def test_primary_fixed_primitives_and_site_support_authoring(tmp_path: Path) -> None:
    path = tmp_path / "primitive-authoring.xml"
    path.write_text(
        """
<mujoco>
  <worldbody>
    <body name="fixture" pos="1 0 0">
      <geom name="sphere" type="sphere" size="0.1"/>
      <geom name="box" type="box" size="0.1 0.2 0.3"/>
      <geom name="cylinder" type="cylinder" size="0.1 0.2"/>
      <geom name="capsule" type="capsule" fromto="0 0 -0.5 0 0 0.5" size="0.1"/>
      <site name="target" type="box" size="0.1 0.1 0.1" pos="0 1 0"/>
    </body>
  </worldbody>
</mujoco>
""".strip(),
        encoding="utf-8",
    )
    document = WorkspaceAdapter(MuJoCoAdapter(path))
    session = Session(document)
    assert session.submit(cmd.Pause())

    fixture = next(node for node in session.nodes if node.name == "fixture")
    target = next(node for node in session.nodes if node.name == "target")
    assert fixture.posable
    assert target.posable
    assert session.submit(cmd.SetPose(fixture.node_id, np.array((2.0, 0.0, 0.0)), np.eye(3)))
    assert session.submit(cmd.SetPose(target.node_id, np.array((2.0, 2.0, 0.0)), np.eye(3)))
    capsule_node = next(node for node in session.nodes if node.name == "capsule")
    assert session.submit(cmd.SetPose(capsule_node.node_id, np.array((2.0, 0.0, 1.0)), np.eye(3)))

    requested = {
        "sphere": np.array((0.2, 0.2, 0.2), np.float32),
        "box": np.array((0.2, 0.3, 0.4), np.float32),
        "cylinder": np.array((0.2, 0.2, 0.3), np.float32),
        "capsule": np.array((0.2, 0.2, 0.3), np.float32),
        "target": np.array((0.4, 0.5, 0.6), np.float32),
    }
    for name, size in requested.items():
        node = next(item for item in session.nodes if item.name == name)
        result = session.submit(cmd.SetGeometrySize(node.node_id, size))
        assert result.ok, f"{name}: {result.message}"

    assert session.submit(cmd.Undo())
    model = document.primary.model
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "target")
    assert model.site_size[site] == pytest.approx((0.1, 0.1, 0.1))
    assert session.submit(cmd.Redo())

    model = document.primary.model
    for name, expected in {
        "sphere": (0.2, 0.0, 0.0),
        "box": (0.2, 0.3, 0.4),
        "cylinder": (0.2, 0.3, 0.0),
        "capsule": (0.2, 0.3, 0.0),
    }.items():
        index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        assert model.geom_size[index] == pytest.approx(expected)
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "target")
    assert model.site_size[site] == pytest.approx((0.4, 0.5, 0.6))

    xml = document.scene_model_xml(0)
    assert xml is not None
    spec = mujoco.MjSpec.from_string(xml)
    assert spec.body("fixture").pos == pytest.approx((2.0, 0.0, 0.0))
    assert spec.site("target").pos == pytest.approx((0.0, 2.0, 0.0))
    capsule = spec.geom("capsule")
    assert capsule.size[:2] == pytest.approx((0.2, 0.3))
    assert capsule.pos == pytest.approx((0.0, 0.0, 1.0))
    assert np.abs(math3d.quat_to_mat3(capsule.quat)[:, 2]) == pytest.approx((0.0, 0.0, 1.0))


def test_site_properties_apply_shape_group_and_endpoints_in_one_rebuild(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "site-properties.xml"
    path.write_text(
        """
<mujoco>
  <worldbody>
    <body name="fixture">
      <site name="marker" type="capsule" fromto="0 0 0 0 0 1" size="0.1" group="1"/>
      <geom type="sphere" size="0.1"/>
    </body>
  </worldbody>
</mujoco>
""".strip(),
        encoding="utf-8",
    )
    document = WorkspaceAdapter(MuJoCoAdapter(path))
    session = Session(document)
    assert session.submit(cmd.Pause())
    node = next(item for item in session.nodes if item.name == "marker")
    current = session.site_properties(node.node_id)
    assert current is not None
    assert current.type == "capsule"
    assert current.group == 1
    assert current.use_from_to
    assert current.from_to == pytest.approx((0.0, 0.0, 0.0, 0.0, 0.0, 1.0))

    invalid = session.submit(
        cmd.SetSiteProperties(node.node_id, "sphere", 2, True, current.from_to)
    )
    assert not invalid.ok
    assert "capsule and cylinder" in invalid.message

    compile_count = 0
    compile_model = document.primary._compile_composed_model

    def counted_compile():
        nonlocal compile_count
        compile_count += 1
        return compile_model()

    monkeypatch.setattr(document.primary, "_compile_composed_model", counted_compile)
    result = session.submit(
        cmd.SetSiteProperties(
            node.node_id,
            "cylinder",
            2,
            True,
            (-1.0, 0.0, 0.25, 1.0, 0.0, 0.25),
        )
    )

    assert result.ok, result.message
    assert compile_count == 1
    updated_node = next(item for item in session.nodes if item.name == "marker")
    updated = session.site_properties(updated_node.node_id)
    assert updated is not None
    assert updated.type == "cylinder"
    assert updated.group == 2
    assert updated.use_from_to
    assert updated.from_to == pytest.approx((-1.0, 0.0, 0.25, 1.0, 0.0, 0.25))
    site = mujoco.mj_name2id(document.primary.model, mujoco.mjtObj.mjOBJ_SITE, "marker")
    assert int(document.primary.model.site_type[site]) == int(mujoco.mjtGeom.mjGEOM_CYLINDER)
    assert int(document.primary.model.site_group[site]) == 2
    assert document.primary.model.site_pos[site] == pytest.approx((0.0, 0.0, 0.25))
    assert document.primary.model.site_size[site, 1] == pytest.approx(1.0)

    assert session.submit(cmd.Undo())
    restored_node = next(item for item in session.nodes if item.name == "marker")
    restored = session.site_properties(restored_node.node_id)
    assert restored is not None
    assert restored.type == "capsule"
    assert restored.group == 1


def test_joint_properties_update_without_topology_recompile(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "joint-properties.xml"
    path.write_text(
        """
<mujoco>
  <compiler angle="degree" autolimits="true"/>
  <worldbody>
    <body name="hinge_body">
      <joint name="hinge" type="hinge" axis="0 0 1" range="-90 90"
             damping="0.1" stiffness="0.2"/>
      <geom type="box" size="0.1 0.1 0.1" mass="1"/>
    </body>
  </worldbody>
</mujoco>
""".strip(),
        encoding="utf-8",
    )
    document = WorkspaceAdapter(MuJoCoAdapter(path))
    session = Session(document)
    assert session.submit(cmd.Pause())
    joint = session.joints[0]
    assert joint.axis == pytest.approx((0.0, 0.0, 1.0))
    assert joint.damping == pytest.approx(0.1)
    assert joint.stiffness == pytest.approx(0.2)
    invalid = session.submit(
        cmd.SetJointProperties(
            joint.joint_id,
            np.zeros(3),
            True,
            (-0.5, 0.75),
            0.4,
            0.6,
        )
    )
    assert not invalid.ok
    assert "axis" in invalid.message.lower()

    compile_count = 0
    compile_model = document.primary._compile_composed_model

    def counted_compile():
        nonlocal compile_count
        compile_count += 1
        return compile_model()

    monkeypatch.setattr(document.primary, "_compile_composed_model", counted_compile)
    source = session.source
    result = session.submit(
        cmd.SetJointProperties(
            joint.joint_id,
            np.array((1.0, 1.0, 0.0)),
            True,
            (-0.5, 0.75),
            0.4,
            0.6,
        )
    )

    assert result.ok, result.message
    assert compile_count == 0
    assert session.source is source
    model = document.primary.model
    assert model.jnt_axis[0] == pytest.approx(np.sqrt(0.5) * np.array((1.0, 1.0, 0.0)))
    assert bool(model.jnt_limited[0])
    assert model.jnt_range[0] == pytest.approx((-0.5, 0.75))
    assert model.dof_damping[0] == pytest.approx(0.4)
    assert model.jnt_stiffness[0] == pytest.approx(0.6)
    spec = mujoco.MjSpec.from_string(document.scene_model_xml(0))
    authored = spec.joint("hinge")
    assert authored.range == pytest.approx((-0.5, 0.75))
    assert authored.damping[0] == pytest.approx(0.4)
    assert authored.stiffness[0] == pytest.approx(0.6)

    assert session.submit(cmd.Undo())
    assert document.primary.model.jnt_axis[0] == pytest.approx((0.0, 0.0, 1.0))
    assert session.submit(cmd.Redo())
    assert document.primary.model.jnt_axis[0] == pytest.approx(
        np.sqrt(0.5) * np.array((1.0, 1.0, 0.0))
    )


def test_ball_joint_properties_apply_one_limit_and_all_rotational_damping(tmp_path: Path) -> None:
    path = tmp_path / "ball-properties.xml"
    path.write_text(
        """
<mujoco>
  <compiler angle="degree" autolimits="true"/>
  <worldbody>
    <body name="ball_body">
      <joint name="ball" type="ball" range="0 120" damping="0.1" stiffness="0.2"/>
      <geom type="sphere" size="0.1" mass="1"/>
    </body>
  </worldbody>
</mujoco>
""".strip(),
        encoding="utf-8",
    )
    document = WorkspaceAdapter(MuJoCoAdapter(path))
    session = Session(document)
    assert session.submit(cmd.Pause())

    result = session.submit(
        cmd.SetJointProperties(0, np.array((0.0, 0.0, 1.0)), False, (0.0, 0.0), 0.5, 0.4)
    )
    assert result.ok, result.message
    assert not bool(document.primary.model.jnt_limited[0])
    assert document.primary.model.jnt_range[0] == pytest.approx((0.0, 0.0))

    result = session.submit(
        cmd.SetJointProperties(0, np.array((0.0, 0.0, 1.0)), True, (-4.0, 1.0), 0.7, 0.8)
    )

    assert result.ok, result.message
    model = document.primary.model
    assert model.jnt_range[0] == pytest.approx((0.0, 1.0))
    assert model.dof_damping[:3] == pytest.approx((0.7, 0.7, 0.7))
    assert model.jnt_stiffness[0] == pytest.approx(0.8)
    spec = mujoco.MjSpec.from_string(document.scene_model_xml(0))
    assert spec.joint("ball").range == pytest.approx((0.0, 1.0))


def test_joint_advanced_properties_rebuild_once_and_preserve_authored_units(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "joint-advanced.xml"
    path.write_text(
        """
<mujoco>
  <compiler angle="degree" autolimits="true"/>
  <worldbody>
    <body name="hinge_body">
      <joint name="hinge" type="hinge" range="-90 90" ref="10" springref="5"
             group="2" armature="0.1" frictionloss="0.2" margin="0.01"
             solreflimit="0.03 1.2" solimplimit="0.8 0.9 0.002 0.6 3"
             solreffriction="0.04 1.3" solimpfriction="0.7 0.95 0.003 0.55 4"
             actuatorfrclimited="auto" actuatorfrcrange="-1 1"/>
      <geom type="box" size="0.1 0.1 0.1" mass="1"/>
    </body>
  </worldbody>
</mujoco>
""".strip(),
        encoding="utf-8",
    )
    document = WorkspaceAdapter(MuJoCoAdapter(path))
    session = Session(document)
    assert session.submit(cmd.Pause())
    current = session.joint_advanced_properties(0)
    assert current is not None
    assert current.reference == pytest.approx(np.radians(10.0))
    assert current.spring_reference == pytest.approx(np.radians(5.0))
    assert current.actuator_force_limit_mode == "auto"

    invalid = session.submit(
        cmd.SetJointAdvancedProperties(
            0,
            2,
            0.1,
            0.2,
            current.reference,
            current.spring_reference,
            0.01,
            current.limit_solver_reference,
            current.limit_solver_impedance,
            current.friction_solver_reference,
            current.friction_solver_impedance,
            "limited",
            (3.0, -2.0),
            False,
        )
    )
    assert not invalid.ok
    assert "upper bound" in invalid.message.lower()

    compile_count = 0
    compile_model = document.primary._compile_composed_model

    def counted_compile():
        nonlocal compile_count
        compile_count += 1
        return compile_model()

    monkeypatch.setattr(document.primary, "_compile_composed_model", counted_compile)
    result = session.submit(
        cmd.SetJointAdvancedProperties(
            joint_id=0,
            group=4,
            armature=0.35,
            friction_loss=0.45,
            reference=np.radians(20.0),
            spring_reference=np.radians(-15.0),
            margin=0.025,
            limit_solver_reference=(0.05, 1.5),
            limit_solver_impedance=(0.6, 0.95, 0.004, 0.7, 2.5),
            friction_solver_reference=(0.06, 1.6),
            friction_solver_impedance=(0.5, 0.9, 0.005, 0.65, 3.5),
            actuator_force_limit_mode="limited",
            actuator_force_range=(-2.0, 3.0),
            actuator_gravity_compensation=True,
        )
    )

    assert result.ok, result.message
    assert compile_count == 1
    updated = session.joint_advanced_properties(0)
    assert updated is not None
    assert updated.group == 4
    assert updated.reference == pytest.approx(np.radians(20.0))
    assert updated.spring_reference == pytest.approx(np.radians(-15.0))
    assert updated.actuator_force_limit_mode == "limited"
    assert updated.actuator_force_range == pytest.approx((-2.0, 3.0))
    assert updated.actuator_gravity_compensation
    assert document.primary.model.qpos0[0] == pytest.approx(np.radians(20.0))
    assert document.primary.model.qpos_spring[0] == pytest.approx(np.radians(-15.0))
    assert bool(document.primary.model.jnt_actfrclimited[0])

    authored = mujoco.MjSpec.from_string(document.scene_model_xml(0)).joint("hinge")
    assert authored.ref == pytest.approx(np.radians(20.0), abs=1e-6)
    assert authored.springref == pytest.approx(np.radians(-15.0), abs=1e-6)
    assert authored.actfrcrange == pytest.approx((-2.0, 3.0))
    assert session.submit(cmd.Undo())
    restored = session.joint_advanced_properties(0)
    assert restored is not None
    assert restored.reference == pytest.approx(np.radians(10.0))
    assert restored.actuator_force_limit_mode == "auto"


def test_geometry_contact_properties_update_without_topology_recompile(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "contact-properties.xml"
    path.write_text(
        """<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 .1" friction=".8 .02 .003"
          contype="2" conaffinity="5" condim="4" priority="3"
          margin=".01" gap=".002" solmix=".6"/>
  </worldbody>
</mujoco>
""",
        encoding="utf-8",
    )
    document = WorkspaceAdapter(MuJoCoAdapter(path))
    session = Session(document)
    assert session.submit(cmd.Pause())
    floor = next(node for node in session.nodes if node.name == "floor")
    initial = session.geometry_properties(floor.node_id)
    assert initial is not None
    assert initial.friction == pytest.approx((0.8, 0.02, 0.003))
    assert initial.contact_dimension == 4

    compile_count = 0
    compile_model = document.primary._compile_composed_model

    def counted_compile():
        nonlocal compile_count
        compile_count += 1
        return compile_model()

    monkeypatch.setattr(document.primary, "_compile_composed_model", counted_compile)
    source = session.source
    generation = session.structure_generation
    edited = cmd.SetGeometryProperties(
        floor.node_id,
        (1.2, 0.04, 0.006),
        7,
        9,
        6,
        11,
        0.03,
        0.004,
        0.25,
        (-120.0, -4.0),
        (0.7, 0.9, 0.004, 0.35, 3.0),
        0.2,
        (1.0, 2.0, 3.0, 0.1, 0.2, 0.3),
    )
    result = session.submit(edited)
    assert result.ok, result.message
    assert compile_count == 0
    assert session.source is source
    assert session.structure_generation == generation + 1

    current = session.geometry_properties(floor.node_id)
    assert current is not None
    assert current.friction == pytest.approx((1.2, 0.04, 0.006))
    assert (
        current.collision_type_mask,
        current.collision_affinity_mask,
        current.contact_dimension,
        current.contact_priority,
    ) == (7, 9, 6, 11)
    assert (current.margin, current.gap, current.solver_mix) == pytest.approx((0.03, 0.004, 0.25))
    assert current.solver_reference == pytest.approx((-120.0, -4.0))
    assert current.solver_impedance == pytest.approx((0.7, 0.9, 0.004, 0.35, 3.0))
    assert current.adhesion == pytest.approx(0.2)
    assert current.surface_velocity == pytest.approx((1.0, 2.0, 3.0, 0.1, 0.2, 0.3))
    geom = mujoco.mj_name2id(document.primary.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    assert document.primary.model.geom_friction[geom] == pytest.approx((1.2, 0.04, 0.006))
    spec_geom = document.primary._root_spec.geom("floor")
    assert spec_geom.friction == pytest.approx((1.2, 0.04, 0.006))
    assert (spec_geom.contype, spec_geom.conaffinity, spec_geom.condim) == (7, 9, 6)
    assert spec_geom.solref == pytest.approx((-120.0, -4.0))
    assert spec_geom.solimp == pytest.approx((0.7, 0.9, 0.004, 0.35, 3.0))
    assert spec_geom.adhesion == pytest.approx(0.2)
    assert spec_geom.surfacevel == pytest.approx((1.0, 2.0, 3.0, 0.1, 0.2, 0.3))

    invalid = session.submit(replace(edited, friction=(-1.0, 0.0, 0.0)))
    assert not invalid.ok
    assert "friction" in invalid.message.lower()
    malformed = session.submit(replace(edited, friction=(1.0,)))
    assert not malformed.ok
    assert "invalid value types" in malformed.message.lower()
    assert session.submit(cmd.Undo())
    restored = session.geometry_properties(floor.node_id)
    assert restored is not None and restored.friction == pytest.approx(initial.friction)
    assert session.submit(cmd.Redo())
    restored = session.geometry_properties(floor.node_id)
    assert restored is not None and restored.friction == pytest.approx((1.2, 0.04, 0.006))


def test_geometry_advanced_properties_recompile_once(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "geometry-advanced.xml"
    path.write_text(
        """<mujoco>
  <worldbody>
    <body name="payload">
      <freejoint/>
      <geom name="payload_geom" type="box" size="0.2 0.3 0.4" density="500"
            group="2" shellinertia="true" fluidshape="ellipsoid"
            fluidcoef="0.6 0.3 1.2 0.8 0.7"/>
    </body>
  </worldbody>
</mujoco>
""",
        encoding="utf-8",
    )
    document = WorkspaceAdapter(MuJoCoAdapter(path))
    session = Session(document)
    assert session.submit(cmd.Pause())
    geom = next(node for node in session.nodes if node.name == "payload_geom")
    initial = session.geometry_advanced_properties(geom.node_id)
    assert initial is not None
    assert initial.visual_group == 2
    assert initial.mass_mode == "density"
    assert initial.density == pytest.approx(500.0)
    assert initial.inertia_mode == "shell"
    assert initial.fluid_ellipsoid

    compile_count = 0
    compile_model = document.primary._compile_composed_model

    def counted_compile():
        nonlocal compile_count
        compile_count += 1
        return compile_model()

    monkeypatch.setattr(document.primary, "_compile_composed_model", counted_compile)
    edited = cmd.SetGeometryAdvancedProperties(
        geom.node_id,
        1,
        "mass",
        5.0,
        700.0,
        "volume",
        False,
        (0.5, 0.25, 1.5, 1.0, 1.0),
    )
    result = session.submit(edited)
    assert result.ok, result.message
    assert compile_count == 1
    current = session.geometry_advanced_properties(geom.node_id)
    assert current is not None
    assert current.visual_group == 1
    assert current.mass_mode == "mass"
    assert current.mass == pytest.approx(5.0)
    assert current.inertia_mode == "volume"
    assert not current.fluid_ellipsoid
    body = mujoco.mj_name2id(document.primary.model, mujoco.mjtObj.mjOBJ_BODY, "payload")
    assert document.primary.model.body_mass[body] == pytest.approx(5.0)

    assert session.submit(cmd.Undo())
    restored = session.geometry_advanced_properties(geom.node_id)
    assert restored is not None
    assert restored.mass_mode == "density"
    assert restored.inertia_mode == "shell"


def test_geometry_resource_import_and_shape_assignment_are_atomic(
    tmp_path: Path, monkeypatch
) -> None:
    from PIL import Image

    path = tmp_path / "resource-shape.xml"
    path.write_text(
        """<mujoco>
  <worldbody><geom name="resource_geom" type="box" size="0.2 0.3 0.4"/></worldbody>
</mujoco>
""",
        encoding="utf-8",
    )
    mesh_path = tmp_path / "tetra.obj"
    mesh_path.write_text(
        """v 0 0 0
v 1 0 0
v 0 1 0
v 0 0 1
f 1 3 2
f 1 2 4
f 1 4 3
f 2 3 4
""",
        encoding="utf-8",
    )
    height_path = tmp_path / "terrain.png"
    Image.fromarray(np.arange(16, dtype=np.uint8).reshape(4, 4)).save(height_path)
    document = WorkspaceAdapter(MuJoCoAdapter(path))
    session = Session(document)
    assert session.submit(cmd.Pause())
    geom = next(node for node in session.nodes if node.name == "resource_geom")

    compile_count = 0
    compile_model = document.primary._compile_composed_model

    def counted_compile():
        nonlocal compile_count
        compile_count += 1
        return compile_model()

    monkeypatch.setattr(document.primary, "_compile_composed_model", counted_compile)
    result = session.submit(
        cmd.ImportModelGeometryResource(geom.node_id, "mesh", mesh_path, "tetra")
    )
    assert result.ok, result.message
    assert compile_count == 1
    shape = session.geometry_shape_properties(geom.node_id)
    assert shape is not None
    assert shape.type == "mesh"
    assert shape.resource_name == "tetra"
    assert shape.mesh_names == ("tetra",)

    result = session.submit(cmd.SetGeometryShape(geom.node_id, "box"))
    assert result.ok, result.message
    assert compile_count == 2
    shape = session.geometry_shape_properties(geom.node_id)
    assert shape is not None and shape.type == "box"

    result = session.submit(
        cmd.ImportModelGeometryResource(geom.node_id, "hfield", height_path, "terrain")
    )
    assert result.ok, result.message
    assert compile_count == 3
    shape = session.geometry_shape_properties(geom.node_id)
    assert shape is not None
    assert shape.type == "hfield"
    assert shape.resource_name == "terrain"
    assert shape.height_field_names == ("terrain",)

    missing = session.submit(cmd.SetGeometryShape(geom.node_id, "mesh", "missing"))
    assert not missing.ok
    assert "unavailable" in missing.message.lower()
    assert session.submit(cmd.Undo())
    restored = session.geometry_shape_properties(geom.node_id)
    assert restored is not None and restored.type == "box"


def test_model_height_field_asset_lifecycle_is_standalone_safe_and_atomic(
    tmp_path: Path, monkeypatch
) -> None:
    from PIL import Image

    path = tmp_path / "asset-lifecycle.xml"
    path.write_text(
        """<mujoco>
  <worldbody><geom name="resource_geom" type="box" size="0.2 0.3 0.4"/></worldbody>
</mujoco>
""",
        encoding="utf-8",
    )
    first_path = tmp_path / "terrain.png"
    second_path = tmp_path / "terrain-replacement.png"
    Image.fromarray(np.arange(16, dtype=np.uint8).reshape(4, 4)).save(first_path)
    Image.fromarray(np.arange(25, dtype=np.uint8).reshape(5, 5)).save(second_path)
    document = WorkspaceAdapter(MuJoCoAdapter(path))
    session = Session(document)
    assert session.submit(cmd.Pause())
    geom = next(node for node in session.nodes if node.name == "resource_geom")

    compile_count = 0
    compile_model = document.primary._compile_composed_model

    def counted_compile():
        nonlocal compile_count
        compile_count += 1
        return compile_model()

    monkeypatch.setattr(document.primary, "_compile_composed_model", counted_compile)
    result = session.submit(
        cmd.ImportModelAsset(
            0,
            "hfield",
            first_path,
            "terrain",
            (("size", "2 3 0.4 0.2"),),
        )
    )
    assert result.ok, result.message
    assert compile_count == 1
    asset = session.model_assets(0)[0]
    assert (asset.type, asset.name, asset.references) == ("hfield", "terrain", ())
    assert {field.name: field.value for field in asset.fields}["size"] == "2 3 0.4 0.2"
    assert asset.data_shape == (4, 4)
    assert asset.preview_shape == (4, 4)
    assert asset.preview_range == pytest.approx((0.0, 1.0))
    shape = session.geometry_shape_properties(geom.node_id)
    assert shape is not None and shape.type == "box"
    assert shape.height_field_names == ("terrain",)

    result = session.submit(cmd.SetGeometryShape(geom.node_id, "hfield", "terrain"))
    assert result.ok, result.message
    assert compile_count == 2
    asset = session.model_assets(0)[0]
    assert asset.references == ("geom resource_geom",)

    result = session.submit(cmd.RenameModelAsset(0, "hfield", "terrain", "terrain_main"))
    assert result.ok, result.message
    assert compile_count == 3
    shape = session.geometry_shape_properties(geom.node_id)
    assert shape is not None and shape.resource_name == "terrain_main"
    blocked = session.submit(cmd.RemoveModelAsset(0, "hfield", "terrain_main"))
    assert not blocked.ok
    assert "used by 1" in blocked.message
    assert compile_count == 3

    assert session.submit(cmd.SetGeometryShape(geom.node_id, "box"))
    assert compile_count == 4
    result = session.submit(cmd.ReplaceModelAssetFile(0, "hfield", "terrain_main", second_path))
    assert result.ok, result.message
    assert compile_count == 5
    asset = session.model_assets(0)[0]
    assert Path(asset.file) == second_path
    assert asset.data_shape == (5, 5)
    assert asset.preview_shape == (5, 5)

    result = session.submit(cmd.DuplicateModelAsset(0, "hfield", "terrain_main", "terrain_copy"))
    assert result.ok, result.message
    assert compile_count == 6
    assert {asset.name for asset in session.model_assets(0)} == {
        "terrain_main",
        "terrain_copy",
    }
    assert session.submit(cmd.RemoveModelAsset(0, "hfield", "terrain_copy"))
    assert session.submit(cmd.RemoveModelAsset(0, "hfield", "terrain_main"))
    assert compile_count == 8
    assert session.model_assets(0) == ()
    assert session.submit(cmd.Undo())
    assert {asset.name for asset in session.model_assets(0)} == {"terrain_main"}


def test_model_asset_reference_graph_repairs_material_and_texture_names() -> None:
    document = workspace()
    assert document.set_scene_model_xml(
        0,
        """<mujoco>
  <asset>
    <texture name="grid" type="2d" builtin="checker" width="8" height="8"/>
    <material name="surface" texture="grid"/>
    <material name="unused" rgba="0.2 0.3 0.4 1"/>
  </asset>
  <worldbody><geom name="floor" type="plane" size="1 1 0.1" material="surface"/></worldbody>
</mujoco>""",
    )
    session = Session(document)
    assert session.submit(cmd.Pause())
    assets = {(asset.type, asset.name): asset for asset in session.model_assets(0)}
    assert assets[("texture", "grid")].references == ("material surface",)
    assert assets[("material", "surface")].references == ("geom floor",)
    assert assets[("material", "unused")].references == ()

    result = session.submit(cmd.RenameModelAsset(0, "texture", "grid", "grid_main"))
    assert result.ok, result.message
    result = session.submit(cmd.RenameModelAsset(0, "material", "surface", "surface_main"))
    assert result.ok, result.message
    xml = document.scene_model_xml(0)
    assert xml is not None
    assert 'texture="grid_main"' in xml
    assert 'material="surface_main"' in xml

    blocked = session.submit(cmd.RemoveModelAsset(0, "material", "surface_main"))
    assert not blocked.ok
    assert "used by 1" in blocked.message
    result = session.submit(cmd.DuplicateModelAsset(0, "material", "surface_main", "surface_copy"))
    assert result.ok, result.message
    copied = next(asset for asset in session.model_assets(0) if asset.name == "surface_copy")
    assert copied.references == ()
    assert session.submit(cmd.RemoveModelAsset(0, "material", "surface_copy"))
    assert session.submit(cmd.RemoveModelAsset(0, "material", "unused"))


def test_body_inertia_properties_recompile_once_and_restore_auto_derivation(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "body-properties.xml"
    path.write_text(
        """<mujoco>
  <worldbody>
    <body name="payload">
      <freejoint/>
      <inertial pos="0.1 0.2 0.3" mass="2" diaginertia="1 1.5 2"/>
      <geom name="payload_geom" type="box" size="0.2 0.3 0.4" density="500"/>
    </body>
  </worldbody>
</mujoco>
""",
        encoding="utf-8",
    )
    document = WorkspaceAdapter(MuJoCoAdapter(path))
    session = Session(document)
    assert session.submit(cmd.Pause())
    payload = next(node for node in session.nodes if node.name == "payload")
    initial = session.body_properties(payload.node_id)
    assert initial is not None
    assert initial.inertia_mode == "diagonal"
    assert initial.mass == pytest.approx(2.0)

    compile_count = 0
    compile_model = document.primary._compile_composed_model

    def counted_compile():
        nonlocal compile_count
        compile_count += 1
        return compile_model()

    monkeypatch.setattr(document.primary, "_compile_composed_model", counted_compile)
    edited = cmd.SetBodyProperties(
        payload.node_id,
        "full",
        3.0,
        (0.2, 0.1, -0.1),
        (1.0, 0.0, 0.0, 0.0),
        (1.0, 1.5, 2.0),
        (2.0, 3.0, 4.0, 0.1, 0.2, 0.3),
        0.25,
        False,
        "allowed",
    )
    result = session.submit(edited)
    assert result.ok, result.message
    assert compile_count == 1
    current = session.body_properties(payload.node_id)
    assert current is not None
    assert current.inertia_mode == "full"
    assert current.mass == pytest.approx(3.0)
    assert current.full_inertia == pytest.approx(edited.full_inertia)
    assert current.gravity_compensation == pytest.approx(0.25)
    assert current.sleep_policy == "allowed"

    result = session.submit(replace(edited, inertia_mode="auto", sleep_policy="auto"))
    assert result.ok, result.message
    assert compile_count == 2
    automatic = session.body_properties(payload.node_id)
    assert automatic is not None
    assert automatic.inertia_mode == "auto"
    assert automatic.mass == pytest.approx(96.0)
    spec_body = document.primary._root_spec.body("payload")
    assert not spec_body.explicitinertial
    assert spec_body.mass == 0.0
    assert np.isnan(spec_body.ipos[0])
    assert np.isnan(spec_body.fullinertia[0])

    invalid = session.submit(replace(edited, full_inertia=(1.0, 1.0, 3.0, 0, 0, 0)))
    assert not invalid.ok
    assert "triangle" in invalid.message.lower()
    assert session.submit(cmd.Undo())
    restored = session.body_properties(payload.node_id)
    assert restored is not None and restored.inertia_mode == "full"
    assert restored.mass == pytest.approx(3.0)


def test_standalone_model_material_lifecycle_does_not_require_geometry(tmp_path: Path) -> None:
    path = tmp_path / "standalone-material.xml"
    path.write_text(
        '<mujoco><worldbody><body name="empty"/></worldbody></mujoco>',
        encoding="utf-8",
    )
    document = WorkspaceAdapter(MuJoCoAdapter(path))
    session = Session(document)
    assert session.submit(cmd.Pause())

    created = session.submit(cmd.CreateModelMaterial(0, "surface"))

    assert created.ok, created.message
    assert created.entity_id >= 0
    asset = next(item for item in session.model_assets(0) if item.name == "surface")
    assert asset.type == "material"
    assert asset.references == ()
    assert asset.runtime_index == created.entity_id
    duplicate = session.submit(cmd.CreateModelMaterial(0, "surface"))
    assert not duplicate.ok

    source = session.source
    assert source is not None
    material = source.materials[created.entity_id]
    edited = replace(
        material,
        rgba=np.array((0.2, 0.4, 0.6, 0.8), np.float32),
        emission=0.25,
        specular=0.75,
        metallic=0.4,
        roughness=0.6,
    )
    result = session.submit(cmd.SetMaterial(created.entity_id, edited))
    assert result.ok, result.message
    xml = document.scene_model_xml(0)
    assert xml is not None
    assert 'name="surface"' in xml
    assert 'rgba="0.2 0.4 0.6 0.8"' in xml
    assert 'metallic="0.4"' in xml
    assert 'roughness="0.6"' in xml

    assert session.submit(cmd.RemoveModelAsset(0, "material", "surface"))
    assert not any(item.name == "surface" for item in session.model_assets(0))
    assert session.submit(cmd.Undo())
    assert any(item.name == "surface" for item in session.model_assets(0))


def test_material_edits_and_bound_copies_preserve_pbr_texture_layers(tmp_path: Path) -> None:
    import xml.etree.ElementTree as ET

    path = tmp_path / "layered-material.xml"
    path.write_text(
        """
<mujoco>
  <asset>
    <texture name="rgb" type="2d" builtin="flat" width="2" height="2"/>
    <texture name="normal" type="2d" builtin="flat" width="2" height="2"/>
    <texture name="orm" type="2d" builtin="flat" width="2" height="2"/>
    <material name="surface" metallic="0.3" roughness="0.7">
      <layer role="rgb" texture="rgb"/>
      <layer role="normal" texture="normal"/>
      <layer role="orm" texture="orm"/>
    </material>
  </asset>
  <worldbody><geom name="box" type="box" size="0.1 0.1 0.1" material="surface"/></worldbody>
</mujoco>
""".strip(),
        encoding="utf-8",
    )
    document = WorkspaceAdapter(MuJoCoAdapter(path))
    session = Session(document)
    assert session.submit(cmd.Pause())
    asset = next(item for item in session.model_assets(0) if item.name == "surface")
    assert dict(asset.texture_layers) == {
        "rgb": "rgb",
        "normal": "normal",
        "orm": "orm",
    }
    source = session.source
    assert source is not None
    material = source.materials[asset.runtime_index]
    assert material.metallic == pytest.approx(0.3)
    assert material.roughness == pytest.approx(0.7)

    result = session.submit(
        cmd.SetMaterial(
            asset.runtime_index,
            replace(
                material,
                rgba=np.array((0.2, 0.4, 0.6, 1.0), np.float32),
                specular=0.8,
            ),
        )
    )
    assert result.ok, result.message
    invalid = session.submit(cmd.SetModelMaterialLayers(0, "surface", (("normal", "missing"),)))
    assert not invalid.ok
    layers = (("rgb", "rgb"), ("roughness", "orm"), ("normal", "normal"))
    updated = session.submit(cmd.SetModelMaterialLayers(0, "surface", layers))
    assert updated.ok, updated.message
    assert dict(
        next(item for item in session.model_assets(0) if item.name == "surface").texture_layers
    ) == dict(layers)
    assert session.submit(cmd.Undo())
    assert "orm" in dict(
        next(item for item in session.model_assets(0) if item.name == "surface").texture_layers
    )
    assert session.submit(cmd.Redo())
    box = next(node for node in session.nodes if node.name == "box")
    copied = session.submit(cmd.AddModelMaterial(box.node_id, "surface_copy", asset.runtime_index))
    assert copied.ok, copied.message

    xml = document.scene_model_xml(0)
    assert xml is not None
    root = ET.fromstring(xml)
    materials = {element.attrib["name"]: element for element in root.findall("asset/material")}
    for name in ("surface", "surface_copy"):
        element = materials[name]
        assert float(element.attrib["metallic"]) == pytest.approx(0.3)
        assert float(element.attrib["roughness"]) == pytest.approx(0.7)
        assert {
            (layer.attrib["role"], layer.attrib["texture"]) for layer in element.findall("layer")
        } == {("rgb", "rgb"), ("normal", "normal"), ("roughness", "orm")}


def test_model_material_creation_binding_and_texture_import(tmp_path: Path, monkeypatch) -> None:
    from PIL import Image

    path = tmp_path / "material-assets.xml"
    path.write_text(
        """
<mujoco>
  <asset>
    <material name="paint" rgba="0.2 0.4 0.6 1" specular="0.3"/>
  </asset>
  <worldbody>
    <geom name="box" type="box" size="0.1 0.1 0.1"/>
  </worldbody>
</mujoco>
""".strip(),
        encoding="utf-8",
    )
    texture_path = tmp_path / "surface.png"
    Image.fromarray(np.full((4, 6, 3), (40, 120, 200), np.uint8)).save(texture_path)
    replacement_texture_path = tmp_path / "surface-replacement.png"
    Image.fromarray(np.full((3, 5, 3), (200, 80, 40), np.uint8)).save(replacement_texture_path)
    cube_path = tmp_path / "studio.png"
    Image.fromarray(np.full((8, 8, 3), (20, 80, 160), np.uint8)).save(cube_path)
    document = WorkspaceAdapter(MuJoCoAdapter(path))
    session = Session(document)
    assert session.submit(cmd.Pause())
    box = next(node for node in session.nodes if node.name == "box")
    paint = session.model_material_indices(0)[0]

    compile_count = 0
    compile_model = document.primary._compile_composed_model

    def counted_compile():
        nonlocal compile_count
        compile_count += 1
        return compile_model()

    monkeypatch.setattr(document.primary, "_compile_composed_model", counted_compile)
    jpeg_path = tmp_path / "surface.jpg"
    Image.fromarray(np.zeros((2, 2, 3), np.uint8)).save(jpeg_path)
    rejected = session.submit(cmd.ImportModelTexture(0, jpeg_path, "jpeg"))
    assert not rejected.ok
    assert "PNG" in rejected.message
    assert compile_count == 0

    created = session.submit(cmd.AddModelMaterial(box.node_id, "paint_copy", paint))
    assert created.ok, created.message
    assert compile_count == 1
    copied = created.entity_id
    assert copied in session.model_material_indices(0)
    model = document.primary.model
    geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "box")
    assert int(model.geom_matid[geom]) == copied
    assert model.mat_rgba[copied] == pytest.approx((0.2, 0.4, 0.6, 1.0))
    instance = next(
        index for index, node_id in enumerate(session.source.geom_node) if node_id == box.node_id
    )
    assert int(session.source.geom_material[instance]) == copied

    bound = session.submit(cmd.SetGeometryMaterial(box.node_id, paint))
    assert bound.ok, bound.message
    assert compile_count == 1
    assert int(document.primary.model.geom_matid[geom]) == paint
    assert int(session.source.geom_material[instance]) == paint

    imported = session.submit(
        cmd.ImportModelTexture(0, texture_path, "surface", material_index=copied)
    )
    assert imported.ok, imported.message
    assert compile_count == 2
    assert "surface" in session.model_texture_names(0)
    model = document.primary.model
    texture = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TEXTURE, "surface")
    assert texture >= 0
    assert int(model.mat_texid[copied, 1]) == texture
    assert session.source.textures["surface"].pixels.shape == (4, 6, 3)

    assert session.submit(cmd.Undo())
    assert "surface" not in session.model_texture_names(0)
    assert session.submit(cmd.Redo())
    assert "surface" in session.model_texture_names(0)

    before_replace_compile = compile_count
    replaced = session.submit(
        cmd.ReplaceModelAssetFile(0, "texture", "surface", replacement_texture_path)
    )
    assert replaced.ok, replaced.message
    assert compile_count == before_replace_compile + 1
    surface_asset = next(
        asset
        for asset in session.model_assets(0)
        if asset.type == "texture" and asset.name == "surface"
    )
    assert Path(surface_asset.file) == replacement_texture_path
    assert session.source.textures["surface"].pixels.shape == (3, 5, 3)
    assert session.submit(cmd.Undo())
    assert session.source.textures["surface"].pixels.shape == (4, 6, 3)
    assert session.submit(cmd.Redo())
    assert session.source.textures["surface"].pixels.shape == (3, 5, 3)

    cube = session.submit(cmd.ImportModelTexture(0, cube_path, "studio", texture_type="cube"))
    assert cube.ok, cube.message
    assert session.source.textures["studio"].type is TextureType.CUBE
    assert session.submit(cmd.SetSkybox("studio"))
    assert session.source.textures["studio"].type is TextureType.SKYBOX
    rejected_binding = session.submit(
        cmd.ImportModelTexture(0, cube_path, "invalid_cube", copied, "cube")
    )
    assert not rejected_binding.ok
    assert "2D" in rejected_binding.message

    before_copy_compile = compile_count
    textured_copy = session.submit(cmd.AddModelMaterial(box.node_id, "textured_copy", copied))
    assert textured_copy.ok, textured_copy.message
    assert compile_count == before_copy_compile + 1
    model = document.primary.model
    texture = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TEXTURE, "surface")
    assert int(model.mat_texid[textured_copy.entity_id, 1]) == texture

    workspace_path = tmp_path / "material-assets.forge.json"
    document.save_scene(workspace_path)
    restored = workspace()
    restored.open_scene(workspace_path)
    assert any(name.endswith("surface") for name in restored.model_texture_names(1))
    restored_model = restored.primary.model
    restored_copy = mujoco.mj_name2id(
        restored_model, mujoco.mjtObj.mjOBJ_MATERIAL, "forge_1_paint_copy"
    )
    restored_texture = mujoco.mj_name2id(
        restored_model, mujoco.mjtObj.mjOBJ_TEXTURE, "forge_1_surface"
    )
    assert int(restored_model.mat_texid[restored_copy, 1]) == restored_texture


def test_model_material_and_texture_choices_stay_model_local() -> None:
    document = workspace()
    first = document.add_scene_model(ASSETS / "test_scene.xml", np.zeros(3), np.eye(3))
    second = document.add_scene_model(
        ASSETS / "test_scene.xml", np.array((2.0, 0.0, 0.0)), np.eye(3)
    )
    session = Session(document)
    assert session.submit(cmd.Pause())

    first_materials = set(session.model_material_indices(first))
    second_materials = set(session.model_material_indices(second))
    assert first_materials
    assert second_materials
    assert first_materials.isdisjoint(second_materials)
    assert all(name.startswith(f"forge_{first}_") for name in session.model_texture_names(first))
    assert all(name.startswith(f"forge_{second}_") for name in session.model_texture_names(second))

    geometry = next(
        node for node in session.nodes if node.model_id == first and node.type is NodeType.GEOM
    )
    result = session.submit(cmd.SetGeometryMaterial(geometry.node_id, next(iter(second_materials))))
    assert not result.ok
    assert "unavailable" in result.message.lower()


def test_mjspec_element_pose_edits_round_trip_in_workspace(tmp_path: Path) -> None:
    document = workspace()
    model_id = document.add_scene_model(
        ASSETS / "test_scene.xml",
        np.array((2.0, 0.0, 0.0)),
        np.eye(3),
    )
    model_node = next(node for node in document.nodes() if node.type is NodeType.MODEL)
    document.add_model_element(model_node.node_id, "body", "fixture")
    body = next(node for node in document.nodes() if node.name == "forge_1_fixture")
    document.add_model_element(body.node_id, "geom:box", "fixture_visual")
    body = next(node for node in document.nodes() if node.name == "forge_1_fixture")
    geom = next(node for node in document.nodes() if node.name == "forge_1_fixture_visual")
    assert document.set_pose(body.node_id, np.array((3.0, 0.0, 0.0)), np.eye(3))
    assert document.set_pose(geom.node_id, np.array((3.0, 1.0, 0.0)), np.eye(3))

    path = tmp_path / "poses.forge.json"
    document.save_scene(path)
    restored = workspace()
    restored.open_scene(path)
    xml = restored.scene_model_xml(model_id)
    assert xml is not None
    spec = mujoco.MjSpec.from_string(xml)
    assert spec.body("fixture").pos == pytest.approx((1.0, 0.0, 0.0))
    assert spec.geom("fixture_visual").pos == pytest.approx((0.0, 1.0, 0.0))


def test_mjcf_camera_and_light_edits_persist_with_model(tmp_path: Path) -> None:
    document = workspace()
    document.add_scene_model(ASSETS / "test_scene.xml", np.zeros(3), np.eye(3))
    camera = document.camera_view(0)
    assert camera is not None
    translated = replace(
        camera,
        eye=camera.eye + np.array((0.5, 0.0, 0.0), np.float32),
        target=camera.target + np.array((0.5, 0.0, 0.0), np.float32),
        fov_y=np.deg2rad(52.0),
    )
    assert document.set_camera_view(0, translated)
    light = document.scene_source().lights.lights[0]
    assert document.set_light(0, replace(light, range=13.0))

    path = tmp_path / "camera-light.forge.json"
    document.save_scene(path)
    restored = workspace()
    restored.open_scene(path)
    restored_camera = restored.camera_view(0)
    assert restored_camera is not None
    assert restored_camera.eye == pytest.approx(translated.eye, abs=1e-5)
    assert np.degrees(restored_camera.fov_y) == pytest.approx(52.0)
    assert restored.scene_source().lights.lights[0].range == pytest.approx(13.0)


def test_model_root_transform_moves_free_bodies_and_model_cameras() -> None:
    document = workspace()
    model_id = document.add_scene_model(ASSETS / "mujoco_visuals.xml", np.zeros(3), np.eye(3))
    session = Session(document)
    assert session.submit(cmd.Pause()).ok
    model_node = next(
        node for node in session.nodes if node.type is NodeType.MODEL and node.model_id == model_id
    )
    before_cameras = {
        camera.name: session.camera_view(camera.camera_id) for camera in session.cameras
    }
    ball = next(node for node in document.nodes() if node.name.endswith("ball"))
    before_ball = document.primary.data.xpos[ball.body_index].copy()

    position = np.array((1.5, -0.75, 0.4))
    rotation = math3d.axis_angle_to_mat3((0.0, 0.0, 1.0), np.deg2rad(35.0))
    assert session.submit(cmd.SetPose(model_node.node_id, position, rotation)).ok

    after_ball = document.primary.data.xpos[ball.body_index]
    assert after_ball == pytest.approx(position + rotation @ before_ball)
    for camera in session.cameras:
        before = before_cameras[camera.name]
        after = session.camera_view(camera.camera_id)
        assert before is not None and after is not None
        assert after.eye == pytest.approx(position + rotation @ before.eye, abs=1e-5)
        assert after.target == pytest.approx(position + rotation @ before.target, abs=1e-5)
        assert after.up == pytest.approx(rotation @ before.up, abs=1e-5)


def test_model_source_supports_complete_mjcf_topology(tmp_path: Path) -> None:
    document = workspace()
    model_id = document.add_scene_model(ASSETS / "test_scene.xml", np.zeros(3), np.eye(3))
    source = """<mujoco model="edited">
  <worldbody>
    <body name="arm">
      <joint name="hinge" type="hinge"/>
      <geom name="link" type="capsule" size="0.03 0.2"/>
      <site name="start" pos="0 0 -0.1"/>
      <site name="end" pos="0 0 0.1"/>
    </body>
  </worldbody>
  <tendon><spatial name="cable"><site site="start"/><site site="end"/></spatial></tendon>
  <actuator><motor name="drive" joint="hinge"/></actuator>
  <sensor><jointpos name="angle" joint="hinge"/></sensor>
  <equality><joint name="lock" joint1="hinge"/></equality>
</mujoco>"""
    assert document.set_scene_model_xml(model_id, source)
    model = document.primary.model
    assert (model.ntendon, model.nu, model.nsensor, model.neq) == (1, 1, 1, 1)

    path = tmp_path / "full-topology.forge.json"
    document.save_scene(path)
    restored = workspace()
    restored.open_scene(path)
    model = restored.primary.model
    assert (model.ntendon, model.nu, model.nsensor, model.neq) == (1, 1, 1, 1)


def test_structured_model_components_edit_and_round_trip(tmp_path: Path) -> None:
    document = workspace()
    model_id = document.add_scene_model(ASSETS / "test_scene.xml", np.zeros(3), np.eye(3))
    source = """<mujoco model="components">
  <worldbody>
    <body name="arm">
      <joint name="hinge" type="hinge"/>
      <geom name="arm_geom" size="0.05"/>
      <site name="start" pos="0 0 -0.1"/>
      <site name="end" pos="0 0 0.1"/>
    </body>
  </worldbody>
</mujoco>"""
    assert document.set_scene_model_xml(model_id, source)

    expected = {
        "actuator": "motor",
        "sensor": "jointpos",
        "tendon": "spatial",
        "equality": "joint",
    }
    for category, subtype in expected.items():
        assert subtype in document.model_component_presets(model_id, category)
        assert document.add_model_component(model_id, category, subtype, category) == 0

    actuator = document.model_components(model_id, "actuator")[0]
    assert actuator.name == "actuator"
    assert {field.name: field.value for field in actuator.fields}["joint"] == "hinge"
    fields = tuple(
        (field.name, "-2 2" if field.name == "ctrlrange" else field.value)
        for field in actuator.fields
    )
    assert document.update_model_component(model_id, "actuator", 0, "drive", fields, ())

    tendon = document.model_components(model_id, "tendon")[0]
    assert tendon.subtype == "spatial"
    assert [item.type for item in tendon.path] == ["site", "site"]
    assert [item.type for item in tendon.path_presets] == ["site", "geom", "pulley"]
    site_preset = tendon.path_presets[0]
    assert site_preset.fields[0].choices == ("start", "end")
    assert document.update_model_component(
        model_id,
        "tendon",
        0,
        "cable",
        tuple((field.name, field.value) for field in tendon.fields),
        tuple(
            (
                item.type,
                tuple((field.name, field.value) for field in item.fields),
            )
            for item in tendon.path
        ),
    )
    assert document.remove_model_component(model_id, "sensor", 0)

    model = document.primary.model
    assert (model.ntendon, model.nu, model.nsensor, model.neq) == (1, 1, 0, 1)
    assert model.actuator_ctrlrange[0] == pytest.approx((-2.0, 2.0))
    path = tmp_path / "structured-components.forge.json"
    document.save_scene(path)
    restored = workspace()
    restored.open_scene(path)
    assert restored.model_components(model_id, "actuator")[0].name == "drive"
    assert restored.model_components(model_id, "tendon")[0].name == "cable"
    assert restored.model_components(model_id, "sensor") == ()


def test_deformable_flex_and_skin_components_are_structurally_editable() -> None:
    document = WorkspaceAdapter(MuJoCoAdapter(ASSETS / "deformables.xml"))
    session = Session(document)
    assert session.submit(cmd.Pause())

    components = session.model_components(0, "deformable")
    assert {item.subtype for item in components} == {"flex", "skin"}
    flex = next(item for item in components if item.subtype == "flex")
    assert {field.name: field for field in flex.fields}["body"].choices == ()
    skin = next(item for item in components if item.subtype == "skin")
    assert skin.name == "ribbon"
    assert [item.type for item in skin.path] == ["bone", "bone"]
    assert {field.name for field in skin.path[0].fields} == {
        "body",
        "bindpos",
        "bindquat",
        "vertid",
        "vertweight",
    }
    assert {"skin_root", "skin_tip"} <= set(
        {field.name: field for field in skin.path[0].fields}["body"].choices
    )
    fields = tuple(
        (field.name, "0.012" if field.name == "inflate" else field.value) for field in skin.fields
    )
    path = tuple(
        (
            item.type,
            tuple(
                (
                    field.name,
                    "0.56 0 0.85" if path_index == 0 and field.name == "bindpos" else field.value,
                )
                for field in item.fields
            ),
        )
        for path_index, item in enumerate(skin.path)
    )
    edited = session.submit(
        cmd.UpdateModelComponent(0, "deformable", skin.component_id, skin.name, fields, path)
    )
    assert edited.ok, edited.message
    assert float(document.primary.model.skin_inflate[0]) == pytest.approx(0.012)
    xml = document.scene_model_source(0)
    assert xml is not None and 'inflate="0.012"' in xml
    assert 'bindpos="0.56 0 0.85"' in xml

    assert session.submit(cmd.Undo())
    restored = next(
        item for item in session.model_components(0, "deformable") if item.subtype == "skin"
    )
    assert {field.name: field.value for field in restored.fields}["inflate"] == "0.004"
    assert session.submit(cmd.Redo())
    restored = next(
        item for item in session.model_components(0, "deformable") if item.subtype == "skin"
    )
    assert {field.name: field.value for field in restored.fields}["inflate"] == "0.012"
    assert session.submit(cmd.RemoveModelComponent(0, "deformable", restored.component_id))
    assert document.primary.model.nskin == 0
    assert session.submit(cmd.Undo())
    assert document.primary.model.nskin == 1


def test_custom_numeric_text_and_tuple_authoring_round_trips(tmp_path: Path) -> None:
    document = workspace()
    assert document.model_component_presets(0, "custom") == ("numeric", "text", "tuple")
    component_ids = {
        subtype: document.add_model_component(0, "custom", subtype, "shared")
        for subtype in ("numeric", "text", "tuple")
    }
    assert component_ids == {"numeric": 0, "text": 1, "tuple": 2}

    components = {item.subtype: item for item in document.model_components(0, "custom")}
    assert {field.name: field.value for field in components["numeric"].fields} == {
        "size": "1",
        "data": "0",
    }
    assert {field.name: field.value for field in components["text"].fields} == {"data": "text"}
    tuple_item = components["tuple"]
    assert [(field.name, field.value) for field in tuple_item.path[0].fields] == [
        ("objtype", "body"),
        ("objname", "world"),
        ("prm", ""),
    ]
    numeric_preset = next(
        preset
        for preset in tuple_item.path_presets
        if {field.name: field.value for field in preset.fields}["objtype"] == "numeric"
    )
    assert {field.name: field.choices for field in numeric_preset.fields}["objname"] == ("shared",)

    assert document.update_model_component(
        0,
        "custom",
        component_ids["numeric"],
        "shared",
        (("size", "3"), ("data", "1 2 3")),
        (),
    )
    assert document.update_model_component(
        0,
        "custom",
        component_ids["text"],
        "shared",
        (("data", "edited value"),),
        (),
    )
    assert document.update_model_component(
        0,
        "custom",
        component_ids["tuple"],
        "shared",
        (),
        (
            ("element", (("objtype", "body"), ("objname", "world"), ("prm", "1"))),
            ("element", (("objtype", "numeric"), ("objname", "shared"), ("prm", "2"))),
        ),
    )
    assert (
        document.primary.model.nnumeric,
        document.primary.model.ntext,
        document.primary.model.ntuple,
    ) == (
        1,
        1,
        1,
    )

    path = tmp_path / "custom-components.forge.json"
    document.save_scene(path)
    restored = workspace()
    restored.open_scene(path)
    restored_components = {item.subtype: item for item in restored.model_components(0, "custom")}
    assert {field.name: field.value for field in restored_components["numeric"].fields}[
        "data"
    ] == "1 2 3"
    assert {field.name: field.value for field in restored_components["text"].fields}[
        "data"
    ] == "edited value"
    assert len(restored_components["tuple"].path) == 2


def test_contact_pair_and_exclude_use_structured_reference_fields(tmp_path: Path) -> None:
    document = workspace()
    model_id = document.add_scene_model(ASSETS / "test_scene.xml", np.zeros(3), np.eye(3))
    source = """<mujoco model="contacts">
  <worldbody>
    <body name="first"><geom name="first_geom" type="sphere" size="0.1"/></body>
    <body name="second"><geom name="second_geom" type="box" size="0.1 0.1 0.1"/></body>
  </worldbody>
</mujoco>"""
    assert document.set_scene_model_xml(model_id, source)
    assert document.model_component_presets(model_id, "contact") == ("pair", "exclude")

    assert document.add_model_component(model_id, "contact", "pair", "explicit_pair") == 0
    assert document.add_model_component(model_id, "contact", "exclude", "body_exclusion") == 1
    pair, exclude = document.model_components(model_id, "contact")
    pair_fields = {field.name: field for field in pair.fields}
    assert pair.subtype == "pair"
    assert pair_fields["geom1"].value == "first_geom"
    assert pair_fields["geom2"].value == "second_geom"
    assert pair_fields["geom1"].choices == ("", "first_geom", "second_geom")
    assert "body1" not in pair_fields
    exclude_fields = {field.name: field for field in exclude.fields}
    assert exclude.subtype == "exclude"
    assert exclude_fields["body1"].choices == ("", "first", "second")
    assert "geom1" not in exclude_fields

    edited_fields = tuple(
        (
            field.name,
            {
                "condim": "4",
                "friction": "1 0.1 0.01 0.2 0.3",
                "margin": "0.01",
                "gap": "0.002",
            }.get(field.name, field.value),
        )
        for field in pair.fields
    )
    assert document.update_model_component(
        model_id, "contact", pair.component_id, pair.name, edited_fields, ()
    )
    assert (document.primary.model.npair, document.primary.model.nexclude) == (1, 1)
    assert int(document.primary.model.pair_dim[0]) == 4
    assert document.primary.model.pair_friction[0] == pytest.approx((1.0, 0.1, 0.01, 0.2, 0.3))

    path = tmp_path / "contacts.forge.json"
    document.save_scene(path)
    restored = workspace()
    restored.open_scene(path)
    assert [item.subtype for item in restored.model_components(model_id, "contact")] == [
        "pair",
        "exclude",
    ]
    assert (restored.primary.model.npair, restored.primary.model.nexclude) == (1, 1)


def test_non_plugin_component_presets_compile_and_energy_sensors_round_trip() -> None:
    adapter = MuJoCoAdapter()
    adapter.new_scene()
    assert adapter.set_scene_model_xml(
        0,
        """<mujoco model="component-catalog">
  <worldbody>
    <body name="hinge_body">
      <joint name="hinge" type="hinge" limited="true" range="-1 1"/>
      <geom name="first_geom" size="0.1"/>
      <site name="first_site"/>
      <camera name="inspection"/>
    </body>
    <body name="ball_body">
      <joint name="ball" type="ball"/>
      <geom name="second_geom" type="box" size="0.1 0.1 0.1"/>
      <site name="second_site"/>
    </body>
  </worldbody>
</mujoco>""",
    )

    actuator_presets = adapter.model_component_presets(0, "actuator")
    assert set(actuator_presets) == {
        "general",
        "motor",
        "position",
        "velocity",
        "intvelocity",
        "orientation",
        "damper",
        "cylinder",
        "muscle",
        "adhesion",
        "dcmotor",
    }
    for index, subtype in enumerate(actuator_presets):
        assert adapter.add_model_component(0, "actuator", subtype, f"actuator{index}") == index
    actuator = adapter.model_components(0, "actuator")[0]
    actuator_fields = {field.name: field for field in actuator.fields}
    assert {"class", "user", "ctrllimited", "forcelimited", "cranksite", "slidersite"} <= set(
        actuator_fields
    )
    assert actuator_fields["joint"].choices == ("", "hinge", "ball")
    assert actuator_fields["class"].choices == ()

    for index, subtype in enumerate(adapter.model_component_presets(0, "tendon")):
        assert adapter.add_model_component(0, "tendon", subtype, f"tendon{index}") == index
    fixed = adapter.model_components(0, "tendon")[0]
    fixed_fields = tuple(
        (
            field.name,
            {"limited": "true", "range": "-1 1"}.get(field.name, field.value),
        )
        for field in fixed.fields
    )
    assert adapter.update_model_component(
        0,
        "tendon",
        0,
        fixed.name,
        fixed_fields,
        tuple(
            (
                item.type,
                tuple((field.name, field.value) for field in item.fields),
            )
            for item in fixed.path
        ),
    )

    sensor_presets = adapter.model_component_presets(0, "sensor")
    assert {
        "touch",
        "camprojection",
        "jointactuatorfrc",
        "ballquat",
        "jointlimitfrc",
        "tendonlimitfrc",
        "actuatorfrc",
        "frameangacc",
        "subtreeangmom",
        "insidesite",
        "distance",
        "normal",
        "fromto",
        "contact",
        "e_potential",
        "e_kinetic",
        "clock",
        "user",
    }.issubset(sensor_presets)
    assert "plugin" not in sensor_presets
    for index, subtype in enumerate(sensor_presets):
        assert adapter.add_model_component(0, "sensor", subtype, f"sensor{index}") == index
    assert "user" in {field.name for field in adapter.model_components(0, "sensor")[0].fields}

    equality_presets = adapter.model_component_presets(0, "equality")
    assert equality_presets == ("joint", "weld", "connect", "tendon")
    for index, subtype in enumerate(equality_presets):
        assert adapter.add_model_component(0, "equality", subtype, f"equality{index}") == index

    assert len(adapter.model_components(0, "actuator")) == len(actuator_presets)
    assert len(adapter.model_components(0, "sensor")) == len(sensor_presets)
    assert {item.subtype for item in adapter.model_components(0, "sensor")} >= {
        "e_potential",
        "e_kinetic",
        "clock",
        "user",
    }
    assert len(adapter.model_components(0, "equality")) == len(equality_presets)


def test_invalid_structured_component_edit_keeps_last_good_model() -> None:
    document = workspace()
    model_id = document.add_scene_model(ASSETS / "actuator_visuals.xml", np.zeros(3), np.eye(3))
    before = document.primary.model
    actuator = document.model_components(model_id, "actuator")[0]
    fields = tuple(
        (field.name, "missing_joint" if field.name == "body" else field.value)
        for field in actuator.fields
    )

    with pytest.raises(ValueError):
        document.update_model_component(model_id, "actuator", 0, actuator.name, fields, ())

    assert document.primary.model is before
    assert document.model_components(model_id, "actuator")[0] == actuator


def test_structured_component_commands_participate_in_undo_redo() -> None:
    document = workspace()
    model_id = document.add_scene_model(ASSETS / "actuator_visuals.xml", np.zeros(3), np.eye(3))
    session = Session(document)
    assert session.submit(cmd.Pause())
    added = session.submit(cmd.AddModelComponent(model_id, "sensor", "jointpos", "angle"))
    assert added
    assert session.can_undo
    assert [
        (item.component_id, item.name) for item in session.model_components(model_id, "sensor")
    ] == [(added.entity_id, "angle")]

    assert session.submit(cmd.Undo())
    assert session.model_components(model_id, "sensor") == ()
    assert session.submit(cmd.Redo())
    assert [
        (item.component_id, item.name) for item in session.model_components(model_id, "sensor")
    ] == [(added.entity_id, "angle")]


def test_model_component_ids_do_not_shift_or_target_a_replacement() -> None:
    document = workspace()
    assert document.set_scene_model_xml(
        0,
        """<mujoco model="component-identity">
  <worldbody><body><joint name="hinge"/><geom size="0.1"/></body></worldbody>
</mujoco>""",
    )
    first_id = document.add_model_component(0, "sensor", "jointpos", "first")
    second_id = document.add_model_component(0, "sensor", "jointpos", "second")
    assert first_id >= 0 and second_id > first_id

    assert document.remove_model_component(0, "sensor", first_id)
    remaining = document.model_components(0, "sensor")
    assert [(item.component_id, item.name) for item in remaining] == [(second_id, "second")]

    assert not document.remove_model_component(0, "sensor", first_id)
    assert [(item.component_id, item.name) for item in document.model_components(0, "sensor")] == [
        (second_id, "second")
    ]

    fields = tuple((field.name, field.value) for field in remaining[0].fields)
    assert document.update_model_component(0, "sensor", second_id, "renamed", fields, ())
    renamed = document.model_components(0, "sensor")
    assert [(item.component_id, item.name) for item in renamed] == [(second_id, "renamed")]

    assert document.set_scene_model_xml(
        0,
        """<mujoco model="replacement">
  <worldbody><body><joint name="hinge"/><geom size="0.1"/></body></worldbody>
  <sensor><jointpos name="replacement" joint="hinge"/></sensor>
</mujoco>""",
    )
    replacement = document.model_components(0, "sensor")
    assert len(replacement) == 1 and replacement[0].component_id != second_id
    assert not document.remove_model_component(0, "sensor", second_id)
    assert document.model_components(0, "sensor") == replacement


def test_model_keyframe_authoring_captures_edits_loads_and_undoes(tmp_path: Path) -> None:
    model_path = tmp_path / "keyframe-authoring.xml"
    model_path.write_text(
        """<mujoco model="keyframe-authoring">
  <worldbody>
    <body mocap="true"><geom size="0.05"/></body>
    <body name="arm">
      <joint name="hinge" type="hinge"/>
      <geom size="0.1"/>
    </body>
    <body><joint type="slide"/><geom size="0.05"/></body>
  </worldbody>
  <actuator><position joint="hinge"/></actuator>
</mujoco>
""",
        encoding="utf-8",
    )
    document = workspace()
    model_position = np.array((4.0, -3.0, 2.0))
    model_rotation = math3d.axis_angle_to_mat3((0.0, 0.0, 1.0), np.pi * 0.5)
    model_id = document.add_scene_model(model_path, model_position, model_rotation)
    session = Session(document)
    assert session.submit(cmd.Pause())

    joint = mujoco.mj_name2id(document.primary.model, mujoco.mjtObj.mjOBJ_JOINT, "forge_1_hinge")
    unnamed_joint = next(index for index in range(document.primary.model.njnt) if index != joint)
    actuator = 0
    body = next(
        index
        for index in range(document.primary.model.nbody)
        if document.primary.model.body_mocapid[index] >= 0
    )
    mocap = int(document.primary.model.body_mocapid[body])
    document.primary.data.qpos[document.primary.model.jnt_qposadr[joint]] = 0.4
    document.primary.data.qvel[document.primary.model.jnt_dofadr[joint]] = -0.2
    document.primary.data.qpos[document.primary.model.jnt_qposadr[unnamed_joint]] = 0.8
    document.primary.data.qvel[document.primary.model.jnt_dofadr[unnamed_joint]] = 0.3
    document.primary.data.ctrl[document.primary.model.actuator_ctrladr[actuator]] = 0.6
    local_mocap_position = np.array((1.0, 2.0, 3.0))
    document.primary.data.mocap_pos[mocap] = (
        local_mocap_position @ model_rotation.T + model_position
    )
    document.primary.data.mocap_quat[mocap] = (1.0, 0.0, 0.0, 0.0)

    added = session.submit(cmd.AddModelKeyframe(model_id, "captured"))
    assert added.ok
    keyframe_id = added.entity_id
    properties = session.keyframe_properties(keyframe_id)
    assert properties is not None
    assert (properties.model_id, properties.name) == (model_id, "captured")
    assert properties.qpos == pytest.approx((0.4, 0.8))
    assert properties.qvel == pytest.approx((-0.2, 0.3))
    assert properties.ctrl == pytest.approx((0.6,))
    assert properties.mocap_position == pytest.approx(local_mocap_position)

    updated = replace(
        properties,
        name="edited",
        time=1.25,
        qpos=(0.25, -0.1),
        qvel=(0.5, 0.2),
        ctrl=(-0.75,),
        mocap_position=(3.0, 2.0, 1.0),
    )
    result = session.submit(cmd.SetModelKeyframe(**updated.__dict__))
    assert result.ok
    assert session.keyframes[keyframe_id].name == "edited"
    assert session.submit(cmd.LoadKeyframe(keyframe_id))
    assert document.primary.data.qpos[document.primary.model.jnt_qposadr[joint]] == pytest.approx(
        0.25
    )
    assert document.primary.data.qvel[document.primary.model.jnt_dofadr[joint]] == pytest.approx(
        0.5
    )
    assert document.primary.data.qpos[
        document.primary.model.jnt_qposadr[unnamed_joint]
    ] == pytest.approx(-0.1)
    assert document.primary.data.qvel[
        document.primary.model.jnt_dofadr[unnamed_joint]
    ] == pytest.approx(0.2)
    assert document.primary.data.ctrl[
        document.primary.model.actuator_ctrladr[actuator]
    ] == pytest.approx(-0.75)
    assert document.primary.data.mocap_pos[mocap] == pytest.approx(
        np.array((3.0, 2.0, 1.0)) @ model_rotation.T + model_position
    )

    assert session.submit(cmd.RemoveModelKeyframe(keyframe_id))
    assert session.keyframes == []
    assert session.submit(cmd.Undo())
    assert [key.name for key in session.keyframes] == ["edited"]
    assert session.submit(cmd.Redo())
    assert session.keyframes == []


def test_schema_model_properties_batch_globals_defaults_and_assets_once() -> None:
    document = workspace()
    adapter = document.primary
    assert document.set_scene_model_xml(
        0,
        """<mujoco model="schema-properties">
  <compiler angle="degree"/>
  <option timestep="0.01"><flag energy="enable"/></option>
  <visual><global fovy="50"/></visual>
  <default><default class="soft"><geom friction="0.5 0.1 0.01"/><joint damping="1"/></default></default>
  <asset>
    <texture name="checker" type="2d" builtin="checker" width="4" height="4"/>
    <material name="paint" metallic="0.2" roughness="0.5"/>
    <mesh name="tetra" vertex="0 0 0  1 0 0  0 1 0  0 0 1"
          face="0 2 1  0 1 3  0 3 2  1 2 3"/>
    <hfield name="terrain" nrow="2" ncol="2" size="1 1 0.2 0.1"
            elevation="0 0 0 1"/>
  </asset>
  <worldbody><body><joint/><geom size="0.1"/></body></worldbody>
</mujoco>""",
    )
    session = Session(document)
    assert session.submit(cmd.Pause())
    groups = {group.group_id: group for group in session.model_property_groups(0)}
    assert {
        "global:model",
        "global:compiler",
        "global:option",
        "global:option/flag",
    } <= groups.keys()
    assert [(field.name, field.value) for field in groups["global:model"].fields] == [
        ("model", "schema-properties")
    ]
    assert {field.name for field in groups["global:option"].fields} >= {
        "timestep",
        "gravity",
        "solver",
    }
    assert {field.name for field in groups["asset:mesh:0"].fields} >= {
        "scale",
        "inertia",
        "smoothnormal",
    }
    assert "vertex" not in {field.name for field in groups["asset:mesh:0"].fields}
    assert {field.name for field in groups["asset:material:0"].fields} >= {
        "metallic",
        "roughness",
        "texuniform",
    }
    texture_fields = {field.name: field for field in groups["asset:texture:0"].fields}
    assert texture_fields["type"].choices == ("", "2d", "cube", "skybox")
    assert texture_fields["colorspace"].choices == ("", "auto", "srgb", "linear")
    soft_geom = next(
        group
        for group in groups.values()
        if group.category == "default" and group.label == "Default soft / geom"
    )

    replace_count = 0
    replace_model_spec = adapter._replace_model_spec

    def counted_replace(model_id, spec):
        nonlocal replace_count
        replace_count += 1
        return replace_model_spec(model_id, spec)

    adapter._replace_model_spec = counted_replace
    updates = (
        ("global:model", (("model", "edited-properties"),)),
        ("global:option", (("timestep", "0.005"), ("gravity", "0 0 -1"))),
        ("global:size", (("nuserdata", "8"),)),
        ("global:visual/global", (("orthographic", "true"), ("fovy", "35"))),
        (soft_geom.group_id, (("friction", "0.8 0.2 0.02"), ("rgba", "0.2 0.3 0.4 1"))),
        ("asset:mesh:0", (("scale", "2 3 4"), ("smoothnormal", "true"))),
        ("asset:hfield:0", (("size", "2 3 0.4 0.2"),)),
        ("asset:material:0", (("metallic", "0.4"), ("roughness", "0.8"))),
        ("asset:texture:0", (("rgb1", "0.2 0.3 0.4"), ("colorspace", "linear"))),
    )
    result = session.submit(cmd.SetModelPropertyGroups(0, updates))
    assert result.ok
    assert replace_count == 1
    assert adapter.model.opt.timestep == pytest.approx(0.005)
    assert adapter.model.opt.gravity == pytest.approx((0.0, 0.0, -1.0))
    edited = {group.group_id: group for group in session.model_property_groups(0)}
    assert edited["global:model"].fields[0].value == "edited-properties"
    assert {field.name: field.value for field in edited["global:visual/global"].fields}[
        "orthographic"
    ] == "true"
    assert {field.name: field.value for field in edited["asset:mesh:0"].fields}["scale"] == "2 3 4"
    assert {field.name: field.value for field in edited["asset:material:0"].fields}[
        "roughness"
    ] == "0.8"
    assert {field.name: field.value for field in edited["asset:texture:0"].fields}[
        "colorspace"
    ] == "linear"
    assert {field.name: field.value for field in edited[soft_geom.group_id].fields}[
        "friction"
    ] == "0.8 0.2 0.02"

    assert session.submit(cmd.Undo())
    assert adapter.model.opt.timestep == pytest.approx(0.01)
    assert session.submit(cmd.Redo())
    assert adapter.model.opt.timestep == pytest.approx(0.005)


def test_default_class_add_nested_edit_remove_and_undo() -> None:
    document = workspace()
    session = Session(document)
    assert session.submit(cmd.Pause())

    base = session.submit(cmd.AddModelDefault(0, -1, "base"))
    assert base.ok
    child = session.submit(cmd.AddModelDefault(0, base.entity_id, "child"))
    assert child.ok
    groups = {group.label: group for group in session.model_property_groups(0)}
    assert "Default base / class" in groups
    child_geom = groups["Default child / geom"]
    assert session.submit(
        cmd.SetModelPropertyGroups(
            0,
            ((child_geom.group_id, (("friction", "0.7 0.2 0.03"),)),),
        )
    )
    edited = {group.label: group for group in session.model_property_groups(0)}
    assert {field.name: field.value for field in edited["Default child / geom"].fields}[
        "friction"
    ] == "0.7 0.2 0.03"

    assert not session.submit(cmd.RemoveModelDefault(0, 0))
    assert session.submit(cmd.RemoveModelDefault(0, base.entity_id))
    assert all("Default base" not in group.label for group in session.model_property_groups(0))
    assert session.submit(cmd.Undo())
    labels = {group.label for group in session.model_property_groups(0)}
    assert {"Default base / class", "Default child / class"} <= labels


def test_noop_model_edits_skip_recompilation() -> None:
    document = workspace()
    model_id = document.add_scene_model(ASSETS / "actuator_visuals.xml", np.zeros(3), np.eye(3))
    compiled = document.primary.model
    assert document.set_scene_model_transform(model_id, np.zeros(3), np.eye(3))
    assert document.primary.model is compiled

    component = document.model_components(model_id, "actuator")[0]
    assert document.update_model_component(
        model_id,
        component.category,
        component.component_id,
        component.name,
        tuple((field.name, field.value) for field in component.fields),
        (),
    )
    assert document.primary.model is compiled


def test_model_transform_preview_moves_frames_without_recompiling(monkeypatch) -> None:
    document = workspace()
    model_id = document.add_scene_model(
        ASSETS / "actuator_visuals.xml", np.array((1.0, 0.0, 0.0)), np.eye(3)
    )
    session = Session(document)
    assert session.submit(cmd.Pause())
    node = next(
        item for item in session.nodes if item.model_id == model_id and item.body_index >= 0
    )
    before_frame = session.tick(FrameNeeds(poses=True), wall_dt=0.0)
    before_position = before_frame.body_xpos[node.body_index].copy()
    before_model = document.primary.model
    before_revision = document.structure_revision
    compile_count = 0
    compile_model = document.primary._compile_composed_model

    def counted_compile():
        nonlocal compile_count
        compile_count += 1
        return compile_model()

    monkeypatch.setattr(document.primary, "_compile_composed_model", counted_compile)

    for x in (1.25, 1.5, 2.0):
        assert session.submit(
            cmd.PreviewSceneModelTransform(model_id, np.array((x, 0.0, 0.0)), np.eye(3))
        )
    preview_frame = session.tick(FrameNeeds(poses=True), wall_dt=0.0)

    assert compile_count == 0
    assert document.primary.model is before_model
    assert document.structure_revision == before_revision
    assert not session.dirty
    assert not session.can_undo
    assert next(item for item in session.scene_models if item.model_id == model_id).position == (
        2.0,
        0.0,
        0.0,
    )
    assert preview_frame.body_xpos[node.body_index] - before_position == pytest.approx(
        (1.0, 0.0, 0.0)
    )

    assert session.submit(cmd.ClearSceneModelTransformPreview(model_id))
    cleared_frame = session.tick(FrameNeeds(poses=True), wall_dt=0.0)
    assert cleared_frame.body_xpos[node.body_index] == pytest.approx(before_position)


def test_model_placement_stays_preview_only_until_explicit_apply(monkeypatch) -> None:
    from forge_viewer.gizmo import GizmoHandle
    from forge_viewer.ui.gizmo import ObjectGizmo

    document = workspace()
    model_id = document.add_scene_model(
        ASSETS / "actuator_visuals.xml", np.array((1.0, 0.0, 0.0)), np.eye(3)
    )
    session = Session(document)
    assert session.submit(cmd.Pause())
    node = next(
        item for item in session.nodes if item.type is NodeType.MODEL and item.model_id == model_id
    )
    assert session.submit(cmd.Select(node.object_id))
    compile_count = 0
    compile_model = document.primary._compile_composed_model

    def counted_compile():
        nonlocal compile_count
        compile_count += 1
        return compile_model()

    monkeypatch.setattr(document.primary, "_compile_composed_model", counted_compile)
    gizmo = ObjectGizmo()
    locked = gizmo.evaluate(session, node)
    assert not locked.ok
    assert "Edit Placement" in locked.reason
    assert gizmo.begin_model_placement(session, model_id)
    assert gizmo.evaluate(session, node).ok
    gizmo._active = GizmoHandle.X
    gizmo._start_edit(session)

    for x in (1.25, 1.5, 2.0):
        result, _ = gizmo._submit_transform(session, node, np.array((x, 0.0, 0.0)), np.eye(3))
        assert result.ok
    gizmo._edit_started = True
    assert compile_count == 0
    assert not session.editing

    gizmo._end(commit=True)

    assert compile_count == 0
    assert not session.can_undo
    assert gizmo.model_placement_active(session, model_id)
    assert next(item for item in session.scene_models if item.model_id == model_id).position == (
        2.0,
        0.0,
        0.0,
    )

    assert gizmo.apply_model_placement(session)

    assert compile_count == 1
    assert not session.editing
    assert session.can_undo
    assert not gizmo.model_placement_active(session, model_id)
    assert next(item for item in session.scene_models if item.model_id == model_id).position == (
        2.0,
        0.0,
        0.0,
    )
    assert session.submit(cmd.Undo())
    assert next(item for item in session.scene_models if item.model_id == model_id).position == (
        1.0,
        0.0,
        0.0,
    )


def test_cancelled_model_placement_discards_preview_without_recompiling(monkeypatch) -> None:
    from forge_viewer.gizmo import GizmoHandle
    from forge_viewer.ui.gizmo import ObjectGizmo

    document = workspace()
    model_id = document.add_scene_model(
        ASSETS / "actuator_visuals.xml", np.array((1.0, 0.0, 0.0)), np.eye(3)
    )
    session = Session(document)
    assert session.submit(cmd.Pause())
    node = next(
        item for item in session.nodes if item.type is NodeType.MODEL and item.model_id == model_id
    )
    assert session.submit(cmd.Select(node.object_id))
    compile_count = 0
    compile_model = document.primary._compile_composed_model

    def counted_compile():
        nonlocal compile_count
        compile_count += 1
        return compile_model()

    monkeypatch.setattr(document.primary, "_compile_composed_model", counted_compile)
    gizmo = ObjectGizmo()
    assert gizmo.begin_model_placement(session, model_id)
    gizmo._active = GizmoHandle.X
    gizmo._start_edit(session)
    result, _ = gizmo._submit_transform(session, node, np.array((2.0, 0.0, 0.0)), np.eye(3))
    assert result.ok
    gizmo._edit_started = True

    gizmo._end(commit=True)
    assert gizmo.model_placement_active(session, model_id)
    assert next(item for item in session.scene_models if item.model_id == model_id).position == (
        2.0,
        0.0,
        0.0,
    )

    assert gizmo.cancel_model_placement(session)

    assert compile_count == 0
    assert not session.editing
    assert not session.can_undo
    assert next(item for item in session.scene_models if item.model_id == model_id).position == (
        1.0,
        0.0,
        0.0,
    )


def test_unchanged_model_placement_exits_without_recompiling(monkeypatch) -> None:
    from forge_viewer.ui.gizmo import ObjectGizmo

    document = workspace()
    model_id = document.add_scene_model(
        ASSETS / "actuator_visuals.xml", np.array((1.0, 0.0, 0.0)), np.eye(3)
    )
    session = Session(document)
    assert session.submit(cmd.Pause())
    compile_count = 0
    compile_model = document.primary._compile_composed_model

    def counted_compile():
        nonlocal compile_count
        compile_count += 1
        return compile_model()

    monkeypatch.setattr(document.primary, "_compile_composed_model", counted_compile)
    gizmo = ObjectGizmo()

    assert gizmo.begin_model_placement(session, model_id)
    assert gizmo.apply_model_placement(session)

    assert compile_count == 0
    assert not session.can_undo
    assert not gizmo.model_placement_active(session, model_id)
