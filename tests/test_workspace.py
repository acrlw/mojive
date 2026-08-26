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
from forge_viewer.types import DEFAULT_MATERIAL, CameraView, Light, LightType, MeshShape
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


def test_workspace_mjcf_export_rejects_lossy_authored_lights(tmp_path: Path) -> None:
    document = workspace()
    document.add_scene_light("panel", Light(type=LightType.AREA))

    with pytest.raises(RuntimeError, match="area light"):
        document.save_scene(tmp_path / "scene.xml")


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
      <geom size="0.05"/>
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
    assert session.submit(cmd.AddModelComponent(model_id, "sensor", "jointpos", "angle"))
    assert session.can_undo
    assert [item.name for item in session.model_components(model_id, "sensor")] == ["angle"]

    assert session.submit(cmd.Undo())
    assert session.model_components(model_id, "sensor") == ()
    assert session.submit(cmd.Redo())
    assert [item.name for item in session.model_components(model_id, "sensor")] == ["angle"]


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
