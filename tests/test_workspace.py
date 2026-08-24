from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from forge_viewer.adapters.base import NodeKind
from forge_viewer.adapters.mujoco_adapter import MuJoCoAdapter
from forge_viewer.adapters.workspace import WorkspaceAdapter
from forge_viewer.types import DEFAULT_MATERIAL, CameraView, Light, LightKind, MeshShape
from forge_viewer.workspace_io import missing_resources

mujoco = pytest.importorskip("mujoco")

ASSETS = Path(__file__).parents[1] / "assets"


def workspace() -> WorkspaceAdapter:
    primary = MuJoCoAdapter()
    primary.new_scene()
    return WorkspaceAdapter(primary)


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
        Light(kind=LightKind.SPOT, position=np.array((1.0, -2.0, 3.0), np.float32)),
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


def test_mjspec_topology_edits_round_trip_in_workspace(tmp_path: Path) -> None:
    document = workspace()
    model_id = document.add_scene_model(ASSETS / "test_scene.xml", np.zeros(3), np.eye(3))
    model_node = next(
        node
        for node in document.nodes()
        if node.kind is NodeKind.MODEL and node.model_id == model_id
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
    model_node = next(node for node in document.nodes() if node.kind is NodeKind.MODEL)
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
