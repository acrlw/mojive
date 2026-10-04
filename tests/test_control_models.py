"""Model authoring is discoverable and round-trips through the control contract."""

import pytest

from mojive.adapters.mujoco import MuJoCoAdapter
from mojive.control.operations import OPERATIONS
from mojive.control.rpc import ControlService, RpcError
from mojive.control.schema import Validator

pytestmark = pytest.mark.physics


@pytest.fixture
def model_service(tmp_path):
    path = tmp_path / "editable.xml"
    path.write_text("""<mujoco><asset><material name="paint" rgba="0.2 0.3 0.4 1"/></asset>
    <worldbody><body name="arm"><joint name="hinge" type="hinge" axis="0 0 1"/>
    <geom name="shape" type="box" size="0.1 0.2 0.3" material="paint"/></body></worldbody>
    <keyframe><key name="rest"/></keyframe></mujoco>""")
    service = ControlService(MuJoCoAdapter(path), path)
    yield service
    service.close()


def invoke(service, method, **params):
    result = service.dispatch(method, params)
    Validator(OPERATIONS[method].output_schema).validate(result)
    return result


def test_discover_model_joint_edit_source_and_undo(model_service):
    service = model_service
    models = invoke(service, "list_models")
    model_id = models["models"][0]["model_id"]
    source = invoke(service, "mujoco.get_model_source", model_id=model_id)
    assert '<joint name="hinge"' in source["mjcf"]
    description = invoke(service, "describe_operations", name="set_joint_properties")["operations"][
        0
    ]
    assert description["available"] and description["transactional"]
    joints = invoke(service, "list_joints", model_id=model_id)
    joint = joints["joints"][0]
    node = invoke(service, "inspect_object", node_id=joint["node_id"])
    assert node["joint_index"] == joint["joint_id"] and node["model_id"] == model_id
    assert node["source_name"] == "hinge"
    values = {
        name: joint[name]
        for name in ("joint_id", "axis", "limited", "range", "damping", "stiffness")
    }
    invoke(
        service,
        "set_joint_properties",
        **{**values, "damping": 0.75},
        expected_document=joints["document"],
    )
    changed = invoke(service, "get_joint_properties", joint_id=joint["joint_id"])
    assert changed["joint"]["damping"] == pytest.approx(0.75)
    assert 'damping="0.75"' in invoke(service, "mujoco.get_model_source", model_id=model_id)["mjcf"]
    invoke(service, "undo", expected_document=changed["document"])
    assert (
        invoke(service, "get_joint_properties", joint_id=joint["joint_id"])["joint"]["damping"]
        == values["damping"]
    )


def test_topology_edit_readback_and_reject_previous_node_epoch(model_service):
    service = model_service
    scene = invoke(service, "get_scene")
    body = next(node for node in scene["objects"] if node["name"] == "arm")
    created = invoke(
        service,
        "add_model_element",
        parent_node_id=body["node_id"],
        element_type="site",
        name="tip",
        expected_document=scene["document"],
    )
    assert invoke(service, "inspect_object", node_id=created["node_id"])["name"] == "tip"
    with pytest.raises(RpcError) as error:
        invoke(
            service,
            "rename_model_element",
            node_id=body["node_id"],
            name="wrong",
            expected_document=scene["document"],
        )
    assert error.value.code == "stale_document"
    renamed = invoke(
        service,
        "rename_model_element",
        node_id=created["node_id"],
        name="tip2",
        expected_document=created["document"],
    )
    node = next(node for node in invoke(service, "get_scene")["objects"] if node["name"] == "tip2")
    invoke(
        service,
        "remove_model_element",
        node_id=node["node_id"],
        expected_document=renamed["document"],
    )
    assert all(node["name"] != "tip2" for node in invoke(service, "get_scene")["objects"])


def test_asset_and_keyframe_edit_readback_recovery(model_service):
    service = model_service
    model_id = invoke(service, "list_models")["models"][0]["model_id"]
    before = invoke(service, "list_model_assets", model_id=model_id)
    assert any(asset["name"] == "paint" for asset in before["assets"])
    invoke(
        service,
        "duplicate_model_asset",
        model_id=model_id,
        asset_type="material",
        name="paint",
        new_name="copy",
        expected_document=before["document"],
    )
    assets = invoke(service, "list_model_assets", model_id=model_id)
    assert {asset["name"] for asset in assets["assets"]} == {"paint", "copy"}
    invoke(
        service,
        "remove_model_asset",
        model_id=model_id,
        asset_type="material",
        name="copy",
        expected_document=assets["document"],
    )
    created = invoke(service, "add_model_keyframe", model_id=model_id, name="pose")
    key = invoke(service, "get_model_keyframe", keyframe_id=created["keyframe_id"])
    invoke(
        service,
        "set_model_keyframe",
        **{**key["keyframe"], "time": 1.5, "qpos": [0.2]},
        expected_document=key["document"],
    )
    changed = invoke(service, "get_model_keyframe", keyframe_id=created["keyframe_id"])
    assert changed["keyframe"]["time"] == 1.5 and changed["keyframe"]["qpos"] == [0.2]
    listed = invoke(service, "list_model_keyframes", model_id=model_id)
    assert {key["name"] for key in listed["keyframes"]} == {"rest", "pose"}
    invoke(
        service,
        "remove_model_keyframe",
        keyframe_id=created["keyframe_id"],
        expected_document=listed["document"],
    )
    invoke(service, "undo")
    assert invoke(service, "get_model_keyframe", keyframe_id=created["keyframe_id"])["keyframe"][
        "qpos"
    ] == [0.2]


def test_model_queries_reject_unknown_id_and_edits_reject_running(model_service):
    for name in (
        "mujoco.get_model_source",
        "list_joints",
        "list_model_assets",
        "list_model_keyframes",
    ):
        with pytest.raises(RpcError) as error:
            invoke(model_service, name, model_id=999)
        assert error.value.code == "not_found"
    invoke(model_service, "resume")
    with pytest.raises(RpcError, match="Pause"):
        invoke(model_service, "add_model_keyframe", model_id=0, name="bad")
    invoke(model_service, "pause")


def test_joint_batch_failure_restores_source_and_does_not_create_history(model_service):
    service = model_service
    joint = invoke(service, "get_joint_properties", joint_id=0)["joint"]
    source = invoke(service, "mujoco.get_model_source", model_id=0)
    values = {
        name: joint[name]
        for name in ("joint_id", "axis", "limited", "range", "damping", "stiffness")
    }
    with pytest.raises(RpcError) as error:
        invoke(
            service,
            "edit_scene",
            expected_document=source["document"],
            operations=[
                {"method": "set_joint_properties", "params": {**values, "damping": 0.6}},
                {"method": "remove_model_element", "params": {"node_id": 999999}},
            ],
        )
    assert error.value.code == "command_failed"
    assert error.value.details == {"index": 1, "method": "remove_model_element"}
    restored = invoke(service, "mujoco.get_model_source", model_id=0)
    assert restored["mjcf"] == source["mjcf"]
    assert restored["document"]["revision"] == source["document"]["revision"]
    assert restored["document"]["structure_revision"] > source["document"]["structure_revision"]
    assert not service.session.can_undo
    assert invoke(service, "get_joint_properties", joint_id=0)["joint"] == joint


def test_asset_import_and_source_replace_are_inspectable_and_undoable(model_service, tmp_path):
    service = model_service
    mesh = tmp_path / "tetra.obj"
    mesh.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n")
    invoke(
        service, "import_model_asset", model_id=0, asset_type="mesh", path=str(mesh), name="tetra"
    )
    assets = invoke(service, "list_model_assets", model_id=0)
    assert any(asset["type"] == "mesh" and asset["name"] == "tetra" for asset in assets["assets"])
    source = invoke(service, "mujoco.get_model_source", model_id=0)
    changed = source["mjcf"].replace('name="hinge"', 'name="pivot"')
    invoke(
        service,
        "mujoco.set_model_source",
        model_id=0,
        mjcf=changed,
        expected_document=source["document"],
    )
    assert invoke(service, "list_joints")["joints"][0]["name"] == "pivot"
    invoke(service, "undo")
    restored = invoke(service, "mujoco.get_model_source", model_id=0)
    assert restored["mjcf"] == source["mjcf"]
    with pytest.raises(RpcError):
        invoke(
            service,
            "mujoco.set_model_source",
            model_id=0,
            mjcf="<mujoco><broken>",
            expected_document=restored["document"],
        )
    assert invoke(service, "mujoco.get_model_source", model_id=0)["mjcf"] == source["mjcf"]
    with pytest.raises(RpcError):
        invoke(
            service,
            "import_model_asset",
            model_id=0,
            asset_type="mesh",
            path=str(tmp_path / "missing.obj"),
            name="missing",
        )
    assert invoke(service, "mujoco.get_model_source", model_id=0)["mjcf"] == source["mjcf"]
