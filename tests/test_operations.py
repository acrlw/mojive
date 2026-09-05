"""Application discovery and editing contracts exercised through the RPC facade."""

import subprocess
import sys
from dataclasses import asdict

import numpy as np
import pytest

from mojive import Scene
from mojive.adapters.static import StaticSceneAdapter
from mojive.control_rpc import ControlService, RpcError
from mojive.control_schema import Validator
from mojive.operations import OPERATIONS, apply_session_operation, document_state
from mojive.types import CameraView, Light, Material


@pytest.fixture
def service():
    value = ControlService(StaticSceneAdapter(Scene()))
    yield value
    value.close()


def invoke(service, method, **params):
    result = service.dispatch(method, params)
    Validator(OPERATIONS[method].output_schema).validate(result)
    return result


def test_rpc_client_import_does_not_initialize_application_or_graphics():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from mojive.control_rpc import RpcClient
for name in ('jsonschema', 'mojive.control', 'mojive.operations', 'mojive.ui.app', 'glfw', 'mujoco', 'moderngl'):
    assert name not in sys.modules, name
""",
        ],
        check=True,
    )


@pytest.mark.parametrize("precondition", ["matching", "identity_only", "old_id", "old_revision"])
def test_native_and_rpc_commands_share_document_preconditions(service, precondition):
    created = invoke(service, "add_scene_object", shape="box", name="before")
    expected = document_state(service.session)
    if precondition == "identity_only":
        expected.pop("revision")
    elif precondition == "old_id":
        expected["id"] = "previous-document"
    elif precondition == "old_revision":
        expected["revision"] -= 1
    params = {"object_id": created["object_id"], "name": "after", "expected_document": expected}
    if precondition.startswith("old_"):
        for call in (
            service.dispatch,
            lambda method, params: apply_session_operation(service.session, method, params),
        ):
            with pytest.raises(RpcError) as error:
                call("rename_scene_entity", params)
            assert error.value.code == "stale_document"
            assert error.value.details["expected"] == expected
            assert service.session.node_by_object_id(created["object_id"]).name == "before"
    else:
        assert apply_session_operation(service.session, "rename_scene_entity", params).ok
        assert service.session.node_by_object_id(created["object_id"]).name == "after"
    assert params["expected_document"] == expected


def test_native_precondition_preserves_typed_camera_values(service):
    view = CameraView()
    params = {"name": "shot", "camera": view, "expected_document": document_state(service.session)}
    assert apply_session_operation(service.session, "add_scene_camera", params).ok
    assert params["camera"] is view


@pytest.mark.parametrize("workspace", [False, True])
def test_inspection_reads_composed_appearance_and_geometry_ids_after_edits(tmp_path, workspace):
    from mojive.adapters.toy import ToyPhysicsAdapter
    from mojive.adapters.workspace import WorkspaceAdapter

    scene = Scene()
    scene.box(
        name="crate", color=(0.2, 0.3, 0.4, 1), material=Material(name="paint", emission=0.25)
    )
    scene.sphere(name="reference")
    adapter = (
        WorkspaceAdapter(ToyPhysicsAdapter(), scene) if workspace else StaticSceneAdapter(scene)
    )
    service = ControlService(adapter)
    try:
        node = next(
            item for item in invoke(service, "get_scene")["objects"] if item["name"] == "crate"
        )
        before = invoke(service, "inspect_object", object_id=node["object_id"])
        assert len(before["geometries"]) == 1
        geometry = before["geometries"][0]
        assert geometry["node_id"] != before["node_id"]
        assert geometry["object_id"] == before["object_id"]
        assert geometry["mesh"]["shape"] == "box"
        assert geometry["rgba"] == pytest.approx([0.2, 0.3, 0.4, 1])
        assert geometry["material"]["name"] == "paint"
        assert geometry["material"]["emission"] == 0.25
        assert geometry["dimensions"]["values"] == pytest.approx(np.asarray(geometry["size"]) * 2)
        edited = invoke(
            service,
            "edit_scene",
            expected_document=before["document"],
            operations=[
                {
                    "method": "set_geometry_color",
                    "params": {"node_id": geometry["node_id"], "rgba": [0.7, 0.2, 0.1, 0.5]},
                },
                {
                    "method": "set_geometry_size",
                    "params": {"node_id": geometry["node_id"], "size": [0.3, 0.4, 0.7]},
                },
            ],
        )
        after = invoke(service, "inspect_object", object_id=node["object_id"])
        assert after["document"] == edited["document"]
        assert after["structure_generation"] > before["structure_generation"]
        assert after["geometries"][0]["rgba"] == pytest.approx([0.7, 0.2, 0.1, 0.5])
        assert after["geometries"][0]["size"] == pytest.approx([0.3, 0.4, 0.7])
        invoke(service, "set_visible", node_id=node["node_id"], visible=False)
        hidden = invoke(service, "inspect_object", node_id=geometry["node_id"])
        assert hidden["visible"] and not hidden["hierarchy_visible"]
        assert hidden["geometries"] == after["geometries"]
        # Returned JSON is an owned snapshot, not a mutable view into Session buffers.
        hidden["geometries"][0]["rgba"][0] = 0
        assert invoke(service, "inspect_object", node_id=geometry["node_id"])["geometries"][0][
            "rgba"
        ][0] == pytest.approx(0.7)
        invoke(service, "undo")
        restored = invoke(service, "inspect_object", object_id=node["object_id"])
        assert restored["geometries"][0]["size"] == geometry["size"]
        assert restored["geometries"][0]["rgba"] == geometry["rgba"]
        invoke(service, "redo")
        if not workspace:
            saved = tmp_path / "scene.mojive.json"
            invoke(service, "save_scene", path=saved)
            invoke(service, "open_scene", path=saved)
            reopened = invoke(service, "inspect_object", object_id=node["object_id"])
            assert reopened["geometries"][0]["rgba"] == after["geometries"][0]["rgba"]
    finally:
        service.close()


def test_result_schemas_reject_incomplete_inspection_and_explain_discovery(service):
    from jsonschema.exceptions import ValidationError

    object_id = invoke(service, "add_scene_object", shape="box")["object_id"]
    node = invoke(service, "inspect_object", object_id=object_id)
    del node["geometries"][0]["material"]
    with pytest.raises(ValidationError):
        Validator(OPERATIONS["inspect_object"].output_schema).validate(node)
    state = invoke(service, "get_state")
    state["camera"]["eye"] = [1, 2]
    with pytest.raises(ValidationError):
        Validator(OPERATIONS["get_state"].output_schema).validate(state)
    for name in (
        "hello",
        "describe_operations",
        "get_state",
        "get_capture_settings",
        "get_viewer_settings",
    ):
        assert OPERATIONS[name].output_schema["properties"]


@pytest.mark.parametrize("method", ["get_scene", "list_objects", "get_bounds"])
def test_queries_observe_updates_from_the_scene_owner(service, method):
    scene = service.session.adapter.scene
    scene.box(name="new object", position=(10, 0, 0))
    scene.add_camera("new camera", CameraView())
    result = invoke(service, method)
    if method == "list_objects":
        assert any(node["name"] == "new object" for node in result)
    else:
        bounds = result["bounds"] if method == "get_scene" else result
        assert bounds["minimum"] == pytest.approx([9.5, -0.5, -0.5])
        assert bounds["maximum"] == pytest.approx([10.5, 0.5, 0.5])
        if method == "get_scene":
            assert any(node["name"] == "new object" for node in result["objects"])
            assert result["cameras"][0]["name"] == "new camera"


@pytest.mark.parametrize("workspace", [False, True])
def test_geometry_child_inspection_uses_the_composed_pose_index(workspace):
    from mojive.adapters.toy import ToyPhysicsAdapter
    from mojive.adapters.workspace import WorkspaceAdapter

    scene = Scene()
    scene.box(name="offset", position=(2, 3, 4))
    adapter = (
        WorkspaceAdapter(ToyPhysicsAdapter(), scene) if workspace else StaticSceneAdapter(scene)
    )
    service = ControlService(adapter)
    try:
        geom = next(node for node in service.session.nodes if node.name == "offset.geom")
        inspected = invoke(service, "inspect_object", node_id=geom.node_id)
        assert inspected["position"] == pytest.approx([2, 3, 4])
    finally:
        service.close()


def test_discovery_has_valid_schemas_defaults_handlers_and_dynamic_availability(service):
    description = invoke(service, "describe_operations")
    for item in description["operations"]:
        Validator.check_schema(item["input_schema"])
        Validator.check_schema(item["output_schema"])
        operation = OPERATIONS[item["name"]]
        if operation.handler:
            assert callable(getattr(service.application, operation.handler))
        for field in item["input_schema"]["properties"].values():
            if "default" in field:
                Validator(field).validate(field["default"])
    by_name = {item["name"]: item for item in description["operations"]}
    assert by_name["add_scene_object"]["available"]
    assert not by_name["undo"]["available"]
    assert "simulation" in by_name["step"]["unavailable_reason"]
    assert "attached viewer" in by_name["capture_viewport"]["unavailable_reason"]
    assert by_name["set_camera"]["alias_of"] == "set_capture_camera"
    invoke(service, "add_scene_object", shape="box", material=asdict(Material()))
    available = invoke(service, "describe_operations", available_only=True)["operations"]
    assert "undo" in {item["name"] for item in available}
    assert all(item["available"] for item in available)
    for method in (
        "hello",
        "get_capabilities",
        "get_scene",
        "get_state",
        "get_bounds",
        "list_objects",
    ):
        invoke(service, method)


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"shape": "bogus"},
        {"shape": "box", "colour": [1, 0, 0, 1]},
        {"shape": "box", "size": [1, 0, 1]},
        {"shape": "box", "position": [1, 2]},
        {"shape": "box", "position": [float("nan"), 0, 0]},
        {"shape": "box", "rotation": [[1, 0, 0]] * 3},
    ],
)
def test_invalid_edits_do_not_create_history_or_objects(service, params):
    with pytest.raises(RpcError) as error:
        service.dispatch("add_scene_object", params)
    assert error.value.code == "invalid_params"
    assert not service.session.can_undo
    assert not len(service.session.source.geom_mesh)


def test_edit_inspect_undo_save_reopen_and_reject_stale_ids(service, tmp_path):
    owner = service.session.adapter.scene
    initial = invoke(service, "get_scene")["document"]
    created = invoke(
        service, "add_scene_object", shape="box", name="crate", expected_document=initial
    )
    inspected = invoke(service, "inspect_object", object_id=created["entity_id"])
    invoke(
        service, "set_pose", node_id=inspected["node_id"], position=[1, 2, 3], rotation=np.eye(3)
    )
    assert invoke(service, "inspect_object", object_id=created["entity_id"])["position"] == [
        1,
        2,
        3,
    ]
    invoke(service, "rename_scene_entity", object_id=created["entity_id"], name="moved")
    invoke(service, "undo")
    assert invoke(service, "inspect_object", object_id=created["entity_id"])["name"] == "crate"
    invoke(service, "redo")
    assert service.session.adapter.scene is owner
    saved = tmp_path / "edited.mojive.json"
    invoke(service, "save_scene", path=str(saved))
    before = invoke(service, "get_scene")["document"]
    assert before["id"] == initial["id"]
    invoke(service, "new_scene")
    invoke(service, "open_scene", path=str(saved))
    current = invoke(service, "get_scene")["document"]
    assert current["id"] != before["id"]
    with pytest.raises(RpcError, match="document changed") as error:
        invoke(
            service, "remove_scene_entity", object_id=created["entity_id"], expected_document=before
        )
    assert error.value.code == "stale_document"
    assert error.value.details["actual"] == current
    assert invoke(service, "inspect_object", object_id=created["entity_id"])["name"] == "moved"
    with pytest.raises(RpcError):
        invoke(service, "open_scene", path=str(tmp_path / "missing.mojive.json"))
    assert invoke(service, "get_scene")["document"] == current


def test_transaction_is_one_undo_record_and_failure_restores_prior_state(service):
    result = invoke(
        service,
        "edit_scene",
        label="Add pair",
        operations=[
            {"method": "add_scene_object", "params": {"shape": shape, "name": shape}}
            for shape in ("box", "sphere")
        ],
    )
    assert len(result["results"]) == 2 and len(service.session.source.geom_mesh) == 2
    invoke(service, "undo")
    assert len(service.session.source.geom_mesh) == 0 and not service.session.can_undo
    invoke(service, "redo")
    original = invoke(service, "get_scene")
    with pytest.raises(RpcError) as error:
        invoke(
            service,
            "edit_scene",
            operations=[
                {"method": "add_scene_object", "params": {"shape": "box", "name": "temporary"}},
                {"method": "remove_scene_entity", "params": {"object_id": 999999}},
            ],
        )
    assert error.value.details == {"index": 1, "method": "remove_scene_entity"}
    current = invoke(service, "get_scene")
    assert current["objects"] == original["objects"]
    assert current["document"] == original["document"]
    assert not service.session.editing
    invoke(service, "undo")
    assert not len(service.session.source.geom_mesh)


def test_transaction_rejects_non_edit_before_mutation(service):
    with pytest.raises(RpcError, match="cannot be part"):
        invoke(
            service,
            "edit_scene",
            operations=[
                {"method": "add_scene_object", "params": {"shape": "box"}},
                {"method": "new_scene", "params": {}},
            ],
        )
    assert not service.session.can_undo and not len(service.session.source.geom_mesh)


def test_camera_light_and_capture_camera_keep_contract_values(service):
    camera = CameraView(
        up=np.array([0, 1, 1], np.float32),
        focal_length=np.array([35, 35]),
        sensor_size=np.array([36, 24]),
    )
    invoke(service, "add_scene_camera", name="shot", camera=camera)
    camera_id = invoke(service, "get_scene")["cameras"][0]["camera_id"]
    invoke(service, "set_capture_camera", camera_id=camera_id)
    settings = invoke(service, "get_capture_settings")
    assert settings["camera"]["up"] == [0, 1, 1]
    assert settings["camera"]["focal_length"] == [35, 35]
    invoke(
        service, "inspect_object", object_id=invoke(service, "get_scene")["cameras"][0]["object_id"]
    )
    light = invoke(service, "add_scene_light", name="key", light=Light())
    assert light["light_id"] >= 0
    invoke(service, "set_capture_camera", eye=[5, 0, 0])
    assert invoke(service, "get_capture_settings")["camera"]["source"] == -1
    with pytest.raises(RpcError, match="far"):
        invoke(service, "set_capture_camera", near=10, far=1)
    assert invoke(service, "get_capture_settings")["camera"]["near"] == camera.near
