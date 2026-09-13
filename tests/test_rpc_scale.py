"""Scale discovery, baked dimensions and history through the public control boundary."""

import json
import subprocess
import sys
import threading

import numpy as np
import pytest

from mojive import Scene
from mojive import commands as cmd
from mojive.adapters.static import StaticSceneAdapter
from mojive.adapters.toy import ToyPhysicsAdapter
from mojive.adapters.workspace import WorkspaceAdapter
from mojive.control.operations import OPERATIONS
from mojive.control.rpc import ControlServer, ControlService, RpcError
from mojive.control.schema import Validator


@pytest.fixture(params=[False, True], ids=["static", "workspace"])
def service(request):
    adapter = StaticSceneAdapter(Scene())
    if request.param:
        adapter = WorkspaceAdapter(adapter, Scene())
    value = ControlService(adapter)
    yield value
    value.close()


def call(service, method, **params):
    result = service.dispatch(method, params)
    Validator(OPERATIONS[method].output_schema).validate(result)
    return result


def create_box(service):
    result = call(
        service, "add_scene_object", shape="box", size=[0.2, 0.3, 0.4], position=[1, 2, 3]
    )
    return call(service, "inspect_object", object_id=result["object_id"])


def test_scale_discovery_describes_capability_validation_and_transaction(service):
    spec = call(service, "describe_operations", name="set_scale")["operations"][0]
    assert spec["available"] and spec["transactional"]
    assert spec["requirements"] == {"capabilities": ["write_scale"], "paused": True}
    assert "expected_document" in spec["input_schema"]["properties"]
    assert "set_scale" in call(service, "hello")["available_methods"]


@pytest.mark.parametrize("target", ["object", "geometry"])
def test_scale_bakes_local_dimensions_and_roundtrips_history_and_file(service, target, tmp_path):
    before = create_box(service)
    node_id = before["node_id"] if target == "object" else before["geometries"][0]["node_id"]
    assert call(service, "inspect_object", node_id=node_id)["scalable"]
    result = call(
        service,
        "set_scale",
        node_id=node_id,
        scale=[2, 0.5, 3],
        expected_document=before["document"],
    )
    assert result["ok"] and result["document"]["revision"] > before["document"]["revision"]

    def check(size):
        current = call(service, "inspect_object", object_id=before["object_id"])
        assert current["scale"] == [1, 1, 1]
        assert current["position"] == before["position"]
        assert current["rotation"] == before["rotation"]
        np.testing.assert_allclose(current["geometries"][0]["size"], size)
        np.testing.assert_allclose(
            current["geometries"][0]["dimensions"]["values"], np.array(size) * 2
        )

    check([0.4, 0.15, 1.2])
    call(service, "undo")
    check([0.2, 0.3, 0.4])
    call(service, "redo")
    check([0.4, 0.15, 1.2])
    path = str(tmp_path / "scaled.mojive.json")
    call(service, "save_scene", path=path)
    call(service, "new_scene")
    call(service, "open_scene", path=path)
    reopened = next(
        item for item in call(service, "get_scene")["objects"] if item["name"] == before["name"]
    )
    before["object_id"] = reopened["object_id"]
    check([0.4, 0.15, 1.2])


@pytest.mark.parametrize(
    ("scale", "code"),
    [
        ([0, 1, 1], "invalid_params"),
        ([-1, 1, 1], "invalid_params"),
        ([True, 1, 1], "invalid_params"),
        (["2", 1, 1], "invalid_params"),
        ([1, 2], "invalid_params"),
        ([float("nan"), 1, 1], "invalid_params"),
        ([float("inf"), 1, 1], "invalid_params"),
        ([1e300, 1, 1], "command_failed"),
        ([1e-300, 1, 1], "command_failed"),
    ],
)
def test_invalid_scale_preserves_dimensions_and_history(service, scale, code):
    before = create_box(service)
    response = service.handle(
        {
            "version": 1,
            "id": "scale",
            "method": "set_scale",
            "params": {"node_id": before["node_id"], "scale": scale},
        }
    )
    assert response["error"]["code"] == code
    assert call(service, "inspect_object", object_id=before["object_id"]) == before
    call(service, "undo")
    assert not call(service, "get_scene")["objects"] or not service.session.source.geom_mesh


def test_stale_or_unsupported_target_cannot_scale(service):
    before = create_box(service)
    stale = {**before["document"], "revision": before["document"]["revision"] - 1}
    with pytest.raises(RpcError) as error:
        call(
            service,
            "set_scale",
            node_id=before["node_id"],
            scale=[2, 2, 2],
            expected_document=stale,
        )
    assert error.value.code == "stale_document"
    world = next(item for item in call(service, "get_scene")["objects"] if item["type"] == "world")
    assert not call(service, "inspect_object", node_id=world["node_id"])["scalable"]
    for node_id in (world["node_id"], 2**30):
        with pytest.raises(RpcError) as error:
            call(service, "set_scale", node_id=node_id, scale=[2, 2, 2])
        assert error.value.code == "command_failed"
    assert call(service, "inspect_object", object_id=before["object_id"]) == before


def test_scale_is_one_transaction_and_rolls_back_a_later_failure(service):
    before = create_box(service)
    operation = {
        "method": "set_scale",
        "params": {"node_id": before["node_id"], "scale": [2, 2, 2]},
    }
    call(service, "edit_scene", operations=[operation, operation])
    np.testing.assert_allclose(service.session.source.geom_size[-1], [0.8, 1.2, 1.6])
    call(service, "undo")
    np.testing.assert_allclose(service.session.source.geom_size[-1], [0.2, 0.3, 0.4])
    with pytest.raises(RpcError) as error:
        call(
            service,
            "edit_scene",
            operations=[
                operation,
                {"method": "set_scale", "params": {"node_id": 2**30, "scale": [2, 2, 2]}},
            ],
        )
    assert error.value.details == {"index": 1, "method": "set_scale"}
    np.testing.assert_allclose(service.session.source.geom_size[-1], [0.2, 0.3, 0.4])


def test_scale_is_unavailable_without_adapter_support():
    service = ControlService(ToyPhysicsAdapter())
    try:
        spec = call(service, "describe_operations", name="set_scale")["operations"][0]
        assert not spec["available"] and "write_scale" in spec["unavailable_reason"]
        node = next(
            item for item in call(service, "get_scene")["objects"] if item["type"] == "geom"
        )
        assert not call(service, "inspect_object", node_id=node["node_id"])["scalable"]
        with pytest.raises(RpcError) as error:
            call(service, "set_scale", node_id=node["node_id"], scale=[2, 2, 2])
        assert error.value.code == "unsupported"
    finally:
        service.close()


def test_scale_requires_paused_session():
    service = ControlService(WorkspaceAdapter(ToyPhysicsAdapter(), Scene()))
    try:
        before = create_box(service)
        assert service.session.submit(cmd.Play()).ok
        assert not service.session.paused
        with pytest.raises(RpcError) as error:
            call(service, "set_scale", node_id=before["node_id"], scale=[2, 2, 2])
        assert error.value.code == "unsupported"
    finally:
        service.close()


@pytest.mark.integration
def test_cli_scales_through_a_real_socket_and_reads_baked_history(tmp_path):
    service = ControlService(StaticSceneAdapter(Scene()))
    server = ControlServer(tmp_path / "scale.sock", service)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:

        def invoke(method, **params):
            process = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "mojive.cli",
                    "control",
                    method,
                    "--socket",
                    str(server.socket_path),
                    "--params-file",
                    "-",
                    "--json",
                ],
                input=json.dumps(params),
                text=True,
                capture_output=True,
                check=True,
                timeout=20,
            )
            assert not process.stderr
            return json.loads(process.stdout)

        created = invoke("add_scene_object", shape="box", size=[0.2, 0.3, 0.4])
        before = invoke("inspect_object", object_id=created["object_id"])
        invoke(
            "set_scale",
            node_id=before["node_id"],
            scale=[2, 3, 4],
            expected_document=before["document"],
        )
        for command, size in (
            (None, [0.4, 0.9, 1.6]),
            ("undo", [0.2, 0.3, 0.4]),
            ("redo", [0.4, 0.9, 1.6]),
        ):
            if command:
                invoke(command)
            current = invoke("inspect_object", object_id=created["object_id"])
            assert current["scale"] == [1, 1, 1]
            np.testing.assert_allclose(current["geometries"][0]["size"], size)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        service.close()
