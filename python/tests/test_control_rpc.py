"""Local control protocol and Session command routing."""

from __future__ import annotations

import json
import socket
import stat
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mojive.adapters.mujoco_adapter import MuJoCoAdapter
from mojive.cli import main
from mojive.control_rpc import (
    PROTOCOL_VERSION,
    ControlServer,
    ControlService,
    RpcClient,
    RpcError,
    ViewerControlService,
)
from mojive.types import CameraView

pytestmark = pytest.mark.physics


@pytest.fixture
def rpc():
    asset = Path("assets/joint_types.xml").resolve()
    service = ControlService(MuJoCoAdapter(asset), asset)
    with tempfile.TemporaryDirectory(prefix="fv-", dir="/tmp") as directory:
        server = ControlServer(Path(directory) / "control.sock", service)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        client = RpcClient(server.socket_path, timeout=1.0)
        try:
            yield client, service, server
        finally:
            client.close()
            server.shutdown()
            server.server_close()
            service.close()
            thread.join(timeout=2.0)


def test_rpc_routes_simulation_and_state_commands(rpc):
    client, service, _ = rpc
    client.call("pause")
    before = client.call("get_state")
    qpos = np.asarray(before["physics"]["qpos"]["values"])
    changed = qpos.copy()
    changed[0] += 0.1
    client.call("set_qpos", {"values": changed.tolist()})
    client.call("step", {"count": 2})
    after = client.call("get_state")

    assert after["paused"]
    assert after["physics"]["qpos"]["values"][0] == pytest.approx(changed[0])
    assert service.session.frame.step == 2


def test_schema_rejection_precedes_control_updates_and_reports_live_availability(rpc):
    client, service, _ = rpc
    assert client.get_state()["paused"]
    before = service.session.adapter.capture_state().ctrl.copy()
    controls = np.full_like(before, 0.25).tolist()
    for count in (0, -1, True, 1.5):
        with pytest.raises(RpcError) as error:
            client.call("step", {"count": count, "ctrl": controls})
        assert error.value.code == "invalid_params"
        assert error.value.details["path"] == "/count"
        np.testing.assert_array_equal(service.session.adapter.capture_state().ctrl, before)
        assert service.session.frame.step == 0
    assert client.describe_operations(name="step")["operations"][0]["available"]
    client.call("resume")
    operation = client.describe_operations(name="step")["operations"][0]
    assert not operation["available"] and "Pause" in operation["unavailable_reason"]
    client.call("pause")


def test_physics_queries_match_schemas_and_report_session_geometry_edits(rpc):
    from mojive.control_schema import Validator
    from mojive.operations import OPERATIONS

    client, _, _ = rpc
    for method in (
        "hello",
        "get_state",
        "get_scene",
        "get_capture_settings",
        "describe_operations",
    ):
        Validator(OPERATIONS[method].output_schema).validate(client.call(method))
    node = next(item for item in client.call("get_scene")["objects"] if item["type"] == "geom")
    before = client.call("inspect_object", {"node_id": node["node_id"]})
    assert before["geometries"]
    client.call(
        "set_geometry_color",
        {
            "node_id": node["node_id"],
            "rgba": [0.2, 0.4, 0.8, 1],
            "expected_document": before["document"],
        },
    )
    after = client.call("inspect_object", {"node_id": node["node_id"]})
    Validator(OPERATIONS["inspect_object"].output_schema).validate(after)
    assert after["geometries"][0]["rgba"] == pytest.approx([0.2, 0.4, 0.8, 1])
    stepped = client.call("step", {"count": 1, "observe": True})
    Validator(OPERATIONS["step"].output_schema).validate(stepped)


def test_rpc_advertises_capabilities_and_supports_atomic_policy_steps(rpc):
    client, _, _ = rpc
    client.call("pause")
    hello = client.hello()
    before = client.get_state()
    ctrl = np.asarray(before["physics"]["ctrl"]["values"])

    observed = client.step(2, ctrl=ctrl, observe=True)

    assert hello["service"] == "mojive.control"
    assert "set_ctrl" in hello["methods"]
    assert "set_shadow_quality" in hello["methods"]
    assert "reset_layout" in hello["methods"]
    assert observed["state"]["step"] == 2
    assert observed["state"]["physics"]["ctrl"]["values"] == pytest.approx(ctrl)


def test_rpc_restores_velocity_and_rejects_wrong_state_shapes(rpc):
    client, _, _ = rpc
    client.call("pause")
    before = client.get_state()
    qvel = np.asarray(before["physics"]["qvel"]["values"])
    changed = qvel + 0.125

    client.call("set_qvel", {"values": changed.tolist()})
    after = client.get_state()

    assert after["physics"]["qvel"]["values"] == pytest.approx(changed)
    with pytest.raises(RpcError, match="Expected qvel shape") as error:
        client.call("set_qvel", {"values": [0.0]})
    assert error.value.code == "invalid_params"


def test_headless_resume_has_a_realtime_scheduler(rpc):
    client, _, _ = rpc
    client.call("resume")
    deadline = time.monotonic() + 1.0
    state = client.get_state()
    while state["step"] == 0 and time.monotonic() < deadline:
        time.sleep(0.01)
        state = client.get_state()
    client.call("pause")

    assert state["step"] > 0


def test_viewer_service_marshals_requests_until_the_ui_thread_pumps(rpc):
    _, service, _ = rpc
    app = SimpleNamespace(camera=service.camera)
    viewer_service = ViewerControlService(SimpleNamespace(session=service.session, app=app))
    response = None

    def request() -> None:
        nonlocal response
        response = viewer_service.handle(
            {"version": PROTOCOL_VERSION, "id": 7, "method": "hello", "params": {}}
        )

    thread = threading.Thread(target=request)
    thread.start()
    deadline = time.monotonic() + 1.0
    while viewer_service._pending.empty() and time.monotonic() < deadline:
        time.sleep(0.001)
    assert thread.is_alive()
    assert viewer_service.pump() == 1
    thread.join(timeout=1.0)
    viewer_service.close()

    assert response["id"] == 7
    assert response["result"]["viewer_attached"] is True


def test_control_socket_is_private_to_the_current_user(rpc):
    client, _, _ = rpc

    assert stat.S_IMODE(client.socket_path.stat().st_mode) == 0o600


def test_rpc_camera_selection_uses_session_overrides(rpc, monkeypatch):
    from mojive import commands as cmd

    client, service, _ = rpc
    view = CameraView(
        eye=np.array([2.0, -3.0, 1.5], np.float32),
        target=np.array([0.2, 0.4, 0.7], np.float32),
    )
    monkeypatch.setattr(service.session.adapter, "set_camera_view", lambda *args: False)
    assert service.session.submit(cmd.SetSceneCamera(0, view)).ok
    client.call("set_camera", {"camera_id": 0})
    assert service.camera.view().eye == pytest.approx(view.eye, abs=1e-6)
    assert service.camera.view().forward() == pytest.approx(view.forward(), abs=1e-6)


def test_rpc_lists_selects_and_inspects_objects(rpc):
    client, _, _ = rpc
    objects = client.call("list_objects")
    selected = next(item for item in objects if item["object_id"])

    result = client.call("select_object", {"object_id": selected["object_id"]})
    inspected = client.call("inspect_object", {"object_id": selected["object_id"]})

    assert result["object"]["node_id"] == selected["node_id"]
    assert inspected["name"] == selected["name"]


def test_rpc_visual_groups_keyframes_and_camera(rpc):
    client, service, _ = rpc
    client.call("pause")
    groups = service.session.adapter.visual_groups()
    group = groups[0]
    client.call(
        "set_visual_group",
        {"category": group.category, "group": 0, "visible": not group.visible[0]},
    )
    camera = client.call("set_camera", {"camera_id": 0})
    render_flag = client.call("set_render_flag", {"name": "shadow", "enabled": False})
    visual_flag = client.call("set_visualization_flag", {"name": "joint", "enabled": True})

    assert camera["source"] == 0
    assert render_flag == {"name": "mjRND_SHADOW", "enabled": False}
    assert visual_flag == {"name": "mjVIS_JOINT", "enabled": True}
    assert service.session.adapter.visual_groups()[0].visible[0] != group.visible[0]


def test_rpc_returns_structured_errors_and_correlates_requests(rpc):
    client, _, server = rpc
    with pytest.raises(RpcError, match="Unknown control method") as error:
        client.call("missing")
    assert error.value.code == "unknown_method"

    request = {"version": PROTOCOL_VERSION + 1, "id": "version", "method": "get_state"}
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.connect(str(server.socket_path))
        connection.sendall(json.dumps(request).encode() + b"\n")
        response = json.loads(connection.makefile().readline())
    assert response["id"] == "version"
    assert response["error"]["code"] == "version_mismatch"


def test_rpc_reuses_one_connection_for_many_requests(rpc):
    client, _, _ = rpc
    first_socket = None
    for _ in range(256):
        assert "physics" in client.call("get_state")
        first_socket = first_socket or client._client
        assert client._client is first_socket


def test_rpc_connection_recovers_after_invalid_request(rpc):
    _, _, server = rpc
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.connect(str(server.socket_path))
        stream = connection.makefile("rwb")
        stream.write(b"not-json\n")
        stream.flush()
        invalid = json.loads(stream.readline())
        stream.write(
            json.dumps(
                {"version": PROTOCOL_VERSION, "id": 9, "method": "get_state", "params": {}}
            ).encode()
            + b"\n"
        )
        stream.flush()
        recovered = json.loads(stream.readline())

    assert invalid["error"]["code"] == "invalid_request"
    assert recovered["id"] == 9
    assert recovered["error"] is None


def test_idle_connection_does_not_block_other_clients(rpc):
    client, _, server = rpc
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as idle:
        idle.connect(str(server.socket_path))
        assert "physics" in client.call("get_state")


def test_rpc_reconnects_on_the_call_after_a_timeout(rpc):
    client, service, _ = rpc
    original = service.dispatch
    delayed = True

    def dispatch(method, params):
        nonlocal delayed
        if delayed and method == "get_state":
            delayed = False
            threading.Event().wait(0.05)
        return original(method, params)

    service.dispatch = dispatch
    client.timeout = 0.01
    with pytest.raises(RpcError, match="timed out") as error:
        client.call("get_state")
    assert error.value.code == "timeout"
    assert client._client is None

    client.timeout = 1.0
    assert "physics" in client.call("get_state")


def test_control_cli_prints_json(rpc, capsys):
    client, _, _ = rpc
    code = main(
        [
            "control",
            "get_state",
            "--socket",
            str(client.socket_path),
            "--json",
        ]
    )
    output = json.loads(capsys.readouterr().out)

    assert code == 0
    assert output["asset"].endswith("joint_types.xml")


def test_rpc_client_timeout():
    with tempfile.TemporaryDirectory(prefix="fv-", dir="/tmp") as directory:
        path = Path(directory) / "silent.sock"
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(str(path))
        listener.listen()

        def accept_without_reply():
            connection, _ = listener.accept()
            connection.recv(4096)
            threading.Event().wait(0.1)
            connection.close()

        thread = threading.Thread(target=accept_without_reply, daemon=True)
        thread.start()
        with pytest.raises(RpcError, match="timed out") as error:
            RpcClient(path, timeout=0.02).call("get_state")
        assert error.value.code == "timeout"
        listener.close()
        thread.join(timeout=1.0)
