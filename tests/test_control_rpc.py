"""Local control protocol and Session command routing."""

from __future__ import annotations

import json
import socket
import tempfile
import threading
from pathlib import Path

import numpy as np
import pytest

from forge_viewer.adapters.mujoco_adapter import MuJoCoAdapter
from forge_viewer.cli import main
from forge_viewer.control_rpc import (
    PROTOCOL_VERSION,
    ControlServer,
    ControlService,
    RpcClient,
    RpcError,
)

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
