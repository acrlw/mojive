"""Real Unix sockets preserve framing, admission and cancellation under pressure."""

import json
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

import pytest

from mojive import cli
from mojive import commands as cmd
from mojive.adapters.toy import ToyPhysicsAdapter
from mojive.control.rpc import (
    ControlServer,
    ControlService,
    RpcClient,
    RpcError,
    RpcLimits,
    ViewerControlService,
)
from mojive.session import Session
from mojive.ui.camera import OrbitCamera

pytestmark = pytest.mark.integration


@contextmanager
def running(tmp_path, *, viewer=False, **budgets):
    session = Session(ToyPhysicsAdapter())
    assert session.submit(cmd.Pause()).ok
    limits = RpcLimits(**budgets)
    service = (
        ViewerControlService(
            SimpleNamespace(session=session, app=SimpleNamespace(camera=OrbitCamera())),
            limits=limits,
        )
        if viewer
        else ControlService(session=session, limits=limits)
    )
    with ControlServer(tmp_path / "budgets.sock", service) as server:
        thread = threading.Thread(
            target=lambda: server.serve_forever(poll_interval=0.01), daemon=True
        )
        thread.start()
        try:
            yield server, service, session
        finally:
            server.shutdown()
            server.server_close()
            service.close()
            thread.join(timeout=2)
            session.release()


def wait_until(predicate):
    deadline = time.monotonic() + 2
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.001)
    assert predicate()


def wire_request(**extra):
    return (
        json.dumps({"version": 1, "id": 1, "method": "step", "params": {"count": 3}, **extra})
        + "\n"
    ).encode()


def read_reply(peer):
    data = bytearray()
    while not data.endswith(b"\n"):
        chunk = peer.recv(65536)
        if not chunk:
            break
        data.extend(chunk)
    return json.loads(data)


def test_oversized_frame_closes_without_parsing_its_suffix(tmp_path):
    with running(tmp_path, max_request_bytes=512) as (server, service, session):
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as peer:
            peer.settimeout(2)
            peer.connect(str(server.socket_path))
            peer.sendall(wire_request(padding="x" * 1024) + wire_request())
            reply = read_reply(peer)
            assert reply["error"]["code"] == "request_too_large" and reply["id"] is None
        assert session.frame.step == 0 and service.stats.snapshot()["counters"]["accepted"] == 0
        with RpcClient(server.socket_path) as client:
            client.step(1)
        assert session.frame.step == 1


@pytest.mark.parametrize("version", [True, 1.0, "1"])
def test_invalid_protocol_version_cannot_execute_mutation(tmp_path, version):
    with (
        running(tmp_path) as (server, _service, session),
        socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as peer,
    ):
        peer.settimeout(2)
        peer.connect(str(server.socket_path))
        peer.sendall(wire_request(version=version))
        assert read_reply(peer)["error"]["code"] == "version_mismatch"
        assert session.frame.step == 0
        peer.sendall(wire_request())
        assert read_reply(peer)["result"]["ok"]
        assert session.frame.step == 3


@pytest.mark.parametrize("value", [float("nan"), object()])
def test_unserializable_result_reports_error_and_keeps_socket_usable(tmp_path, monkeypatch, value):
    with running(tmp_path) as (server, service, session), RpcClient(server.socket_path) as client:
        dispatch = service.dispatch

        def invalid_result(method, params):
            result = dispatch(method, params)
            return {"invalid": value} if method == "step" else result

        monkeypatch.setattr(service, "dispatch", invalid_result)
        with pytest.raises(RpcError) as error:
            client.step(1)
        assert error.value.code == "internal_error"
        assert "serialize" in str(error.value)
        assert session.frame.step == 1
        connection = client._client
        assert client.hello()["service"] == "mojive.control"
        assert client._client is connection


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity", "1e999"])
def test_nonfinite_request_envelope_cannot_execute_mutation(tmp_path, value):
    with (
        running(tmp_path) as (server, _service, session),
        socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as peer,
    ):
        peer.settimeout(2)
        peer.connect(str(server.socket_path))
        peer.sendall(wire_request().replace(b'"id": 1', f'"id": {value}'.encode()))
        assert read_reply(peer)["error"]["code"] == "invalid_request"
        assert session.frame.step == 0
        peer.sendall(wire_request())
        assert read_reply(peer)["result"]["ok"]


def test_large_write_receives_transport_rejection_without_retry(tmp_path):
    with running(tmp_path, max_request_bytes=512) as (server, service, session):
        with RpcClient(server.socket_path) as client:
            with pytest.raises(RpcError) as error:
                client.call("step", {"count": 3, "padding": "x" * (2 * 1024 * 1024)})
            assert error.value.code == "request_too_large"
            assert client._client is None and session.frame.step == 0
        assert service.stats.snapshot()["counters"]["oversize_requests"] == 1


def test_connection_limit_rejects_before_starting_work_and_releases_slots(tmp_path):
    with running(tmp_path, max_connections=1) as (server, service, session):
        with RpcClient(server.socket_path) as first:
            first.hello()
            with RpcClient(server.socket_path) as second:
                with pytest.raises(RpcError) as error:
                    second.step(5)
                assert error.value.code == "busy"
                assert session.frame.step == 0
        wait_until(lambda: service.stats.snapshot()["counters"]["connections"] == 0)
        with RpcClient(server.socket_path) as next_client:
            next_client.step(1)
        stats = service.stats.snapshot()["counters"]
        assert stats["peak_connections"] == 1 and stats["connections_rejected"] == 1
        assert session.frame.step == 1


def test_disconnect_without_deadline_cancels_unstarted_mutation(tmp_path):
    with running(tmp_path, viewer=True) as (server, service, session):
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as peer:
            peer.connect(str(server.socket_path))
            peer.sendall(wire_request())
            wait_until(lambda: service.stats.snapshot()["counters"]["queued_requests"] == 1)
        wait_until(lambda: service.stats.snapshot()["counters"]["cancelled"] == 1)
        assert service.pump() == 1 and session.frame.step == 0
        stats = service.stats.snapshot()["counters"]
        assert stats["queued_bytes"] == 0 and stats["inflight_requests"] == 0


def test_configured_connection_burst_fits_the_listen_backlog(tmp_path):
    count = 16
    with running(tmp_path, max_connections=count) as (server, service, _session):
        barrier = threading.Barrier(count)

        def connect(_index):
            barrier.wait(timeout=5)
            with RpcClient(server.socket_path) as client:
                result = client.hello()
                barrier.wait(timeout=5)  # Keep all admitted sockets alive together.
                return result["service"]

        with ThreadPoolExecutor(max_workers=count) as pool:
            assert list(pool.map(connect, range(count))) == ["mojive.control"] * count
        wait_until(lambda: service.stats.snapshot()["counters"]["connections"] == 0)
        stats = service.stats.snapshot()["counters"]
        assert stats["peak_connections"] == count and stats["connections_rejected"] == 0


def test_server_and_client_bound_responses_before_exposing_data(tmp_path):
    with running(tmp_path, max_response_bytes=512) as (server, service, _session):
        with RpcClient(server.socket_path) as client:
            with pytest.raises(RpcError) as error:
                client.hello()
            assert error.value.code == "response_too_large"
            assert client.step(1)["ok"]  # Complete error framing preserves the connection.
        assert service.stats.snapshot()["counters"]["oversize_responses"] == 1
    with (
        running(tmp_path) as (server, _service, _session),
        RpcClient(server.socket_path, limits=RpcLimits(max_response_bytes=512)) as client,
    ):
        with pytest.raises(RpcError) as error:
            client.hello()
        assert error.value.code == "response_too_large" and client._client is None
        client.limits = replace(client.limits, max_response_bytes=256 * 1024 * 1024)
        assert client.hello()["service"] == "mojive.control"


def test_cli_reads_live_stats_and_validates_limit_files_before_connecting(tmp_path, capsys):
    path = tmp_path / "limits.json"
    path.write_text(json.dumps({"max_response_bytes": 8192}))
    with running(tmp_path, max_connections=3) as (server, _service, session):
        assert (
            cli.main(
                [
                    "control",
                    "get_rpc_stats",
                    "--socket",
                    str(server.socket_path),
                    "--json",
                    "--limits-file",
                    str(path),
                ]
            )
            == 0
        )
        stats = json.loads(capsys.readouterr().out)
        assert stats["limits"]["max_connections"] == 3 and stats["mode"] == "headless"
        assert session.frame.step == 0
    path.write_text('{"max_connections": 0}')
    assert cli.main(["control", "get_rpc_stats", "--json", "--limits-file", str(path)]) == 2
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "invalid_params"
