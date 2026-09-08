"""Local RPC ownership and execution deadlines without a graphics context."""

from __future__ import annotations

import socket
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from mojive import commands as cmd
from mojive.adapters.toy import ToyPhysicsAdapter
from mojive.control_rpc import (
    PROTOCOL_VERSION,
    ControlServer,
    ControlService,
    RpcClient,
    RpcError,
    ViewerControlService,
)
from mojive.session import Session
from mojive.ui.camera import OrbitCamera


def test_async_application_result_is_resolved_by_the_socket_worker(tmp_path):
    pending = Future()
    app = SimpleNamespace(request_capture_async=lambda *args, **kwargs: pending)
    service = ControlService(ToyPhysicsAdapter(), app=app)
    with ControlServer(tmp_path / "async.sock", service) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with (
                RpcClient(server.socket_path) as client,
                ThreadPoolExecutor(max_workers=1) as worker,
            ):
                result = worker.submit(client.call, "capture_viewport")
                pending.set_exception(OSError("Capture destination is unavailable"))
                with pytest.raises(RpcError, match="Capture destination") as error:
                    result.result(timeout=2)
                assert error.value.code == "capture_failed"
                assert client.hello()["viewer_attached"]
        finally:
            server.shutdown()
            thread.join(timeout=2)
            service.close()


pytestmark = pytest.mark.integration


@pytest.fixture
def service():
    value = ControlService(ToyPhysicsAdapter())
    value.session.submit(cmd.Pause())
    yield value
    value.close()


def test_rpc_preserves_existing_file_and_symlink(tmp_path, service):
    path = tmp_path / "document.txt"
    path.write_text("user document")
    link = tmp_path / "control.sock"
    link.symlink_to(path)
    for target in (path, link):
        with pytest.raises(FileExistsError, match="not a socket"):
            ControlServer(target, service)
        assert path.read_text() == "user document"
        assert link.is_symlink()


def test_rpc_rejects_an_active_socket_and_preserves_its_owner(tmp_path, service):
    path = tmp_path / "control.sock"
    with ControlServer(path, service):
        with pytest.raises(FileExistsError, match="already in use"):
            ControlServer(path, service)
        assert path.is_socket()
    assert not path.exists()


def test_rpc_reclaims_a_stale_socket(tmp_path, service):
    path = tmp_path / "stale.sock"
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as stale:
        stale.bind(str(path))
    with ControlServer(path, service) as server:
        assert server.socket.getsockname() == str(path)
    assert not path.exists()


def test_rpc_close_preserves_a_replacement_socket(tmp_path, service):
    path = tmp_path / "control.sock"
    first = ControlServer(path, service)
    path.unlink()
    try:
        with ControlServer(path, service):
            first.server_close()
            assert path.is_socket()
    finally:
        first.server_close()


def request(deadline):
    return {
        "version": PROTOCOL_VERSION,
        "id": 1,
        "method": "step",
        "params": {"count": 3},
        "deadline": deadline,
    }


def test_headless_service_rejects_expired_mutations(service):
    response = service.handle(request(time.monotonic() - 1))
    assert response["error"]["code"] == "deadline_exceeded"
    assert service.session.frame.step == 0


@pytest.mark.parametrize("deadline", [True, "1", float("nan"), float("inf")])
def test_service_rejects_invalid_deadlines(service, deadline):
    response = service.handle(request(deadline))
    assert response["error"]["code"] == "invalid_request"
    assert service.session.frame.step == 0


@pytest.fixture
def viewer_service():
    session = Session(ToyPhysicsAdapter())
    session.submit(cmd.Pause())
    service = ViewerControlService(
        SimpleNamespace(session=session, app=SimpleNamespace(camera=OrbitCamera()))
    )
    yield service, session
    service.close()
    session.release()


def test_disconnected_client_does_not_leave_a_late_step(tmp_path, viewer_service):
    service, session = viewer_service
    server = ControlServer(tmp_path / "viewer.sock", service)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with RpcClient(server.socket_path, timeout=0.05) as client:
            with pytest.raises(RpcError) as error:
                client.step(3)
            assert error.value.code in {"timeout", "deadline_exceeded"}
        assert session.frame.step == 0
        # The UI resumes after the client's request deadline.
        assert service.pump() == 1
        assert session.frame.step == 0
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_started_request_reports_unknown_completion_instead_of_cancellation(viewer_service):
    service, session = viewer_service
    started, finish = threading.Event(), threading.Event()
    dispatch = service._core.dispatch

    def delayed(method, params):
        started.set()
        assert finish.wait(2)
        return dispatch(method, params)

    service._core.dispatch = delayed
    responses = []
    worker = threading.Thread(
        target=lambda: responses.append(service.handle(request(time.monotonic() + 0.2)))
    )
    worker.start()
    # Wait for queue insertion, without consuming the request on this thread.
    pending = service._pending.get(timeout=1)
    service._pending.put(pending)
    pump = threading.Thread(target=service.pump)
    pump.start()
    try:
        assert started.wait(1)
        worker.join(timeout=1)
        assert responses[0]["error"]["code"] == "completion_unknown"
        assert session.frame.step == 0
    finally:
        finish.set()
        pump.join(timeout=2)
        worker.join(timeout=2)
    assert session.frame.step == 3


def test_closing_viewer_service_releases_queued_requests(viewer_service):
    service, session = viewer_service
    responses = []
    worker = threading.Thread(target=lambda: responses.append(service.handle(request(None))))
    worker.start()
    pending = service._pending.get(timeout=1)
    service._pending.put(pending)
    service.close()
    worker.join(timeout=1)
    assert responses[0]["error"]["code"] == "unavailable"
    assert session.frame.step == 0
