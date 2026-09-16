"""Request admission, frame fairness and bounded diagnostics without sockets."""

import time
from concurrent.futures import Future
from dataclasses import fields
from types import SimpleNamespace

import pytest

from mojive import commands as cmd
from mojive.adapters.toy import ToyPhysicsAdapter
from mojive.control.operations import OPERATIONS
from mojive.control.rpc import ControlService, RpcLimits, ViewerControlService
from mojive.control.schema import Validator
from mojive.session import Session
from mojive.ui.camera import OrbitCamera


def request(method="step", params=None, **extra):
    return {"version": 1, "id": 1, "method": method, "params": params or {}, **extra}


@pytest.fixture
def make_service():
    owned = []

    def make(**limits):
        session = Session(ToyPhysicsAdapter())
        assert session.submit(cmd.Pause()).ok
        app = SimpleNamespace(camera=OrbitCamera())
        service = ViewerControlService(
            SimpleNamespace(session=session, app=app), limits=RpcLimits(**limits)
        )
        owned.append((service, session))
        return service, session

    yield make
    for service, session in owned:
        service.close()
        session.release()


@pytest.mark.parametrize("name", [item.name for item in fields(RpcLimits)])
@pytest.mark.parametrize("value", [0, -1, True, float("nan"), float("inf")])
def test_invalid_budgets_fail_at_construction(name, value):
    with pytest.raises(ValueError):
        RpcLimits(**{name: value})


def test_full_queue_rejects_without_writing_and_recovers_after_drain(make_service):
    service, session = make_service(max_pending_requests=1)
    first = service.submit(request())
    rejected = service.submit(request(params={"count": 10}))
    assert rejected["error"]["code"] == "busy"
    assert session.frame.step == 0
    assert service.pump() == 1
    assert first.result()["error"] is None and session.frame.step == 1
    second = service.submit(request())
    assert service.pump() == 1
    assert second.result()["error"] is None and session.frame.step == 2
    stats = service.stats.snapshot()["counters"]
    assert stats["queue_rejected"] == 1 and stats["queued_requests"] == 0
    assert stats["inflight_requests"] == 0 and stats["queued_bytes"] == 0


def test_queue_byte_budget_is_independent_of_item_count(make_service):
    service, session = make_service(max_pending_bytes=150)
    first = service.submit(request(), request_bytes=100)
    rejected = service.submit(request(), request_bytes=100)
    assert rejected["error"]["code"] == "busy"
    assert first.cancel()
    assert service.pump() == 1 and session.frame.step == 0
    assert service.stats.snapshot()["counters"]["queued_bytes"] == 0
    assert isinstance(service.submit(request(), request_bytes=100), Future)


def test_time_budget_leaves_valid_work_for_subsequent_frames(make_service, monkeypatch):
    service, session = make_service(requests_per_pump=8, pump_budget_ms=3)
    clock = [100.0]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    dispatch = service._core.dispatch

    def slow(method, params):
        result = dispatch(method, params)
        clock[0] += 0.002
        return result

    monkeypatch.setattr(service._core, "dispatch", slow)
    pending = [service.submit(request()) for _ in range(5)]
    assert service.pump() == 2 and session.frame.step == 2
    assert sum(value.done() for value in pending) == 2
    assert service.pump() == 2 and session.frame.step == 4
    assert service.pump() == 1 and session.frame.step == 5
    stats = service.stats.snapshot()
    assert stats["counters"]["pump_budget_exhausted"] == 2
    assert stats["timings"]["queue_wait_ms"]["maximum"] == pytest.approx(8)
    assert stats["timings"]["pump_ms"]["maximum"] == pytest.approx(4)


def test_count_budget_and_cancelled_work_do_not_starve_following_frames(make_service):
    service, session = make_service(requests_per_pump=1)
    first, second = service.submit(request()), service.submit(request())
    assert first.cancel()
    assert service.pump() == 1 and session.frame.step == 0 and not second.done()
    assert service.pump() == 1 and session.frame.step == 1


def test_deadline_and_disconnect_are_checked_immediately_before_execution(
    make_service, monkeypatch
):
    service, session = make_service()
    clock = [100.0]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    expired = service.submit(request(deadline=101.0))
    disconnected = service.submit(request(), disconnected=lambda: True)
    clock[0] = 102.0
    assert service.pump() == 2
    assert expired.cancelled() and disconnected.cancelled() and session.frame.step == 0
    assert service.stats.snapshot()["counters"]["expired"] == 1


@pytest.mark.parametrize("viewer", [False, True])
def test_async_work_retains_capacity_until_actual_completion(make_service, viewer):
    source = Future()
    if viewer:
        service, _session = make_service(max_inflight_requests=1)
        service._core.application.app.request_capture_async = lambda *args, **kwargs: source
    else:
        service = ControlService(
            ToyPhysicsAdapter(),
            limits=RpcLimits(max_inflight_requests=1),
            app=SimpleNamespace(request_capture_async=lambda *args, **kwargs: source),
        )
    try:
        pending = service.submit(request("capture_viewport"))
        if viewer:
            assert service.pump() == 1
        assert not pending.cancel()  # Execution already started, including asynchronous capture.
        assert service.submit(request("hello"))["error"]["code"] == "busy"
        source.set_result({"path": "finished"})
        assert pending.result()["result"] == {"path": "finished"}
        assert service.stats.snapshot()["counters"]["inflight_requests"] == 0
        following = service.submit(request("hello"))
        if viewer:
            service.pump()
            following = following.result()
        assert following["error"] is None
    finally:
        if not viewer:
            service.close()


def test_diagnostics_are_bounded_and_do_not_change_the_document(make_service):
    service, session = make_service()
    before = session.document_revision
    for _ in range(300):
        service.submit(request("hello"))
        service.pump()
    pending = service.submit(request("get_rpc_stats"))
    service.pump()
    result = pending.result()["result"]
    Validator(OPERATIONS["get_rpc_stats"].output_schema).validate(result)
    assert result["sample_limit"] == 256
    assert result["timings"]["handler_ms"]["count"] == 256
    assert result["counters"]["started"] == 301
    assert result["limits"]["requests_per_pump"] == 8
    assert session.document_revision == before
    assert not service.stats.snapshot()["counters"]["inflight_requests"]


def test_deferred_document_write_holds_following_requests_and_preserves_failure(
    make_service, monkeypatch
):
    service, session = make_service()
    source = Future()
    dispatch = service._core.dispatch

    def deferred(method, params):
        return source if method == "save_scene" else dispatch(method, params)

    monkeypatch.setattr(service._core, "dispatch", deferred)
    first = service.submit(request("save_scene", {"path": "unused"}))
    following = service.submit(request())
    assert service.pump() == 1
    assert service.pump() == 0 and session.frame.step == 0
    from mojive.control.errors import ControlError

    source.set_exception(ControlError("command_failed", "save fixture failed"))
    assert first.result()["error"] == {"code": "command_failed", "message": "save fixture failed"}
    assert service.pump() == 1 and session.frame.step == 1
    assert following.result()["error"] is None
