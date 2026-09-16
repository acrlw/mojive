"""RPC backlog leaves real Viewer frames available for keyboard interaction."""

from concurrent.futures import Future

import pytest
from imgui_bundle import imgui

from mojive import Scene, build_scene
from mojive import commands as cmd
from mojive.control.rpc import RpcLimits

pytestmark = pytest.mark.gpu


def test_backlog_keeps_keyboard_editing_live_and_drains(tmp_path, monkeypatch):
    monkeypatch.setenv("MOJIVE_SETTINGS", str(tmp_path / "settings.json"))
    scene = Scene()
    box = scene.box()
    with build_scene(scene, width=960, height=640, vsync=False, show_window=False) as viewer:
        for _ in range(12):
            viewer.sync()
        imgui.internal.focus_window(imgui.internal.find_window_by_name("Viewport"))
        viewer.sync()
        viewer.session.submit(cmd.Select(box.object_id))
        server = viewer.start_rpc(
            tmp_path / "budget.sock",
            limits=RpcLimits(max_pending_requests=64, requests_per_pump=2),
        )
        service = server.service
        pending = [
            service.submit({"version": 1, "id": index, "method": "hello"}) for index in range(64)
        ]
        assert all(isinstance(result, Future) for result in pending)
        rejected = service.submit(
            {
                "version": 1,
                "id": 64,
                "method": "rename_scene_entity",
                "params": {"object_id": box.object_id, "name": "rejected"},
            }
        )
        assert rejected["error"]["code"] == "busy"
        io = imgui.get_io()
        io.add_key_event(imgui.Key.delete, True)
        viewer.sync()
        io.add_key_event(imgui.Key.delete, False)
        viewer.sync()
        assert viewer.session.source.instance_count == 0
        assert 0 < sum(result.done() for result in pending) <= 4
        assert service.stats.snapshot()["counters"]["queued_requests"] >= 60
        for _ in range(64):
            if all(result.done() for result in pending):
                break
            viewer.sync()
        assert all(result.done() and result.result()["error"] is None for result in pending)
        counts = service.stats.snapshot()["counters"]
        assert counts["queued_requests"] == counts["queued_bytes"] == 0
        assert counts["inflight_requests"] == 0 and counts["queue_rejected"] == 1
        assert counts["pump_budget_exhausted"] > 0


@pytest.mark.parametrize("method", ["load", "mujoco.set_model_source", "save_scene"])
def test_document_rpc_keeps_presentation_live_and_preserves_request_order(
    tmp_path, monkeypatch, method
):
    import threading
    import time

    from mojive.app.composition import build_editor
    from mojive.scene.assets import resolve

    monkeypatch.setenv("MOJIVE_SETTINGS", str(tmp_path / "settings.json"))
    entered, release = threading.Event(), threading.Event()
    with build_editor(width=960, height=640, vsync=False, show_window=False) as viewer:
        for _ in range(4):
            viewer.sync()
        service = viewer.start_rpc(tmp_path / "document.sock").service
        app = viewer.app
        load = app._load_model

        def blocked(command):
            entered.set()
            assert release.wait(5)
            return load(command)

        monkeypatch.setattr(app, "_load_model", blocked)
        params = {
            "load": {"path": str(resolve("joint_types"))},
            "mujoco.set_model_source": {
                "model_id": 0,
                "mjcf": '<mujoco><worldbody><geom size="0.2"/></worldbody></mujoco>',
            },
            "save_scene": {"path": str(tmp_path / "saved.mojive.json")},
        }[method]
        pending = service.submit({"version": 1, "id": 1, "method": method, "params": params})
        following = service.submit({"version": 1, "id": 2, "method": "get_state"})
        try:
            viewer.sync()
            assert entered.wait(2)
            before = app._frame_index
            for _ in range(3):
                viewer.sync()
            assert app._frame_index == before + 3
            assert not pending.done() and not following.done()
        finally:
            release.set()
        deadline = time.monotonic() + 10
        while not following.done() and time.monotonic() < deadline:
            viewer.sync()
        assert pending.done() and following.done()
        assert pending.result()["error"] is None
        assert following.result()["error"] is None
        assert pending.result()["result"]["document"]["id"] == viewer.session.document_id
        if method == "save_scene":
            assert (tmp_path / "saved.mojive.json").is_file()


def test_unstarted_document_rpc_is_completed_when_viewer_closes(tmp_path, monkeypatch):
    monkeypatch.setenv("MOJIVE_SETTINGS", str(tmp_path / "settings.json"))
    viewer = build_scene(Scene(), vsync=False, show_window=False)
    service = viewer.start_rpc(tmp_path / "close.sock").service
    future = service.submit(
        {
            "version": 1,
            "id": 1,
            "method": "save_scene",
            "params": {"path": str(tmp_path / "unused.mojive.json")},
        }
    )
    assert service.pump() == 1
    assert not future.done()
    viewer.release()
    assert future.done() and future.result()["error"]["code"] == "command_failed"
    assert not (tmp_path / "unused.mojive.json").exists()
