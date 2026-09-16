"""RPC backlog leaves real Viewer frames available for keyboard interaction."""

from concurrent.futures import Future

import pytest
from imgui_bundle import imgui

from mojive import Scene, build_scene
from mojive import commands as cmd
from mojive.control.rpc import RpcLimits
from mojive.tools.ui_runtime import _click

pytestmark = pytest.mark.gpu


def test_backlog_keeps_keyboard_editing_live_and_drains(tmp_path, monkeypatch):
    monkeypatch.setenv("MOJIVE_SETTINGS", str(tmp_path / "settings.json"))
    scene = Scene()
    box = scene.box()
    with build_scene(scene, width=960, height=640, vsync=False, show_window=False) as viewer:
        for _ in range(12):
            viewer.sync()
        x, y, width, height = viewer.app._viewport_rect
        _click(viewer, (x + width * 0.5, y + height * 0.5))
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
