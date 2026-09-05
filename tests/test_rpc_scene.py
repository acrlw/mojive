"""Scene control validation and capture ownership without a physics engine."""

from __future__ import annotations

import threading

import numpy as np
import pytest

from mojive import Scene
from mojive.adapters.static import StaticSceneAdapter
from mojive.control_rpc import ControlService
from mojive.render.backend import RenderFlag


@pytest.fixture
def service():
    scene = Scene()
    scene.box()
    value = ControlService(StaticSceneAdapter(scene))
    yield value
    value.close()


@pytest.mark.parametrize("value", ["false", "true", 0, 1, None, [], {}])
def test_invalid_visibility_cannot_mutate_scene(service, value):
    response = service.handle(
        {
            "version": 1,
            "id": "visibility",
            "method": "set_visible",
            "params": {"node_id": 1, "visible": value},
        }
    )
    assert response["id"] == "visibility"
    assert response["error"]["code"] == "invalid_params"
    assert service.session.node(1).visible is True


@pytest.mark.parametrize("params", [{}, {"node_id": 1}, {"visible": False}])
def test_missing_visibility_parameters_report_client_error(service, params):
    response = service.handle(
        {
            "version": 1,
            "method": "set_visible",
            "params": params,
        }
    )
    assert response["error"]["code"] == "invalid_params"
    assert service.session.node(1).visible is True


@pytest.mark.parametrize("dimension", [0, -1, 1.5, True, "96", None])
def test_capture_rejects_invalid_dimensions_before_starting_graphics(service, dimension):
    response = service.handle(
        {
            "version": 1,
            "method": "capture",
            "params": {"width": dimension},
        }
    )
    assert response["error"]["code"] == "invalid_params"
    assert service.application._capture_service._renderer is None


def test_capture_flags_do_not_require_mujoco_and_reject_string_booleans(service):
    response = service.handle(
        {
            "version": 1,
            "method": "set_render_flag",
            "params": {"name": "mjRND_SHADOW", "enabled": "false"},
        }
    )
    assert response["error"]["code"] == "invalid_params"
    assert not service.application._capture_service.flags
    service.dispatch("set_render_flag", {"name": "shadow", "enabled": False})
    service.dispatch("set_visualization_flag", {"name": "flex_edge", "enabled": True})
    assert service.application._capture_service.flags == {
        RenderFlag.SHADOW: False,
        RenderFlag.FLEXEDGE: True,
    }


def test_capture_reuses_resources_and_releases_on_the_graphics_thread(
    service, monkeypatch, tmp_path
):
    events = []

    class Renderer:
        def __init__(self, source, *, width, height):
            self.width, self.height = width, height
            self.record("create")

        def record(self, operation):
            events.append((operation, threading.get_ident()))

        def set_scene(self, source):
            self.record("structure")

        def update(self, frame, *, camera):
            self.record("update")

        def render(self, *, product):
            self.record("render")
            return np.zeros((self.height, self.width, 3), np.uint8)

        def resize(self, width, height):
            self.width, self.height = width, height
            self.record("resize")

        def close(self):
            self.record("close")

    monkeypatch.setattr("mojive.session_capture.SceneRenderer", Renderer)
    params = {"width": 32, "height": 24, "output": str(tmp_path / "capture.png")}
    service.dispatch("capture", params)
    service.dispatch("capture", params)
    service.dispatch("capture", {**params, "width": 48})
    service.dispatch("set_visible", {"node_id": 1, "visible": False})
    service.dispatch("capture", params)
    service.close()
    names = [name for name, _ in events]
    assert names.count("create") == names.count("close") == 1
    assert names.count("structure") == 1
    assert names.count("resize") == 2
    assert names.count("render") == 4
    threads = {thread for _, thread in events}
    assert len(threads) == 1 and threading.get_ident() not in threads
