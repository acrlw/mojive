"""Exercise the shared camera, joint, geometry and menu specimens through real input."""

import json
from contextlib import closing
from copy import deepcopy
from dataclasses import fields

import numpy as np
import pytest
from examples.ui_design.native.design_guide import DesignGuide
from examples.ui_design.native.guide_capture import capture_interactions
from PIL import Image

from mojive.app.ui.window import create_window
from mojive.render.geometry import geometry_view
from mojive.scene.state import camera_bookmark
from mojive.ui.window import WindowConfig

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("language", ("zh_CN", "en"))
def test_context_capture_uses_shared_controls_and_restores_sample(backend_name, tmp_path, language):
    with (
        create_window(
            WindowConfig(width=1440, height=1000, docking=False, ini_path="", show_on_start=False),
            backend_name,
        ) as window,
        closing(DesignGuide(window, tmp_path, language=language)) as guide,
    ):
        guide.animate = False
        for _ in range(4):
            guide.frame()
        app = guide.sample.app
        session = app.session
        panel = app.panels.get("camera")
        original = {
            "page": guide.page,
            "values": deepcopy(guide.values),
            "selected": deepcopy(guide.selected),
            "selection": session.selected,
            "camera": camera_bookmark(app.camera, app.camera.view()),
            "geometry": geometry_view(app.backend),
            "preferences": deepcopy(app.preferences.values),
            "bookmarks": panel.bookmark_directory,
            "joint": next(joint for joint in session.joints if joint.name == "hinge_limited"),
            "physics": session.adapter.capture_state(),
            "can_undo": session.can_undo,
            "sensor": app.panels.get("Sensors").sensor_index,
        }
        capture_interactions(guide, context_only=True)
        report = json.loads((tmp_path / "context-interactions.json").read_text())
        assert len(report["directions"]) == 7
        assert len(report["bookmarks"]) == 2
        assert report["geometry"] == ["default", "visual", "collision", "both"]
        assert [sensor["dimension"] for sensor in report["sensors"]] == [3, 14]
        names = (
            "camera-directions",
            "camera-bookmarks-empty",
            "camera-bookmarks-populated",
            "camera-bookmarks-menu",
            "camera-bookmarks-narrow",
            "menu-add",
            "joint-unlimited",
            "joint-limited",
            "color-picker-edited",
            "color-picker-restored",
            "inertia-automatic",
            "sensors-xyz",
            "sensors-components",
            *("content-geometry-" + value for value in report["geometry"]),
            *("scene-geometry-" + value for value in report["geometry"]),
        )
        for name in names:
            with Image.open(tmp_path / (name + ".png")) as image:
                assert image.width >= 1440 and image.height >= 1000
        assert guide.page == original["page"]
        assert guide.values == original["values"]
        assert guide.selected == original["selected"]
        assert session.selected == original["selection"]
        assert geometry_view(app.backend) == original["geometry"]
        assert app.preferences.values == original["preferences"]
        assert panel.bookmark_directory == original["bookmarks"]
        assert not list(tmp_path.glob("camera-capture-*"))
        assert not panel.bookmark_directory.exists()
        current = next(joint for joint in session.joints if joint.name == "hinge_limited")
        assert current.limited == original["joint"].limited
        assert current.range == original["joint"].range
        assert session.can_undo == original["can_undo"]
        assert app.panels.get("Sensors").sensor_index == original["sensor"]
        restored_camera = camera_bookmark(app.camera, app.camera.view())
        for key in ("eye", "target", "up", "yaw", "pitch", "distance", "fov_y"):
            np.testing.assert_allclose(restored_camera[key], original["camera"][key], atol=1e-6)
        restored_state = session.adapter.capture_state()
        for field in fields(original["physics"]):
            np.testing.assert_allclose(
                getattr(restored_state, field.name), getattr(original["physics"], field.name)
            )
