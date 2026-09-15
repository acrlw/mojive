"""The product dock layout adapts to scale and preserves saved personal layouts."""

from pathlib import Path

import pytest
from imgui_bundle import imgui
from PIL import Image

from mojive.app.composition import build
from mojive.config import LayoutConfig, ViewerConfig
from mojive.scene.assets import resolve
from mojive.tools.ui_runtime import _click, _item_center, _open_main_menu

pytestmark = [pytest.mark.gpu, pytest.mark.physics]


def _settle(viewer):
    for _ in range(10):
        viewer.sync()


def _assert_default_layout(viewer):
    windows = {
        name: imgui.internal.find_window_by_name(name)
        for name in (
            "Hierarchy",
            "Viewport",
            "Control",
            "Joints",
            "Camera",
            "Inspector",
            "Keyframes",
            "Output",
        )
    }
    assert all(window is not None and window.dock_node is not None for window in windows.values())
    hierarchy, viewport, control, inspector, timeline = (
        windows[name].dock_node
        for name in ("Hierarchy", "Viewport", "Control", "Inspector", "Keyframes")
    )
    root = imgui.internal.dock_builder_get_node(viewer.window.dockspace_id)
    assert timeline.pos.x == hierarchy.pos.x == root.pos.x
    assert hierarchy.pos.y == viewport.pos.y == control.pos.y
    assert timeline.pos.x + timeline.size.x == viewport.pos.x + viewport.size.x
    assert hierarchy.pos.y + hierarchy.size.y == viewport.pos.y + viewport.size.y
    assert timeline.pos.y > viewport.pos.y + viewport.size.y
    assert control.pos.x == inspector.pos.x > timeline.pos.x + timeline.size.x
    assert inspector.pos.y + inspector.size.y == timeline.pos.y + timeline.size.y
    assert control.size.x / root.size.x == pytest.approx(0.23, abs=0.01)
    assert timeline.size.y / root.size.y == pytest.approx(0.28, abs=0.01)
    assert hierarchy.size.x / timeline.size.x == pytest.approx(0.23, abs=0.01)
    assert inspector.size.y / root.size.y == pytest.approx(0.52, abs=0.01)
    assert windows["Joints"].dock_node.id_ == windows["Camera"].dock_node.id_ == control.id_
    assert windows["Output"].dock_node.id_ == timeline.id_
    assert control.selected_tab_id == windows["Joints"].tab_id
    assert timeline.selected_tab_id == windows["Keyframes"].tab_id


@pytest.mark.parametrize(
    "width,height,scale", ((1600, 1000, 1.0), (2400, 1500, 1.5), (3840, 1978, 2.5))
)
def test_default_layout_and_reset_match_the_reviewed_dock_arrangement(
    backend_name, monkeypatch, tmp_path, width, height, scale
):
    monkeypatch.setenv("MOJIVE_UI_SCALE", str(scale))
    monkeypatch.setenv("MOJIVE_SETTINGS", str(tmp_path / "settings.json"))
    viewer = build(
        resolve("joint_gizmo"),
        renderer=backend_name,
        vsync=False,
        width=width,
        height=height,
        show_window=False,
    )
    try:
        _settle(viewer)
        _assert_default_layout(viewer)
        output = Path("output/default-layout")
        output.mkdir(parents=True, exist_ok=True)
        Image.fromarray(viewer.window.read_frame()[::-1, :, :3]).save(
            output / f"default-{backend_name}-{scale}.png"
        )
        hierarchy = imgui.internal.find_window_by_name("Hierarchy")
        imgui.internal.dock_context_process_undock_window(
            imgui.get_current_context(), hierarchy, True
        )
        _settle(viewer)
        assert hierarchy.dock_node is None
        _open_main_menu(viewer, "Window")
        _click(viewer, _item_center(viewer, "menu_item", "Reset Layout"))
        _settle(viewer)
        _assert_default_layout(viewer)
    finally:
        viewer.release()


def test_saved_personal_layout_takes_precedence(backend_name, monkeypatch, tmp_path):
    monkeypatch.setenv("MOJIVE_UI_SCALE", "1")
    monkeypatch.setenv("MOJIVE_SETTINGS", str(tmp_path / "settings.json"))
    path = tmp_path / "personal.ini"
    config = ViewerConfig(layout=LayoutConfig(path=path))
    for first in (True, False):
        viewer = build(
            resolve("joint_gizmo"),
            renderer=backend_name,
            config=config,
            width=1600,
            height=1000,
            show_window=False,
        )
        try:
            _settle(viewer)
            hierarchy = imgui.internal.find_window_by_name("Hierarchy")
            if first:
                imgui.internal.dock_context_process_undock_window(
                    imgui.get_current_context(), hierarchy, True
                )
                _settle(viewer)
                imgui.save_ini_settings_to_disk(str(path))
            assert hierarchy.dock_node is None
            assert imgui.internal.find_window_by_name("Viewport").dock_node is not None
        finally:
            viewer.release()
