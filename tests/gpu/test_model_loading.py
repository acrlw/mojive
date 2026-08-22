from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.gpu

pytest.importorskip("glfw")
pytest.importorskip("mujoco")

from forge_viewer.assets import resolve  # noqa: E402
from forge_viewer.commands import CommandResult  # noqa: E402
from forge_viewer.composition import build, build_scene  # noqa: E402
from forge_viewer.scene import Scene  # noqa: E402
from forge_viewer.ui.app import MODEL_FILTERS  # noqa: E402


@pytest.fixture(scope="module")
def viewer():
    instance = build(resolve("empty"), paused=True, vsync=False, width=960, height=640)
    try:
        yield instance
    finally:
        instance.release()


def _click(viewer, point: tuple[float, float]) -> None:
    from imgui_bundle import imgui

    io = imgui.get_io()
    io.add_mouse_pos_event(*point)
    viewer.sync()
    io.add_mouse_button_event(0, True)
    viewer.sync()
    io.add_mouse_button_event(0, False)
    viewer.sync()


def _item_center(viewer, function_name: str, label: str) -> tuple[float, float]:
    from imgui_bundle import imgui

    original = getattr(imgui, function_name)
    found = []

    def spy(item_label, *args, **kwargs):
        result = original(item_label, *args, **kwargs)
        if item_label == label:
            lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
            found.append(((lo.x + hi.x) * 0.5, (lo.y + hi.y) * 0.5))
        return result

    setattr(imgui, function_name, spy)
    try:
        viewer.sync()
    finally:
        setattr(imgui, function_name, original)
    assert found
    return found[-1]


def test_file_menu_opens_model_browser(viewer, monkeypatch):
    opened = []
    viewer.sync()
    monkeypatch.setattr(viewer.app, "_open_model_dialog", lambda: opened.append(True))
    _click(viewer, _item_center(viewer, "begin_menu", "File"))
    _click(viewer, _item_center(viewer, "menu_item", "Open Model (MJCF / URDF)..."))
    assert opened


def test_add_model_dialog_filters_formats_and_accepts_multiple_files(viewer, monkeypatch):
    from imgui_bundle import portable_file_dialogs

    paths = ["/tmp/robot.xml", "/tmp/arm.urdf"]
    opened = {}

    class Dialog:
        def ready(self, _timeout):
            return True

        def result(self):
            return paths

    def open_file(title, default_path, filters, options):
        opened.update(title=title, default_path=default_path, filters=filters, options=options)
        return Dialog()

    loaded = []
    monkeypatch.setattr(portable_file_dialogs, "open_file", open_file)
    monkeypatch.setattr(
        viewer.app,
        "add_model",
        lambda path: loaded.append(Path(path)) or CommandResult.good(),
    )

    viewer.app._open_model_dialog("add")
    viewer.app._poll_model_dialog()

    assert opened["title"] == "Add MJCF or URDF models"
    assert opened["filters"] == MODEL_FILTERS
    assert opened["options"] == portable_file_dialogs.opt.multiselect
    assert loaded == [Path(path) for path in paths]


def test_runtime_model_loading_rebuilds_gpu_scene(viewer):
    viewer.sync()
    assert viewer.backend.stats.instances == 0

    viewer.window._file_drag_active = True
    viewer.window._on_file_drop(None, [str(resolve("test_scene.xml"))])
    assert not viewer.window.file_drag_active
    viewer.sync()
    assert viewer.session.asset_path == resolve("test_scene.xml")

    for name in ("test_scene.urdf",):
        result = viewer.app.load_model(resolve(name))
        assert result.ok, result.message
        viewer.sync()
        assert viewer.session.asset_path == resolve(name)
        assert viewer.backend.stats.instances == viewer.session.source.instance_count
        assert viewer.backend.stats.instances > 0


def test_static_scene_file_menu_renders():
    instance = build_scene(Scene(), vsync=False, width=960, height=640)
    try:
        instance.sync()
        _click(instance, _item_center(instance, "begin_menu", "File"))
        _item_center(instance, "menu_item", "New Scene")
    finally:
        instance.release()
