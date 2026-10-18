"""Real workspace windows preserve activated menu hover and command routing."""

from pathlib import Path

import numpy as np
import pytest
from imgui_bundle import imgui
from PIL import Image

from mojive import Scene
from mojive.app.composition import build_scene
from mojive.config import LayoutConfig, ViewerConfig

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("language,scale", (("en", 1), ("zh_CN", 2)))
def test_main_menu_hover_switches_and_native_submenus_commit(
    language, scale, backend_name, monkeypatch, tmp_path
):
    monkeypatch.setenv("MOJIVE_SETTINGS", str(tmp_path / "settings.json"))
    monkeypatch.setenv("MOJIVE_UI_SCALE", str(scale))
    scene = Scene()
    scene.box(name="Menu interaction")
    with build_scene(
        scene,
        renderer=backend_name,
        width=1100 * scale,
        height=750 * scale,
        show_window=False,
        config=ViewerConfig(
            layout=LayoutConfig(persistence=False), debug_server=False, threaded_physics=False
        ),
    ) as viewer:
        viewer.app.set_language(language)
        events = []
        process = viewer.window._input.process_inputs

        def inputs():
            process()
            io = imgui.get_io()
            for method, args in events:
                getattr(io, method)(*args)
            events.clear()
            io.delta_time = 1 / 60

        monkeypatch.setattr(viewer.window._input, "process_inputs", inputs)

        def frame(count=3):
            for _ in range(count):
                viewer.sync()

        def event(method, *args):
            events.append((method, args))
            frame(1)

        def hover(rect):
            x, y, width, height = rect
            event("add_mouse_pos_event", x + width / 2, y + height / 2)
            frame(4)

        def click(rect):
            hover(rect)
            event("add_mouse_button_event", 0, True)
            event("add_mouse_button_event", 0, False)
            frame()

        def capture(name):
            destination = Path("output/ui-customization/menus") / backend_name
            destination.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.asarray(viewer.window.read_frame())[::-1]).save(
                destination / f"{name}-{language}-{scale}x.png"
            )

        frame(8)
        menus = viewer.app.workspace.menus
        hover(menus.menu_rects["File"])
        assert not menus.observed
        click(menus.menu_rects["File"])
        assert set(menus.observed) == {"File"}
        for name in ("Window", "Edit", "Help", "View", "File", "Window"):
            hover(menus.menu_rects[name])
            assert set(menus.observed) == {name}
        capture("hover-window")
        hover(menus.menu_rects["View"])
        projection = next(
            row for row in menus.observed["View"]["rows"] if row["label"] == "Projection"
        )
        hover(projection["rect"])
        frame(8)
        assert "View/Projection" in menus.observed
        capture("hover-projection")
        orthographic = next(
            row
            for row in menus.observed["View/Projection"]["rows"]
            if row["label"] == "Orthographic"
        )
        click(orthographic["rect"])
        assert viewer.app.camera.orthographic
        assert not menus.observed
        hover(menus.menu_rects["Help"])
        assert not menus.observed
        click(menus.menu_rects["Help"])
        assert "Help" in menus.observed
        event("add_key_event", imgui.Key.escape, True)
        event("add_key_event", imgui.Key.escape, False)
        frame()
        hover(menus.menu_rects["File"])
        assert not menus.observed
