"""Muted mouse outlines retain readable contrast with unchanged canonical geometry."""

from copy import deepcopy
from math import floor
from pathlib import Path

import pytest
from imgui_bundle import imgui
from PIL import Image

from mojive.app.composition import build_workspace
from mojive.assets import resolve
from mojive.config import LayoutConfig, ViewerConfig
from mojive.design.themes import select_theme
from mojive.tools.native_ui_review import _NativeInput
from mojive.tools.native_workspace import capture
from mojive.ui.viewport_widgets import hints
from mojive.ui.workspace.widget_icons import GLYPHS

pytestmark = pytest.mark.gpu
OUTPUT = Path(__file__).resolve().parents[2] / "output/status-mouse-clarity"


@pytest.mark.parametrize(
    "scale,theme,canvas",
    ((1, "dark", 22), (1.25, "dark", 22), (2, "dark", 22), (1.25, "light", 22)),
)
def test_muted_mouse_outline_pixels_preserve_the_shared_geometry(
    backend_name, monkeypatch, tmp_path, scale, theme, canvas
):
    monkeypatch.setenv("MOJIVE_SETTINGS", str(tmp_path / "settings.json"))
    monkeypatch.setenv("MOJIVE_DESIGN_SPEC", str(tmp_path / "design.json"))
    monkeypatch.setenv("MOJIVE_UI_SCALE", str(scale))
    target = OUTPUT / f"{backend_name}-{scale:g}-{theme}-{canvas}"
    target.mkdir(parents=True, exist_ok=True)
    with (
        build_workspace(
            resolve("joint_gizmo"),
            renderer=backend_name,
            show_window=False,
            width=1800,
            height=1100,
            vsync=False,
            config=ViewerConfig(
                layout=LayoutConfig(persistence=False), debug_server=False, threaded_physics=False
            ),
        ) as viewer,
        _NativeInput(viewer) as inputs,
    ):
        app = viewer.app
        app.set_language("en")
        spec = deepcopy(app.design.spec)
        spec["parts"]["status_hints"] = {"iconSize": canvas}
        app.design.update_spec(spec)
        select_theme(app.design, theme, persist=False)
        calls = []
        original = hints.draw_packaged_icon

        def record(draw, center, size, name, color, **options):
            window = imgui.get_current_context().current_window
            if (
                window is not None
                and "application_status" in window.name
                and name.startswith("status-mouse-")
            ):
                calls.append((center, size, name, color, options["stroke_color"]))
            return original(draw, center, size, name, color, **options)

        with monkeypatch.context() as patch:
            patch.setattr(hints, "draw_packaged_icon", record)
            inputs.sync(10)
            calls.clear()
            image_path = target / "status.png"
            capture(viewer, image_path)
        mouse_calls = calls[-3:]
        assert {item[2] for item in mouse_calls} == {
            "status-mouse-left",
            "status-mouse-right",
            "status-mouse-wheel",
        }
        window = imgui.internal.find_window_by_name("Status###application_status")
        with Image.open(image_path) as image:
            density = image.width / imgui.get_io().display_size.x
            bounds = (*window.pos, window.pos.x + window.size.x, window.pos.y + window.size.y)
            bar = image.crop(tuple(round(value * density) for value in bounds))
            background = bar.getpixel((2, bar.height // 2))[:3]
            for center, size, _name, fill, shell in mouse_calls:
                assert size == pytest.approx(canvas * scale)
                assert shell[:3] == fill[:3]
                assert shell[3] == pytest.approx(fill[3] * 0.8)
                # The canonical lower-left wall is flat at grid (5, 14), away from fills.
                wall = (center[0] - window.pos.x - 7 * size / 24) * density
                x = floor(wall)
                y = floor((center[1] - window.pos.y + 2 * size / 24) * density)
                stroke = next(
                    c["width"] for c in GLYPHS[_name]["contours"] if c["stroke"] != "none"
                )
                half = stroke * size / 48 * density
                fringe = min(1, stroke * size / 48) * density
                # Below one pixel, a nearest pixel center can land in the authored AA fringe.
                coverage = min(1, max(0, (half + fringe - abs(x + 0.5 - wall)) / fringe))
                alpha = shell[3] * coverage
                expected = tuple(
                    round(bg * (1 - alpha) + fg * 255 * alpha)
                    for bg, fg in zip(background, shell[:3], strict=True)
                )
                assert bar.getpixel((x, y))[:3] == pytest.approx(expected, abs=2)
            bar.save(image_path)
            bar.crop((0, 0, min(850, bar.width), bar.height)).save(target / "hints.png")
