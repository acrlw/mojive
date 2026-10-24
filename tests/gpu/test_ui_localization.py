"""Hidden production frames verify localized empty states and live language changes."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mojive.app.composition import build_editor
from mojive.config import LayoutConfig, ViewerConfig
from mojive.ui.imgui_draw import ImguiDraw2D

pytestmark = [pytest.mark.gpu, pytest.mark.physics]
OUTPUT = Path(__file__).resolve().parents[2] / "output/ui-localization"


@pytest.mark.parametrize("scale", (1.0, 1.25))
def test_empty_workspace_switches_language_without_stale_text(
    backend_name, monkeypatch, tmp_path, scale
):
    monkeypatch.setenv("MOJIVE_SETTINGS", str(tmp_path / "settings.json"))
    monkeypatch.setenv("MOJIVE_UI_SCALE", str(scale))
    viewer = build_editor(
        renderer=backend_name,
        show_window=False,
        width=1440,
        height=900,
        vsync=False,
        config=ViewerConfig(
            layout=LayoutConfig(persistence=False), debug_server=False, threaded_physics=False
        ),
    )
    painted = []
    original = ImguiDraw2D.text

    def text(draw, position, color, value, **options):
        painted.append(value)
        return original(draw, position, color, value, **options)

    monkeypatch.setattr(ImguiDraw2D, "text", text)
    try:
        for language in ("en", "zh_CN", "en"):
            viewer.app.set_language(language)
            for _ in range(8):
                painted.clear()
                viewer.sync()
            expected = (
                "打开场景或添加对象以开始编辑。"
                if language == "zh_CN"
                else "Open a scene or add an object to begin."
            )
            assert expected in painted
            if language == "zh_CN":
                assert "Open a scene or add an object to begin." not in painted
            OUTPUT.mkdir(parents=True, exist_ok=True)
            pixels = viewer.window.read_frame()
            assert pixels is not None
            Image.fromarray(np.asarray(pixels)[::-1]).save(
                OUTPUT / f"empty-{language}-{scale:g}-{backend_name}.png"
            )
    finally:
        viewer.release()
