"""Disabled rails dim without revealing the track through their round handle."""

from pathlib import Path

import numpy as np
import pytest
from imgui_bundle import imgui
from PIL import Image

from mojive.app.ui.window import create_window
from mojive.ui.imgui_draw import ImguiDraw2D
from mojive.ui.panels.value_cards import draw_value_rail
from mojive.ui.theme import THEME
from mojive.ui.window import WindowConfig

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("scale", (0.65, 1.0, 1.5, 2.5))
def test_disabled_handle_stays_opaque_and_preserves_rail_contrast(backend_name, scale):
    with create_window(
        WindowConfig(
            width=round(340 * scale),
            height=round(144 * scale),
            ui_scale=scale,
            vsync=False,
            docking=False,
            ini_path="",
            show_on_start=False,
        ),
        backend_name,
    ) as window:
        for _ in range(3):
            window.begin_frame()
            draw = ImguiDraw2D(imgui.get_foreground_draw_list())
            draw.rect_filled((0, 0), window.size_points, THEME.bg_window)
            for row, (name, hover, press, disabled) in enumerate(
                (
                    ("Normal", False, False, False),
                    ("Hover", True, False, False),
                    ("Pressed", True, True, False),
                    ("Disabled", False, False, True),
                )
            ):
                y = (25 + row * 32) * scale
                draw.text((8 * scale, y - 7 * scale), THEME.text, name)
                imgui.begin_disabled(disabled)
                draw_value_rail(
                    draw,
                    (100 * scale, y),
                    (300 * scale, y),
                    220 * scale,
                    5 * scale,
                    THEME,
                    scale,
                    hovered=hover,
                    pressed=press,
                    alpha=imgui.get_style().alpha,
                    disabled=disabled,
                )
                imgui.end_disabled()
            pixels = window.end_frame(readback=True)[::-1].copy()

        def sample(x, y):
            return pixels[round(y * window.pixel_scale), round(x * window.pixel_scale), :3]

        for row in range(4):
            y = (25 + row * 32) * scale
            center = sample(220 * scale, y).astype(int)
            shaft = sample(170 * scale, y).astype(int)
            off_shaft = sample(220 * scale, y + 1.5 * scale).astype(int)
            assert np.all(center > shaft)
            assert np.max(np.abs(center - off_shaft)) <= 1
        assert np.all(sample(220 * scale, 121 * scale) < sample(220 * scale, 25 * scale))
        output = Path("output/value-rail-disabled")
        output.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pixels).save(output / f"states-{backend_name}-{scale}.png")
