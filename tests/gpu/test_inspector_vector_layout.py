"""Native Inspector preserves translated captions across its responsive widths."""

from contextlib import closing
from pathlib import Path

import pytest
from examples.ui_design.native.design_guide import DesignGuide
from imgui_bundle import imgui
from PIL import Image

from mojive.app.ui.window import create_window
from mojive.ui.imgui_draw import ImguiDraw2D
from mojive.ui.window import WindowConfig
from mojive.ui.workspace.design import DESIGN
from mojive.ui.workspace.panel_host import panel

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("language", ("en", "zh_CN"))
def test_native_transform_captions_fit_at_default_narrow_and_wide_widths(
    backend_name, tmp_path, monkeypatch, language
):
    with (
        create_window(
            WindowConfig(width=1440, height=1000, docking=False, ini_path="", show_on_start=False),
            backend_name,
        ) as window,
        closing(DesignGuide(window, tmp_path, language=language)) as guide,
    ):
        guide.page, guide.animate = "Inspector", False
        for _ in range(3):
            guide.frame()
        width, layouts, painted = 320, [], []
        vector, text = guide.ui.vector, ImguiDraw2D.text

        def observe_vector(*args, **options):
            layouts.append(options["layout"])
            return vector(*args, **options)

        def observe_text(draw, position, color, value, **options):
            painted.append(value)
            return text(draw, position, color, value, **options)

        def native_panel(*_args):
            imgui.set_next_window_pos((100, 100))
            imgui.set_next_window_size((width * guide.s, 700 * guide.s))
            imgui.push_style_var(
                imgui.StyleVar_.window_padding,
                (DESIGN.panel_padding_x * guide.s, DESIGN.panel_padding_y * guide.s),
            )
            try:
                guide.sample._drawn.add("Inspector")
                panel(guide.sample.app.workspace, "Inspector", guide.sample.context)
            finally:
                imgui.pop_style_var()

        monkeypatch.setattr(guide.ui, "vector", observe_vector)
        monkeypatch.setattr(ImguiDraw2D, "text", observe_text)
        monkeypatch.setattr(guide, "draw_contents", native_panel)
        for width in (320, 220, 640):
            for _ in range(3):
                layouts.clear()
                painted.clear()
                pixels = guide.frame(readback=True)
            assert len(layouts) == 3 and all(layout == layouts[0] for layout in layouts)
            assert len({field[2] for field in layouts[0].fields}) == 1
            for label in ("Position", "Rotation", "Scale"):
                assert guide.tr(label) in painted
            assert "(m)" in painted and "(°)" in painted
            assert not any(value.endswith("…") for value in painted)
            if language == "zh_CN":
                output = Path("output/ui_design/parity-v5") / f"inspector-captions-{backend_name}"
                output.mkdir(parents=True, exist_ok=True)
                Image.fromarray(pixels[::-1]).save(output / f"width-{width}.png")
