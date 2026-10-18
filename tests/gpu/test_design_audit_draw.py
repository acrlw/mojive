"""Post-submit audit samples retain clipping, glyph ink and atlas-independent bitmaps."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from imgui_bundle import imgui

from mojive import Scene
from mojive.app.composition import build_scene
from mojive.config import LayoutConfig, ViewerConfig
from mojive.tools.design_audit_draw import NativeDrawSamples

pytestmark = pytest.mark.gpu


def test_post_submit_samples_observe_header_fonts_and_menu_ink(backend_name, monkeypatch, tmp_path):
    monkeypatch.setenv("MOJIVE_SETTINGS", str(tmp_path / "settings.json"))
    monkeypatch.setenv("MOJIVE_DESIGN_SPEC", str(tmp_path / "design.json"))
    with build_scene(
        Scene(),
        renderer=backend_name,
        width=1400,
        height=900,
        show_window=False,
        config=ViewerConfig(layout=LayoutConfig(persistence=False), debug_server=False),
    ) as viewer:
        viewer.app.set_language("en")
        runtime, ui = viewer.app.design, viewer.app.workspace.ui
        runtime.observe_components = True

        def frame():
            for _ in range(5):
                viewer.sync()

        def samples():
            scale = (
                tuple(imgui.get_draw_data().framebuffer_scale)
                if backend_name == "opengl"
                else (1, 1)
            )
            sampler = NativeDrawSamples(
                SimpleNamespace(_editor_window=lambda window: False), clip_scale=scale
            )
            records = list(ui.design_primitives.observed) + list(runtime.observed)
            return {
                row["key"]: sampler.sample({**row, "visibleHit": row["hit"]})[0]
                for row in records
                if row["key"] in ("menu-File", "panel-header-Scene")
            }

        frame()
        before = samples()
        size_before = tuple((tex.width, tex.height) for tex in imgui.get_io().fonts.tex_list)
        assert set(before) == {"menu-File", "panel-header-Scene"}
        assert all(np.count_nonzero(rows[:, 3]) > 0 for rows in before.values())
        initial_spec = deepcopy(runtime.spec)
        spec = deepcopy(initial_spec)
        spec["parts"]["menu_trigger"] = {"textColor": "#e8b04fff"}
        spec["parts"]["panel_header"] = {"fontFamily": "mono"}
        runtime.update_spec(spec)
        frame()
        after = samples()
        assert all(not np.array_equal(before[key], after[key]) for key in before)
        glyph = after["menu-File"][after["menu-File"][:, 3].astype(bool)]
        assert set(glyph[:, 2].astype(np.uint32) & 0xFFFFFF) == {0x4FB0E8}
        # First use of a larger face grows the atlas after the baseline, as a
        # workbench family edit can. Existing glyphs must still compare equal.
        font = ui.mono
        baked = font.get_font_baked(160)
        for code in range(33, 127):
            baked.find_glyph(code)
        runtime.update_spec(initial_spec)
        frame()
        restored = samples()
        size_after = tuple((tex.width, tex.height) for tex in imgui.get_io().fonts.tex_list)
        assert size_after != size_before
        for key in before:
            np.testing.assert_array_equal(restored[key], before[key])
