"""HiDPI window and overlay scaling regressions."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.gpu

pytest.importorskip("glfw")

from forge_viewer.composition import build_scene  # noqa: E402
from forge_viewer.scene import Scene  # noqa: E402
from forge_viewer.ui import viewcube  # noqa: E402


def test_view_gizmo_and_font_share_the_explicit_ui_scale(monkeypatch):
    from imgui_bundle import imgui

    monkeypatch.setenv("FORGE_VIEWER_UI_SCALE", "2")
    viewer = build_scene(Scene(), vsync=False, width=960, height=640)
    try:
        viewer.sync()
        scale = viewer.window.style_scale

        assert scale == pytest.approx(2.0)
        assert viewer.window.font_report.size_pt == pytest.approx(
            viewer.window.config.font_size_pt * scale
        )
        assert imgui.get_style().font_scale_dpi == pytest.approx(1.0)
        radii = {ball.radius for ball in viewer.app.view_cube.balls}
        assert len(radii) == 1
        assert next(iter(radii)) == pytest.approx(viewcube.BALL_PT * scale)
    finally:
        viewer.release()
