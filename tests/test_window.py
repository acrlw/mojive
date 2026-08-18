"""Window scaling tests."""

from __future__ import annotations

import pytest

from forge_viewer.ui.window import layout_scale


@pytest.mark.parametrize(
    ("ui_scale", "framebuffer_scale", "expected"),
    (
        (1.0, 1.0, 1.0),
        (2.0, 1.0, 2.0),
        (2.0, 2.0, 1.0),
        (1.5, 1.0, 1.5),
    ),
)
def test_layout_scale_separates_desktop_and_framebuffer_scaling(
    ui_scale: float, framebuffer_scale: float, expected: float
) -> None:
    assert layout_scale(ui_scale, framebuffer_scale) == pytest.approx(expected)
