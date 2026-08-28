"""Window scaling tests."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from imgui_bundle import imgui

from forge_viewer.ui.app import _fit_image_rect, _prepare_modal, _toggle_angle_input
from forge_viewer.ui.window import layout_scale, resolve_context_api, resolve_ui_scales
from forge_viewer.ui.window_wgpu import _scissor_rect_for_target


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


@pytest.mark.parametrize(
    ("content_scale", "framebuffer_scale", "override", "expected"),
    (
        (1.0, 1.0, None, (1.0, 1.0)),
        (2.0, 1.0, None, (2.0, 2.0)),
        (2.0, 2.0, None, (2.0, 1.0)),
        (2.0, 2.0, 2.0, (4.0, 2.0)),
        (2.0, 1.0, 2.0, (2.0, 2.0)),
    ),
)
def test_explicit_ui_scale_controls_layout_space(
    content_scale: float,
    framebuffer_scale: float,
    override: float | None,
    expected: tuple[float, float],
) -> None:
    assert resolve_ui_scales(content_scale, framebuffer_scale, override) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("requested", "expected"),
    (
        ("auto", "native"),
        ("", "native"),
        ("native", "native"),
        ("glfw", "native"),
        ("egl", "egl"),
    ),
)
def test_resolve_context_api(requested: str, expected: str) -> None:
    assert resolve_context_api(requested) == expected


def test_resolve_context_api_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="Unsupported FORGE_VIEWER_GL"):
        resolve_context_api("gles")


def test_viewport_image_is_aspect_fitted_while_render_target_resize_is_pending() -> None:
    assert _fit_image_rect((10.0, 20.0), (800.0, 800.0), (1600, 900)) == pytest.approx(
        (10.0, 195.0, 800.0, 450.0)
    )
    assert _fit_image_rect((10.0, 20.0), (800.0, 450.0), (1600, 900)) == pytest.approx(
        (10.0, 20.0, 800.0, 450.0)
    )


def test_wgpu_scissor_is_scaled_and_clamped_to_a_resized_target() -> None:
    scissor = _scissor_rect_for_target(
        (0.0, 40.0, 3840.0, 1978.0),
        (0.0, 0.0),
        (1.0, 1.0),
        (3840, 1938),
        (1600, 1000),
    )
    assert scissor == (0, 20, 1600, 980)


def test_wgpu_scissor_discards_clips_outside_the_resized_target() -> None:
    assert (
        _scissor_rect_for_target(
            (4000.0, 2000.0, 4100.0, 2100.0),
            (0.0, 0.0),
            (1.0, 1.0),
            (3840, 1938),
            (1600, 1000),
        )
        is None
    )


def test_precise_angle_input_toggles_units_without_changing_the_angle() -> None:
    radians, unit = _toggle_angle_input(180.0, "degrees")
    assert unit == "radians"
    assert radians == pytest.approx(np.pi)

    degrees, unit = _toggle_angle_input(radians, unit)
    assert unit == "degrees"
    assert degrees == pytest.approx(180.0)


def test_modal_layout_tracks_the_current_viewport_and_enforces_readable_width(
    monkeypatch,
) -> None:
    positions = []
    constraints = []
    center = imgui.ImVec2(700.0, 450.0)
    monkeypatch.setattr(
        imgui,
        "get_main_viewport",
        lambda: SimpleNamespace(get_center=lambda: center, work_size=imgui.ImVec2(800.0, 600.0)),
    )
    monkeypatch.setattr(imgui, "set_next_window_pos", lambda *args: positions.append(args))
    monkeypatch.setattr(
        imgui,
        "set_next_window_size_constraints",
        lambda *args: constraints.append(args),
    )

    _prepare_modal(440.0)

    position, condition, pivot = positions[0]
    assert (position.x, position.y) == pytest.approx((700.0, 450.0))
    assert condition == imgui.Cond_.always.value
    assert (pivot.x, pivot.y) == pytest.approx((0.5, 0.5))
    minimum, maximum = constraints[0]
    assert (minimum.x, minimum.y) == pytest.approx((440.0, 0.0))
    assert maximum.x == pytest.approx(440.0)
    assert maximum.y > 1e30
