"""Audit crops convert native window points without rescaling captured pixels."""

from concurrent.futures import Future
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from mojive.tools.design_audit import ParameterAudit
from mojive.ui.window import Window


def audit_capture(tmp_path, pixels, scale, rect):
    future = Future()
    future.set_result({"image": pixels})
    calls = []

    def request(**kwargs):
        calls.append(kwargs)
        return future

    window = SimpleNamespace(_pixel_scale=scale)
    window.points_to_pixels = lambda value: Window.points_to_pixels(window, value)
    audit = object.__new__(ParameterAudit)
    audit.viewer = SimpleNamespace(
        app=SimpleNamespace(request_capture_async=request), window=window
    )
    audit.input = SimpleNamespace(frames=lambda: calls.append("frame"))
    audit.bench = SimpleNamespace(_viewport_rect=rect)
    audit.output = tmp_path / "audit.json"
    return audit, calls


@pytest.mark.parametrize(
    "scale,right,size",
    ((1, 1460, (1920, 1080)), (2, 1460, (3840, 2160)), (1.25, 100.4, (240, 160))),
)
def test_native_editor_boundary_is_converted_to_framebuffer_pixels(tmp_path, scale, right, size):
    width, height = size
    pixels = np.zeros((height, width, 3), dtype=np.uint8)
    expected_right = round(right * scale)
    pixels[0, 0] = (12, 34, 56)
    pixels[13, expected_right - 1] = (78, 90, 123)
    # This location was wrongly cropped away on the actual 2x Retina capture.
    if scale == 2:
        pixels[100, 1516] = (200, 100, 50)
    audit, calls = audit_capture(tmp_path, pixels, scale, (right, 0, 460, height / scale))
    path = audit.capture("form_popup", "paddingLeft", "before")
    actual = np.asarray(Image.open(path))
    assert actual.shape == (height, expected_right, 3)
    assert np.array_equal(actual, pixels[:, :expected_right])
    assert calls == [{"surface": "window", "memory": True}, "frame"]


@pytest.mark.parametrize(
    "rect,include_editor", ((None, False), ((90, 0, 30, 80), True), ((300, 0, 30, 80), False))
)
def test_full_window_capture_and_outside_editor_keep_original_pixel_extent(
    tmp_path, rect, include_editor
):
    pixels = np.arange(120 * 80 * 3, dtype=np.uint8).reshape((80, 120, 3))
    audit, _calls = audit_capture(tmp_path, pixels, 2, rect)
    actual = np.asarray(
        Image.open(audit.capture("form_popup", "radius", "after", include_editor=include_editor))
    )
    assert np.array_equal(actual, pixels)


def test_crop_rejects_an_empty_production_region_instead_of_saving_fake_pixels(tmp_path):
    pixels = np.zeros((80, 120, 3), dtype=np.uint8)
    audit, _calls = audit_capture(tmp_path, pixels, 2, (-4, 0, 30, 80))
    with pytest.raises(RuntimeError, match="outside the captured window"):
        audit.capture("form_popup", "radius", "after")
    assert not list(tmp_path.rglob("*.png"))
