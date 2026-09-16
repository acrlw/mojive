"""Viewport image presentation without scene editing, camera gestures or gizmos."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from imgui_bundle import imgui

from mojive.types import ViewportImage


def fit_image_rect(
    position: tuple[float, float],
    available: tuple[float, float],
    image_size: tuple[int, int],
) -> tuple[float, float, float, float]:
    """Aspect-fit a render target inside its current viewport panel."""

    x, y = float(position[0]), float(position[1])
    width, height = max(float(available[0]), 1.0), max(float(available[1]), 1.0)
    image_width, image_height = max(int(image_size[0]), 1), max(int(image_size[1]), 1)
    scale = min(width / image_width, height / image_height)
    fitted_width = image_width * scale
    fitted_height = image_height * scale
    return (
        x + (width - fitted_width) * 0.5,
        y + (height - fitted_height) * 0.5,
        fitted_width,
        fitted_height,
    )


@dataclass
class ViewportSurface:
    """Own the panel's content bounds and present a render target with its declared orientation."""

    position: tuple[float, float] = (0.0, 0.0)
    size: tuple[float, float] = (640.0, 480.0)

    def image_rect(self, image_size: tuple[int, int]) -> tuple[float, float, float, float]:
        return fit_image_rect(self.position, self.size, image_size)

    def begin(self, title: str) -> None:
        if title != "Viewport":
            title += "###Viewport"
        imgui.push_style_var(imgui.StyleVar_.window_padding, imgui.ImVec2(0.0, 0.0))
        imgui.begin(title, None, imgui.WindowFlags_.no_scrollbar.value)
        imgui.pop_style_var()
        pos = imgui.get_cursor_screen_pos()
        size = imgui.get_content_region_avail()
        if not imgui.is_window_docked():
            # Floating scenes stop before resize grips; clipping must not change their aspect.
            inset = float(math.ceil(imgui.get_style().window_border_size * 0.5))
            pos = imgui.ImVec2(pos.x + inset, pos.y)
            size = imgui.ImVec2(size.x - 2.0 * inset, size.y - inset)
        self.position = (float(pos.x), float(pos.y))
        self.size = (max(float(size.x), 1.0), max(float(size.y), 1.0))

    def draw_image(
        self, image: ViewportImage, texture_ref: Callable[[ViewportImage], imgui.ImTextureRef]
    ) -> tuple[float, float, float, float]:
        rect = self.image_rect((image.width, image.height))
        x, y, width, height = rect
        uv0 = imgui.ImVec2(0.0, 1.0) if image.flip_y else imgui.ImVec2(0.0, 0.0)
        uv1 = imgui.ImVec2(1.0, 0.0) if image.flip_y else imgui.ImVec2(1.0, 1.0)
        imgui.set_cursor_screen_pos(imgui.ImVec2(x, y))
        imgui.push_clip_rect((x, y), (x + width, y + height), False)
        try:
            imgui.image(texture_ref(image), imgui.ImVec2(width, height), uv0, uv1)
        finally:
            imgui.pop_clip_rect()
        return rect
