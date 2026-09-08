"""Export production mouse hint geometry with black and transparent shells."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from ..ui.theme import THEME, Theme
from ..ui.viewport_widgets import (
    OVERLAY_GEOMETRY,
    draw_mouse_hint_glyph,
    mouse_button_geometry,
    mouse_wheel_geometry,
)
from .tool_icons import _PillowDraw2D

_SUPERSAMPLE = 4
_GLYPH_SCALE_FOR_CANVAS = 0.03125
_TRANSPARENT = (0, 0, 0, 0)
_BLACK = (0, 0, 0, 255)
_CHECKER = ((92, 96, 104, 255), (132, 137, 147, 255))


def _checker_preview(icon: Image.Image) -> Image.Image:
    size = icon.width
    cell = max(8, size // 16)
    preview = Image.new("RGBA", icon.size, (0, 0, 0, 255))
    draw = ImageDraw.Draw(preview)
    for y in range(0, size, cell):
        for x in range(0, size, cell):
            draw.rectangle(
                (x, y, min(size, x + cell), min(size, y + cell)),
                fill=_CHECKER[(x // cell + y // cell) & 1],
            )
    preview.alpha_composite(icon)
    return preview


def _render_mouse_icon(size: int, button: str, *, black_shell: bool) -> Image.Image:
    working_size = int(size) * _SUPERSAMPLE
    scale = working_size * _GLYPH_SCALE_FOR_CANVAS
    width = OVERLAY_GEOMETRY.hint_mouse_width * scale
    height = OVERLAY_GEOMETRY.hint_control_height * scale
    x = (working_size - width) * 0.5
    y = (working_size - height) * 0.5
    center_y = working_size * 0.5
    outline_width = OVERLAY_GEOMETRY.hint_mouse_stroke * scale
    radius = min(width * 0.22, height * 0.18)
    image = Image.new("RGBA", (working_size, working_size), _TRANSPARENT)
    draw = _PillowDraw2D(image, scale)
    theme = Theme(text=THEME.text, primary=THEME.primary)

    if not black_shell:
        draw_mouse_hint_glyph(
            draw,
            x,
            center_y,
            button,
            "",
            theme,
            scale,
            pixel_size=float(_SUPERSAMPLE),
        )
        return image.resize((size, size), Image.Resampling.LANCZOS)

    # The diagnostic follows the same construction as production, but paints
    # the omitted knockout pixels black so their exact width can be inspected.
    draw.rect(
        (x, y),
        (x + width, y + height),
        theme.text,
        outline_width,
        rounding=radius,
    )
    if button in {"left", "right"}:
        geometry = mouse_button_geometry(
            x,
            y,
            width,
            height,
            button,
            outline_width=outline_width,
            geometry=OVERLAY_GEOMETRY,
        )
        assert geometry is not None
        gap = outline_width * OVERLAY_GEOMETRY.hint_mouse_button_shell_ratio
        path = tuple(geometry.fill)
        pixels = ImageDraw.Draw(image)
        pixels.line(
            (*path, path[0]),
            fill=_BLACK,
            width=max(1, round(gap * 2.0)),
            joint="curve",
        )
        pixels.polygon(path, fill=_BLACK)
        draw.convex_fill(path, theme.primary)
    else:
        wheel = mouse_wheel_geometry(
            x,
            y,
            width,
            height,
            outline_width=outline_width,
            pixel_size=float(_SUPERSAMPLE),
            geometry=OVERLAY_GEOMETRY,
        )
        draw.rect_filled(
            (wheel.lo[0] - wheel.gap, wheel.lo[1] - wheel.gap),
            (wheel.hi[0] + wheel.gap, wheel.hi[1] + wheel.gap),
            tuple(channel / 255.0 for channel in _BLACK),
            rounding=wheel.rounding + wheel.gap,
        )
        draw.rect_filled(wheel.lo, wheel.hi, theme.primary, rounding=wheel.rounding)
    return image.resize((size, size), Image.Resampling.LANCZOS)


def export_icons(output: Path, size: int = 1024) -> tuple[Path, ...]:
    """Write left, right, and wheel icons using production geometry."""

    output.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for button in ("left", "right", "wheel"):
        for shell_name, black_shell in (("black-shell", True), ("transparent-shell", False)):
            icon = _render_mouse_icon(size, button, black_shell=black_shell)
            path = output / f"mouse-{button}-{shell_name}.png"
            icon.save(path)
            written.append(path)
            preview = output / f"mouse-{button}-{shell_name}-preview.png"
            _checker_preview(icon).save(preview)
            written.append(preview)
    return tuple(written)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, default=Path("output/mouse-icons-1024"))
    parser.add_argument("--size", type=int, default=1024)
    args = parser.parse_args()
    for path in export_icons(args.output, args.size):
        print(path)


if __name__ == "__main__":
    main()
