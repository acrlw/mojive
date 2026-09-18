"""Cached alpha-centroid measurements of the current Icon Library candidates."""

from functools import lru_cache

import numpy as np
from PIL import Image

from mojive.geometry2d.curves import CORNER_SMOOTHING
from mojive.tools.tool_icons import _PillowDraw2D
from mojive.ui.icons import ICON_GRID, IconStyle, draw_concept_icon, draw_icon
from mojive.ui.panels.filters import severity_meshes

_CANVAS = 80.0
_RESOLUTION = 256
_COORDINATES = (np.arange(_RESOLUTION) + 0.5) * _CANVAS / _RESOLUTION - _CANVAS / 2

MIRROR_PAIRS = (
    ("playback-previous", "playback-next"),
    ("transport-previous", "transport-next"),
    ("transport-first", "transport-last"),
    ("key-previous", "key-next"),
    ("status-mouse-left", "status-mouse-right"),
)
MIRROR_PARTNERS = {a: b for pair in MIRROR_PAIRS for a, b in (pair, pair[::-1])}


def update_manual_offset(offsets, name, value, *, link_mirrored=False):
    """Edit or reset a glyph, optionally reflecting its translation onto its partner."""
    if value is None:
        offsets.pop(name, None)
    else:
        offsets[name] = tuple(value)
    partner = MIRROR_PARTNERS.get(name) if link_mirrored else None
    if partner is not None:
        if value is None:
            offsets.pop(partner, None)
        else:
            offsets[partner] = (-value[0], value[1])


class _MaskDraw(_PillowDraw2D):
    def rect(self, lo, hi, color, width=1.0, *, rounding=0.0, smoothing=None):
        super().rect(
            lo,
            hi,
            color,
            width,
            rounding=rounding,
            smoothing=CORNER_SMOOTHING if smoothing is None else smoothing,
        )

    def indexed_fill(self, points, indices, color, *, origin=None, **_kwargs):
        # Triangles encode the holes. Paint their union instead of filling the outer contour.
        if origin is not None:
            points = tuple((x + origin[0], y + origin[1]) for x, y in points)
        for i in range(0, len(indices), 3):
            self.convex_fill(tuple(points[j] for j in indices[i : i + 3]), color)


def rasterized_ink(name: str, style: IconStyle | None = None):
    unit = _RESOLUTION / _CANVAS
    image = Image.new("RGBA", (_RESOLUTION, _RESOLUTION))
    draw = _MaskDraw(image, unit)
    center, size, white = (_RESOLUTION / 2,) * 2, ICON_GRID * unit, (1.0,) * 4
    if style is None:
        draw_icon(draw, center, size, name, white)
    elif name in {"status-info", "status-warning", "status-error"}:
        for vertices, indices, _outline, _hole in severity_meshes(
            size, name.removeprefix("status-")
        ):
            draw.indexed_fill(vertices, indices, white, origin=center)
    else:
        draw_concept_icon(
            draw,
            center,
            size,
            name,
            white,
            padding=style.padding,
            mouse_width=style.mouse_width,
            stroke_width=style.stroke_width,
            rotate_ring_gap_ratio=style.rotate_ring_gap_ratio,
            rotate_ring_cap=style.rotate_ring_cap,
            tuning=style.tuning,
            alignment=style.alignment,
        )
    weights = np.asarray(image.getchannel("A"), dtype=float) / 255.0
    mass = weights.sum()
    centroid = (
        float(weights.sum(axis=0) @ _COORDINATES / mass),
        float(weights.sum(axis=1) @ _COORDINATES / mass),
    )
    return weights, centroid


@lru_cache(maxsize=256)
def candidate_centroid(name: str, style: IconStyle) -> tuple[float, float]:
    # Position, preview size and alignment strength deliberately do not enter this key.
    # Retain only two coordinates per candidate, not a raster for every slider value.
    return rasterized_ink(name, style)[1]
