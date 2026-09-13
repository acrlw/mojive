"""Submit shared drag-link geometry through the neutral Draw2D protocol."""

from __future__ import annotations

import math

from mojive.geometry2d.curves import CORNER_SMOOTHING
from mojive.geometry2d.drag_link import smooth_drag_link_mesh
from mojive.ui.paint_protocol import Draw2D


def draw_drag_link(
    overlay: Draw2D,
    start,
    end,
    core,
    edge,
    width: float,
    radius: float,
    edge_width: float,
    *,
    smoothing: float = CORNER_SMOOTHING,
) -> None:
    """Draw the shared implicit drag-link outline through a backend-neutral triangle mesh."""
    dx, dy = float(end[0] - start[0]), float(end[1] - start[1])
    distance = math.hypot(dx, dy)
    ux, uy = (dx / distance, dy / distance) if distance > 1e-9 else (1.0, 0.0)
    x, y = float(start[0]), float(start[1])

    for level, color in ((edge_width, edge), (0.0, core)):
        vertices, indices, outline, hole = smooth_drag_link_mesh(
            distance, radius, width, smoothing, level
        )
        overlay.indexed_fill(
            vertices, indices, color, outline=outline, hole=hole, origin=(x, y), direction=(ux, uy)
        )
