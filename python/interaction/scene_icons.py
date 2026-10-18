"""Canonical scene-marker centerlines compiled from the adopted UI SVG masters."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from mojive.geometry2d.compiler import Contour2D
from mojive.geometry2d.curves import CURVE_TOLERANCE
from mojive.geometry2d.mesh import compile_stroke, tessellate_contours
from mojive.geometry2d.path import PathBuilder2D
from mojive.geometry2d.style import StrokeStyle

SCENE_ICON_GRID = 24.0
SCENE_ICON_SIZE_PT = 24.0


@dataclass(frozen=True)
class SceneIconStroke:
    """One round stroke on the canonical grid, centered at its authored anchor."""

    points: np.ndarray
    width: float
    closed: bool


@lru_cache(maxsize=2)
def scene_icon_strokes(name: str) -> tuple[SceneIconStroke, ...]:
    """Read the camera or light master without importing a UI or render implementation."""
    if name not in ("camera", "light"):
        raise ValueError(f"Unknown scene icon: {name!r}")
    glyph = json.loads((Path(__file__).with_name("resources") / "scene_icons.json").read_text())[
        name
    ]
    strokes = []
    for contour in glyph["contours"]:
        if (
            contour["fill"] != "none"
            or contour["stroke"] == "none"
            or contour["cap"] != "round"
            or contour["join"] != "round"
            or contour["opacity"] != 1
            or contour["stroke_opacity"] != 1
            or "stroke_paths" in contour
        ):
            raise ValueError(f"Scene icon {name!r} must use round, unfilled strokes")
        points = np.asarray(contour["points"], np.float64) - SCENE_ICON_GRID / 2
        points.setflags(write=False)
        strokes.append(SceneIconStroke(points, float(contour["width"]), contour["closed"]))
    return tuple(strokes)


@lru_cache(maxsize=2)
def scene_icon_data(name: str):
    """Pack paths once; frame publication transforms one contiguous vertex array."""
    strokes = scene_icon_strokes(name)
    points = np.concatenate(tuple(stroke.points for stroke in strokes))
    points.setflags(write=False)
    spans, start = [], 0
    for stroke in strokes:
        stop = start + len(stroke.points)
        spans.append((start, stop, stroke.width, stroke.closed))
        start = stop
    return points, tuple(spans)


@lru_cache(maxsize=2)
def scene_icon_geometry(name: str) -> np.ndarray:
    """Expand the canonical round strokes once, preserving caps and inner boundaries."""
    strokes = scene_icon_strokes(name)
    width = strokes[0].width
    if any(stroke.width != width for stroke in strokes):
        raise ValueError(f"Scene icon {name!r} must use one canonical stroke width")
    builder = PathBuilder2D()
    for stroke in strokes:
        builder.move_to(*stroke.points[0])
        for point in stroke.points[1:]:
            builder.line_to(*point)
        if stroke.closed:
            builder.close()
    mesh = compile_stroke(
        builder.finish(), StrokeStyle(width, cap="round", join="round"), tolerance=CURVE_TOLERANCE
    )
    # Stroke expansion retains segment/cap vertices inside the union. Only its
    # boundary is needed for the static glyph, as in the workspace icon cache.
    edges = {int(start): int(end) for start, end in mesh.boundary_edges}
    if len(edges) != len(mesh.boundary_edges) or set(edges) != set(edges.values()):
        raise ValueError(f"Scene icon {name!r} must have closed, nonbranching boundaries")
    contours = []
    while edges:
        start = current = next(iter(edges))
        outline = []
        while True:
            outline.append(mesh.positions[current])
            current = edges.pop(current)
            if current == start:
                break
        contours.append(Contour2D(outline, True))
    compact = tessellate_contours(contours)
    triangles = compact.positions[compact.indices].reshape(-1, 3, 2)
    triangles.setflags(write=False)
    return triangles
