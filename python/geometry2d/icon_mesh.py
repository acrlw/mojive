"""Compile canonical icon stroke unions before size-dependent UI painting."""

from functools import lru_cache

from mojive.geometry2d.compiler import Contour2D
from mojive.geometry2d.curves import CURVE_TOLERANCE
from mojive.geometry2d.mesh import compile_stroke, tessellate_contours
from mojive.geometry2d.path import PathBuilder2D
from mojive.geometry2d.polygons import signed_polygon_area
from mojive.geometry2d.style import StrokeStyle


@lru_cache(maxsize=512)
def stroke_mesh(paths, closed, width, cap, join, miter_limit):
    """Compile one canonical stroke union, retaining exterior and hole boundaries."""
    builder = PathBuilder2D()
    closures = (closed,) * len(paths) if isinstance(closed, bool) else closed
    for path, is_closed in zip(paths, closures, strict=True):
        builder.move_to(*path[0])
        for point in path[1:]:
            builder.line_to(*point)
        if is_closed:
            builder.close()
    mesh = compile_stroke(
        builder.finish(),
        StrokeStyle(width, cap=cap, join=join, miter_limit=miter_limit),
        tolerance=CURVE_TOLERANCE,
    )
    vertices = tuple(map(tuple, mesh.positions))
    edges = {int(start): int(end) for start, end in mesh.boundary_edges}
    if len(edges) != len(mesh.boundary_edges) or set(edges) != set(edges.values()):
        raise ValueError("Icon stroke boundary must consist of closed, nonbranching contours")
    boundaries = []
    while edges:
        start = current = next(iter(edges))
        outline = []
        while True:
            outline.append(vertices[current])
            current = edges.pop(current)
            if current == start:
                break
        outline = tuple(outline)
        # Native boundary edges have the filled region on their left. Negative
        # winding therefore marks a hole, whose AA fringe must extend inward.
        boundaries.append((outline, bool(signed_polygon_area(outline) < 0)))
    # Stroke unions retain the interior vertices of their segment/cap primitives.
    # Only the union boundary is needed for the cached UI mesh and its AA fringe.
    compact = tessellate_contours(Contour2D(outline, True) for outline, _inside in boundaries)
    return tuple(map(tuple, compact.positions)), tuple(map(int, compact.indices)), tuple(boundaries)


def compile_icon_layers(contours):
    """Keep fill/translucent order while merging compatible opaque stroke contours."""
    layers = []
    stroke_key, paths, closed = None, [], []

    def flush():
        nonlocal stroke_key
        if stroke_key is None:
            return
        width, cap, join, miter_limit, opacity = stroke_key
        vertices, indices, boundaries = stroke_mesh(
            tuple(paths), tuple(closed), width, cap, join, miter_limit
        )
        layers.append(
            {
                "background": False,
                "opacity": opacity,
                "stroke": {"vertices": vertices, "indices": indices, "boundaries": boundaries},
            }
        )
        stroke_key = None
        paths.clear()
        closed.clear()

    for contour in contours:
        points = tuple(map(tuple, contour["points"]))
        if len(points) < 2:
            continue
        if contour["fill"] != "none":
            flush()
            layers.append(
                {
                    "background": contour["fill"].startswith("var"),
                    "opacity": contour["opacity"] * contour["fill_opacity"],
                    "fill": points,
                }
            )
        if contour["stroke"] != "none":
            opacity = contour["opacity"] * contour["stroke_opacity"]
            key = (
                contour["width"],
                contour["cap"],
                contour["join"],
                contour["miter_limit"],
                opacity,
            )
            if stroke_key != key:
                flush()
            stroke_key = key
            strokes = tuple(
                tuple(map(tuple, path)) for path in contour.get("stroke_paths", (points,))
            )
            paths.extend(strokes)
            closed.extend([contour["closed"] and "stroke_paths" not in contour] * len(strokes))
            if opacity != 1:
                flush()
    flush()
    return layers
