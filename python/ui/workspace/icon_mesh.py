"""Prepare canonical icon draw operations without retaining native draw lists."""

from functools import lru_cache

from imgui_bundle import imgui

from mojive.geometry2d.compiler import Contour2D
from mojive.geometry2d.curves import CURVE_TOLERANCE
from mojive.geometry2d.mesh import compile_stroke, tessellate_contours
from mojive.geometry2d.path import PathBuilder2D
from mojive.geometry2d.polygons import signed_polygon_area
from mojive.geometry2d.style import StrokeStyle
from mojive.ui.icon_draw import ImguiIconDraw
from mojive.ui.imgui_draw import ImguiDraw2D, _concave_indices


class _MeshRecorder:
    def __init__(self, scale):
        self.flags = imgui.ImDrawListFlags_.anti_aliased_fill.value
        self.operations = []
        self.scale = scale

    def vertices(self, points):
        return [imgui.ImVec2(point.x * self.scale, point.y * self.scale) for point in points]

    def add_indexed_fill(self, points, indices, color):
        self.operations.append((False, self.vertices(points), indices))

    def add_poly_fringe(self, inner, outer, color):
        self.operations.append((True, self.vertices(inner), self.vertices(outer)))

    def add_concave_poly_filled(self, points, color):
        path = tuple((point.x, point.y) for point in points)
        self.add_indexed_fill(points, _concave_indices(path), color)


@lru_cache(maxsize=512)
def _stroke_mesh(paths, closed, width, cap, join, miter_limit):
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
        boundaries.append((outline, signed_polygon_area(outline) < 0))
    # Stroke unions retain the interior vertices of their segment/cap primitives.
    # Only the union boundary is needed for the cached UI mesh and its AA fringe.
    compact = tessellate_contours(Contour2D(outline, True) for outline, _inside in boundaries)
    return tuple(map(tuple, compact.positions)), tuple(map(int, compact.indices)), tuple(boundaries)


@lru_cache(maxsize=512)
def prepared_icon(name, size):
    """Reuse triangulation, stroke expansion and AA rings at each physical size."""
    from mojive.ui.workspace.widget_icons import ICON_GRID, ICON_STROKE, icon_geometry

    if size <= 0:
        return ()
    scale = size / ICON_GRID
    recorder = _MeshRecorder(scale)
    # Build cap/join tessellation on the authored grid. Rebuilding at each display
    # size changes the chord tolerance relative to the stroke, especially below 1 pt.
    painter = ImguiIconDraw(ImguiDraw2D(recorder), min(1.0 / scale, ICON_STROKE / 2))
    groups = []
    stroke_key, stroke_paths, stroke_closed = None, [], []

    def flush_strokes():
        nonlocal stroke_key
        if stroke_key is None:
            return
        thickness, cap, join, miter_limit, opacity = stroke_key
        recorder.operations = []
        vertices, indices, boundaries = _stroke_mesh(
            tuple(stroke_paths), tuple(stroke_closed), thickness, cap, join, miter_limit
        )
        painter.indexed_fill(vertices, indices, (1.0, 1.0, 1.0, 1.0))
        for boundary, inside in boundaries:
            painter._write_anti_alias_fringe(boundary, 0xFFFFFFFF, inside=inside)
        groups.append((False, opacity, tuple(recorder.operations)))
        stroke_key = None
        stroke_paths.clear()
        stroke_closed.clear()

    for contour, points, paths, thickness in icon_geometry(name, ICON_GRID):
        if len(points) < 2:
            continue
        if contour["fill"] != "none":
            flush_strokes()
            recorder.operations = []
            painter.fringed_concave_fill(points, (1.0, 1.0, 1.0, 1.0))
            groups.append(
                (
                    contour["fill"].startswith("var"),
                    contour["opacity"] * contour["fill_opacity"],
                    tuple(recorder.operations),
                )
            )
        if contour["stroke"] != "none":
            opacity = contour["opacity"] * contour["stroke_opacity"]
            key = (thickness, contour["cap"], contour["join"], contour["miter_limit"], opacity)
            if stroke_key != key:
                flush_strokes()
            stroke_key = key
            stroke_paths.extend(paths)
            stroke_closed.extend([contour["closed"] and "stroke_paths" not in contour] * len(paths))
            # Adjacent opaque strokes form one silhouette, including at T-junctions.
            # Applying a fringe to each source contour leaves internal seams and
            # darkens intersections when the complete glyph is disabled or faded.
            # Translucent SVG strokes retain their authored compositing order.
            if opacity != 1:
                flush_strokes()
    flush_strokes()
    return tuple(groups)
