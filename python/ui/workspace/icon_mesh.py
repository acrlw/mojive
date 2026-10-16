"""Prepare canonical icon draw operations without retaining native draw lists."""

from functools import lru_cache

from imgui_bundle import imgui

from mojive.ui.icon_draw import ImguiIconDraw
from mojive.ui.imgui_draw import ImguiDraw2D, _concave_indices


class _MeshRecorder:
    def __init__(self):
        self.flags = imgui.ImDrawListFlags_.anti_aliased_fill.value
        self.operations = []

    def add_indexed_fill(self, points, indices, color):
        self.operations.append((False, points, indices))

    def add_poly_fringe(self, inner, outer, color):
        self.operations.append((True, inner, outer))

    def add_concave_poly_filled(self, points, color):
        path = tuple((point.x, point.y) for point in points)
        self.add_indexed_fill(points, _concave_indices(path), color)


@lru_cache(maxsize=512)
def prepared_icon(name, size):
    """Reuse triangulation, stroke expansion and AA rings at each physical size."""
    from mojive.ui.workspace.widget_icons import ICON_GRID, ICON_STROKE, icon_geometry

    recorder = _MeshRecorder()
    painter = ImguiIconDraw(ImguiDraw2D(recorder), min(1.0, ICON_STROKE * size / (2 * ICON_GRID)))
    groups = []
    for contour, points, paths, thickness in icon_geometry(name, size):
        if len(points) < 2:
            continue
        if contour["fill"] != "none":
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
            recorder.operations = []
            closed = contour["closed"] and "stroke_paths" not in contour
            for path in paths:
                painter.polyline(
                    path, (1.0, 1.0, 1.0, 1.0), thickness, closed=closed, cap="round", smoothing=0.0
                )
            groups.append(
                (False, contour["opacity"] * contour["stroke_opacity"], tuple(recorder.operations))
            )
    return tuple(groups)
