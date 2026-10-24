"""Primitive interior seams remain inside a continuous exterior silhouette."""

from pathlib import Path

import numpy as np
import pytest
from examples.ui_design.native.compile_icons import compile_svg
from imgui_bundle import imgui

from mojive.geometry2d.icon_mesh import compile_icon_layers
from mojive.geometry2d.mesh import compile_stroke
from mojive.geometry2d.path import PathBuilder2D
from mojive.geometry2d.style import StrokeStyle

MASTERS = Path(__file__).parents[1] / "examples/ui_design/native/svg/mojive"


def contains(points, polygon):
    points, vertices = np.asarray(points), np.asarray(polygon)
    inside = np.zeros(len(points), dtype=bool)
    previous = vertices[-1]
    for current in vertices:
        x1, y1 = previous
        x2, y2 = current
        if y1 != y2:
            crossing = (y1 > points[:, 1]) != (y2 > points[:, 1])
            boundary = x1 + (points[:, 1] - y1) * (x2 - x1) / (y2 - y1)
            inside ^= crossing & (points[:, 0] < boundary)
        previous = current
    return inside


@pytest.mark.parametrize("name", ("box", "cone"))
def test_interior_strokes_cannot_protrude_through_the_primitive_outer_silhouette(name):
    glyph = compile_svg((MASTERS / f"shape-{name}.svg").read_text())
    shell, *seams = glyph["contours"]
    assert shell["closed"]
    assert all(not seam["closed"] for seam in seams)
    outline = next(
        edge
        for edge, inside in compile_icon_layers([shell])[0]["stroke"]["boundaries"]
        if not inside
    )
    for seam in seams:
        path = PathBuilder2D().move_to(*seam["points"][0])
        for point in seam["points"][1:]:
            path.line_to(*point)
        stroke = StrokeStyle(
            seam["width"], cap=seam["cap"], join=seam["join"], miter_limit=seam["miter_limit"]
        )
        mesh = compile_stroke(path.finish(), stroke)
        # The full expanded seam is inside the expanded shell, keeping all joined caps contained at every production size.
        for point in mesh.positions:
            if contains([point], outline)[0]:
                continue
            # Shared tangent vertices can lie on the boundary within float32 precision.
            assert min(np.linalg.norm(point - np.asarray(vertex)) for vertex in outline) < 1e-5, (
                name,
                point,
            )


def covered(point, vertices, indices):
    triangles = np.asarray(vertices)[np.asarray(indices).reshape(-1, 3)]
    edges = np.roll(triangles, -1, axis=1) - triangles
    offsets = np.asarray(point) - triangles
    cross = edges[:, :, 0] * offsets[:, :, 1] - edges[:, :, 1] * offsets[:, :, 0]
    return ((cross >= -1e-8).all(axis=1) | (cross <= 1e-8).all(axis=1)).any()


@pytest.mark.parametrize(
    "name,voids",
    (
        ("sphere", ((6, 9), (18, 9))),
        ("ellipsoid", ((6, 9.5), (18, 9.5))),
        ("cylinder", ((6.2, 6.0), (17.8, 6.0))),
    ),
)
@pytest.mark.parametrize("size", (14, 16, 18, 24, 56, 112))
def test_tangent_joins_leave_no_inward_cap_lobes_in_the_upper_opening(name, voids, size):
    from mojive.ui.workspace.icon_mesh import prepared_icon

    context = imgui.create_context()
    try:
        groups = prepared_icon("shape-" + name, size)
        for _accent, _opacity, operations in groups:
            for fringe, vertices, indices in operations:
                if fringe:
                    continue
                vertices = [(vertex.x, vertex.y) for vertex in vertices]
                for point in voids:
                    assert not covered(np.asarray(point) * size / 24, vertices, indices)
    finally:
        imgui.destroy_context(context)
