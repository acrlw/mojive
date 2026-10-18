"""Reject malformed masters before their geometry reaches the native painter."""

import json
import math
from itertools import pairwise
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pytest
from examples.ui_design.native.compile_icons import (
    COMPILED_ICONS,
    ROOT,
    compile_svg,
    library_masters,
)

from mojive.ui.workspace.widget_icons import ICON_STROKE


def test_previous_rotate_arc_exceeds_declared_canvas():
    source = (Path(__file__).parent / "fixtures/ui_design/invalid_rotate.svg").read_text()
    with pytest.raises(ValueError, match="exceed its viewBox"):
        compile_svg(source)


@pytest.mark.parametrize("canvas", ("0 0 48 24", "-24 -12 48 24", "0 0 24 48"))
def test_rectangular_canvas_keeps_proportions_and_center(canvas):
    x, y, w, h = map(float, canvas.split())
    svg = (
        f'<svg viewBox="{canvas}" fill="none" stroke="currentColor" stroke-width="2">'
        f'<rect x="{x + 4}" y="{y + 4}" width="{w - 8}" height="{h - 8}"/></svg>'
    )
    glyph = compile_svg(svg)
    left, top, right, bottom = glyph["bounds"]
    assert (right - left) / (bottom - top) == pytest.approx((w - 6) / (h - 6))
    assert (left + right, top + bottom) == pytest.approx((24, 24))
    assert glyph["contours"][0]["width"] == pytest.approx(2 * 24 / max(w, h))


def test_all_masters_compile_inside_their_canvas():
    sources = json.loads((ROOT / "icons-source.json").read_text())
    compiled = json.loads(COMPILED_ICONS.read_text())
    assert compiled == json.loads(
        json.dumps({name: compile_svg(svg) for name, svg in sources.items()})
    )
    for name, glyph in compiled.items():
        for contour in glyph["contours"]:
            points = contour["points"]
            if contour["closed"]:
                points = [*points, points[0]]
            assert all(math.dist(a, b) > 1e-6 for a, b in pairwise(points)), name
        for size in (14, 16, 18, 24, 56, 112):
            bounds = [v * size / 24 for v in glyph["bounds"]]
            assert min(bounds) >= 0 and max(bounds) <= size, name
    eye = compiled["eye"]["bounds"]
    mouse = compiled["mouse"]["bounds"]
    assert eye[2] - eye[0] > eye[3] - eye[1]
    assert mouse[2] - mouse[0] < mouse[3] - mouse[1]
    left, top, right, bottom = compiled["rotate"]["bounds"]
    assert (left + right, top + bottom) == pytest.approx((24, 24))
    assert 17 <= right - left <= 22
    assert 17 <= bottom - top <= 22


def test_joint_projection_and_sort_masters_share_the_grid_and_have_distinct_silhouettes():
    sources = json.loads((ROOT / "icons-source.json").read_text())
    names = (
        "joint-hinge",
        "joint-slide",
        "joint-ball",
        "joint-free",
        "perspective",
        "orthographic",
        "sort",
    )
    assert len({sources[name] for name in names}) == len(names)
    for name in names:
        glyph = compile_svg(sources[name])
        assert glyph["viewBox"] == [0, 0, 24, 24]
        assert all(
            contour["width"] == ICON_STROKE
            for contour in glyph["contours"]
            if contour["stroke"] != "none"
        )
        assert sources[name] not in (sources["eye"], sources["sliders"], sources["pose"])


def test_dash_and_alpha_survive_compilation():
    sources = json.loads((ROOT / "icons-source.json").read_text())
    perturb = compile_svg(sources["perturb"])["contours"][-1]
    assert len(perturb["stroke_paths"]) == 3
    assert perturb["stroke_paths"][0][-1] != perturb["stroke_paths"][1][0]
    shading = compile_svg(sources["shading"])["contours"][-1]
    assert shading["opacity"] == 0.5
    assert shading["stroke"] == "none"


def test_svg_stroke_caps_joins_and_element_overrides_survive_compilation():
    glyph = compile_svg(
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M5 5H15V15"/>'
        '<path d="M7 7H17V17" stroke-linecap="square" '
        'stroke-linejoin="bevel" stroke-miterlimit="2"/></svg>'
    )
    first, second = glyph["contours"]
    assert (first["cap"], first["join"], first["miter_limit"]) == ("round", "round", 4)
    assert (second["cap"], second["join"], second["miter_limit"]) == ("square", "bevel", 2)


@pytest.fixture
def stroke_operations(monkeypatch):
    from mojive.ui.workspace import icon_mesh, widget_icons

    count = 0

    def prepare(path, *, size=24, width=1.5, cap="round", join="round"):
        nonlocal count
        name = f"__stroke_test_{count}"
        count += 1
        glyph = compile_svg(
            '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            f'stroke-width="{width}" stroke-linecap="{cap}" stroke-linejoin="{join}">'
            f'<path d="{path}"/></svg>'
        )
        monkeypatch.setitem(widget_icons.GLYPHS, name, glyph)
        return tuple(
            operation
            for _background, _opacity, operations in icon_mesh.prepared_icon(name, size)
            for operation in operations
        )

    yield prepare
    icon_mesh.prepared_icon.cache_clear()
    widget_icons.icon_geometry.cache_clear()


def _opaque_triangles(operations):
    return np.concatenate(
        [
            np.asarray([(point.x, point.y) for point in vertices])[np.asarray(indices)].reshape(
                -1, 3, 2
            )
            for fringe, vertices, indices in operations
            if not fringe and indices
        ]
    )


def _distance_to_segments(samples, points):
    distance = np.full(len(samples), np.inf)
    for a, b in pairwise(np.asarray(points)):
        t = np.clip(((samples - a) @ (b - a)) / np.dot(b - a, b - a), 0, 1)
        distance = np.minimum(distance, np.linalg.norm(samples - a - t[:, None] * (b - a), axis=1))
    return distance


@pytest.mark.parametrize("size", (14, 16, 18, 24))
def test_prepared_round_projection_corners_stay_inside_the_stroke_radius(stroke_operations, size):
    operations = stroke_operations("M5.2 9 18.8 5.9v12.2L5.2 15Z", size=size)
    vertices = _opaque_triangles(operations).reshape(-1, 2) * 24 / size
    assert (*vertices.min(axis=0), *vertices.max(axis=0)) == pytest.approx(
        (4.45, 5.15, 19.55, 18.85), abs=1e-5
    )
    centerline = ((5.2, 9), (18.8, 5.9), (18.8, 18.1), (5.2, 15), (5.2, 9))
    # A miter tip lies farther from the centerline than half the stroke width.
    assert _distance_to_segments(vertices, centerline).max() <= 0.75 + 1e-5


def test_prepared_crossing_stroke_is_one_coverage_union(stroke_operations):
    operations = stroke_operations("M5 5 19 19 5 19 19 5Z")
    triangles = _opaque_triangles(operations)
    samples = np.random.default_rng(32).uniform(3, 21, (2500, 2))
    distance = _distance_to_segments(samples, ((5, 5), (19, 19), (5, 19), (19, 5), (5, 5)))
    samples = samples[np.abs(distance - 0.75) > 0.025]
    expected = (distance[np.abs(distance - 0.75) > 0.025] < 0.75).astype(int)
    coverage = np.zeros(len(samples), dtype=int)
    for triangle in triangles:
        inside = np.ones(len(samples), dtype=bool)
        for a, b in zip(triangle, np.roll(triangle, -1, axis=0), strict=True):
            cross = (b[0] - a[0]) * (samples[:, 1] - a[1]) - (b[1] - a[1]) * (samples[:, 0] - a[0])
            inside &= cross >= 0
        coverage += inside
    np.testing.assert_array_equal(coverage, expected)


@pytest.mark.parametrize("size", (14, 16, 18, 24, 56, 112))
def test_separate_stroke_contours_have_only_the_combined_silhouette_fringe(stroke_operations, size):
    operations = stroke_operations("M6 6H18V18H6Z M6 12H18", size=size)
    triangles = _opaque_triangles(operations) * 24 / size
    samples = np.random.default_rng(42).uniform(4, 20, (3000, 2))
    outer = ((6, 6), (18, 6), (18, 18), (6, 18), (6, 6))
    crossbar = ((6, 12), (18, 12))

    def distance(points):
        return np.minimum(
            _distance_to_segments(points, outer), _distance_to_segments(points, crossbar)
        )

    expected = distance(samples)
    keep = np.abs(expected - 0.75) > 0.03
    samples, expected = samples[keep], (expected[keep] < 0.75).astype(int)
    coverage = np.zeros(len(samples), dtype=int)
    for triangle in triangles:
        inside = np.ones(len(samples), dtype=bool)
        for a, b in zip(triangle, np.roll(triangle, -1, axis=0), strict=True):
            inside &= (b[0] - a[0]) * (samples[:, 1] - a[1]) - (b[1] - a[1]) * (
                samples[:, 0] - a[0]
            ) >= 0
        coverage += inside
    np.testing.assert_array_equal(coverage, expected)
    fringes = [operation for operation in operations if operation[0]]
    assert len(fringes) == 3  # One exterior boundary and two real holes.
    for _fringe, edge, _fade in fringes:
        points = np.asarray([(point.x, point.y) for point in edge]) * 24 / size
        # A per-contour fringe crosses the interior of the T-junction at distance 0.
        assert distance(points).min() >= 0.72


def test_translucent_strokes_keep_authored_compositing_order(monkeypatch):
    from mojive.ui.workspace import icon_mesh, widget_icons

    name = "__translucent_stroke_layers"
    glyph = compile_svg(
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="1.5" stroke-opacity="0.5">'
        '<path d="M5 5L19 19"/><path d="M5 19L19 5"/></svg>'
    )
    monkeypatch.setitem(widget_icons.GLYPHS, name, glyph)
    layers = icon_mesh.prepared_icon(name, 24)
    assert len(layers) == 2
    assert all(opacity == 0.5 for _background, opacity, _operations in layers)


def test_prepared_closed_stroke_antialiases_outside_and_into_its_hole(stroke_operations):
    from mojive.geometry2d.polygons import signed_polygon_area

    operations = stroke_operations("M6 6H18V18H6Z", width=2)
    boundaries = [operation for operation in operations if operation[0]]
    assert len(boundaries) == 2
    for _fringe, edge, fade in boundaries:
        edge = np.asarray([(point.x, point.y) for point in edge])
        fade = np.asarray([(point.x, point.y) for point in fade])
        hole = signed_polygon_area(tuple(map(tuple, edge))) < 0
        if hole:
            assert np.all(fade.min(axis=0) > edge.min(axis=0))
            assert np.all(fade.max(axis=0) < edge.max(axis=0))
        else:
            assert np.all(fade.min(axis=0) < edge.min(axis=0))
            assert np.all(fade.max(axis=0) > edge.max(axis=0))


def test_stroke_tessellation_is_cached_on_the_canonical_grid(stroke_operations, monkeypatch):

    from mojive.geometry2d import icon_mesh as compiler

    compiler.stroke_mesh.cache_clear()
    compile_stroke = compiler.compile_stroke
    calls = []

    def compile_once(path, style, **kwargs):
        calls.append((path, style))
        return compile_stroke(path, style, **kwargs)

    monkeypatch.setattr(compiler, "compile_stroke", compile_once)
    areas = []
    for size in (14, 16, 18, 24):
        triangles = _opaque_triangles(stroke_operations("M5 5H19V19", size=size))
        a, b = triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
        areas.append(np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]).sum() / 2 * (24 / size) ** 2)
    assert len(calls) == 1
    assert calls[0][1].width == 1.5
    assert areas == pytest.approx([areas[-1]] * 4, rel=1e-6)


@pytest.mark.parametrize("size", (14, 16, 18, 24, 56, 112))
def test_hinge_arm_caps_stay_inside_the_crossbar_outer_faces(stroke_operations, size):
    from mojive.ui.workspace.widget_icons import GLYPHS

    contours = GLYPHS["joint-hinge"]["contours"]
    pivot = np.asarray(contours[0]["points"]).mean(axis=0)
    lines = sorted(
        (contour for contour in contours if len(contour["points"]) == 2),
        key=lambda contour: np.linalg.norm(np.asarray(contour["points"]).mean(axis=0) - pivot),
    )
    assert len(lines) == 4
    arms, bars = lines[:2], lines[2:]
    for arm in arms:
        points = np.asarray(arm["points"])
        endpoint = max(points, key=lambda point: np.linalg.norm(point - pivot))
        bar = min(
            bars, key=lambda item: np.linalg.norm(np.asarray(item["points"]).mean(0) - endpoint)
        )
        a, b = np.asarray(bar["points"])
        junction = (a + b) / 2
        np.testing.assert_allclose(endpoint, junction, atol=1e-6)
        normal = np.array((a[1] - b[1], b[0] - a[0])) / np.linalg.norm(b - a)
        if normal @ (junction - pivot) < 0:
            normal = -normal
        path = "M" + "L".join(f"{x} {y}" for x, y in points)
        operations = stroke_operations(
            path, size=size, width=arm["width"], cap=arm["cap"], join=arm["join"]
        )
        vertices = _opaque_triangles(operations).reshape(-1, 2) * 24 / size
        # The complete round cap is contained by the crossbar's outer stroke face.
        assert ((vertices - junction) @ normal).max() <= bar["width"] / 2 + 1e-5


def test_curved_stroke_mesh_discards_internal_union_primitive_vertices():
    from mojive.geometry2d.icon_mesh import stroke_mesh as _stroke_mesh
    from mojive.ui.workspace.widget_icons import icon_geometry

    for contour, _points, paths, width in icon_geometry("settings", 24):
        vertices, indices, boundaries = _stroke_mesh(
            paths, contour["closed"], width, contour["cap"], contour["join"], contour["miter_limit"]
        )
        boundary_count = sum(len(points) for points, _inside in boundaries)
        assert len(vertices) <= boundary_count
        assert len(indices) < boundary_count * 6


@pytest.mark.parametrize(
    "name", ("settings", "camera", "mouse", "circle", "perspective", "orthographic")
)
@pytest.mark.parametrize("size", (14, 16, 18, 24, 112))
def test_curved_icon_boundaries_do_not_amplify_tangent_notches_into_aa_teeth(name, size):
    from mojive.ui.workspace.icon_mesh import prepared_icon

    fringe_width = min(1, ICON_STROKE * size / 48)
    for _background, _opacity, operations in prepared_icon(name, size):
        for fringe, edge, fade in operations:
            if not fringe:
                continue
            edge = np.asarray([(point.x, point.y) for point in edge])
            fade = np.asarray([(point.x, point.y) for point in fade])
            directions = np.roll(edge, -1, axis=0) - edge
            directions /= np.linalg.norm(directions, axis=1)[:, None]
            incoming = np.roll(directions, 1, axis=0)
            turns = np.arctan2(
                incoming[:, 0] * directions[:, 1] - incoming[:, 1] * directions[:, 0],
                (incoming * directions).sum(axis=1),
            )
            # These authored contours are smooth. Tiny rectangle tips in a
            # sampled-circle union introduced ~90-degree turns and 50% AA spikes.
            assert np.abs(turns).max() < math.radians(20)
            assert np.linalg.norm(fade - edge, axis=1).max() <= fringe_width * 1.02 + 1e-5


def test_sidebar_masters_share_the_grid_weight_and_compact_envelopes():
    sources = json.loads((ROOT / "icons-source.json").read_text())
    names = (
        "scene",
        "control",
        "assets",
        "camera",
        "performance",
        "layers",
        "terminal",
        "settings",
    )
    for name in names:
        source = ET.fromstring(sources[name])
        glyph = compile_svg(sources[name])
        assert glyph["viewBox"] == [0, 0, 24, 24]
        assert source.attrib["stroke-linecap"] == source.attrib["stroke-linejoin"] == "round"
        assert all(
            contour["width"] == ICON_STROKE
            for contour in glyph["contours"]
            if contour["stroke"] != "none"
        )
        left, top, right, bottom = glyph["bounds"]
        assert max(right - left, bottom - top) <= 22, name


def test_imported_masters_preserve_library_geometry_and_aliases():
    sources = json.loads((ROOT / "icons-source.json").read_text())
    imported = library_masters()
    assert all(sources[name] == svg for name, svg in imported.items())
    for library in ("lucide", "tabler"):
        assert "Copyright" in (ROOT / "svg" / library / "LICENSE").read_text()
    for first, second in (("assets", "folder"), ("settings", "gear")):
        assert sources[first] == sources[second]


def test_geometry_modes_have_distinct_library_silhouettes_from_visibility_and_layers():
    sources = json.loads((ROOT / "icons-source.json").read_text())
    names = ("geometry-default", "visual", "collision", "both", "eye", "layers", "mujoco-visuals")
    assert len({sources[name] for name in names}) == len(names)
    for name in names[:4]:
        glyph = compile_svg(sources[name])
        assert glyph["viewBox"] == [0, 0, 24, 24]
        assert all(
            contour["width"] == ICON_STROKE
            for contour in glyph["contours"]
            if contour["stroke"] != "none"
        )


def test_default_geometry_flag_preserves_its_opening_without_internal_aa_seams():
    from mojive.geometry2d.icon_mesh import stroke_mesh as _stroke_mesh
    from mojive.ui.workspace.widget_icons import icon_geometry

    contour, _points, paths, width = next(iter(icon_geometry("geometry-default", 24)))
    _vertices, _indices, boundaries = _stroke_mesh(
        paths, contour["closed"], width, contour["cap"], contour["join"], contour["miter_limit"]
    )
    # The flag joins its pole once; one exterior and one flag opening remain.
    assert sorted(inside for _points, inside in boundaries) == [False, True]
    hole = np.asarray(next(points for points, inside in boundaries if inside))
    assert hole[:, 0].max() - hole[:, 0].min() > 10
    assert hole[:, 1].max() - hole[:, 1].min() > 7
