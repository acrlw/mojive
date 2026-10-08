"""Reject malformed masters before their geometry reaches the native painter."""

import json
import math
from itertools import pairwise
from pathlib import Path

import pytest
from examples.ui_design.native.compile_icons import ROOT, compile_svg


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
    compiled = json.loads((ROOT / "icons.json").read_text())
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
    assert compiled["rotate"]["bounds"] == pytest.approx([3.125, 3.125, 20.875, 20.875])


def test_dash_and_alpha_survive_compilation():
    sources = json.loads((ROOT / "icons-source.json").read_text())
    perturb = compile_svg(sources["perturb"])["contours"][-1]
    assert len(perturb["stroke_paths"]) == 3
    assert perturb["stroke_paths"][0][-1] != perturb["stroke_paths"][1][0]
    shading = compile_svg(sources["shading"])["contours"][-1]
    assert shading["opacity"] == 0.5
    assert shading["stroke"] == "none"
