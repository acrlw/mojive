"""Fixed SVG topology is packaged once; display size only changes its AA fringe."""

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from mojive.geometry2d import icon_mesh as compiler
from mojive.ui.workspace import icon_mesh, widget_icons
from tests.curve_assertions import assert_paths_close


def _assert_stroke_matches(mesh, reference):
    # Native tessellation normalizes to float32. Equivalent triangulations and
    # boundary start vertices can differ between compiler/platform combinations.
    tolerance = 24 * np.finfo(np.float32).eps * 16

    def area(value):
        vertices, indices = np.asarray(value["vertices"]), np.asarray(value["indices"])
        assert np.isfinite(vertices).all()
        assert indices.min() >= 0 and indices.max() < len(vertices)
        triangle_indices = indices.reshape(-1, 3)
        edges = Counter(
            tuple(sorted((int(a), int(b))))
            for triangle in triangle_indices
            for a, b in zip(triangle, np.roll(triangle, -1), strict=True)
        )
        assert max(edges.values()) <= 2
        boundary_length = sum(
            np.linalg.norm(vertices[a] - vertices[b])
            for (a, b), count in edges.items()
            if count == 1
        )
        outline_length = sum(
            np.linalg.norm(np.asarray(points) - np.roll(points, 1, axis=0), axis=1).sum()
            for points, _inside in value["boundaries"]
        )
        assert boundary_length == pytest.approx(outline_length, abs=len(vertices) * tolerance)
        triangles = vertices[triangle_indices]
        a, b = triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
        areas = (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]) / 2
        assert areas.min() >= -(tolerance**2)
        return areas.sum()

    vertices, expected = np.asarray(mesh["vertices"]), np.asarray(reference["vertices"])
    np.testing.assert_allclose(vertices.min(axis=0), expected.min(axis=0), atol=tolerance, rtol=0)
    np.testing.assert_allclose(vertices.max(axis=0), expected.max(axis=0), atol=tolerance, rtol=0)
    remaining = list(reference["boundaries"])
    perimeter = 0.0
    for points, hole in mesh["boundaries"]:
        points = np.asarray(points)
        # Match by geometric bounds, independently of contour order and start.
        match = next(
            i
            for i, (outline, inside) in enumerate(remaining)
            if hole == inside
            and np.allclose(np.min(outline, axis=0), points.min(axis=0), atol=tolerance, rtol=0)
            and np.allclose(np.max(outline, axis=0), points.max(axis=0), atol=tolerance, rtol=0)
        )
        outline, _inside = remaining.pop(match)
        assert_paths_close(points, outline, tolerance)
        perimeter += np.linalg.norm(points - np.roll(points, 1, axis=0), axis=1).sum()
    assert not remaining
    assert area(mesh) == pytest.approx(area(reference), abs=perimeter * tolerance)


def test_packaged_meshes_match_every_canonical_master():
    layers = icon_mesh._packaged_layers()
    for name, glyph in widget_icons.GLYPHS.items():
        expected = compiler.compile_icon_layers(glyph["contours"])
        assert len(layers[name]) == len(expected), name
        for actual, compiled in zip(layers[name], expected, strict=True):
            assert actual["background"] == compiled["background"]
            assert actual["opacity"] == compiled["opacity"]
            if "fill" in compiled:
                np.testing.assert_allclose(actual["fill"], compiled["fill"], atol=1e-6, rtol=0)
                continue
            mesh, reference = actual["stroke"], compiled["stroke"]
            _assert_stroke_matches(mesh, reference)


def test_mesh_comparison_accepts_equivalent_diagonals_and_rejects_missing_fill():
    points = ((0, 0), (2, 0), (2, 2), (0, 2))
    reference = {
        "vertices": points,
        "indices": (0, 1, 2, 0, 2, 3),
        "boundaries": ((points, False),),
    }
    reordered = (*points[2:], *points[:2])
    mesh = {
        "vertices": reordered,
        "indices": (0, 1, 3, 1, 2, 3),
        "boundaries": ((reordered, False),),
    }
    _assert_stroke_matches(mesh, reference)
    for indices in ((0, 1, 3), (0, 1, 3, 0, 1, 3)):
        mesh["indices"] = indices
        with pytest.raises(AssertionError):
            _assert_stroke_matches(mesh, reference)


def test_first_paint_and_scale_changes_do_not_compile_strokes(monkeypatch):
    def unexpected_compilation(*args, **kwargs):
        pytest.fail("An unchanged packaged icon compiled geometry on first paint")

    monkeypatch.setattr(compiler, "compile_stroke", unexpected_compilation)
    compiler.stroke_mesh.cache_clear()
    icon_mesh.prepared_icon.cache_clear()
    from imgui_bundle import imgui

    context = imgui.create_context()
    try:
        for name in widget_icons.GLYPHS:
            for size in (14, 18, 24, 31.5):
                assert icon_mesh.prepared_icon(name, size)
    finally:
        imgui.destroy_context(context)


@pytest.mark.parametrize("mismatch", ("version", "source", "inventory"))
def test_stale_mesh_package_fails_before_paint(tmp_path, monkeypatch, mismatch):
    source = Path(icon_mesh.__file__).with_name("icons.json").read_bytes()
    meshes = {
        "version": 1,
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "icons": dict(icon_mesh._packaged_layers()),
    }
    if mismatch == "version":
        meshes["version"] = 0
    elif mismatch == "source":
        source += b" "
    else:
        meshes["icons"].pop("settings")
    (tmp_path / "icons.json").write_bytes(source)
    (tmp_path / "icon_meshes.json").write_text(json.dumps(meshes))
    monkeypatch.setattr(icon_mesh, "__file__", str(tmp_path / "icon_mesh.py"))
    icon_mesh._packaged_layers.cache_clear()
    try:
        with pytest.raises(ValueError, match="make ui-design-icons"):
            icon_mesh._packaged_layers()
    finally:
        icon_mesh._packaged_layers.cache_clear()


def test_failed_compilation_leaves_existing_resource_set_untouched(monkeypatch):
    from examples.ui_design.native import compile_icons

    def failed_compilation(*args, **kwargs):
        raise ValueError("Invalid stroke topology")

    def premature_write(*args, **kwargs):
        pytest.fail("Resources were changed before every mesh compiled successfully")

    monkeypatch.setattr(compiler, "compile_icon_layers", failed_compilation)
    monkeypatch.setattr(Path, "write_text", premature_write)
    with pytest.raises(ValueError, match="Invalid stroke topology"):
        compile_icons.main()
