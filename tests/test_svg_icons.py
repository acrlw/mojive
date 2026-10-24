"""SVG assets preserve production placement, holes and independent file edits."""

from dataclasses import replace
from xml.etree import ElementTree as ET

import numpy as np
import pytest

from mojive.geometry2d.polygons import signed_polygon_area
from mojive.tools.svg_icons import (
    ICON_NAMES,
    export_icon,
    export_icons,
    parse_svg,
    prepare_svg,
)
from mojive.tools.ui_feasibility.svg_icons import SvgStudy
from mojive.ui.icons import drawing as icon_drawing
from mojive.ui.workspace.icon_mesh import _packaged_layers
from mojive.ui.workspace.widget_icons import COMPILED_ALIASES, GLYPHS


@pytest.mark.parametrize("name", ICON_NAMES)
def test_exported_production_assets_have_finite_geometry_and_preserve_holes(name):
    source = export_icon(name)
    icon = parse_svg(source)
    bounds = np.array(icon.bounds)
    assert np.isfinite(bounds).all()
    assert np.all(bounds[:2] >= -0.01)
    assert np.all(bounds[2:] <= 24.01)
    for size in (14, 24, 56, 112):
        prepared = prepare_svg(icon, size)
        for shape, vertices, indices, loops, _contours in prepared:
            if shape.primitive != "fill":
                assert np.isfinite(shape.stroke_width)
                if shape.circle:
                    assert np.isfinite(_contours).all()
                continue
            points = np.array(vertices)
            triangles = points[np.array(indices).reshape(-1, 3)]
            ab, ac = triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
            area = np.abs(ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0]).sum() / 2
            expected = sum(signed_polygon_area(path) for path in shape.contours) * (size / 24) ** 2
            assert area == pytest.approx(expected, rel=2e-5, abs=2e-5)
            assert len(loops) == len(shape.contours)
            assert sum(inside for _points, inside in loops) == sum(
                signed_polygon_area(path) < 0 for path in shape.contours
            )
        assert prepare_svg(icon, size) is prepared


def test_svg_export_ignores_retained_experimental_optical_translation(monkeypatch):
    original = parse_svg(export_icon("playback-play"))
    monkeypatch.setattr(icon_drawing, "production_icon_offset", lambda _name: (2.0, -1.0))
    shifted = parse_svg(export_icon("playback-play"))
    assert np.array(shifted.bounds) - np.array(original.bounds) == pytest.approx([0, 0, 0, 0])


@pytest.mark.parametrize("kind", ("info", "warning", "error"))
def test_svg_severity_uses_the_reviewed_output_master(kind):
    name = COMPILED_ALIASES[f"status-{kind}"]
    icon = parse_svg(export_icon(f"status-{kind}"))
    layers = _packaged_layers()[name]
    assert len(icon.shapes) == len(layers)
    for shape, layer in zip(icon.shapes, layers, strict=True):
        mesh = layer.get("mesh", layer.get("stroke"))
        assert len(shape.contours) == len(mesh["boundaries"])
        for actual, (expected, _inside) in zip(shape.contours, mesh["boundaries"], strict=True):
            assert np.asarray(actual) == pytest.approx(np.asarray(expected), abs=1e-7)


def test_svg_diagnostics_have_distinct_frames_and_open_marker_surrounds():
    icons = [parse_svg(export_icon(f"status-{kind}")) for kind in ("info", "warning", "error")]
    assert len({repr(icon.shapes[0].contours) for icon in icons}) == 3
    for icon in icons:
        assert any(signed_polygon_area(path) < 0 for path in icon.shapes[0].contours)


@pytest.mark.parametrize("name", ("tool-scale", "tool-snap"))
def test_tool_exports_keep_canonical_placement_without_per_icon_fitting(name):
    icon = parse_svg(export_icon(name))
    bounds = GLYPHS[COMPILED_ALIASES[name]]["bounds"]
    assert icon.bounds == pytest.approx(bounds, abs=0.02)


def test_rectangular_viewbox_is_centered_without_stretching():
    source = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="10 20 40 20">'
        '<path d="M 10 20 L 50 20 L 50 40 L 10 40 Z"/></svg>'
    )
    icon = parse_svg(source)
    assert icon.bounds == pytest.approx((0, 6, 24, 18))


@pytest.mark.parametrize(
    "unsupported",
    (
        '<path d="M 0 0 C 1 1 2 2 3 3 Z"/>',
        '<path d="M 0 0 L 1 0 L 0 1 Z" transform="translate(1)"/>',
        '<path d="M 0 0 L 1 0 L 0 1 Z" stroke="red"/>',
        '<image href="icon.png"/>',
        '<path fill="none" stroke="currentColor" d="M 0 0 L 1 0 L 1 1 Z M 2 2 L 3 3"/>',
    ),
)
def test_unsupported_svg_features_fail_explicitly(unsupported):
    with pytest.raises(ValueError):
        parse_svg(
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">{unsupported}</svg>'
        )


@pytest.mark.integration
def test_reload_uses_file_edits_and_retains_previous_set_after_invalid_svg(tmp_path):
    manifest = export_icons(tmp_path)
    assert len(manifest["icons"]) == len(ICON_NAMES)
    study = SvgStudy(directory=tmp_path)
    study.reload()
    original = study.icons["playback-play"]
    file = tmp_path / "playback-play.svg"
    root = ET.fromstring(file.read_text())
    path = next(element for element in root if element.tag.endswith("}path"))
    path.set("d", "M 8 8 L 16 8 L 12 16 Z")
    file.write_text(ET.tostring(root, encoding="unicode"))
    study.reload()
    assert not study.error
    assert study.icons["playback-play"].bounds == pytest.approx((8, 8, 16, 16))
    assert prepare_svg(study.icons["playback-play"], 24) != prepare_svg(original, 24)
    complete = study.icons
    file.write_text("<svg>")
    study.reload()
    assert study.error
    assert study.icons is complete


def test_prepared_geometry_is_independent_of_color_role():
    icon = parse_svg(export_icon("status-mouse-left"))
    assert {shape.role for shape in icon.shapes} == {"foreground"}
    edited = replace(icon, shapes=tuple(replace(shape, role="accent") for shape in icon.shapes))
    for before, after in zip(prepare_svg(icon, 24), prepare_svg(edited, 24), strict=True):
        assert before[1:] == after[1:]
