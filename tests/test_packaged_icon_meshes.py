"""Fixed SVG topology is packaged once; display size only changes its AA fringe."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from mojive.geometry2d import icon_mesh as compiler
from mojive.ui.workspace import icon_mesh, widget_icons


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
            assert mesh["indices"] == list(reference["indices"])
            np.testing.assert_allclose(mesh["vertices"], reference["vertices"], atol=1e-6, rtol=0)
            for (points, hole), (outline, inside) in zip(
                mesh["boundaries"], reference["boundaries"], strict=True
            ):
                assert hole == inside
                np.testing.assert_allclose(points, outline, atol=1e-6, rtol=0)


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
