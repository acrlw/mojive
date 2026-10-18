"""Scene markers preserve the adopted UI masters through world projection."""

import json
import tomllib
from pathlib import Path

import numpy as np
import pytest
from examples.ui_design.native.compile_icons import (
    COMPILED_ICONS,
    SCENE_ICONS,
    compile_svg,
    library_masters,
)

from mojive.interaction.gizmo import project, scene_icon_paths, scene_icon_triangles
from mojive.interaction.scene_icons import (
    SCENE_ICON_GRID,
    scene_icon_data,
    scene_icon_geometry,
    scene_icon_strokes,
)
from mojive.types import CameraView


def test_packaged_scene_masters_match_the_svg_compiler_and_workspace_inventory():
    inventory = json.loads(COMPILED_ICONS.read_text())
    subset = json.loads(SCENE_ICONS.read_text())
    sources = library_masters()

    assert set(subset) == {"camera", "light"}
    for name, glyph in subset.items():
        assert glyph == inventory[name]
        assert glyph == json.loads(json.dumps(compile_svg(sources[name])))
    project_config = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
    assert (
        "python/interaction/resources/scene_icons.json"
        in project_config["tool"]["hatch"]["build"]["artifacts"]
    )


@pytest.mark.parametrize("name", ("camera", "light"))
@pytest.mark.parametrize("orthographic", (False, True))
@pytest.mark.parametrize("size", (14, 18, 24, 56, 112))
@pytest.mark.parametrize("ui_scale", (1.0, 1.5))
def test_world_markers_project_to_the_complete_uniformly_scaled_master(
    name, orthographic, size, ui_scale
):
    camera = CameraView(
        eye=np.zeros(3, np.float32),
        target=np.array((0.0, 0.0, -1.0), np.float32),
        up=np.array((0.0, 1.0, 0.0), np.float32),
        orthographic=orthographic,
        ortho_height=4.0,
        aspect=2.0,
    )
    rect = (0.0, 0.0, 1600.0, 800.0)
    anchors = np.array(((-0.4, 0.2, -2.0), (0.7, -0.3, -9.0)), np.float32)
    screen_anchors = project(camera, anchors, rect)
    scaled_size = size * ui_scale
    paths = scene_icon_paths(name, anchors, camera, rect[3], scaled_size)

    for (points, width, closed), master in zip(paths, scene_icon_strokes(name), strict=True):
        screen = project(camera, points, rect).reshape(len(anchors), -1, 3)
        expected_offsets = master.points * scaled_size / SCENE_ICON_GRID
        expected = screen_anchors[:, None, :2] + expected_offsets
        np.testing.assert_allclose(screen[:, :, :2], expected, atol=0.0002, rtol=0)
        assert width == pytest.approx(master.width * scaled_size / SCENE_ICON_GRID)
        assert closed is master.closed
    triangles = scene_icon_triangles(name, anchors, camera, rect[3], scaled_size)
    screen = project(camera, triangles, rect).reshape(len(anchors), -1, 3)
    expected_offsets = scene_icon_geometry(name).reshape(-1, 2) * scaled_size / SCENE_ICON_GRID
    expected = screen_anchors[:, None, :2] + expected_offsets
    np.testing.assert_allclose(screen[:, :, :2], expected, atol=0.0002, rtol=0)


@pytest.mark.parametrize("name", ("camera", "light"))
def test_master_arrays_are_reused_and_read_only(name):
    first = scene_icon_strokes(name)
    assert scene_icon_strokes(name) is first
    packed, spans = scene_icon_data(name)
    assert scene_icon_data(name)[0] is packed
    assert not packed.flags.writeable
    mesh = scene_icon_geometry(name)
    assert scene_icon_geometry(name) is mesh
    assert not mesh.flags.writeable
    for stroke, (start, stop, width, closed) in zip(first, spans, strict=True):
        assert not stroke.points.flags.writeable
        np.testing.assert_array_equal(packed[start:stop], stroke.points)
        assert (width, closed) == (stroke.width, stroke.closed)


def test_unknown_scene_master_does_not_silently_alias_an_unrelated_symbol():
    with pytest.raises(ValueError, match="Unknown scene icon"):
        scene_icon_strokes("capture-image")
