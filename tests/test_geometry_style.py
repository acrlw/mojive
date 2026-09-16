"""Geometry display settings validate values without changing authored materials."""

from dataclasses import asdict

import numpy as np
import pytest

from mojive import GeometryStyle, GeometryView, Scene
from mojive.render.builder import SceneSourceBuilder
from mojive.types import GeometryRole


@pytest.mark.parametrize(
    "field,value",
    [
        ("collision_color", (1, 0)),
        ("collision_color", (0, float("nan"), 1)),
        ("collision_color", (0, 2, 1)),
        ("visual_opacity", -0.1),
        ("collision_opacity", float("inf")),
        ("visual_opacity", 1.1),
    ],
)
def test_style_rejects_invalid_values(field, value):
    with pytest.raises(ValueError):
        GeometryStyle(**{field: value})


def test_style_round_trips_json_color_lists():
    style = GeometryStyle(collision_color=[0.2, 0.4, 0.6], visual_opacity=1)
    assert style.collision_color == (0.2, 0.4, 0.6)
    assert GeometryStyle(**asdict(style)) == style


def test_style_preserves_authored_colors_and_shared_geometry():
    scene = Scene()
    scene.box(name="visual", position=(1, 0, 0))
    scene.box(name="collision", position=(2, 0, 0))
    scene.box(name="shared", position=(3, 0, 0))
    source = scene.source
    source.geom_role = np.array(
        [GeometryRole.VISUAL, GeometryRole.COLLISION, GeometryRole.BOTH], np.uint8
    )
    before = source.geom_rgba.copy()
    style = GeometryStyle((0.2, 0.4, 0.6), visual_opacity=0.8, collision_opacity=0.2)
    builder = SceneSourceBuilder()
    builder.set_source(source)
    options = {
        "static": True,
        "skin": True,
        "flex_face": False,
        "flex_skin": True,
        "visual_geometry": True,
        "collision_geometry": True,
        "geometry_style": style,
    }
    assert builder.set_visual_options(**options)
    colors = builder.scene.colors
    np.testing.assert_allclose(sorted(colors[:, 3]), [-3.2, -1.8, 1])
    assert any(np.allclose(row[:3], np.power(style.collision_color, 2.2)) for row in colors)
    assert not builder.set_visual_options(**options)
    np.testing.assert_array_equal(source.geom_rgba, before)
    # A local Both override uses the same display settings as the global view.
    for node in source.nodes:
        node.geometry_view = GeometryView.BOTH
    options.update(visual_geometry=False, collision_geometry=False)
    builder.set_visual_options(**options)
    np.testing.assert_array_equal(builder.scene.colors, colors)
