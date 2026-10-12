"""Native orientation widget drawing and hit regions share their scale."""

from unittest.mock import Mock

import numpy as np
import pytest
from examples.ui_design.native.scene_preview import ReferenceViewCube

from mojive import CameraView
from mojive.ui import viewcube


@pytest.mark.parametrize(
    "rect,effective_scale", [((0, 0, 342, 424), 1.4), ((0, 0, 1200, 900), 2.0)]
)
def test_reference_viewcube_draw_and_hit_regions_use_the_same_scale(rect, effective_scale):
    camera = CameraView(eye=np.array([4.0, -4.0, 3.0]), target=np.zeros(3))
    cube = ReferenceViewCube(selection_padding=1.2)
    cube.update(camera, rect, (-100, -100), 2.0)
    assert cube.reference_scale == effective_scale
    center = viewcube.widget_center(rect, effective_scale)
    expected = viewcube.layout(
        camera, center, viewcube.RADIUS_PT * effective_scale, viewcube.BALL_PT * effective_scale
    )
    for actual, reference in zip(cube.balls, expected, strict=True):
        assert (actual.axis, actual.sign) == (reference.axis, reference.sign)
        assert actual.screen == pytest.approx(reference.screen)
        assert actual.radius == pytest.approx(reference.radius)
    front = min(cube.balls, key=lambda ball: ball.depth)
    hovered = cube.update(camera, rect, front.screen, 2.0)
    assert hovered is not None
    assert (hovered.axis, hovered.sign) == (front.axis, front.sign)
    overlay = Mock()
    cube.draw(overlay, 2.0)
    backdrop = overlay.circle_filled.call_args_list[0]
    assert backdrop.args[0] == pytest.approx(center)
    assert backdrop.args[1] == pytest.approx(viewcube.BACKDROP_RADIUS_PT * effective_scale)
