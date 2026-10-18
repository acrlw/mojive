"""Authored navigation frames retain their size without native corner clamping."""

from copy import deepcopy

import pytest

from mojive.ui.workspace.menu_geometry import native_epsilon
from mojive.ui.workspace.widgets import color
from tests.test_activity_rail_insets import rail as rail
from tests.test_menu_corner_offsets import _corner_point, _paint_covers


@pytest.mark.parametrize("dimension", ("width", "height"))
@pytest.mark.parametrize("g3", (False, True))
def test_small_authored_rail_frames_reserve_the_actual_parent_offset(
    rail, monkeypatch, dimension, g3
):
    ui, runtime = rail.ui, rail.runtime
    spec = deepcopy(runtime.spec)
    spec["native"]["g3"] = g3
    spec["parts"].setdefault("rail", {})["radius"] = 40.25
    spec["parts"]["icon_button"] = {
        dimension: 20,
        "iconSize": 24 if dimension == "width" else 12,
    }
    runtime.update_spec(spec)
    painted = []
    rectangle = ui.rect

    def capture(x, y, width, height, tint, r=None, **options):
        draw = options.get("draw") or ui.draw
        ink = tint if options.get("color_resolved") else ui.ink(tint)
        result = rectangle(x, y, width, height, tint, r, **options)
        painted.append(((x, y, width, height), (r or 0) * ui.s, color(ink), draw))
        return result

    monkeypatch.setattr(ui, "rect", capture)
    rail.frame(4)
    bounds = ui.hits["rail-Settings"]
    record = rail.record
    assert bounds[2 if dimension == "width" else 3] == pytest.approx(20 * ui.s)
    assert record["outerRadius"] == 40.25
    assert len(set(record["padding"])) == 1
    inset = record["padding"][0]
    assert inset > 8 * ui.s
    assert record["cornerInset"] == pytest.approx(inset)
    assert record["itemRadius"] == pytest.approx(record["outerRadius"] - inset)
    assert record["itemRadius"] <= min(bounds[2:]) / 2 - 1
    ox, oy, width, height = record["outer"]
    x, y, child_width, child_height = bounds
    tolerance = native_epsilon(oy + height, y + child_height)
    assert x - ox == pytest.approx(inset, abs=tolerance)
    assert oy + height - y - child_height == pytest.approx(inset, abs=tolerance)
    assert width >= child_width + 2 * inset
    child = next(p for p in reversed(painted) if p[0] == pytest.approx(bounds))
    assert child[1] == pytest.approx(record["itemRadius"])
    if not g3:
        _bounds, radius, ink, draw = child
        diagonal = radius * (1 - 0.5**0.5)
        for right, bottom in ((False, False), (True, False), (False, True), (True, True)):
            assert not _paint_covers(
                draw, _corner_point(bounds, right, bottom, diagonal - 0.8), ink
            )
            assert _paint_covers(draw, _corner_point(bounds, right, bottom, diagonal + 0.8), ink)


@pytest.mark.parametrize("radius", (16, 16.25))
def test_saved_or_live_authored_rail_radius_remains_in_native_window_points(rail, radius):
    rail.frame(3)
    spec = deepcopy(rail.runtime.spec)
    spec["parts"].setdefault("rail", {})["radius"] = radius
    rail.runtime.update_spec(spec)
    rail.frame(3)
    assert rail.record["outerRadius"] == radius
    assert rail.record["itemRadius"] == pytest.approx(max(0, radius - rail.record["cornerInset"]))
    assert rail.record["padding"][0] == pytest.approx(6 * rail.ui.s)
