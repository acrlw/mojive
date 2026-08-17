from __future__ import annotations

import pytest

from forge_viewer.render.forge.passes.outline import OUTLINE_RADIUS, circular_offsets
from forge_viewer.render.forge.picking import (
    pick,
    viewport_point_to_ndc,
    viewport_point_to_target_pixel,
)

PANEL = (100.0, 50.0, 800.0, 600.0)
RECT = (100.0, 150.0, 800.0, 400.0)
TARGET = (1600, 800)


def test_viewport_point_to_target_pixel_at_2x_dpi_with_letterbox():

    x, y, w, h = RECT
    cases = {
        "top left": ((x, y), (0, 799)),
        "top right": ((x + w, y), (1599, 799)),
        "bottom left": ((x, y + h), (0, 0)),
        "bottom right": ((x + w, y + h), (1599, 0)),
        "center": ((x + w / 2, y + h / 2), (800, 399)),
    }
    for point, expect in cases.values():
        assert viewport_point_to_target_pixel(point, RECT, TARGET) == expect


def test_target_pixel_flips_y():

    x, y, w, h = RECT
    top = viewport_point_to_target_pixel((x + w / 2, y + 1.0), RECT, TARGET)
    bottom = viewport_point_to_target_pixel((x + w / 2, y + h - 1.0), RECT, TARGET)
    assert top is not None and bottom is not None
    assert top[1] > bottom[1]

    assert top[1] == 797, top
    assert bottom[1] == 1, bottom


def test_points_on_the_black_bars_are_not_on_the_picture():

    px, py, pw, _ph = PANEL
    assert viewport_point_to_target_pixel((px + pw / 2, py + 10.0), RECT, TARGET) is None
    assert viewport_point_to_target_pixel((px - 5.0, py + 300.0), RECT, TARGET) is None


def test_ndc_is_dpi_free():

    x, y, w, h = RECT
    assert viewport_point_to_ndc((x + w / 2, y + h / 2), RECT) == pytest.approx((0.0, 0.0))
    assert viewport_point_to_ndc((x, y), RECT) == pytest.approx((-1.0, 1.0))
    assert viewport_point_to_ndc((x + w, y + h), RECT) == pytest.approx((1.0, -1.0))


class _Counter:
    def __init__(self, result: int) -> None:
        self.result = result
        self.calls = 0
        self.args: list[tuple[float, float]] = []

    def __call__(self, a: float, b: float) -> int:
        self.calls += 1
        self.args.append((a, b))
        return self.result


def test_level_one_hit_stops_there():

    gpu = _Counter(42)
    ray = _Counter(7)
    near = _Counter(9)
    x, y, w, h = RECT
    r = pick((x + w / 2, y + h / 2), RECT, TARGET, gpu, ray, near)

    assert (r.object_id, r.level) == (42, 1)
    assert gpu.calls == 1
    assert (ray.calls, near.calls) == (0, 0)
    assert gpu.args == [(800, 399)]


def test_falls_through_in_order():

    x, y, w, h = RECT
    center = (x + w / 2, y + h / 2)

    ray, near = _Counter(7), _Counter(9)
    r = pick(center, RECT, TARGET, _Counter(0), ray, near)
    assert (r.object_id, r.level) == (7, 2)
    assert (ray.calls, near.calls) == (1, 0)
    assert ray.args == [(0.0, 0.0)]

    ray, near = _Counter(0), _Counter(9)
    r = pick(center, RECT, TARGET, _Counter(0), ray, near)
    assert (r.object_id, r.level) == (9, 3)
    assert (ray.calls, near.calls) == (1, 1)

    r = pick(center, RECT, TARGET, _Counter(0), _Counter(0), _Counter(0))
    assert (r.object_id, r.level, r.hit) == (0, 0, False)


def test_missing_callbacks_do_not_crash():

    x, y, w, h = RECT
    r = pick((x + w / 2, y + h / 2), RECT, TARGET, _Counter(0))
    assert (r.object_id, r.level) == (0, 0)


def test_click_outside_the_rect_is_a_miss_and_reads_nothing():

    gpu, ray, near = _Counter(42), _Counter(7), _Counter(9)
    r = pick((PANEL[0] + 5.0, PANEL[1] + 5.0), RECT, TARGET, gpu, ray, near)
    assert (r.object_id, r.level, r.pixel) == (0, 0, None)
    assert (gpu.calls, ray.calls, near.calls) == (0, 0, 0)


def test_root_node_counts_as_a_miss():

    ray, near = _Counter(7), _Counter(9)
    x, y, w, h = RECT
    r = pick((x + w / 2, y + h / 2), RECT, TARGET, _Counter(1), ray, near, root_id=1)
    assert (r.object_id, r.hit) == (0, False)
    assert r.level == 1
    assert (ray.calls, near.calls) == (0, 0)

    r = pick((x + w / 2, y + h / 2), RECT, TARGET, _Counter(5), ray, near, root_id=1)
    assert r.object_id == 5


def test_root_filter_applies_to_every_level():

    x, y, w, h = RECT
    center = (x + w / 2, y + h / 2)
    r = pick(center, RECT, TARGET, _Counter(0), _Counter(1), _Counter(9), root_id=1)
    assert (r.object_id, r.level) == (0, 2)


def test_neighborhood_is_circular_not_square():

    offsets = circular_offsets(3)
    assert len(offsets) == 29
    assert len(set(offsets)) == len(offsets)
    assert len(list(range(-3, 4))) ** 2 == 49

    assert (3, 0) in offsets and (2, 2) in offsets
    assert (3, 1) not in offsets and (3, 3) not in offsets


def test_the_two_passes_register_themselves_under_the_right_names():

    from forge_viewer.render.forge import backend as fb
    from forge_viewer.render.forge.passes import idbuffer, outline  # noqa: F401

    reg = fb.registered()
    assert reg["id"]().name == "id"
    assert reg["outline"]().name == "outline"

    assert fb.PASS_ORDER.index("outline") < fb.PASS_ORDER.index("debug")
    assert fb.PASS_ORDER.index("debug") < fb.PASS_ORDER.index("gizmo")
    assert fb.PASS_ORDER.index("id") > fb.PASS_ORDER.index("opaque")


def test_neighborhood_radius_is_the_pixel_width():

    assert OUTLINE_RADIUS == 3
    for r in (1, 2, 3, 5):
        offsets = circular_offsets(r)
        assert max(abs(dx) for dx, _ in offsets) == r
        assert max(abs(dy) for _, dy in offsets) == r
        assert (0, 0) in offsets
