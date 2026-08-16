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
        "左上": ((x, y), (0, 799)),
        "右上": ((x + w, y), (1599, 799)),
        "左下": ((x, y + h), (0, 0)),
        "右下": ((x + w, y + h), (1599, 0)),
        "正中": ((x + w / 2, y + h / 2), (800, 399)),
    }
    for name, (point, expect) in cases.items():
        assert viewport_point_to_target_pixel(point, RECT, TARGET) == expect, f"{name} 换算错了"


def test_target_pixel_flips_y():

    x, y, w, h = RECT
    top = viewport_point_to_target_pixel((x + w / 2, y + 1.0), RECT, TARGET)
    bottom = viewport_point_to_target_pixel((x + w / 2, y + h - 1.0), RECT, TARGET)
    assert top is not None and bottom is not None
    assert top[1] > bottom[1], "画面上方的点必须落在更大的行号上（GL 原点在左下）"

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
    assert gpu.calls == 1, "第 1 级只读一个像素，只该问一次"
    assert (ray.calls, near.calls) == (0, 0), "第 1 级已经与画面逐像素一致了，后面不该再问"
    assert gpu.args == [(800, 399)], "第 1 级拿到的必须是换算过的目标像素，不是窗口坐标"


def test_falls_through_in_order():

    x, y, w, h = RECT
    center = (x + w / 2, y + h / 2)

    ray, near = _Counter(7), _Counter(9)
    r = pick(center, RECT, TARGET, _Counter(0), ray, near)
    assert (r.object_id, r.level) == (7, 2), "ID buffer 空的时候该落到射线那一级"
    assert (ray.calls, near.calls) == (1, 0)
    assert ray.args == [(0.0, 0.0)], "射线那一级吃的是 NDC，中心点就是 (0, 0)"

    ray, near = _Counter(0), _Counter(9)
    r = pick(center, RECT, TARGET, _Counter(0), ray, near)
    assert (r.object_id, r.level) == (9, 3), "前两级都空才轮到兜底"
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
    assert (r.object_id, r.hit) == (0, False), "点到地面必须是没点中"
    assert r.level == 1, "作出结论的是第 1 级"
    assert (ray.calls, near.calls) == (0, 0), "第 1 级已经作了结论，后两级不该再问"

    r = pick((x + w / 2, y + h / 2), RECT, TARGET, _Counter(5), ray, near, root_id=1)
    assert r.object_id == 5


def test_root_filter_applies_to_every_level():

    x, y, w, h = RECT
    center = (x + w / 2, y + h / 2)
    r = pick(center, RECT, TARGET, _Counter(0), _Counter(1), _Counter(9), root_id=1)
    assert (r.object_id, r.level) == (0, 2)


def test_neighborhood_is_circular_not_square():

    offsets = circular_offsets(3)
    assert len(offsets) == 29, f"圆形邻域应当是 29 个像素，实为 {len(offsets)}"
    assert len(set(offsets)) == len(offsets), "偏移表里有重复"
    assert len(list(range(-3, 4))) ** 2 == 49, "方形邻域是 49——这是被换掉的那个方案"

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
