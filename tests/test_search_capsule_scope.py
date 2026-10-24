"""Search capsules keep their actual background and focus shape inside panels."""

from copy import deepcopy
from math import sqrt

import pytest
from imgui_bundle import imgui

from mojive.design.native.geometry import Rect
from mojive.geometry2d.curves import smooth_capsule_points
from mojive.ui import compound_fields
from mojive.ui.workspace.widgets import color
from tests.test_component_appearance import appearance as appearance
from tests.test_settings_corner_chain import _paint_covers

pytestmark = pytest.mark.parametrize("appearance", (1, 1.2, 1.25, 2.25), indirect=True)


def _capsule_diagonal(bounds, runtime):
    x, y, width, height = bounds
    if not runtime.spec["native"]["g3"]:
        return (min(width, height) / 2 - 1) * (1 - sqrt(0.5))
    points = smooth_capsule_points(
        x, y, width, height, runtime.spec["native"]["g3_capsule_smoothing"]
    )
    candidates = []
    for a, b in zip(points, (*points[1:], points[0]), strict=True):
        start, end = a[0] + a[1] - x - width - y, b[0] + b[1] - x - width - y
        if start * end <= 0 and start != end:
            t = start / (start - end)
            inset = x + width - a[0] - t * (b[0] - a[0])
            if 0 <= inset <= height / 2:
                candidates.append(inset)
    return min(candidates)


@pytest.mark.parametrize("g3", (False, True))
@pytest.mark.parametrize("height", (28, 40))
@pytest.mark.parametrize("parent_radius", (0, 3))
def test_actual_search_background_and_keyboard_focus_remain_capsules(
    appearance, monkeypatch, g3, height, parent_radius
):
    ui, runtime, update, frame = appearance
    runtime.observe_components = True
    update({"search": {"height": height}})
    spec = deepcopy(runtime.spec)
    spec["native"]["g3"] = g3
    runtime.update_spec(spec)
    io = imgui.get_io()
    io.config_flags |= imgui.ConfigFlags_.nav_enable_keyboard
    painted, focused = [], []
    rectangle, focus = ui.rect, compound_fields.draw_focus_frame
    request_focus = False

    def rect(x, y, width, height, tint, r=None, **options):
        resolved = tint if options.get("color_resolved") else ui.ink(tint)
        painted.append(((x, y, width, height), r, color(resolved)))
        return rectangle(x, y, width, height, tint, r, **options)

    def focus_frame(lo, hi, **options):
        draw = imgui.get_window_draw_list()
        first = len(draw.vtx_buffer)
        thickness = max(1, imgui.get_font_size() / 13)
        focus(lo, hi, **options)
        vertices = [(v.pos.x, v.pos.y, v.col) for v in list(draw.vtx_buffer)[first:]]
        if vertices:
            focused.append((tuple(lo), tuple(hi), options["rounding"], thickness, vertices))

    monkeypatch.setattr(ui, "rect", rect)
    monkeypatch.setattr(compound_fields, "draw_focus_frame", focus_frame)

    def paint():
        nonlocal request_focus
        ui.design.begin_frame()
        painted.clear()
        focused.clear()
        with ui.corner_scope(parent_radius * ui.s, ("input_text",)):
            if request_focus:
                imgui.set_keyboard_focus_here()
                request_focus = False
            ui.search("capsule-search", 30 * ui.s, 40 * ui.s, 300 * ui.s, "Filter")
            ui.input_text("ordinary-text", 30 * ui.s, 110 * ui.s, 300 * ui.s, "Value")

    for _ in range(3):
        frame(paint)
    bounds = ui.hits["capsule-search"]
    search = [p for p in painted if p[0] == pytest.approx(bounds)]
    assert search and all(p[1] * ui.s == pytest.approx(bounds[3] / 2) for p in search)
    ordinary = [p for p in painted if p[0] == pytest.approx(ui.hits["ordinary-text"])]
    assert ordinary and all(p[1] == pytest.approx(parent_radius) for p in ordinary)
    draw = imgui.internal.find_window_by_name("Appearance").draw_list
    x, y, width, actual_height = bounds
    diagonal = _capsule_diagonal(bounds, runtime)
    assert not _paint_covers(draw, (x + width - diagonal + 0.9, y + diagonal - 0.9), search[-1][2])
    assert _paint_covers(draw, (x + width - diagonal - 0.9, y + diagonal + 0.9), search[-1][2])

    request_focus = True
    for _ in range(3):
        frame(paint)
    assert imgui.get_current_context().nav_cursor_visible
    lo, hi, rounding, thickness, vertices = next(
        sample for sample in focused if sample[0] == pytest.approx(bounds[:2])
    )
    assert rounding == pytest.approx(actual_height / 2)
    inset = thickness / 2 + 0.5
    # Inspect the real stroke mesh, including native PathRect's one-unit cap.
    # Its opaque outer arc must clear the corner by the capsule's diagonal.
    actual_radius = actual_height / 2 - inset - 1
    expected = 2 * (inset + actual_radius * (1 - sqrt(0.5))) - thickness / sqrt(2)
    corner_vertices = [
        (hi[0] - vx) + (vy - lo[1])
        for vx, vy, ink in vertices
        if vx >= hi[0] - actual_height / 2 and vy <= lo[1] + actual_height / 2 and ink >> 24 == 255
    ]
    assert corner_vertices
    assert min(corner_vertices) == pytest.approx(expected, abs=1)


@pytest.mark.parametrize("parent_disabled", (False, True))
def test_disabled_search_keeps_glyph_and_clear_hit_regions_inert(appearance, parent_disabled):
    ui, _runtime, _update, frame = appearance
    enabled = False
    value = "Filter"

    def paint():
        nonlocal value
        ui.design.begin_frame()
        imgui.begin_disabled(parent_disabled and not enabled)
        result = ui.design.primitives.input_text(
            "disabled-search",
            Rect(30 * ui.s, 40 * ui.s, 300 * ui.s, 28 * ui.s),
            value,
            enabled=enabled or parent_disabled,
            parts=("input_text", "search"),
            icon="search",
        )
        imgui.end_disabled()
        value = result.value

    def click(point):
        io = imgui.get_io()
        io.add_mouse_pos_event(*point)
        frame(paint)
        io.add_mouse_button_event(0, True)
        frame(paint)
        io.add_mouse_button_event(0, False)
        frame(paint)
        frame(paint)

    for _ in range(3):
        frame(paint)
    x, y, width, height = ui.hits["disabled-search"]
    for point in ((x + height / 2, y + height / 2), (x + width - height / 2, y + height / 2)):
        click(point)
        assert value == "Filter"
        assert imgui.get_current_context().active_id == 0
    enabled = True
    frame(paint)
    click((x + width - height / 2, y + height / 2))
    assert value == ""
