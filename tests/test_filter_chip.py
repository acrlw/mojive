"""Capsule filter content shares measured geometry with the native scene and guide."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
from imgui_bundle import imgui

from mojive.design.groups import group_scope
from mojive.design.native.geometry import Rect
from mojive.design.native.presentation import DesignPresentation
from mojive.design.runtime import DesignRuntime
from mojive.design.specification import DEFAULT
from mojive.design.store import SpecificationStore
from mojive.ui.imgui_draw import ImguiDraw2D
from mojive.ui.workspace.filter_chip import draw_filter_chip, filter_chip_size
from mojive.ui.workspace.widgets import UI


@pytest.fixture(params=(1, 1.2, 2.25))
def chips(request, tmp_path, monkeypatch):
    context = imgui.create_context()
    io = imgui.get_io()
    io.set_ini_filename(None)
    io.display_size = (900 * request.param, 700 * request.param)
    io.delta_time = 1 / 60
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    ui = UI(
        SimpleNamespace(style_scale=request.param, font_report=SimpleNamespace(cjk_path=None)),
        portable_fonts=True,
    )
    runtime = DesignRuntime(SpecificationStore(tmp_path / "design.json"))
    ui.design = DesignPresentation(ui, runtime)
    recorded, icons, captions = [], [], []
    original_record = ui.record_component_box
    original_icon = ui.icon
    original_text = ui.component_text

    def record(part, box, **options):
        recorded.append(box)
        return original_record(part, box, **options)

    def icon(name, x, y, size, *args, **options):
        icons.append(Rect(x, y, size * ui.s, size * ui.s))
        return original_icon(name, x, y, size, *args, **options)

    def caption(part, rect, value, *args, **options):
        captions.append((rect, value))
        return original_text(part, rect, value, *args, **options)

    monkeypatch.setattr(ui, "record_component_box", record)
    monkeypatch.setattr(ui, "icon", icon)
    monkeypatch.setattr(ui, "component_text", caption)

    def paint(count, *, owner="Scene", maximum_width=None, origin=(30, 50)):
        recorded.clear()
        icons.clear()
        captions.clear()
        imgui.new_frame()
        imgui.set_next_window_pos((0, 0))
        imgui.set_next_window_size(io.display_size)
        imgui.begin("Capsule test")
        try:
            with group_scope(owner):
                width, height = filter_chip_size(ui, count, maximum_width=maximum_width)
                draw_filter_chip(
                    ui,
                    "chip",
                    Rect(origin[0] * ui.s, origin[1] * ui.s, width, height),
                    count,
                    "geom",
                    "Geometries",
                    (150, 190, 140, 255),
                    False,
                )
            return recorded[-1], icons[-1], captions[-1]
        finally:
            imgui.end()
            imgui.render()

    try:
        yield ui, runtime, paint
    finally:
        imgui.destroy_context(context)


@pytest.mark.parametrize("height,icon_size", ((28, 16), (36, 16), (44, 20)))
def test_default_glyph_centers_on_capsule_end_circle_and_count_keeps_its_gap(
    chips, height, icon_size
):
    ui, runtime, paint = chips
    spec = deepcopy(runtime.spec)
    spec["parts"]["scene_type_filters_filter"] = {"height": height, "iconSize": icon_size}
    runtime.update_spec(spec)
    for count in (1, 52, 999):
        box, icon, (label, value) = paint(count)
        assert value == str(count)
        assert icon.x + icon.width / 2 == pytest.approx(box.frame.x + box.frame.height / 2)
        assert icon.y + icon.height / 2 == pytest.approx(box.frame.y + box.frame.height / 2)
        assert label.x - icon.right == pytest.approx(8 * ui.s)
        assert box.frame.right - label.right == pytest.approx((height - icon_size) / 2 * ui.s)
        assert box.padding[0] == pytest.approx((height - icon_size) / 2 * ui.s)
        assert label.width > 0


def test_capsule_group_padding_gap_and_fixed_width_remain_independent(chips):
    ui, runtime, paint = chips
    scene_before = paint(52)
    peer_before = paint(52, owner="Inspector")
    spec = deepcopy(runtime.spec)
    spec["parts"]["scene_type_filters_filter"] = {
        "height": 36,
        "paddingLeft": 13,
        "paddingRight": 17,
        "gap": 11,
        "widthMode": "fixed",
        "width": 100,
    }
    runtime.update_spec(spec)
    box, icon, (label, _value) = paint(52)
    assert box.frame.width == pytest.approx(100 * ui.s)
    assert box.frame.height == pytest.approx(36 * ui.s)
    assert icon.x - box.frame.x == pytest.approx(13 * ui.s)
    assert label.x - icon.right == pytest.approx(11 * ui.s)
    assert box.padding[2] == pytest.approx(17 * ui.s)
    assert paint(52, owner="Inspector") == peer_before
    assert paint(52, maximum_width=75 * ui.s)[0].frame.width == pytest.approx(75 * ui.s)
    assert box != scene_before[0]


@pytest.mark.parametrize("family", ("system", "portable", "mono"))
@pytest.mark.parametrize("font_size", (12, 14))
def test_fractional_chip_origins_keep_complete_emitted_counts(
    chips, monkeypatch, family, font_size
):
    ui, runtime, paint = chips
    ui.prepare_fonts("portable", (family,))
    spec = deepcopy(runtime.spec)
    spec["parts"]["scene_type_filters_filter"] = {"fontFamily": family, "fontSize": font_size}
    runtime.update_spec(spec)
    emitted = []
    original = ImguiDraw2D.text

    def text(draw, position, color, value, **options):
        emitted.append(value)
        return original(draw, position, color, value, **options)

    monkeypatch.setattr(ImguiDraw2D, "text", text)
    for origin in ((20, 50), (20.2, 50), (110.2, 50), (290.6, 50), (740.8, 50)):
        for count in (6, 11, 19, 21, 23, 45, 52, 999):
            emitted.clear()
            box, _icon, (label, value) = paint(count, origin=origin)
            assert value == str(count)
            assert emitted == [str(count)]
            assert label.x >= box.content.x
            assert label.right <= box.content.right + 1e-6


def test_g3_is_opt_in_without_removing_saved_smoothing_options():
    assert DEFAULT["native"] == {
        "g3": False,
        "g3_smoothing": 0.618,
        "g3_capsule_smoothing": 0.382,
        "g3_capsule_optical": True,
    }
