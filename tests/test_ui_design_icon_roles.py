"""Native icon roles preserve the canonical family's proportions and weight."""

from types import SimpleNamespace

import pytest
from imgui_bundle import imgui

from mojive.ui.workspace import widgets
from mojive.ui.workspace.widgets import (
    GLYPHS,
    ICON_GRID,
    ICON_ROLES,
    ICON_STROKE,
    UI,
    icon_geometry,
    icon_role_canvas,
)


@pytest.fixture(params=(0.65, 1.0, 1.25, 1.5, 2.25))
def role_ui(request):
    context = imgui.create_context()
    io = imgui.get_io()
    io.set_ini_filename(None)
    io.display_size = (800 * request.param, 400 * request.param)
    io.delta_time = 1 / 60
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    ui = UI.__new__(UI)
    ui.s, ui.hits, ui.translate = request.param, {}, str
    ui.body = ui.bold = ui.mono = io.fonts.add_font_default()
    imgui.get_style().font_size_base = 14 * ui.s

    def frame(paint):
        imgui.new_frame()
        imgui.set_next_window_pos((0, 0))
        imgui.set_next_window_size(io.display_size)
        imgui.begin("Icon roles", flags=imgui.WindowFlags_.no_title_bar)
        result = paint()
        imgui.end()
        imgui.render()
        return result

    frame(lambda: None)
    try:
        yield SimpleNamespace(ui=ui, io=io, frame=frame)
    finally:
        imgui.destroy_context(context)


def _bounds(draw, first, *, ink=False):
    points = [
        draw.vtx_buffer[index].pos
        for index in range(first, len(draw.vtx_buffer))
        if not ink or draw.vtx_buffer[index].col >> 24
    ]
    assert points
    return (
        min(point.x for point in points),
        min(point.y for point in points),
        max(point.x for point in points),
        max(point.y for point in points),
    )


def _vertices(draw, first):
    return [
        (draw.vtx_buffer[index].pos.x, draw.vtx_buffer[index].pos.y, draw.vtx_buffer[index].col)
        for index in range(first, len(draw.vtx_buffer))
    ]


def _opaque_area(draw, first_index):
    area = 0.0
    indices = list(draw.idx_buffer)[first_index:]
    for index in range(0, len(indices), 3):
        a, b, c = (draw.vtx_buffer[i] for i in indices[index : index + 3])
        if all(vertex.col >> 24 == 255 for vertex in (a, b, c)):
            area += (
                abs(
                    (b.pos.x - a.pos.x) * (c.pos.y - a.pos.y)
                    - (b.pos.y - a.pos.y) * (c.pos.x - a.pos.x)
                )
                / 2
            )
    return area


@pytest.mark.parametrize("role", tuple(ICON_ROLES))
def test_icon_role_uses_one_scale_for_all_master_dimensions(role):
    spec = ICON_ROLES[role]
    for name, glyph in GLYPHS.items():
        canvas = icon_role_canvas(role)
        assert canvas == spec.canvas_size
        for contour, _points, _paths, width in icon_geometry(name, canvas):
            assert width == pytest.approx(contour["width"] * canvas / ICON_GRID)
        expected_stroke = (
            1.0
            if name in ("status-mouse-left", "status-mouse-right", "status-mouse-wheel")
            else ICON_STROKE
        )
        assert all(
            contour["width"] == expected_stroke
            for contour in glyph["contours"]
            if contour["stroke"] != "none"
        )
    assert (
        ICON_ROLES["activity"].canvas_size
        >= ICON_ROLES["title"].canvas_size
        > ICON_ROLES["toolbar"].canvas_size
        >= ICON_ROLES["subtle"].canvas_size
        > ICON_ROLES["status"].canvas_size
    )
    assert ICON_ROLES["toolbar"].canvas_size >= ICON_ROLES["inline"].canvas_size
    assert all(spec.canvas_size <= spec.slot_size for spec in ICON_ROLES.values())


@pytest.mark.parametrize("role", tuple(ICON_ROLES))
def test_every_glyph_retains_master_geometry_and_fits_its_role_at_supported_scales(role_ui, role):
    ui, frame = role_ui.ui, role_ui.frame
    spec = ICON_ROLES[role]
    x, y, slot = 80 * ui.s, 90 * ui.s, spec.slot_size * ui.s

    def paint():
        for name in GLYPHS:
            first = len(ui.draw.vtx_buffer)
            ui.role_icon(name, x, y, role=role)
            vertices = _vertices(ui.draw, first)
            left, top, right, bottom = _bounds(ui.draw, first)
            ink_left, ink_top, ink_right, ink_bottom = _bounds(ui.draw, first, ink=True)
            assert (
                ink_left >= x and ink_top >= y and ink_right <= x + slot and ink_bottom <= y + slot
            ), (name, role)
            # Transparent fringe vertices can extend beyond the visible mark at tiny scales.
            assert (
                left >= x - 1 and top >= y - 1 and right <= x + slot + 1 and bottom <= y + slot + 1
            ), (name, role)
            # A role places the master unchanged. It does not fit or recenter the
            # tessellated ink independently for Close, Play or a mitered outline.
            first = len(ui.draw.vtx_buffer)
            inset = (spec.slot_size - spec.canvas_size) * ui.s / 2
            ui.icon(name, x + inset, y + inset, spec.canvas_size)
            assert _vertices(ui.draw, first) == vertices, (name, role)

    frame(paint)


@pytest.mark.parametrize("role", tuple(ICON_ROLES))
def test_role_changes_glyph_hierarchy_without_moving_caption_or_click_target(
    role_ui, role, monkeypatch
):
    ui, io, frame = role_ui.ui, role_ui.io, role_ui.frame
    centers = {}
    role_icon, text = ui.role_icon, ui.text

    def observe(kind, draw, *args, **kwargs):
        first = len(ui.draw.vtx_buffer)
        draw(*args, **kwargs)
        left, top, right, bottom = _bounds(ui.draw, first)
        centers[kind] = ((left + right) / 2, (top + bottom) / 2)

    monkeypatch.setattr(ui, "role_icon", lambda *a, **kw: observe("icon", role_icon, *a, **kw))
    monkeypatch.setattr(ui, "text", lambda *a, **kw: observe("text", text, *a, **kw))

    def paint():
        clicked = ui.button(
            "role-action", 40 * ui.s, 50 * ui.s, 110, 36, icon="folder", label="H", icon_role=role
        )
        return clicked, imgui.get_item_id()

    frame(paint)
    hit, identifier = frame(paint)
    assert not hit
    assert centers["icon"][1] == pytest.approx(centers["text"][1], abs=0.75 * ui.s)
    assert ui.hits["role-action"] == pytest.approx((40 * ui.s, 50 * ui.s, 110 * ui.s, 36 * ui.s))
    io.add_mouse_pos_event(95 * ui.s, 68 * ui.s)
    frame(paint)
    io.add_mouse_button_event(0, True)
    assert not frame(paint)[0]
    io.add_mouse_button_event(0, False)
    clicked, clicked_id = frame(paint)
    assert clicked and clicked_id == identifier


@pytest.mark.parametrize("name", ("scene", "joint-hinge", "joint-free"))
def test_labeled_actions_balance_ink_height_with_caption_and_preserve_hit_area(
    role_ui, name, monkeypatch
):
    ui, frame = role_ui.ui, role_ui.frame
    bounds = {}
    icon, text = ui.role_icon, ui.text

    def observe(kind, draw, *args, **kwargs):
        first = len(ui.draw.vtx_buffer)
        draw(*args, **kwargs)
        bounds[kind] = _bounds(ui.draw, first, ink=True)

    monkeypatch.setattr(ui, "role_icon", lambda *a, **kw: observe("icon", icon, *a, **kw))
    monkeypatch.setattr(ui, "text", lambda *a, **kw: observe("text", text, *a, **kw))
    frame(lambda: ui.button("caption", 40 * ui.s, 50 * ui.s, 110, icon=name, label="H"))
    _, top, _, bottom = bounds["icon"]
    _, text_top, _, text_bottom = bounds["text"]
    inline_height = bottom - top
    assert 0.6 * (text_bottom - text_top) <= inline_height <= text_bottom - text_top + 4 * ui.s
    assert (top + bottom) / 2 == pytest.approx((text_top + text_bottom) / 2, abs=ui.s)
    assert bounds["text"][0] - bounds["icon"][2] >= 4 * ui.s
    assert ui.hits["caption"] == pytest.approx((40 * ui.s, 50 * ui.s, 110 * ui.s, 28 * ui.s))
    frame(lambda: ui.button("tool", 40 * ui.s, 90 * ui.s, icon=name))
    assert bounds["icon"][3] - bounds["icon"][1] > inline_height * 1.25
    assert ui.hits["tool"] == pytest.approx((40 * ui.s, 90 * ui.s, 28 * ui.s, 28 * ui.s))


def test_explicit_legacy_icon_size_keeps_its_canvas_semantics(role_ui, monkeypatch):
    ui, frame = role_ui.ui, role_ui.frame
    sizes = []
    original = ui.icon

    def icon(name, x, y, size=16, *args, **kwargs):
        sizes.append(size)
        original(name, x, y, size, *args, **kwargs)

    monkeypatch.setattr(ui, "icon", icon)
    frame(lambda: ui.button("legacy", 40 * ui.s, 50 * ui.s, icon="camera", icon_size=18))
    assert sizes == [18]


def test_role_geometry_translates_after_window_moves_and_colors_change(role_ui):
    ui, frame = role_ui.ui, role_ui.frame

    def paint(x, y, c):
        first = len(ui.draw.vtx_buffer)
        ui.role_icon("select", x, y, c=c, role="toolbar")
        return _bounds(ui.draw, first, ink=True)

    first = frame(lambda: paint(80 * ui.s, 90 * ui.s, widgets.TEXT))
    second = frame(lambda: paint(420 * ui.s, 180 * ui.s, widgets.SAGE))
    assert second == pytest.approx(
        tuple(
            value + delta for value, delta in zip(first, (340 * ui.s, 90 * ui.s) * 2, strict=True)
        ),
        abs=0.01,
    )


def test_role_icon_fits_custom_small_slots_and_skips_empty_slots(role_ui):
    ui, frame = role_ui.ui, role_ui.frame

    def paint():
        first = len(ui.draw.vtx_buffer)
        ui.role_icon("play", 80 * ui.s, 90 * ui.s, 8 * ui.s, 10 * ui.s, role="toolbar")
        left, top, right, bottom = _bounds(ui.draw, first, ink=True)
        # The compiled master contains the optical placement. No second fitter
        # rescales or recenters its triangle for compact controls.
        expected = tuple(
            (80 if index % 2 == 0 else 91) * ui.s + value * 8 * ui.s / ICON_GRID
            for index, value in enumerate(GLYPHS["play"]["bounds"])
        )
        assert (left, top, right, bottom) == pytest.approx(expected, abs=0.01)
        first = len(ui.draw.vtx_buffer)
        ui.role_icon("play", 0, 0, 0, 10, role="toolbar")
        assert len(ui.draw.vtx_buffer) == first

    frame(paint)


@pytest.mark.parametrize("role", ("toolbar", "subtle"))
def test_close_pin_and_playback_preserve_relative_optical_weight(role_ui, role):
    ui, frame = role_ui.ui, role_ui.frame
    spec = ICON_ROLES[role]
    boxes = {}

    def paint():
        for name in ("close", "pin", "previous", "next", "reset", "play", "pause"):
            first = len(ui.draw.vtx_buffer)
            ui.role_icon(name, 80 * ui.s, 90 * ui.s, role=role)
            boxes[name] = _bounds(ui.draw, first, ink=True)
            assert all(
                width == pytest.approx(ICON_STROKE * spec.canvas_size * ui.s / ICON_GRID)
                for contour, _points, _paths, width in icon_geometry(name, spec.canvas_size * ui.s)
                if contour["stroke"] != "none"
            )

    frame(paint)

    def height(name):
        return boxes[name][3] - boxes[name][1]

    assert height("close") < height("pin") * 0.75
    for name in ("play", "pause"):
        bounds = GLYPHS[name]["bounds"]
        assert height(name) == pytest.approx(
            (bounds[3] - bounds[1]) * spec.canvas_size * ui.s / ICON_GRID, abs=0.01
        )
    assert height("previous") == pytest.approx(height("next"), abs=0.01)
    assert height("previous") <= height("play") * 1.1
    assert height("reset") < height("pin")
    for name, (left, top, right, bottom) in boxes.items():
        bx, by, br, bb = GLYPHS[name]["bounds"]
        inset = (spec.slot_size - spec.canvas_size) / 2
        assert ((left + right) / 2, (top + bottom) / 2) == pytest.approx(
            (
                (80 + inset + (bx + br) / 2 * spec.canvas_size / ICON_GRID) * ui.s,
                (90 + inset + (by + bb) / 2 * spec.canvas_size / ICON_GRID) * ui.s,
            ),
            abs=0.15 * ui.s,
        ), name


def test_context_aliases_submit_identical_canonical_geometry(role_ui):
    from mojive.ui.workspace.widget_icons import COMPILED_ALIASES

    ui, frame = role_ui.ui, role_ui.frame

    def paint():
        for alias, master in COMPILED_ALIASES.items():
            first = len(ui.draw.vtx_buffer)
            ui.icon(alias, 80 * ui.s, 90 * ui.s, 18)
            actual = _vertices(ui.draw, first)
            first = len(ui.draw.vtx_buffer)
            ui.icon(master, 80 * ui.s, 90 * ui.s, 18)
            assert _vertices(ui.draw, first) == actual, (alias, master)

    frame(paint)


@pytest.mark.parametrize("name", ("close", "pin", "previous", "next", "reset", "play", "pause"))
def test_actual_stroke_and_fill_area_scale_without_a_one_pixel_floor(role_ui, name):
    ui, frame = role_ui.ui, role_ui.frame
    canvas = ICON_ROLES["subtle"].canvas_size

    def paint(size):
        first = len(ui.draw.idx_buffer)
        ui.icon(name, 80 * ui.s, 90 * ui.s, size)
        return _opaque_area(ui.draw, first)

    actual = frame(lambda: paint(canvas))
    canonical = frame(lambda: paint(ICON_GRID / ui.s))
    assert actual > 0
    assert actual == pytest.approx(canonical * (canvas * ui.s / ICON_GRID) ** 2, rel=0.003)


@pytest.mark.parametrize("size", (14, 24, 56, 112))
def test_sidebar_family_preserves_geometry_at_actual_small_and_large_sizes(role_ui, size):
    ui, frame = role_ui.ui, role_ui.frame
    names = (
        "scene",
        "control",
        "assets",
        "pose",
        "camera",
        "chart",
        "layers",
        "terminal",
        "settings",
    )
    x, y = 80 * ui.s, 90 * ui.s

    def paint():
        for name in names:
            first_vertex, first_index = len(ui.draw.vtx_buffer), len(ui.draw.idx_buffer)
            ui.icon(name, x, y, size)
            left, top, right, bottom = _bounds(ui.draw, first_vertex, ink=True)
            assert x <= left < right <= x + size * ui.s, (name, size)
            assert y <= top < bottom <= y + size * ui.s, (name, size)
            current_area = _opaque_area(ui.draw, first_index)
            first_index = len(ui.draw.idx_buffer)
            ui.icon(name, x, y, ICON_GRID / ui.s)
            canonical_area = _opaque_area(ui.draw, first_index)
            assert current_area == pytest.approx(
                canonical_area * (size * ui.s / ICON_GRID) ** 2, rel=0.003
            ), (name, size)

    frame(paint)


@pytest.mark.parametrize("size", (14, 16, 18, 24, 56, 112))
def test_library_symbols_preserve_opaque_geometry_at_every_size(role_ui, size):
    ui, frame = role_ui.ui, role_ui.frame
    x, y = 80 * ui.s, 90 * ui.s
    factor = size * ui.s / ICON_GRID
    names = (
        "visual",
        "collision",
        "both",
        "pose",
        "joint-hinge",
        "joint-slide",
        "joint-ball",
        "joint-free",
        "perspective",
        "orthographic",
    )

    def paint():
        for name in names:
            first_vertex, first_index = len(ui.draw.vtx_buffer), len(ui.draw.idx_buffer)
            ui.icon(name, x, y, size)
            area = _opaque_area(ui.draw, first_index)
            left, top, right, bottom = _bounds(ui.draw, first_vertex, ink=True)
            assert area > 0
            assert x <= left < right <= x + size * ui.s
            assert y <= top < bottom <= y + size * ui.s
            first_index = len(ui.draw.idx_buffer)
            ui.icon(name, x, y, ICON_GRID / ui.s)
            canonical_area = _opaque_area(ui.draw, first_index)
            assert area == pytest.approx(canonical_area * factor**2, rel=0.003), name

    frame(paint)
