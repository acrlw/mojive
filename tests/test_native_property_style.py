"""Native property families share responsive geometry and switch hit regions."""

from types import SimpleNamespace

import pytest
from imgui_bundle import imgui

from mojive.ui.workspace.design import DESIGN
from mojive.ui.workspace.inspector_style import NativeInspectorPresentation
from mojive.ui.workspace.property_style import (
    NativePropertyPresentation,
    property_controls,
    property_toggle,
)
from mojive.ui.workspace.widgets import UI


@pytest.fixture(params=(1.0, 2.0, 2.25))
def properties(request):
    context = imgui.create_context()
    io = imgui.get_io()
    io.set_ini_filename(None)
    io.display_size = (800 * request.param, 650 * request.param)
    io.delta_time = 1 / 60
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    ui = UI.__new__(UI)
    ui.s = request.param
    ui.body = ui.bold = ui.mono = io.fonts.add_font_default()
    ui.translate = str
    ui.hits, ui.numeric_edits = {}, {}
    generic = NativePropertyPresentation(ui)
    inspector = NativeInspectorPresentation(ui)
    inspector.ctx = SimpleNamespace(tr=str)

    def frame(draw):
        ui.hits.clear()
        ui.numeric_edits.clear()
        imgui.new_frame()
        imgui.set_next_window_pos((0, 0))
        imgui.set_next_window_size(io.display_size)
        imgui.begin("Property families", flags=imgui.WindowFlags_.no_title_bar)
        with property_controls(ui):
            draw()
        imgui.end()
        imgui.render()

    try:
        yield SimpleNamespace(ui=ui, generic=generic, inspector=inspector, io=io, frame=frame)
    finally:
        imgui.destroy_context(context)


@pytest.mark.parametrize("width", (145, 340))
def test_scalar_labels_and_fields_align_across_presentations(properties, width):
    state, origins = properties, {}

    def draw():
        for index, presentation in enumerate((state.generic, state.inspector)):
            imgui.set_cursor_screen_pos((40 * state.ui.s, (40 + index * 150) * state.ui.s))
            imgui.begin_child(f"family-{index}", (width * state.ui.s, 120 * state.ui.s))
            top = imgui.get_cursor_screen_pos()
            imgui.push_item_width(width * state.ui.s)
            if presentation.begin_table(f"property-family-{index}"):
                presentation.row("Distance")
                pos = imgui.get_cursor_screen_pos()
                origins[index] = (pos.x - top.x, pos.y - top.y)
                if index:
                    presentation.numeric("##inspector", 1, 0.01, 0, 10, "%.3f", 0)
                else:
                    presentation.numeric("##generic", 1, 0.01, 0, 10, "%.3f")
                imgui.end_table()
            imgui.pop_item_width()
            imgui.end_child()

    for _ in range(3):
        state.frame(draw)
    generic_label = state.ui.hits["property-label-Distance"]
    # Both presenters expose the same semantic label key, so compare the field
    # origins against their known table origins as well as the resulting bounds.
    assert origins[1][0] == pytest.approx(origins[0][0], abs=state.ui.s)
    assert origins[1][1] == pytest.approx(origins[0][1], abs=0.01)
    a, b = state.ui.hits["##generic"], state.ui.hits["##inspector"]
    assert a[2:] == pytest.approx(b[2:], abs=state.ui.s)
    assert generic_label[3] == pytest.approx(DESIGN.body_size * state.ui.s)


@pytest.mark.parametrize("width", (220, 480))
def test_numeric_arrays_use_the_same_responsive_grid(properties, width):
    state = properties
    values = (0.1, 0.2, 0.3, 0.4)

    def draw():
        for index, presentation in enumerate((state.generic, state.inspector)):
            imgui.set_cursor_screen_pos((40 * state.ui.s, (40 + index * 150) * state.ui.s))
            imgui.begin_child(f"family-{index}", (width * state.ui.s, 120 * state.ui.s))
            imgui.push_item_width(width * state.ui.s)
            if index:
                presentation.numbers("##inspector", values, 0.01, 0, 1, "%.3f", 0)
            else:
                presentation.numbers("##generic", values, 0.01, 0, 1, "%.3f")
            imgui.pop_item_width()
            imgui.end_child()

    for _ in range(3):
        state.frame(draw)
    for index in range(len(values)):
        a, b = state.ui.hits[f"##generic-{index}"], state.ui.hits[f"##inspector-{index}"]
        assert b == pytest.approx((a[0], a[1] + 150 * state.ui.s, a[2], a[3]), abs=1)


def test_numeric_array_geometry_stays_fixed_when_value_gains_digits(properties):
    state, values = properties, [0.1, 0.2, 0.3, 0.4]

    def draw():
        for index, presentation in enumerate((state.generic, state.inspector)):
            imgui.set_cursor_screen_pos((40 * state.ui.s, (40 + index * 150) * state.ui.s))
            imgui.begin_child(f"family-{index}", (300 * state.ui.s, 120 * state.ui.s))
            imgui.push_item_width(300 * state.ui.s)
            if index:
                presentation.numbers("##inspector", values, 0.01, 0, 0, "%.3f", 0)
            else:
                presentation.numbers("##generic", values, 0.01, 0, 0, "%.3f")
            imgui.pop_item_width()
            imgui.end_child()

    for _ in range(3):
        state.frame(draw)
    before = dict(state.ui.hits)
    values[0] = 1000
    state.frame(draw)
    for prefix in ("##generic", "##inspector"):
        for index in range(len(values)):
            key = f"{prefix}-{index}-frame"
            assert state.ui.hits[key] == before[key]


@pytest.mark.parametrize("width", (180, 280, 310, 620))
@pytest.mark.parametrize("count,fmt", ((3, "%.5f"), (3, "%.5g"), (2, "%.4g")))
@pytest.mark.parametrize("properties", (1.0, 1.2, 2.0, 2.25), indirect=True)
def test_plain_numeric_group_reserves_only_its_actual_rows(properties, width, count, fmt):
    state = properties
    values = (1.0, 0.005, 0.0001)[:count]

    def draw():
        imgui.set_cursor_screen_pos((40 * state.ui.s, 40 * state.ui.s))
        imgui.begin_child("contact-group", (width * state.ui.s, 580 * state.ui.s))
        if state.inspector.begin_table("insp_geometry_contact"):
            state.inspector.row("Friction", components=count)
            state.inspector.numbers("##friction", values, 0.005, 0, 1e6, fmt, 0)
            state.inspector.row("Contact dimension")
            state.inspector.numeric("##dimension", 3, 1, 1, 6, "%d", 0, integer=True)
            state.inspector.end_table()
        imgui.end_child()

    for _ in range(3):
        state.frame(draw)
    fields = [state.ui.hits[f"##friction-{index}-frame"] for index in range(count)]
    bottom = max(y + h for _x, y, _w, h in fields)
    next_label = state.ui.hits["property-label-Contact dimension"]
    assert 0 <= next_label[1] - bottom <= 2 * DESIGN.row_gap * state.ui.s
    first_label = state.ui.hits["property-label-Friction"]
    assert first_label[0] == pytest.approx(next_label[0])
    if width == 620:
        assert len({y for _x, y, _w, _h in fields}) == 1


@pytest.mark.parametrize("width", (280, 600))
def test_related_scalar_and_xyz_rows_share_columns_across_formats(properties, width):
    state, node = properties, SimpleNamespace(node_id=7)
    rows = (
        ("position", (0.2, 0, 0), "%.5f"),
        ("rotation", (0, -90, 0), "%.2f"),
        ("inertia", (0.0533713, 0.0533713, 0.00288568), "%.6g"),
    )

    def draw():
        imgui.set_cursor_screen_pos((40 * state.ui.s, 40 * state.ui.s))
        imgui.begin_child("inertial-group", (width * state.ui.s, 580 * state.ui.s))
        if state.inspector.begin_table("body_inertial_properties"):
            state.inspector.row("Mass")
            state.inspector.numeric("##mass", 2.9, 0.01, 0, 10, "%.6g", 0)
            for name, values, fmt in rows:
                state.inspector.vector_row(
                    node, name, name, values, editable=True, speed=0.01, fmt=fmt
                )
            imgui.end_table()
        imgui.end_child()

    for _ in range(3):
        state.frame(draw)
    hits, scale = state.ui.hits, state.ui.s
    label = hits["property-label-Mass"]
    mass = hits["##mass-frame"]
    reference = None
    for name, _, _ in rows:
        vector_label = hits["vector-label-" + name]
        fields = [hits[f"##{name}_{axis}_7-frame"] for axis in range(3)]
        assert vector_label[0] == pytest.approx(label[0])
        assert vector_label[2] == pytest.approx(label[2])
        assert fields[0][0] == pytest.approx(mass[0])
        assert fields[0][1] - vector_label[1] == pytest.approx(mass[1] - label[1])
        geometry = [(x, y - fields[0][1], w, h) for x, y, w, h in fields]
        if reference is None:
            reference = geometry
        assert geometry == pytest.approx(reference)
        assert fields[-1][0] + fields[-1][2] <= mass[0] + mass[2] + scale
    if width == 600:
        assert label[2] < mass[2]
        assert reference[0][1] == reference[1][1] == reference[2][1]
    else:
        assert label[2] == pytest.approx(mass[2])


def test_switch_rows_share_spacing_and_clickable_labels(properties):
    state, values = properties, [False, False]

    def draw():
        imgui.set_cursor_screen_pos((40 * state.ui.s, 40 * state.ui.s))
        values[0] = property_toggle(state.ui, "first", "Visible", values[0], width=250 * state.ui.s)
        values[1] = state.generic.toggle(
            "second", "Transform handles", values[1], width=250 * state.ui.s
        )

    for _ in range(3):
        state.frame(draw)
    a, b = state.ui.hits["first"], state.ui.hits["second"]
    assert b[1] - a[1] - a[3] == pytest.approx(DESIGN.row_gap * state.ui.s)
    for index, key in enumerate(("first-label", "second")):
        x, y, w, h = state.ui.hits[key]
        for value in (True, False):
            state.io.add_mouse_pos_event(x + w / 2, y + h / 2)
            state.frame(draw)
            state.io.add_mouse_button_event(0, True)
            state.frame(draw)
            state.io.add_mouse_button_event(0, False)
            state.frame(draw)
            assert values[index] is value
