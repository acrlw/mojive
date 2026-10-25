"""Section headings and production body labels share one native content column."""

from contextlib import contextmanager

import pytest
from imgui_bundle import imgui

from mojive.design.groups import group_part, group_scope
from mojive.ui.imgui_draw import ImguiDraw2D
from mojive.ui.workspace.menu_geometry import native_epsilon, native_unit
from mojive.ui.workspace.property_style import (
    NativePropertyPresentation,
    advance,
    property_controls,
    property_toggle,
)
from mojive.ui.workspace.section_columns import section_box, section_column
from mojive.ui.workspace.window_chrome import fit_native_content_width
from tests.test_component_appearance import appearance as appearance

pytestmark = pytest.mark.parametrize("appearance", (1, 1.1, 1.2, 1.25, 2.25), indirect=True)


def _horizontal_state():
    window = imgui.internal.get_current_window()
    return (
        imgui.get_cursor_screen_pos().x,
        imgui.get_content_region_avail().x,
        window.work_rect.min.x,
        window.work_rect.max.x,
        window.parent_work_rect.min.x,
        window.parent_work_rect.max.x,
        window.content_region_rect.min.x,
        window.content_region_rect.max.x,
    )


@contextmanager
def _child(ui, width, name="Section column"):
    imgui.set_cursor_screen_pos((30 * ui.s, 30 * ui.s))
    imgui.begin_child(name, (width * ui.s, 580 * ui.s))
    fit_native_content_width(imgui.internal.get_current_window())
    try:
        yield
    finally:
        imgui.end_child()


def _record(ui, runtime, key):
    records = (*runtime.observed, *ui.design.primitives.observed)
    return next(record for record in reversed(records) if record["key"] == key)


def _close(actual, expected):
    assert actual == pytest.approx(expected, abs=native_epsilon(actual, expected))


def _table(presentation, key, label):
    assert presentation.begin_table(key)
    presentation.readonly_value(label, "0.850")
    presentation.end_table()


@pytest.mark.parametrize("owner", ("Cameras", "Inspector", "Settings/Camera"))
@pytest.mark.parametrize("width", (220, 620))
def test_production_table_labels_share_heading_column_and_restore_native_width(
    appearance, owner, width
):
    ui, runtime, _update, frame = appearance
    runtime.observe_components = True
    presentation = NativePropertyPresentation(ui, prefix="alignment")
    snapshots = []

    def draw():
        runtime.begin_frame()
        with _child(ui, width), group_scope(owner), property_controls(ui):
            before = _horizontal_state()
            with section_column(ui):
                content = _horizontal_state()
                assert presentation.section("Position", imgui.TreeNodeFlags_.default_open)
                _table(presentation, "alignment-first", "Position value")
                assert presentation.section("Visibility", imgui.TreeNodeFlags_.default_open)
                _table(presentation, "alignment-second", "Visible value")
            after = _horizontal_state()
            snapshots.append((before, content, after))

    for _ in range(3):
        frame(draw)
    before, content, after = snapshots[-1]
    for actual, expected in zip(after, before, strict=True):
        _close(actual, expected)
    for heading, label, table in (
        ("Position", "Position value", "alignment-first"),
        ("Visibility", "Visible value", "alignment-second"),
    ):
        header = _record(ui, runtime, "section-" + heading)
        caption = _record(ui, runtime, "alignment-label-" + label)
        row = _record(ui, runtime, table + "/row/" + label)
        _close(header["hit"][0], before[0])
        _close(header["labelRect"][0], content[0])
        _close(caption["content"][0], header["labelRect"][0])
        for field in row["fields"]:
            assert field[0] >= content[0] - native_epsilon(*field, *content)
            assert field[0] + field[2] <= content[0] + content[1] + native_epsilon(*field, *content)
    for snapshot in snapshots[1:]:
        for current, initial in zip(snapshot, snapshots[0], strict=True):
            for actual, expected in zip(current, initial, strict=True):
                _close(actual, expected)


def test_section_column_preserves_authored_asymmetric_header_and_body_offsets(appearance):
    ui, runtime, update, frame = appearance
    runtime.observe_components = True
    owner = "Settings/Camera"
    update(
        {
            group_part(owner, "section_header"): {
                "marginLeft": 7,
                "marginRight": 9,
                "paddingLeft": 10,
                "paddingRight": 14,
            },
            group_part(owner, "section_body"): {
                "marginLeft": 3,
                "marginRight": 5,
                "paddingLeft": 2,
                "paddingRight": 4,
            },
            "form_label": {"marginLeft": 1, "paddingLeft": 2},
        }
    )
    presentation = NativePropertyPresentation(ui, prefix="authored")
    before = []

    def draw():
        runtime.begin_frame()
        with _child(ui, 620), group_scope(owner), property_controls(ui):
            before[:] = _horizontal_state()
            with section_column(ui):
                assert presentation.section("Camera", imgui.TreeNodeFlags_.default_open)
                _table(presentation, "authored-table", "Projection")

    for _ in range(3):
        frame(draw)
    header = _record(ui, runtime, "section-Camera")
    body = _record(ui, runtime, "section-body/authored-table")
    caption = _record(ui, runtime, "authored-label-Projection")
    _close(header["hit"][0] - before[0], native_unit(7 * ui.s))
    _close(
        header["labelRect"][0] - before[0],
        native_unit(7 * ui.s) + native_unit(10 * ui.s),
    )
    _close(body["content"][0] - header["labelRect"][0], 5 * ui.s)
    _close(caption["content"][0] - body["content"][0], 3 * ui.s)
    _close(
        header["content"][0] + header["content"][2] - body["content"][0] - body["content"][2],
        9 * ui.s,
    )


@pytest.mark.parametrize(
    "labels",
    [("Geometry view", "Collision opacity", "Shape"), ("几何显示", "碰撞体不透明度", "形状")],
)
def test_settings_page_shares_field_edges_across_sections_and_keeps_collapsed_width(
    appearance, labels
):
    ui, runtime, _update, frame = appearance
    runtime.observe_components = True
    fields = NativePropertyPresentation(ui, prefix="alignment", fit_labels=True)
    show_middle = True

    def draw():
        runtime.begin_frame()
        with (
            _child(ui, 620),
            group_scope("Settings/MuJoCo Visuals"),
            property_controls(ui),
            section_column(ui),
            fields.label_column("visuals"),
        ):
            for index, label in enumerate(labels):
                if index == 1 and not show_middle:
                    continue
                _table(fields, f"aligned-{index}", label)

    for _ in range(3):
        frame(draw)
    regions = [
        _record(ui, runtime, f"aligned-{i}/row/{label}")["fields"][1]
        for i, label in enumerate(labels)
    ]
    expected = regions[0][0], regions[0][2]
    for region in regions[1:]:
        assert (region[0], region[2]) == pytest.approx(expected)
    show_middle = False
    for _ in range(3):
        frame(draw)
    region = _record(ui, runtime, "aligned-0/row/" + labels[0])["fields"][1]
    assert (region[0], region[2]) == pytest.approx(expected)


def test_manual_vector_switch_and_text_share_section_column(appearance, monkeypatch):
    ui, runtime, _update, frame = appearance
    runtime.observe_components = True
    presentation = NativePropertyPresentation(ui, prefix="manual")
    text_origins = []
    original = ImguiDraw2D.text

    def text(draw, position, color, value, **options):
        if value == "Raw body label":
            text_origins.append(position[0])
        return original(draw, position, color, value, **options)

    monkeypatch.setattr(ImguiDraw2D, "text", text)

    def draw():
        runtime.begin_frame()
        with _child(ui, 620), group_scope("Inspector"), property_controls(ui), section_column(ui):
            assert presentation.section("Transform", imgui.TreeNodeFlags_.default_open)
            position = imgui.get_cursor_screen_pos()
            width = imgui.get_content_region_avail().x
            layout = ui.vector_layout(width, format="%.3f")
            ui.vector(
                "manual-vector",
                position.x,
                position.y,
                width,
                "Position",
                (0.85, 0, 0.55),
                "m",
                layout=layout,
            )
            advance(ui, position.x, position.y + layout.height + 8 * ui.s)
            property_toggle(ui, "manual-switch", "Visible", True)
            position = imgui.get_cursor_screen_pos()
            ui.text(position.x, position.y, "Raw body label", translate=False)

    for _ in range(3):
        frame(draw)
    heading_x = _record(ui, runtime, "section-Transform")["labelRect"][0]
    _close(_record(ui, runtime, "manual-vector-label")["content"][0], heading_x)
    _close(_record(ui, runtime, "manual-switch")["labelRect"][0], heading_x)
    assert text_origins
    for actual in text_origins:
        _close(actual, heading_x)


def test_short_header_reset_slot_does_not_shrink_the_body_column(appearance):
    ui, runtime, _update, frame = appearance
    runtime.observe_components = True
    presentation = NativePropertyPresentation(ui, prefix="reset")
    column = []

    def draw():
        runtime.begin_frame()
        with _child(ui, 620), group_scope("Inspector"), property_controls(ui), section_column(ui):
            pos = imgui.get_cursor_screen_pos()
            width = imgui.get_content_region_avail().x
            column[:] = (pos.x, width)
            ui.section("Transform", pos.x, pos.y, width - 32 * ui.s, True)
            advance(ui, pos.x, pos.y + ui.section_offset(42) * ui.s)
            _table(presentation, "reset-table", "Position")

    for _ in range(3):
        frame(draw)
    header = _record(ui, runtime, "section-Transform")
    caption = _record(ui, runtime, "reset-label-Position")
    body = _record(ui, runtime, "section-body/reset-table")
    _close(header["labelRect"][0], caption["content"][0])
    _close(body["content"][0], column[0])
    _close(body["content"][2], column[1])
    _close(
        body["content"][0] + body["content"][2] - header["content"][0] - header["content"][2],
        32 * ui.s,
    )


def test_nested_columns_do_not_compound_and_new_child_has_its_own_column(appearance):
    ui, runtime, _update, frame = appearance
    runtime.observe_components = True
    states = {}

    def draw():
        runtime.begin_frame()
        with _child(ui, 620), group_scope("Cameras"):
            states["outer-before"] = _horizontal_state()
            with section_column(ui):
                states["outer"] = _horizontal_state()
                with section_column(ui):
                    states["nested"] = _horizontal_state()
                    p = imgui.get_cursor_screen_pos()
                    ui.section("Outer", p.x, p.y, imgui.get_content_region_avail().x, True)
                    advance(ui, p.x, p.y + 44 * ui.s)
                states["after-nested"] = _horizontal_state()
                imgui.begin_child("Independent child", (0, 200 * ui.s))
                fit_native_content_width(imgui.internal.get_current_window())
                try:
                    states["child-before"] = _horizontal_state()
                    with section_column(ui):
                        states["child"] = _horizontal_state()
                        p = imgui.get_cursor_screen_pos()
                        ui.section("Inner", p.x, p.y, imgui.get_content_region_avail().x, True)
                    states["child-after"] = _horizontal_state()
                finally:
                    imgui.end_child()
            states["outer-after"] = _horizontal_state()

    for _ in range(3):
        frame(draw)
    for key in ("nested", "after-nested"):
        for actual, expected in zip(states[key], states["outer"], strict=True):
            _close(actual, expected)
    for before, after in (("outer-before", "outer-after"), ("child-before", "child-after")):
        for actual, expected in zip(states[after], states[before], strict=True):
            _close(actual, expected)
    _close(states["child"][0] - states["child-before"][0], native_unit(6 * ui.s))
    _close(_record(ui, runtime, "section-Outer")["labelRect"][0], states["outer"][0])
    _close(_record(ui, runtime, "section-Inner")["labelRect"][0], states["child"][0])


def test_section_column_restores_horizontal_state_after_exception(appearance):
    ui, runtime, _update, frame = appearance
    runtime.observe_components = True

    def draw():
        runtime.begin_frame()
        with _child(ui, 620), group_scope("Inspector"):
            before = _horizontal_state()
            with pytest.raises(RuntimeError, match="Interrupted section"), section_column(ui):
                imgui.dummy((10 * ui.s, 24 * ui.s))
                raise RuntimeError("Interrupted section")
            for actual, expected in zip(_horizontal_state(), before, strict=True):
                _close(actual, expected)
            p = imgui.get_cursor_screen_pos()
            ui.section("Standalone", p.x, p.y, imgui.get_content_region_avail().x, True)
            header = _record(ui, runtime, "section-Standalone")
            _close(header["hit"][0], before[0])
            _close(header["labelRect"][0] - header["hit"][0], 6 * ui.s)

    frame(draw)


def test_nested_owned_header_padding_keeps_its_body_aligned_and_restores_parent(appearance):
    ui, runtime, update, frame = appearance
    runtime.observe_components = True
    update(
        {
            group_part("Inspector", "section_header"): {
                "paddingLeft": 10,
                "paddingRight": 14,
            }
        }
    )
    presentation = NativePropertyPresentation(ui, prefix="nested-owned")
    states = {}

    def draw():
        runtime.begin_frame()
        with _child(ui, 620), group_scope("Cameras"), property_controls(ui):
            states["before"] = _horizontal_state()
            with section_column(ui):
                states["outer"] = _horizontal_state()
                assert presentation.section("Outer owner", imgui.TreeNodeFlags_.default_open)
                _table(presentation, "outer-owned-table", "Outer label")
                with group_scope("Inspector"), section_column(ui):
                    assert presentation.section("Inner owner", imgui.TreeNodeFlags_.default_open)
                    _table(presentation, "inner-owned-table", "Inner label")
                states["after-inner"] = _horizontal_state()
            states["after"] = _horizontal_state()

    for _ in range(3):
        frame(draw)
    for before, after in (("outer", "after-inner"), ("before", "after")):
        for actual, expected in zip(states[after], states[before], strict=True):
            _close(actual, expected)
    for section, label in (("Outer owner", "Outer label"), ("Inner owner", "Inner label")):
        header = _record(ui, runtime, "section-" + section)
        body = _record(ui, runtime, "nested-owned-label-" + label)
        _close(header["labelRect"][0], body["content"][0])
        assert header["hit"][0] >= states["before"][0] - native_epsilon(*header["hit"])
        assert header["hit"][0] + header["hit"][2] <= sum(states["before"][:2]) + native_epsilon(
            *header["hit"]
        )


def test_nested_fixed_header_width_reserves_the_same_body_budget(appearance):
    ui, runtime, update, frame = appearance
    runtime.observe_components = True
    update(
        {
            group_part("Cameras", "section_header"): {"width": 300, "widthMode": "auto"},
            group_part("Inspector", "section_header"): {"width": 300, "widthMode": "fixed"},
        }
    )
    presentation = NativePropertyPresentation(ui, prefix="fixed-width")
    states = {}

    def draw():
        runtime.begin_frame()
        with _child(ui, 620), group_scope("Cameras"), property_controls(ui), section_column(ui):
            states["outer"] = _horizontal_state()
            with group_scope("Inspector"), section_column(ui):
                states["inner"] = _horizontal_state()
                assert presentation.section("Fixed width", imgui.TreeNodeFlags_.default_open)
                _table(presentation, "fixed-width-table", "Position")
            states["restored"] = _horizontal_state()

    for _ in range(3):
        frame(draw)
    header = _record(ui, runtime, "section-Fixed width")
    body = _record(ui, runtime, "section-body/fixed-width-table")
    _close(header["hit"][2], 300 * ui.s)
    _close(body["content"][0], header["content"][0])
    _close(body["content"][2], header["content"][2])
    _close(states["inner"][1], header["content"][2])
    assert states["inner"][1] < states["outer"][1]
    for actual, expected in zip(states["restored"], states["outer"], strict=True):
        _close(actual, expected)


@pytest.mark.parametrize(
    ("width", "header_style"),
    (
        (5, {}),
        (0.5, {}),
        (160, {"paddingLeft": 400, "paddingRight": 600}),
        (5, {"marginLeft": 400, "marginRight": 600}),
        (
            620,
            {"width": 5, "widthMode": "fixed", "paddingLeft": 400, "paddingRight": 600},
        ),
    ),
    ids=("tiny", "fractional", "oversized-padding", "oversized-margin", "fixed-width"),
)
def test_exhausted_horizontal_budget_keeps_native_origin_and_restores_scope(
    appearance, width, header_style
):
    ui, runtime, update, frame = appearance
    update({group_part("Cameras", "section_header"): header_style})

    def draw():
        runtime.begin_frame()
        with _child(ui, 620), group_scope("Cameras"):
            window = imgui.internal.get_current_window()
            rects = (window.work_rect, window.parent_work_rect, window.content_region_rect)
            saved = [(rect.min.x, rect.max.x) for rect in rects]
            origin = imgui.get_cursor_screen_pos()
            # Set the actual native work budget; BeginChild clamps tiny window sizes.
            for rect in rects:
                rect.max.x = origin.x + width
            try:
                before = _horizontal_state()
                box = section_box(ui, origin.x, origin.y, width, native_horizontal=True)
                for rect in (box.frame, box.content):
                    assert rect.width >= 0
                    assert rect.x >= origin.x - native_epsilon(rect.x, origin.x)
                    assert rect.right <= origin.x + width + native_epsilon(rect.right, origin.x)
                    _close(rect.x - origin.x, native_unit(rect.x - origin.x))
                with section_column(ui) as column:
                    assert column.width >= 0
                    assert column.x >= origin.x - native_epsilon(column.x, origin.x)
                    assert column.right <= origin.x + width + native_epsilon(column.right, origin.x)
                    _close(column.x, box.content.x)
                    _close(imgui.get_cursor_screen_pos().x, column.x)
                    imgui.dummy((0, 10 * ui.s))
                    _close(imgui.get_cursor_screen_pos().x, column.x)
                for actual, expected in zip(_horizontal_state(), before, strict=True):
                    _close(actual, expected)
            finally:
                for rect, (low, high) in zip(rects, saved, strict=True):
                    rect.min.x, rect.max.x = low, high

    frame(draw)
