"""Switch rows retain readable captions and a common trailing control edge."""

from types import SimpleNamespace

import pytest
from imgui_bundle import imgui

from mojive import ViewportLayers
from mojive.design.groups import group_scope
from mojive.design.native.geometry import Rect
from mojive.ui.workspace.property_style import property_toggle, switch_form_layout
from mojive.ui.workspace.scene_panels import layers_panel
from tests.test_component_appearance import appearance as appearance


@pytest.mark.parametrize("width", (150, 420))
@pytest.mark.parametrize("custom", (False, True))
def test_real_rows_align_switches_and_preserve_label_clicks(appearance, width, custom):
    ui, runtime, update, frame = appearance
    runtime.observe_components = True
    update({"switch": {"switchWidth": 44, "switchHeight": 24, "switchThumb": 20}})
    if custom:
        update(
            {
                "switch": {
                    "switchWidth": 56,
                    "switchHeight": 28,
                    "switchThumb": 20,
                    "marginLeft": 7,
                    "marginRight": 11,
                    "paddingLeft": 5,
                    "paddingRight": 9,
                },
                "form_row": {"marginLeft": 3, "marginRight": 13},
            }
        )
    labels, values, vertices = ("Grid", "Selection highlight"), [False, False], {}

    def draw():
        ui.design.begin_frame()
        ui.hits.clear()
        for index, label in enumerate(labels):
            draw = imgui.get_window_draw_list()
            start = len(draw.vtx_buffer)
            imgui.set_cursor_screen_pos((40 * ui.s, (40 + index * 160) * ui.s))
            values[index] = property_toggle(
                ui, f"row-{index}", label, values[index], width=width * ui.s
            )
            vertices[index] = [
                (vertex.pos.x, vertex.pos.y, vertex.col) for vertex in list(draw.vtx_buffer)[start:]
            ]
        return {row["key"]: row for row in ui.design.primitives.observed if "switchRect" in row}

    for _ in range(3):
        rows = frame(draw)
    right = (40 + width - (13 + 11 + 9 if custom else 0)) * ui.s
    for index, _label in enumerate(labels):
        row = rows[f"row-{index}"]
        x, y, w, h = row["switchRect"]
        assert x + w == pytest.approx(right)
        assert w == pytest.approx((56 if custom else 44) * ui.s)
        assert h == pytest.approx((28 if custom else 24) * ui.s)
        track_x = [vx for vx, vy, _color in vertices[index] if y <= vy <= y + h]
        assert track_x and right - ui.s <= max(track_x) <= right + ui.s
        if row["textPresent"]:
            lx, ly, lw, lh = row["labelRect"]
            assert lx + lw <= x - row["gap"] + 1e-4
            assert ly + lh >= y
            target = (lx + min(lw / 2, 20 * ui.s), ly + lh / 2)
        else:
            lx, ly, lw, lh = ui.hits[f"row-{index}-label-text"]
            assert ly + lh <= y
            target = (lx + min(lw / 2, 20 * ui.s), ly + lh / 2)
        io = imgui.get_io()
        io.add_mouse_pos_event(*target)
        frame(draw)
        io.add_mouse_button_event(0, True)
        frame(draw)
        io.add_mouse_button_event(0, False)
        frame(draw)
        assert values[index] is True
        io.add_mouse_pos_event(x + w / 2, y + h / 2)
        frame(draw)
        io.add_mouse_button_event(0, True)
        frame(draw)
        io.add_mouse_button_event(0, False)
        frame(draw)
        assert values[index] is False


def test_layer_panel_rows_share_trailing_edge_and_disabled_named_layer_stays_readonly(appearance):
    ui, runtime, _update, frame = appearance
    runtime.observe_components = True
    app = SimpleNamespace(viewport_layers=ViewportLayers(debug_3d=False))
    writes = []
    ctx = SimpleNamespace(
        tr=str,
        set_viewport_layers=writes.append,
        backend=SimpleNamespace(
            debug=SimpleNamespace(
                layers=lambda: (SimpleNamespace(name="trajectory", primitives=(1,)),)
            )
        ),
    )
    workspace = SimpleNamespace(ui=ui, s=ui.s, app=app, _panel_context=ctx)

    def draw():
        ui.design.begin_frame()
        with group_scope("Layers"):
            layers_panel(workspace, 40 * ui.s, 20 * ui.s, 340 * ui.s)
        return [row for row in ui.design.primitives.observed if "switchRect" in row]

    for _ in range(3):
        rows = frame(draw)
    assert len(rows) == 8
    assert [row["switchRect"][0] + row["switchRect"][2] for row in rows] == pytest.approx(
        [380 * ui.s] * len(rows)
    )
    named = next(row for row in rows if row["key"] == "drawing-layer-trajectory")
    x, y, w, h = named["hit"]
    io = imgui.get_io()
    io.add_mouse_pos_event(x + w / 2, y + h / 2)
    frame(draw)
    io.add_mouse_button_event(0, True)
    frame(draw)
    io.add_mouse_button_event(0, False)
    frame(draw)
    assert writes == []


def test_inline_primitive_caption_uses_the_same_content_edge(appearance):
    ui, _runtime, update, frame = appearance
    update({"switch": {"paddingRight": 9, "marginRight": 11}})

    def draw():
        ui.design.begin_frame()
        for index, label in enumerate(("Grid", "Selection highlight")):
            ui.design.primitives.switch(
                f"inline-{index}",
                Rect(40 * ui.s, (40 + index * 60) * ui.s, 400 * ui.s, 40 * ui.s),
                False,
                label=label,
            )
        return ui.design.primitives.observed

    rows = frame(draw)
    for row in rows:
        x, _y, w, _h = row["switchRect"]
        assert x + w == pytest.approx((40 + 400 - 11 - 9) * ui.s)


def test_switch_dimensions_do_not_change_numeric_row_geometry(appearance):
    ui, _runtime, update, frame = appearance
    before = frame(lambda: ui.form_layout(400 * ui.s, field_part="input_number"))
    update({"switch": {"switchWidth": 64, "marginRight": 15}})
    after = frame(lambda: ui.form_layout(400 * ui.s, field_part="input_number"))
    assert after == before
    layout = frame(lambda: switch_form_layout(ui, 400 * ui.s, "Visible"))
    assert layout.fields[0][2] >= (64 + 15) * ui.s


@pytest.mark.parametrize("target", ("label", "switch"))
def test_wrapped_switch_row_retains_its_description_tooltip(appearance, monkeypatch, target):
    ui, runtime, _update, frame = appearance
    runtime.observe_components = True
    tooltips = []
    monkeypatch.setattr(ui, "tooltip", lambda text, **_options: tooltips.append(text))

    def draw():
        ui.design.begin_frame()
        imgui.set_cursor_screen_pos((40 * ui.s, 40 * ui.s))
        property_toggle(
            ui,
            "wrapped",
            "Selection highlight",
            False,
            width=150 * ui.s,
            tooltip="Highlight selected content",
        )
        return ui.design.primitives.observed[-1]

    row = frame(draw)
    assert not row["textPresent"]
    x, y, w, h = ui.hits["wrapped-label-text"] if target == "label" else row["switchRect"]
    io = imgui.get_io()
    io.delta_time = 1
    io.add_mouse_pos_event(x + w / 2, y + h / 2)
    for _ in range(3):
        frame(draw)
    assert "Highlight selected content" in tooltips
