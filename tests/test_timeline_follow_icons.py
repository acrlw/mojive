"""Timeline follow modes reach the reviewed glyph family through the real controls."""

import json
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree as ET

import pytest
from examples.ui_design.native.timeline_panel import NativeTimelineToolbar as ReferenceToolbar
from imgui_bundle import imgui

from mojive.ui.keyframe_editor.controller import TimelineEditor
from mojive.ui.workspace import widgets
from mojive.ui.workspace.timeline_panel import NativeTimelineToolbar
from mojive.ui.workspace.widgets import UI


@pytest.mark.parametrize(
    "name, original",
    (
        ("follow", "follow"),
        ("follow-off", "pin"),
        ("follow-page", "follow"),
        ("follow-locked", "lock"),
        ("perspective", "perspective"),
        ("orthographic", "orthographic"),
        ("sort", "sort"),
        ("search", "search"),
    ),
)
def test_reviewed_shapes_preserve_the_pre_guide_production_baseline(name, original):
    from examples.ui_design.native.compile_icons import compile_svg

    from mojive.ui.workspace.widget_icons import GLYPHS, ICON_STROKE

    fixture = Path(__file__).parent / "fixtures/ui_design/pre_guide_icons.json"
    baseline = json.loads(fixture.read_text())
    master = ET.fromstring(baseline["icons"][original])
    master.set("stroke-width", str(ICON_STROKE))
    approved = compile_svg(ET.tostring(master, encoding="unicode"))
    assert GLYPHS[name] == json.loads(json.dumps(approved))


@pytest.mark.parametrize("toolbar_type", (NativeTimelineToolbar, ReferenceToolbar))
@pytest.mark.parametrize("scale", (1.0, 2.0))
def test_follow_controls_paint_distinct_modes_and_preserve_standalone_follow(
    monkeypatch, toolbar_type, scale
):
    context = imgui.create_context()
    try:
        io = imgui.get_io()
        io.set_ini_filename(None)
        io.display_size, io.delta_time = (500 * scale, 200 * scale), 1 / 60
        io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
        ui = UI.__new__(UI)
        ui.s, ui.hits, ui.translate = scale, {}, str
        ui.body = ui.bold = ui.mono = io.fonts.add_font_default()
        imgui.get_style().font_size_base = 14 * scale
        toolbar = toolbar_type(SimpleNamespace(ui=ui, s=scale))
        editor = TimelineEditor()
        painted, original = [], widgets.draw_icon

        def observe(draw, name, *args, **kwargs):
            first = len(draw.vtx_buffer)
            original(draw, name, *args, **kwargs)
            assert len(draw.vtx_buffer) > first
            painted.append(name)

        monkeypatch.setattr(widgets, "draw_icon", observe)

        def frame(*, callbacks=False):
            painted.clear()
            imgui.new_frame()
            imgui.set_next_window_pos((0, 0))
            imgui.set_next_window_size(io.display_size)
            imgui.begin("Timeline follow")
            toolbar._draw_follow(editor, None, 30 * scale, 40 * scale)
            if callbacks:
                for index, mode in enumerate(("off", "page", "locked")):
                    x = (180 + index * 32) * scale
                    toolbar.draw_follow_icon(
                        ui.draw,
                        (x, 40 * scale),
                        (x + 28 * scale, 68 * scale),
                        (1, 1, 1, 1),
                        scale,
                        "key-follow-" + mode,
                        "",
                    )
                toolbar.draw_command_icon(
                    ui.draw, (330 * scale, 54 * scale), "follow", (1, 1, 1, 1), scale
                )
            imgui.end()
            imgui.render()

        expected = ["follow-off", "follow-page", "follow-locked"]
        frame(callbacks=True)
        assert painted == [*expected, *expected, "follow"]
        for index, mode in enumerate(("off", "page", "locked")):
            x, y, width, height = ui.hits[f"timeline-follow-{index}"]
            io.add_mouse_pos_event(x + width / 2, y + height / 2)
            frame()
            io.add_mouse_button_event(0, True)
            frame()
            io.add_mouse_button_event(0, False)
            frame()
            assert editor.follow_mode == mode
            assert painted == expected
    finally:
        imgui.destroy_context(context)
