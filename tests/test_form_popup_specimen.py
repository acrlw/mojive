"""New form popup catalogue parts retain real editable production references."""

from copy import deepcopy
from itertools import product
from types import SimpleNamespace

import pytest
from imgui_bundle import imgui

from mojive.design.native.geometry import Rect
from mojive.design.native.presentation import DesignPresentation
from mojive.design.native.specimen_forms import form_reference
from mojive.design.native.specimens import ComponentSpecimens
from mojive.design.runtime import DesignRuntime
from mojive.design.store import SpecificationStore
from mojive.ui.localization import Language, Localizer
from mojive.ui.workspace.menu_geometry import native_unit
from mojive.ui.workspace.widgets import UI


@pytest.fixture(params=tuple(product((1, 1.2, 1.25, 2, 2.25), ("en", "zh_CN"))))
def reference(request, tmp_path):
    scale, language = request.param
    context = imgui.create_context()
    io = imgui.get_io()
    io.set_ini_filename(None)
    io.display_size, io.delta_time = (1000 * scale, 700 * scale), 1 / 60
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    ui = UI(
        SimpleNamespace(style_scale=scale, font_report=SimpleNamespace(cjk_path=None)),
        portable_fonts=True,
    )
    ui.translate = Localizer(Language(language)).text
    runtime = DesignRuntime(SpecificationStore(tmp_path / "design.json"))
    runtime.observe_components = True
    ui.design = DesignPresentation(ui, runtime)
    specimens = ComponentSpecimens(runtime)
    state = SimpleNamespace(ui=ui, runtime=runtime, specimens=specimens)

    def frame(part, count=3):
        for _ in range(count):
            imgui.new_frame()
            ui.design.begin_frame()
            ui.hits.clear()
            imgui.set_next_window_pos((0, 0))
            imgui.set_next_window_size(io.display_size)
            imgui.begin("Form reference", flags=imgui.WindowFlags_.no_title_bar)
            try:
                size = specimens.measure(ui, part, 420 * scale)
                record = specimens.paint(ui, part, Rect(30 * scale, 50 * scale, *size))
                imgui.dummy((0, 0))
            finally:
                imgui.end()
                imgui.render()
        return record

    state.frame = frame
    try:
        yield state
    finally:
        imgui.destroy_context(context)


@pytest.mark.parametrize("part", ("form_popup", "timeline_form_popup"))
def test_registered_form_references_use_complete_production_geometry(reference, part):
    state = reference
    record = state.frame(part)
    assert record["part"] == part and state.specimens.last[part] == record
    frame, content = Rect(*record["frame"]), Rect(*record["content"])
    assert len(record["children"]) == 3
    rows = [Rect(*row) for row in record["children"]]
    assert rows[0].x == content.x and rows[0].y == content.y
    assert rows[-1].right == content.right and rows[-1].bottom == content.bottom
    assert frame.width > 200 * state.ui.s and frame.height > 100 * state.ui.s
    controls = {row["key"]: row for row in record["controls"]}
    assert {
        "timeline-pause-at-end",
        "##timeline-countdown",
        "##timeline-end_hold",
    } <= controls.keys()
    for key in ("##timeline-countdown", "##timeline-end_hold"):
        field = Rect(*state.ui.hits[key + "-frame"])
        assert field.right == content.right
        assert field.y >= content.y and field.bottom <= content.bottom
    assert not imgui.get_current_context().open_popup_stack


@pytest.mark.parametrize("part", ("form_popup", "timeline_form_popup"))
def test_reference_switch_and_recording_drag_edit_only_local_sample_state(reference, part):
    state = reference
    state.frame(part)
    _toolbar, ctx, _layout = form_reference(state.specimens, state.ui)
    original = ctx.recording_config.countdown
    io = imgui.get_io()

    def event(method, *args):
        getattr(io, method)(*args)
        state.frame(part, 1)

    x, y, width, height = state.ui.hits["timeline-pause-at-end"]
    event("add_mouse_pos_event", x + width / 2, y + height / 2)
    event("add_mouse_button_event", 0, True)
    event("add_mouse_button_event", 0, False)
    assert ctx.session.state_take_pause_at_end is False
    assert state.specimens.values["form_popup/pause"] is False
    x, y, width, height = state.ui.hits["##timeline-countdown"]
    event("add_mouse_pos_event", x + width / 2, y + height / 2)
    event("add_mouse_button_event", 0, True)
    event("add_mouse_pos_event", x + width / 2 + 30 * state.ui.s, y + height / 2)
    assert ctx.recording_config.countdown != original
    event("add_key_event", imgui.Key.escape, True)
    assert ctx.recording_config.countdown == original
    event("add_key_event", imgui.Key.escape, False)
    event("add_mouse_button_event", 0, False)
    state.specimens.values.clear()
    state.frame(part)
    assert ctx.session.state_take_pause_at_end is True
    assert ctx.recording_config.countdown == original


def test_timeline_form_reference_keeps_its_local_padding_override(reference):
    state = reference
    spec = deepcopy(state.runtime.spec)
    spec["parts"]["timeline_form_popup"] = {"paddingLeft": 5, "paddingRight": 19}
    state.runtime.update_spec(spec)
    base = state.frame("form_popup")
    local = state.frame("timeline_form_popup")
    left = right = native_unit(19 * state.ui.s)
    assert local["frame"][2] - base["frame"][2] == (left + right - 2 * native_unit(8 * state.ui.s))
    assert local["content"][0] - local["frame"][0] == left
    assert local["frame"][0] + local["frame"][2] - sum(local["content"][::2]) == right
