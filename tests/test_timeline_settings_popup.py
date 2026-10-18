"""Production Timeline form popup owns equal insets and complete row budgets."""

from copy import deepcopy
from itertools import product
from types import SimpleNamespace

import pytest
from imgui_bundle import imgui

from mojive.config import RecordingConfig
from mojive.design.groups import group_scope
from mojive.design.native.presentation import DesignPresentation
from mojive.design.runtime import DesignRuntime
from mojive.design.store import SpecificationStore
from mojive.ui.localization import Language, Localizer
from mojive.ui.panels import PanelContext
from mojive.ui.workspace.menu_geometry import native_unit
from mojive.ui.workspace.timeline_panel import NativeTimelineToolbar
from mojive.ui.workspace.widgets import UI


@pytest.fixture(
    params=tuple(product((1, 1.2, 1.25, 2.25), ("en", "zh_CN"), ("default", "current")))
)
def form(request, tmp_path, monkeypatch):
    scale, language, profile = request.param
    monkeypatch.setenv("MOJIVE_SETTINGS", str(tmp_path / "settings.json"))
    context = imgui.create_context()
    io = imgui.get_io()
    io.set_ini_filename(None)
    io.display_size, io.delta_time = (900 * scale, 600 * scale), 1 / 60
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    ui = UI(
        SimpleNamespace(style_scale=scale, font_report=SimpleNamespace(cjk_path=None)),
        portable_fonts=True,
    )
    ui.translate = Localizer(Language(language)).text
    runtime = DesignRuntime(SpecificationStore(tmp_path / "design.json"))
    if profile == "current":
        spec = deepcopy(runtime.spec)
        spec["tokens"].update(font_family="portable", radius_popup=8)
        spec["parts"]["menu"] = {"paddingX": 6, "paddingY": 6, "radius": 12}
        runtime.update_spec(spec)
    runtime.observe_components = True
    ui.design = DesignPresentation(ui, runtime)
    session = SimpleNamespace(state_take_pause_at_end=True)
    ctx = PanelContext(
        session, None, style_scale=scale, translate=ui.translate, recording_config=RecordingConfig()
    )
    ctx.set_take_pause_at_end = lambda value: setattr(session, "state_take_pause_at_end", value)
    state = SimpleNamespace(
        ui=ui,
        runtime=runtime,
        ctx=ctx,
        toolbar=NativeTimelineToolbar(SimpleNamespace(ui=ui, s=scale)),
        writes=[],
    )

    def update(config, *, persist=True):
        ctx.recording_config = config
        state.writes.append((config, persist))

    ctx.set_recording_config = update

    def frame(count=3):
        for _ in range(count):
            imgui.new_frame()
            ui.design.begin_frame()
            ui.hits.clear()
            imgui.set_next_window_pos((0, 0))
            imgui.set_next_window_size(io.display_size)
            imgui.begin("Form host", flags=imgui.WindowFlags_.no_title_bar)
            with group_scope("Timeline"):
                if not imgui.is_popup_open("Timeline settings"):
                    imgui.open_popup("Timeline settings")
                state.toolbar._draw_settings(ctx)
            imgui.end()
            imgui.render()
        state.record = next(
            r for r in reversed(runtime.observed) if r["key"] == "timeline-settings-popup"
        )
        state.window = (
            imgui.get_current_context()
            .open_popup_stack[len(imgui.get_current_context().open_popup_stack) - 1]
            .window
        )

    state.frame = frame
    try:
        yield state
    finally:
        imgui.destroy_context(context)


@pytest.mark.parametrize("recording", (False, True))
def test_form_actual_four_insets_and_compact_radius_match_at_every_scale(form, recording):
    state = form
    if not recording:
        state.ctx.recording_config = None
    state.frame()
    r, window, scale = state.record, state.window, state.ui.s
    rows = r["rows"]
    pad = native_unit(8 * scale)
    assert rows[0][0] - window.pos.x == pad
    assert rows[0][1] - window.pos.y == pad
    assert window.pos.x + window.size.x - rows[-1][0] - rows[-1][2] == pad
    assert window.pos.y + window.size.y - rows[-1][1] - rows[-1][3] == pad
    assert r["padding"] == [pad] * 4 or r["padding"] == (pad,) * 4
    assert window.window_rounding == pytest.approx(r["innerRadius"] + pad)
    assert window.scrollbar_sizes.x == window.scrollbar_sizes.y == 0
    assert rows[0][3] == rows[-1][3]
    if recording:
        assert rows[1][1] - rows[0][1] - rows[0][3] == native_unit(12 * scale)
        assert rows[2][1] - rows[1][1] - rows[1][3] == native_unit(6 * scale)
        right = window.pos.x + window.size.x - pad
        for key in ("##timeline-countdown", "##timeline-end_hold"):
            x, y, width, height = state.ui.hits[key + "-frame"]
            assert x + width == right
            assert height == r["rows"][1][3]
            assert y >= rows[1][1]
        assert (
            state.ui.hits["##timeline-end_hold-frame"][1]
            + state.ui.hits["##timeline-end_hold-frame"][3]
            == window.pos.y + window.size.y - pad
        )


@pytest.mark.parametrize("radius", (0, 22))
def test_form_default_radius_is_independent_of_padding_and_numeric_corners(form, radius):
    state = form
    spec = deepcopy(state.runtime.spec)
    spec["parts"]["timeline_form_popup"] = {
        "paddingX": 14,
        "paddingY": 14,
        "rowGap": 4,
        "groupGap": 16,
    }
    spec["parts"]["input_number"] = {"radius": 9}
    state.runtime.update_spec(spec)
    state.frame()
    r = state.record
    assert tuple(r["padding"]) == (native_unit(14 * state.ui.s),) * 4
    assert r["rowGap"] == native_unit(4 * state.ui.s)
    assert r["groupGap"] == native_unit(16 * state.ui.s)
    assert r["innerRadius"] == pytest.approx(9 * state.ui.s)
    assert r["radius"] == pytest.approx(9 * state.ui.s + r["padding"][0])
    spec["parts"]["timeline_form_popup"]["radius"] = radius
    state.runtime.update_spec(spec)
    state.frame()
    assert state.record["radius"] == pytest.approx(radius * state.ui.s)
    assert state.record["innerRadius"] == pytest.approx(
        max(0, radius * state.ui.s - state.record["padding"][0])
    )


def test_real_recording_drag_cancels_and_video_disables_the_pause_switch(form):
    state = form
    state.frame()
    io = imgui.get_io()

    def event(name, *args):
        getattr(io, name)(*args)
        state.frame(1)

    x, y, width, height = state.ui.hits["##timeline-countdown"]
    original = state.ctx.recording_config.countdown
    event("add_mouse_pos_event", x + width / 2, y + height / 2)
    event("add_mouse_button_event", 0, True)
    event("add_mouse_pos_event", x + width / 2 + 30 * state.ui.s, y + height / 2)
    assert state.ctx.recording_config.countdown != original
    event("add_key_event", imgui.Key.escape, True)
    assert state.ctx.recording_config.countdown == original
    assert state.toolbar._recording_edit is None
    assert state.writes[-1][1] is False
    event("add_key_event", imgui.Key.escape, False)
    event("add_mouse_button_event", 0, False)
    state.ctx.take_video_active = True
    state.frame()
    x, y, width, height = state.ui.hits["timeline-pause-at-end"]
    before = state.ctx.session.state_take_pause_at_end
    event("add_mouse_pos_event", x + width / 2, y + height / 2)
    event("add_mouse_button_event", 0, True)
    event("add_mouse_button_event", 0, False)
    assert state.ctx.session.state_take_pause_at_end is before


def test_authored_form_radius_fits_the_actual_numeric_frame(form):
    state = form
    spec = deepcopy(state.runtime.spec)
    spec["parts"]["timeline_form_popup"] = {"paddingX": 8, "paddingY": 8, "radius": 24}
    state.runtime.update_spec(spec)
    state.frame()
    record, scale = state.record, state.ui.s
    field = state.ui.hits["##timeline-end_hold-frame"]
    assert record["radius"] == pytest.approx(24 * scale)
    assert record["innerRadius"] <= min(field[2], field[3]) / 2 - 1
    assert record["radius"] == pytest.approx(record["innerRadius"] + record["padding"][0])
    assert field[3] == record["rows"][-1][3]
