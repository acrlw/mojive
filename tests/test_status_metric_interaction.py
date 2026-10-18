"""Status metric tooltips and pointer actions through real ImGui input."""

from types import SimpleNamespace

import pytest
from imgui_bundle import imgui

from mojive import RecordingInfo
from mojive.ui.app import status as status_module
from mojive.ui.input_bindings import DEFAULT_INPUT_BINDINGS
from mojive.ui.localization import Language, Localizer
from mojive.ui.pointer_bindings import PointerAction
from mojive.ui.viewport_widgets import StatusLayout, localized_viewport_labels
from mojive.ui.viewport_widgets.model import StatusMetric
from mojive.ui.viewport_widgets.status import format_simulation_metric


@pytest.fixture(params=(Language.ENGLISH, Language.SIMPLIFIED_CHINESE))
def metric_input(request, monkeypatch):
    context = imgui.create_context()
    io = imgui.get_io()
    io.set_ini_filename(None)
    io.display_size = (800, 500)
    io.delta_time = 1 / 60
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    localizer = Localizer(request.param)
    app = SimpleNamespace(
        window=SimpleNamespace(style_scale=1, painter=lambda: None, pixels_to_points=float),
        session=SimpleNamespace(
            adapter=SimpleNamespace(
                caps=SimpleNamespace(simulation=True, external_clock=False, clock_control=True),
                timestep=lambda: 0.002,
            ),
            replay_info=None,
            state_take_playing=False,
            paused=True,
            frame=SimpleNamespace(time=1234.125, step=123_456_789_001, physics_hz=60),
        ),
        recording=RecordingInfo(),
        panels={},
        _status_panel="Output",
        design=SimpleNamespace(workbench=SimpleNamespace(pick=False)),
        localizer=localizer,
        _viewport_labels=localized_viewport_labels(localizer.text),
        theme=None,
        backend=SimpleNamespace(caps=SimpleNamespace(name="opengl")),
        _status_metric_mode="steps",
        _frame_rate=SimpleNamespace(value=59.8),
        passive_mode=False,
        external_status="",
        external_paused=None,
        _status_tool_hints=lambda **kwargs: (),
        input_bindings=DEFAULT_INPUT_BINDINGS,
    )
    state = SimpleNamespace(
        app=app,
        io=io,
        tooltips=[],
        copied=[],
        point=(150, 486),
        readouts={
            "backend": ("Backend", "OpenGL"),
            "delta": ("Δt", "0.00200000000000000004 s"),
            "physics": (app._viewport_labels.physics, "100.12500000000001 Hz"),
            "render": (app._viewport_labels.render, "59.799999999999997 FPS"),
        },
        readout_points={
            key: (300 + index * 120, 486)
            for index, key in enumerate(("backend", "delta", "physics", "render"))
        },
    )

    def draw_status(_draw, origin, _width, height, *_args, **kwargs):
        _caption, exact = format_simulation_metric(
            kwargs["metric_mode"], kwargs["sim_time"], kwargs["step"], kwargs["labels"]
        )
        return StatusLayout(
            metric_rect=(origin[0] + 100, origin[1], origin[0] + 200, origin[1] + height),
            metric_exact=exact,
            metrics=tuple(
                StatusMetric(
                    key,
                    label,
                    value,
                    (
                        origin[0] + state.readout_points[key][0] - 40,
                        origin[1],
                        origin[0] + state.readout_points[key][0] + 40,
                        origin[1] + height,
                    ),
                )
                for key, (label, value) in state.readouts.items()
            ),
        )

    def toggle():
        app._status_metric_mode = "time" if app._status_metric_mode == "steps" else "steps"

    def frame(count=1, *, loading=False):
        for _ in range(count):
            state.tooltips.clear()
            imgui.new_frame()
            status_module._Status._paint_application_status_bar(app, loading=loading)
            imgui.render()

    def event(method, *args):
        getattr(io, method)(*args)
        frame()

    def click(button=0):
        event("add_mouse_button_event", button, True)
        event("add_mouse_button_event", button, False)

    app._toggle_status_metric = toggle
    state.frame, state.event, state.click = frame, event, click
    monkeypatch.setattr(status_module, "draw_status", draw_status)
    monkeypatch.setattr(
        status_module,
        "status_tooltip",
        lambda _ui, *lines, **_options: state.tooltips.append("\n".join(lines)),
    )
    monkeypatch.setattr(imgui, "set_clipboard_text", state.copied.append)
    try:
        frame(3)
        event("add_mouse_pos_event", *state.point)
        frame(30)
        yield state
    finally:
        imgui.destroy_context(context)


@pytest.mark.parametrize("mode", ("steps", "time"))
def test_hover_reveals_current_exact_value_and_localized_actions(metric_input, mode):
    state, app = metric_input, metric_input.app
    app._status_metric_mode = mode
    state.frame()
    labels = app._viewport_labels
    exact = "123456789001" if mode == "steps" else "1234.125 s"
    label = labels.steps if mode == "steps" else labels.time
    switch = labels.show_time if mode == "steps" else labels.show_steps
    assert state.tooltips == [f"{label} {exact}\n{switch}\n{labels.copy_exact}"]
    app.session.frame.step += 1
    app.session.frame.time += 0.125
    state.frame()
    assert (
        "123456789002" in state.tooltips[0] if mode == "steps" else "1234.25 s" in state.tooltips[0]
    )
    assert not state.copied


def test_left_click_switches_once_and_right_click_copies_unabridged_value(metric_input):
    state, app = metric_input, metric_input.app
    state.click(1)
    assert state.copied == ["123456789001"]
    assert app._status_metric_mode == "steps"
    state.click()
    assert app._status_metric_mode == "time"
    state.frame(3)
    assert app._status_metric_mode == "time"
    state.click(1)
    assert state.copied == ["123456789001", "1234.125 s"]


@pytest.mark.parametrize("key", ("metric", "physics"))
def test_metric_tooltip_waits_while_copy_remains_immediate(metric_input, key):
    state = metric_input
    state.event("add_mouse_pos_event", 400, 100)
    state.frame(30)
    point = state.point if key == "metric" else state.readout_points[key]
    state.event("add_mouse_pos_event", *point)
    state.frame(2)
    assert not state.tooltips
    state.click(1)
    exact = "123456789001" if key == "metric" else state.readouts[key][1]
    assert state.copied == [exact]
    assert not state.tooltips
    state.frame(30)
    assert exact in state.tooltips[0]


def test_copy_uses_remapped_pointer_and_requires_hover(metric_input):
    state, app = metric_input, metric_input.app
    app.input_bindings = app.input_bindings.remap_pointer(PointerAction.STATUS_COPY, ("middle",))
    state.click(1)
    assert not state.copied
    state.click(2)
    assert state.copied == ["123456789001"]
    state.event("add_mouse_pos_event", 400, 100)
    state.click(2)
    assert state.copied == ["123456789001"]
    assert not state.tooltips


def test_workbench_alt_picking_suppresses_switch_and_remapped_copy(metric_input):
    state, app = metric_input, metric_input.app
    app.design.workbench.pick = True
    app.input_bindings = app.input_bindings.remap_pointer(PointerAction.STATUS_COPY, ("alt+right",))
    state.event("add_key_event", imgui.Key.mod_alt, True)
    state.click()
    state.click(1)
    assert app._status_metric_mode == "steps"
    assert not state.copied
    state.event("add_key_event", imgui.Key.mod_alt, False)
    state.click()
    assert app._status_metric_mode == "time"


def test_loading_status_does_not_offer_stale_metric_actions(metric_input):
    state = metric_input
    state.frame(loading=True)
    assert not state.tooltips
    assert not state.copied


@pytest.mark.parametrize("key", ("backend", "delta", "physics", "render"))
def test_readonly_metric_hover_and_copy_keep_exact_caption(metric_input, key):
    state, app = metric_input, metric_input.app
    state.event("add_mouse_pos_event", *state.readout_points[key])
    state.frame(2)
    label, exact = state.readouts[key]
    assert state.tooltips == [f"{label} {exact}\n{app._viewport_labels.copy_exact}"]
    state.click()
    assert app._status_metric_mode == "steps"
    assert not state.copied
    state.click(1)
    assert state.copied == [exact]
    assert app._status_metric_mode == "steps"
    state.readouts[key] = (label, "123.45678901234567 units")
    state.frame()
    assert state.tooltips == [
        f"{label} 123.45678901234567 units\n{app._viewport_labels.copy_exact}"
    ]
    state.click(1)
    assert state.copied == [exact, "123.45678901234567 units"]


def test_readonly_metric_copy_honors_remapping_and_hover(metric_input):
    state, app = metric_input, metric_input.app
    app.input_bindings = app.input_bindings.remap_pointer(PointerAction.STATUS_COPY, ("middle",))
    state.event("add_mouse_pos_event", *state.readout_points["physics"])
    state.frame(2)
    state.click(1)
    assert not state.copied
    state.click(2)
    assert state.copied == [state.readouts["physics"][1]]
    state.event("add_mouse_pos_event", 400, 100)
    state.click(2)
    assert state.copied == [state.readouts["physics"][1]]
    assert not state.tooltips


def test_readonly_metric_workbench_picking_suppresses_copy(metric_input):
    state, app = metric_input, metric_input.app
    state.event("add_mouse_pos_event", *state.readout_points["delta"])
    state.frame(2)
    app.design.workbench.pick = True
    app.input_bindings = app.input_bindings.remap_pointer(PointerAction.STATUS_COPY, ("alt+right",))
    state.event("add_key_event", imgui.Key.mod_alt, True)
    state.click(1)
    assert not state.copied
    state.event("add_key_event", imgui.Key.mod_alt, False)
    app.input_bindings = DEFAULT_INPUT_BINDINGS
    state.click(1)
    assert state.copied == [state.readouts["delta"][1]]


def test_loading_status_does_not_offer_readonly_metric_actions(metric_input):
    state = metric_input
    state.event("add_mouse_pos_event", *state.readout_points["render"])
    state.frame(2, loading=True)
    state.io.add_mouse_button_event(1, True)
    state.frame(loading=True)
    state.io.add_mouse_button_event(1, False)
    state.frame(loading=True)
    assert not state.tooltips
    assert not state.copied


def test_omitted_readonly_metric_has_no_stale_hit_target(metric_input):
    state = metric_input
    state.event("add_mouse_pos_event", *state.readout_points["physics"])
    state.frame(2)
    del state.readouts["physics"]
    state.frame()
    state.click(1)
    assert not state.tooltips
    assert not state.copied
