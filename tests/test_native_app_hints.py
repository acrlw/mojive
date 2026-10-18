"""Native app notices share quiet workspace hints without changing legacy help."""

from math import floor
from pathlib import Path
from types import SimpleNamespace

import pytest
from imgui_bundle import imgui

from mojive.capture import RecordingInfo, RecordingPhase
from mojive.ui.app.model_edits import _ModelEdits
from mojive.ui.app.status import _Status
from mojive.ui.imgui_draw import ImguiDraw2D
from mojive.ui.theme import THEME
from mojive.ui.viewport_widgets.model import DEFAULT_VIEWPORT_LABELS
from tests.test_workspace_tooltips import native_state
from tests.test_workspace_tooltips import tooltip_frame as tooltip_frame


def _app(ui, native):
    return SimpleNamespace(
        workspace=SimpleNamespace(ui=ui) if native else None,
        window=SimpleNamespace(style_scale=ui.s, painter=ImguiDraw2D, pixels_to_points=float),
        localizer=SimpleNamespace(text=lambda text: "Localized " + text),
        theme=THEME,
        _viewport_rect=(0, 0, 900 * ui.s, 500 * ui.s),
    )


def _paint(state, action):
    def paint():
        imgui.push_font(state.ui.bold, 36 * state.ui.s)
        try:
            before = native_state()
            action()
            assert native_state() == before
        finally:
            imgui.pop_font()

    state.frame(paint)


def _expect(state, text, native, shown, raw):
    if native:
        rows = [row for row in state.rows if row[0] == text]
        assert len(rows) == int(shown)
        if rows:
            assert rows[0][1:3] == (state.ui.body.font_id, floor(12 * state.ui.s + 0.5))
        assert not raw
        assert not state.translations
    else:
        assert raw == ([text] if shown else [])


@pytest.mark.parametrize("native", (False, True))
@pytest.mark.parametrize("hovered", (False, True))
@pytest.mark.parametrize(
    "phase,target,label",
    (
        (RecordingPhase.RECORDING, "pause", "Pause Recording"),
        (RecordingPhase.PAUSED, "pause", "Resume Recording"),
        (RecordingPhase.RECORDING, "stop", "Stop Recording"),
        (RecordingPhase.COUNTDOWN, "stop", "Cancel Recording"),
    ),
)
def test_recording_actions_use_the_real_status_hint_path(
    tooltip_frame, monkeypatch, native, hovered, phase, target, label
):
    state, raw = tooltip_frame, []
    app = _app(state.ui, native)
    app.session = SimpleNamespace(
        frame=SimpleNamespace(time=0, step=0, physics_hz=480),
        adapter=SimpleNamespace(
            caps=SimpleNamespace(simulation=True, external_clock=False), timestep=lambda: 0.002
        ),
        replay_info=None,
        state_take_playing=False,
        paused=True,
    )
    app.recording = RecordingInfo(phase=phase, duration=3, countdown_remaining=2)
    app.panels = SimpleNamespace(get=lambda _key: None)
    app.input_bindings = SimpleNamespace(
        pointer_match=lambda *_a, **_k: False, pointer_frame=lambda: None
    )
    app.backend = SimpleNamespace(caps=SimpleNamespace(name="opengl"))
    app.design, app._status_panel = None, ""
    app._status_metric_mode, app._frame_rate = "time", SimpleNamespace(value=59.8)
    app.passive_mode, app.external_status = False, ""
    app._viewport_labels, app._status_tool_hints = DEFAULT_VIEWPORT_LABELS, lambda **_k: ()

    def is_hovered(*_args, **_kwargs):
        return hovered and imgui.get_item_id() == imgui.get_id("##status_recording_" + target)

    monkeypatch.setattr(imgui, "is_item_hovered", is_hovered)
    monkeypatch.setattr(imgui, "set_tooltip", raw.append)
    _paint(state, lambda: _Status._paint_application_status_bar(app))
    _expect(state, "Localized " + label, native, hovered, raw)


@pytest.mark.parametrize("native", (False, True))
@pytest.mark.parametrize("hovered", (False, True))
def test_file_receipt_hint_keeps_the_exact_path_and_native_help_font(
    tooltip_frame, monkeypatch, native, hovered
):
    state, raw = tooltip_frame, []
    app = _app(state.ui, native)
    path = Path("/tmp/captured scene.png")
    message = SimpleNamespace(text="Saved " + str(path), copy_text=str(path))
    monkeypatch.setattr(imgui, "is_item_hovered", lambda *_a, **_k: hovered)
    monkeypatch.setattr(imgui, "set_tooltip", raw.append)
    _paint(state, lambda: _Status._draw_file_status(app, ImguiDraw2D(), message, path))
    _expect(state, "Localized Ctrl+click to reveal file\n" + str(path), native, hovered, raw)


@pytest.mark.parametrize("native", (False, True))
@pytest.mark.parametrize("hovered", (False, True))
def test_pending_edit_error_hint_uses_delayed_native_help_and_legacy_item_hint(
    tooltip_frame, monkeypatch, native, hovered
):
    state, raw = tooltip_frame, []
    app = _app(state.ui, native)
    app.model_edits = SimpleNamespace(error="The model changed; discard the pending edits")
    app._viewport_hint_bottom = lambda: 480 * state.ui.s
    monkeypatch.setattr(imgui, "is_item_hovered", lambda *_a, **_k: hovered)
    monkeypatch.setattr(imgui, "set_item_tooltip", raw.append)
    _paint(state, lambda: _ModelEdits._draw_pending_model_edits(app))
    _expect(state, app.model_edits.error, native, hovered if native else True, raw)
