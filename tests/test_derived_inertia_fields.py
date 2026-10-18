"""Automatically computed inertia stays readable and copyable without becoming editable."""

from dataclasses import replace
from itertools import pairwise

import pytest
from imgui_bundle import imgui

from mojive import commands as cmd
from tests.test_ui_design_components import _click_control
from tests.test_ui_design_components import components as components
from tests.test_ui_design_components import model_components as model_components


@pytest.mark.parametrize("language", ("en", "zh_CN"))
@pytest.mark.parametrize("width", (180, 320))
@pytest.mark.parametrize("access", ("paused", "running", "unsupported"))
def test_derived_inertia_is_copyable_and_rejects_text_edits(
    model_components, monkeypatch, language, width, access
):
    state, session = model_components, model_components.ctx.session
    ui, observed, notes = state.study.ui, {}, []
    body = next(node for node in session.nodes if node.name == "hinge_body")
    assert session.submit(cmd.SelectNode(body.node_id)).ok
    initial = session.body_properties(body.node_id)
    assert initial.inertia_mode == "auto"
    if access == "running":
        assert session.submit(cmd.Play()).ok
    elif access == "unsupported":
        monkeypatch.setattr(
            session.adapter, "caps", replace(session.adapter.caps, model_properties=False)
        )
    input_text = ui.input_text
    text_wrapped = imgui.text_wrapped

    def observe(key, x, y, field_width, value, **kwargs):
        observed[key] = (value, kwargs)
        return input_text(key, x, y, field_width, value, **kwargs)

    def note(text):
        notes.append(text)
        return text_wrapped(text)

    monkeypatch.setattr(ui, "input_text", observe)
    monkeypatch.setattr(imgui, "text_wrapped", note)
    state.frame(body, "Physics", count=3, width=width)
    keys = tuple("##body-derived-" + name for name in ("mass", "position", "rotation", "inertia"))
    bounds = []
    for key in keys:
        text, options = observed[key]
        assert text
        assert options["mono"]
        assert options["flags"] & imgui.InputTextFlags_.read_only
        x, y, field_width, height = ui.hits[key]
        assert 0 <= x < x + field_width <= width * ui.s + 1
        bounds.append((y, height))
    assert all(a + ah <= b for (a, ah), (b, _) in pairwise(bounds))
    assert state.ctx.tr("Calculated from geometry") in notes
    if access == "unsupported":
        assert state.ctx.tr("Backend does not support body property editing") in notes
        assert state.ctx.tr("Pause the simulation to edit model body properties") not in notes
    elif access == "running":
        assert state.ctx.tr("Pause the simulation to edit model body properties") in notes

    _click_control(state, body, ui.hits[keys[0]], tab="Physics", width=width)
    state.io.add_input_characters_utf8("999999")
    state.frame(body, "Physics", count=3, width=width)
    assert observed[keys[0]][0] == repr(float(initial.mass))
    assert session.body_properties(body.node_id) == initial
    assert state.inspector._body_property_edit == initial
    assert not session.can_undo and not session.editing

    for key, values in ((keys[0], (initial.mass,)), (keys[3], initial.diagonal_inertia)):
        exact = ", ".join(repr(float(value)) for value in values)
        assert observed[key][0] == exact
        _click_control(state, body, ui.hits[key], tab="Physics", width=width)
        modifier = imgui.Key.mod_super if state.io.config_mac_osx_behaviors else imgui.Key.mod_ctrl
        state.io.add_key_event(modifier, True)
        state.frame(body, "Physics", width=width)
        for action in (imgui.Key.a, imgui.Key.c):
            state.io.add_key_event(action, True)
            state.frame(body, "Physics", width=width)
            state.io.add_key_event(action, False)
            state.frame(body, "Physics", width=width)
        state.io.add_key_event(modifier, False)
        state.frame(body, "Physics", width=width)
        assert imgui.get_clipboard_text() == exact
        assert ui.hits[key][3] == pytest.approx(bounds[keys.index(key)][1])

        x, y, field_width, height = ui.hits[key]
        state.io.add_mouse_pos_event(x + field_width / 2, y + height / 2)
        state.frame(body, "Physics", width=width)
        state.io.add_mouse_button_event(1, True)
        state.frame(body, "Physics", width=width)
        state.io.add_mouse_button_event(1, False)
        state.frame(body, "Physics", width=width)
        assert imgui.get_clipboard_text() == exact
    assert session.body_properties(body.node_id) == initial
    assert not session.can_undo and not session.editing
