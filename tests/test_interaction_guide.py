"""Interaction Guide shares status input descriptors and keeps complete row geometry."""

from itertools import pairwise

import pytest

from mojive.ui.input_bindings import DEFAULT_INPUT_BINDINGS, InputAction
from mojive.ui.pointer_bindings import PointerAction
from mojive.ui.viewport_widgets.hints import pointer_tool_hint
from mojive.ui.workspace.interaction_guide import draw_interaction_rows, interaction_rows
from mojive.ui.workspace.menu_geometry import native_epsilon
from tests.test_component_appearance import appearance as appearance


def test_guide_reuses_primary_status_mouse_descriptors_and_current_keys():
    bindings = DEFAULT_INPUT_BINDINGS.remap(InputAction.FRAME_SCENE, "h")
    bindings = bindings.remap_pointer(PointerAction.PAN, ("alt+middle",))
    rows = {row.title: row for row in interaction_rows(bindings)}
    assert rows["Frame all / focus"].hints[0].control == "H"
    assert rows["Simulation playback"].hints[0].control == bindings.label(InputAction.TOGGLE_PAUSE)
    for title, action in (
        ("Orbit", PointerAction.ORBIT),
        ("Pan", PointerAction.PAN),
        ("Precise gizmo input", PointerAction.GIZMO_VALUE),
    ):
        assert rows[title].hints[0] == pointer_tool_hint(action, bindings, "", hint_id=action.value)
    assert rows["Pan"].hints[0].modifier == "Alt"
    assert rows["Precise gizmo input"].hints[0].suffix == "×2"
    assert interaction_rows(bindings) is interaction_rows(bindings)


@pytest.mark.parametrize("mac", (False, True))
def test_guide_keeps_platform_shortcuts_and_unbound_actions_explicit(mac):
    bindings = DEFAULT_INPUT_BINDINGS.remap(InputAction.FRAME_SCENE, None)
    rows = {row.title: row for row in interaction_rows(bindings, mac)}
    assert rows["Frame all / focus"].hints[0].kind == "text"
    assert rows["Frame all / focus"].hints[0].label == "Unbound"
    modifier = "Cmd" if mac else "Ctrl"
    assert rows["Settings"].hints[-1].control == modifier + "+,"
    assert rows["Undo / redo"].hints[-1].control == modifier + "+Shift+Z"


@pytest.mark.parametrize("width", (270, 460))
def test_guide_wraps_complete_status_indicators_inside_each_row(appearance, width):
    ui, runtime, update, frame = appearance
    update({"dialog_body": {"rowGap": 7}})
    runtime.observe_components = True
    fields = frame(lambda: draw_interaction_rows(ui, DEFAULT_INPUT_BINDINGS, width * ui.s))
    for previous, following in pairwise(fields):
        assert following[1] - previous[1] - previous[3] == pytest.approx(7 * ui.s)
    records = [row for row in runtime.observed if row["kind"] == "interaction_hint"]
    assert len(records) == len(interaction_rows(DEFAULT_INPUT_BINDINGS))
    for row in records:
        x, y, w, h = row["hit"]
        for fx, fy, fw, fh in row["fields"]:
            tolerance = native_epsilon(x, y, w, h, fx, fy, fw, fh)
            assert x - tolerance <= fx and fx + fw <= x + w + tolerance
            assert y - tolerance <= fy and fy + fh <= y + h + tolerance
