"""Offscreen numeric ink can be omitted without changing native field ownership."""

from types import SimpleNamespace

import pytest
from imgui_bundle import imgui

from mojive.design.native import controls
from mojive.design.native.geometry import Rect
from tests.test_component_appearance import appearance as appearance

pytestmark = pytest.mark.parametrize("appearance", (1, 1.2, 1.25, 2.25), indirect=True)


def _mesh(draw):
    return (
        tuple((tuple(vertex.pos), tuple(vertex.uv), vertex.col) for vertex in draw.vtx_buffer),
        tuple(draw.idx_buffer),
    )


@pytest.mark.parametrize("axis", (None, "X"))
@pytest.mark.parametrize("integer", (False, True))
@pytest.mark.parametrize("empty", (False, True))
def test_offscreen_number_retains_native_submission_records_hits_and_extent(
    appearance, monkeypatch, axis, integer, empty
):
    ui, runtime, _update, frame = appearance
    runtime.observe_components = True
    painter = ui.design.primitives
    calls, snapshots = [], []
    drag_name = "drag_int" if integer else "drag_float"
    drag = getattr(imgui, drag_name)

    def submit(*args, **kwargs):
        calls.append(args[0])
        return drag(*args, **kwargs)

    monkeypatch.setattr(imgui, drag_name, submit)
    box = Rect(70.625 * ui.s, 90.25 * ui.s, 0 if empty else 130 * ui.s, 32 * ui.s)

    def draw():
        painter.observed.clear()
        ui.hits.clear()
        ui.numeric_edits.clear()
        imgui.push_clip_rect((20 * ui.s, 20 * ui.s), (40 * ui.s, 40 * ui.s), True)
        try:
            result = painter.input_number("culled-number", box, 7, axis=axis, integer=integer)
            native_id = imgui.get_item_id()
            cursor = tuple(imgui.internal.get_current_window().dc.cursor_max_pos)
            records = tuple(
                {name: value for name, value in record.items() if name != "submittedFrame"}
                for record in painter.observed
            )
            snapshots.append((result, dict(ui.hits), records, cursor, native_id))
        finally:
            imgui.pop_clip_rect()

    for _ in range(3):
        frame(draw)
    actual = snapshots[-1]
    assert len(calls) == 3
    assert calls == ["###culled-number"] * 3
    assert actual[4] != 0
    assert "culled-number" in ui.numeric_edits
    assert not ui.numeric_edits["culled-number"].active
    assert actual[1]["culled-number-frame"] == actual[0].box.frame.tuple()
    assert bool(actual[2][0]["nativeMinimumHit"]) is empty
    assert actual[2][0]["hit"] == actual[0].box.frame.tuple()
    if axis:
        assert "culled-number-reset" in actual[1]

    def unexpected(*_args, **_kwargs):
        pytest.fail("Fully clipped numeric fields must not prepare custom ink")

    with monkeypatch.context() as no_ink:
        no_ink.setattr(painter, "aligned_text", unexpected)
        no_ink.setattr(painter, "focus_frame", unexpected)
        no_ink.setattr(painter, "field_background", unexpected)
        no_ink.setattr(controls, "fit_numeric_text", unexpected)
        frame(draw)
    assert snapshots[-1] == actual
    with monkeypatch.context() as baseline:
        baseline.setattr(painter, "_field_paint_visible", lambda _frame: True)
        frame(draw)
    assert snapshots[-1] == actual


@pytest.mark.parametrize("radius", (0, 6, 40))
@pytest.mark.parametrize("clip", ("full", "narrow", "corner", "aa", "outside"))
def test_partial_clip_and_fractional_aa_keep_exact_baseline_numeric_ink(
    appearance, monkeypatch, radius, clip
):
    ui, _runtime, update, frame = appearance
    update({"input_number": {"radius": radius}})
    painter = ui.design.primitives
    rectangle = Rect(70.625 * ui.s, 90.25 * ui.s, 130 * ui.s, 32 * ui.s)
    snapshots, visible = [], []
    original = painter._field_paint_visible

    def visibility(box):
        result = original(box)
        visible.append(result)
        return result

    monkeypatch.setattr(painter, "_field_paint_visible", visibility)

    def draw():
        box = painter.field_box(painter.style(("input_number", "axis_input")), rectangle)
        right, bottom = box.frame.right, box.frame.bottom
        boundary = {
            "full": (rectangle.x - 10, rectangle.y - 10, right + 10, bottom + 10),
            "narrow": (right - 0.125, rectangle.y, right + 0.625, bottom),
            "corner": (right - 0.375, bottom - 0.625, right + 1, bottom + 1),
            "aa": (right + 0.25, rectangle.y, right + 1.25, bottom),
            "outside": (right + 2.25, rectangle.y, right + 3.25, bottom),
        }[clip]
        imgui.push_clip_rect(boundary[:2], boundary[2:], True)
        try:
            painter.input_number("edge-number", rectangle, 123.456789, axis="X")
            snapshots.append(_mesh(ui.draw))
        finally:
            imgui.pop_clip_rect()

    for _ in range(3):
        frame(draw)
    candidate = snapshots[-1]
    assert visible[-1] is (clip != "outside")
    with monkeypatch.context() as baseline:
        baseline.setattr(painter, "_field_paint_visible", lambda _frame: True)
        frame(draw)
    if clip == "outside":
        assert len(candidate[0]) < len(snapshots[-1][0])
    else:
        assert candidate == snapshots[-1]


@pytest.mark.parametrize("axis", (None, "X"))
def test_active_drag_and_text_input_keep_native_ownership_when_parent_clip_moves_away(
    appearance, axis
):
    ui, _runtime, _update, frame = appearance
    state = SimpleNamespace(value=1.0, clipped=False, changed=False)
    io = imgui.get_io()

    def draw():
        if state.clipped:
            imgui.push_clip_rect((0, 0), (10, 10), True)
        try:
            state.changed, state.value = ui.input(
                "active-culled-number",
                70.625 * ui.s,
                90.25 * ui.s,
                130 * ui.s,
                state.value,
                axis=axis,
                speed=0.1,
            )
        finally:
            if state.clipped:
                imgui.pop_clip_rect()

    def event(name, *args):
        getattr(io, name)(*args)
        frame(draw)

    for _ in range(3):
        frame(draw)
    x, y, width, height = ui.hits["active-culled-number"]
    x, y = x + width / 2, y + height / 2
    event("add_mouse_pos_event", x, y)
    event("add_mouse_button_event", 0, True)
    assert ui.numeric_edits["active-culled-number"].activated
    state.clipped = True
    event("add_mouse_pos_event", x + 30 * ui.s, y)
    assert state.changed and state.value > 1
    assert ui.numeric_edits["active-culled-number"].active
    event("add_mouse_button_event", 0, False)
    assert ui.numeric_edits["active-culled-number"].committed
    assert ui.numeric_edits["active-culled-number"].deactivated

    state.clipped = False
    frame(draw)
    modifier = imgui.Key.mod_super if io.config_mac_osx_behaviors else imgui.Key.mod_ctrl
    event("add_key_event", modifier, True)
    event("add_mouse_pos_event", x, y)
    event("add_mouse_button_event", 0, True)
    event("add_mouse_button_event", 0, False)
    event("add_key_event", modifier, False)
    assert ui.numeric_edits["active-culled-number"].active
    state.clipped = True
    event("add_input_characters_utf8", "9.250")
    assert ui.numeric_edits["active-culled-number"].active
    event("add_key_event", imgui.Key.enter, True)
    assert state.value == pytest.approx(9.25)
    assert ui.numeric_edits["active-culled-number"].committed
    event("add_key_event", imgui.Key.enter, False)
