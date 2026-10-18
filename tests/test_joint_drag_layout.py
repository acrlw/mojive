"""Unbounded joint fields share the native centered numeric caption layout."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
from imgui_bundle import imgui

from mojive.design.native.presentation import DesignPresentation
from mojive.design.runtime import DesignRuntime
from mojive.design.store import SpecificationStore
from mojive.ui.workspace.widgets import UI


@pytest.fixture(params=(1.0, 2.0))
def joint_field(request, tmp_path, monkeypatch):
    context = imgui.create_context()
    io = imgui.get_io()
    io.set_ini_filename(None)
    io.display_size = (900 * request.param, 350 * request.param)
    io.delta_time = 1 / 60
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    ui = UI(SimpleNamespace(style_scale=request.param, font_report=SimpleNamespace(cjk_path=None)))
    runtime = DesignRuntime(SpecificationStore(tmp_path / "design.json"))
    runtime.observe_components = True
    ui.design = DesignPresentation(ui, runtime)
    state = SimpleNamespace(
        ui=ui, runtime=runtime, value=0.0, width=500.0, unit="rad", bounds=None, painted=[]
    )
    original_text = ui.text

    def text(x, y, value, color, size, **options):
        state.painted.append(
            (
                value,
                x,
                y,
                ui.measure(value, size, **{k: v for k, v in options.items() if k != "max_width"}),
                size * ui.s,
            )
        )
        return original_text(x, y, value, color, size, **options)

    monkeypatch.setattr(ui, "text", text)

    def frame():
        imgui.new_frame()
        imgui.set_next_window_pos((0, 0))
        imgui.set_next_window_size(io.display_size)
        imgui.begin("Joint field", flags=imgui.WindowFlags_.no_title_bar)
        ui.design.begin_frame()
        ui.hits.clear()
        state.painted.clear()
        state.edit = ui.rail(
            "joint",
            40 * ui.s,
            50 * ui.s,
            state.width * ui.s,
            state.value,
            state.bounds,
            initial=0.0,
            show_reset=False,
            unit=state.unit,
            fmt="%.3f",
            angular_degrees=state.unit == "rad",
        )
        state.value = state.edit.value
        state.records = {row["key"]: row for row in ui.design.primitives.observed}
        imgui.end()
        imgui.render()

    state.frame = frame
    for _ in range(3):
        frame()
    try:
        yield state
    finally:
        imgui.destroy_context(context)


@pytest.mark.parametrize("width", (96, 320, 700))
@pytest.mark.parametrize("unit", ("rad", "m", ""))
@pytest.mark.parametrize("value", (0.0, 1.25, -12345.678))
def test_unbounded_joint_caption_is_centered_without_changing_field_bounds(
    joint_field, width, unit, value
):
    state = joint_field
    state.width, state.unit = width, unit
    state.frame()
    baseline = dict(state.ui.hits)
    state.value = value
    state.frame()
    assert "joint-track" not in state.ui.hits
    assert state.ui.hits["joint-value"] == baseline["joint-value"]
    assert state.value == value
    numeric = state.records["##joint-value"]
    cx, cy, cw, ch = numeric["content"]
    left = min(row[1] for row in state.painted)
    right = max(row[1] + row[3] for row in state.painted)
    assert (left + right) / 2 == pytest.approx(cx + cw / 2, abs=0.01)
    for _text, x, y, text_width, text_height in state.painted:
        assert y + text_height / 2 == pytest.approx(cy + ch / 2, abs=0.01)
        assert cx - 0.01 <= x <= x + text_width <= cx + cw + 0.01


def test_bounded_joint_retains_separate_track_and_stable_trailing_unit(joint_field):
    state = joint_field
    state.bounds = (-1.5, 1.5)
    state.frame()
    track = state.ui.hits["joint-track"]
    field = state.ui.hits["joint-value"]
    unit_x = state.painted[-1][1]
    assert track[0] + track[2] < field[0]
    assert field[2] < state.width * state.ui.s / 2
    state.value = 1.25
    state.frame()
    assert state.ui.hits["joint-track"] == track
    assert state.ui.hits["joint-value"] == field
    assert state.painted[-1][1] == unit_x


@pytest.mark.parametrize("align", ("left", "center", "right"))
def test_unbounded_caption_respects_authored_alignment_and_content_insets(joint_field, align):
    state = joint_field
    spec = deepcopy(state.runtime.spec)
    spec["parts"]["input_number"] = {
        "textAlign": align,
        "paddingLeft": 16,
        "paddingRight": 24,
        "paddingTop": 6,
        "paddingBottom": 10,
    }
    state.runtime.update_spec(spec)
    state.frame()
    numeric = state.records["##joint-value"]
    cx, cy, cw, ch = numeric["content"]
    left = min(row[1] for row in state.painted)
    right = max(row[1] + row[3] for row in state.painted)
    if align == "left":
        assert left == pytest.approx(cx)
    elif align == "right":
        assert right == pytest.approx(cx + cw)
    else:
        assert (left + right) / 2 == pytest.approx(cx + cw / 2)
    for _text, _x, y, _width, height in state.painted:
        assert y + height / 2 == pytest.approx(cy + ch / 2)


def test_unbounded_joint_drag_preserves_native_editing_and_centered_caption(joint_field):
    state = joint_field
    x, y, width, height = state.ui.hits["joint-value"]
    io = imgui.get_io()
    io.add_mouse_pos_event(x + width / 2, y + height / 2)
    state.frame()
    io.add_mouse_button_event(0, True)
    state.frame()
    for offset in (20, 40, 60):
        io.add_mouse_pos_event(x + width / 2 + offset * state.ui.s, y + height / 2)
        state.frame()
    assert state.value != 0
    assert state.ui.hits["joint-value"] == (x, y, width, height)
    numeric = state.records["##joint-value"]
    cx, _cy, cw, _ch = numeric["content"]
    left = min(row[1] for row in state.painted)
    right = max(row[1] + row[3] for row in state.painted)
    assert (left + right) / 2 == pytest.approx(cx + cw / 2, abs=0.01)
    io.add_mouse_button_event(0, False)
    state.frame()
