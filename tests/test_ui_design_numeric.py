"""Native reference numeric input without a window or GPU renderer."""

from types import SimpleNamespace

import pytest
from examples.ui_design.native.widgets import UI
from imgui_bundle import imgui

from mojive.ui.imgui_draw import ImguiDraw2D
from mojive.ui.localization import Language, Localizer


@pytest.fixture(params=(1.0, 2.0, 2.25))
def numeric(request):
    context = imgui.create_context()
    io = imgui.get_io()
    io.set_ini_filename(None)
    io.display_size = (800 * request.param, 400 * request.param)
    io.delta_time = 1 / 60
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    io.config_flags |= imgui.ConfigFlags_.nav_enable_keyboard
    language = request.node.callspec.params.get("language")
    if language is not None:
        report = SimpleNamespace(cjk_path=None)
        if language == "zh_CN":
            from mojive.text.sources import resolve_font_sources

            sources = resolve_font_sources(allow_download=False)
            if len(sources) < 2:
                pytest.skip("CJK font is unavailable")
            report.cjk_path, report.cjk_index = sources[1].path, sources[1].index
        ui = UI(SimpleNamespace(style_scale=request.param, font_report=report))
    else:
        ui = UI.__new__(UI)
        ui.body = ui.bold = ui.mono = io.fonts.add_font_default()
    ui.s = request.param
    ui.translate = Localizer(Language(language)).text if language is not None else str
    ui.hits, ui.numeric_edits = {}, {}
    state = SimpleNamespace(
        ui=ui,
        io=io,
        value=1.0,
        axis="X",
        disabled=False,
        vector=False,
        vector_width=330,
        vector_layout=None,
        values=[1.0, 2.0, 3.0],
        reset_value=0.0,
        reset_values=(0.0, 0.0, 0.0),
        width=110,
        format="%.3f",
    )

    def frame(count=1):
        for _ in range(count):
            imgui.new_frame()
            imgui.set_next_window_pos((0, 0))
            imgui.set_next_window_size(io.display_size)
            imgui.begin("Numeric input", flags=imgui.WindowFlags_.no_title_bar)
            imgui.begin_disabled(state.disabled)
            ui.numeric_edits.clear()
            if state.vector:
                state.changed, state.values = ui.vector(
                    "##vector",
                    40 * ui.s,
                    50 * ui.s,
                    state.vector_width * ui.s,
                    "Position",
                    state.values,
                    speed=0.1,
                    reset_values=state.reset_values,
                    format=state.format,
                    layout=state.vector_layout,
                )
            else:
                state.changed, state.value = ui.input(
                    "##number",
                    40 * ui.s,
                    50 * ui.s,
                    state.width * ui.s,
                    state.value,
                    axis=state.axis,
                    speed=0.1,
                    reset_value=state.reset_value,
                    format=state.format,
                )
            imgui.end_disabled()
            imgui.end()
            imgui.render()

    def event(method, *args):
        getattr(io, method)(*args)
        frame()

    def point(key="##number"):
        x, y, w, h = ui.hits[key]
        return x + w / 2, y + h / 2

    def text_input(text):
        event("add_key_event", imgui.Key.mod_ctrl, True)
        event("add_mouse_pos_event", *point())
        event("add_mouse_button_event", 0, True)
        event("add_mouse_button_event", 0, False)
        event("add_key_event", imgui.Key.mod_ctrl, False)
        event("add_input_characters_utf8", text)

    state.frame, state.event, state.point, state.text_input = frame, event, point, text_input
    frame(3)
    try:
        yield state
    finally:
        imgui.destroy_context(context)


@pytest.mark.parametrize("axis", ("X", "R", "A", None))
def test_numeric_drag_updates_while_active_and_publishes_native_completion(numeric, axis):
    numeric.axis = axis
    numeric.frame()
    x, y = numeric.point()
    numeric.event("add_mouse_pos_event", x, y)
    numeric.event("add_mouse_button_event", 0, True)
    assert numeric.ui.numeric_edits["##number"].activated
    numeric.event("add_mouse_pos_event", x + 25 * numeric.ui.s, y)
    assert numeric.changed and numeric.value > 1.0
    assert numeric.ui.numeric_edits["##number"].active
    numeric.event("add_mouse_button_event", 0, False)
    edit = numeric.ui.numeric_edits["##number"]
    assert edit.deactivated and edit.committed and not edit.canceled
    assert numeric.ui.hits["##number"][3] == pytest.approx(28 * numeric.ui.s, abs=1)


@pytest.mark.parametrize("key, expected", ((imgui.Key.enter, 2.75), (imgui.Key.escape, 1.0)))
def test_numeric_text_uses_native_validation_and_escape_cancel(numeric, key, expected):
    numeric.text_input("2.750")
    assert numeric.ui.numeric_edits["##number"].active
    assert numeric.value == 1.0
    numeric.event("add_key_event", key, True)
    assert numeric.value == pytest.approx(expected)
    edit = numeric.ui.numeric_edits["##number"]
    assert edit.deactivated
    assert edit.canceled is (key == imgui.Key.escape)
    assert edit.committed is (key == imgui.Key.enter)
    numeric.event("add_key_event", key, False)


def test_numeric_disabled_drag_and_untouched_precision(numeric):
    numeric.value = 1.123456789
    numeric.disabled = True
    numeric.frame()
    x, y = numeric.point()
    numeric.event("add_mouse_pos_event", x, y)
    numeric.event("add_mouse_button_event", 0, True)
    numeric.event("add_mouse_pos_event", x + 25 * numeric.ui.s, y)
    numeric.event("add_mouse_button_event", 0, False)
    assert numeric.value == 1.123456789
    assert not numeric.changed
    assert not numeric.ui.numeric_edits["##number"].activated


def test_numeric_drag_escape_publishes_transaction_cancel(numeric):
    x, y = numeric.point()
    numeric.event("add_mouse_pos_event", x, y)
    numeric.event("add_mouse_button_event", 0, True)
    numeric.event("add_mouse_pos_event", x + 25 * numeric.ui.s, y)
    assert numeric.value > 1.0
    numeric.event("add_key_event", imgui.Key.escape, True)
    edit = numeric.ui.numeric_edits["##number"]
    assert edit.canceled and edit.deactivated and not edit.active
    numeric.event("add_key_event", imgui.Key.escape, False)
    numeric.event("add_mouse_button_event", 0, False)


def test_numeric_double_click_enters_native_text_edit(numeric):
    numeric.event("add_mouse_pos_event", *numeric.point())
    for _ in range(2):
        numeric.event("add_mouse_button_event", 0, True)
        numeric.event("add_mouse_button_event", 0, False)
    numeric.event("add_input_characters_utf8", "4.500")
    assert numeric.value == 1.0
    numeric.event("add_key_event", imgui.Key.enter, True)
    assert numeric.value == pytest.approx(4.5)
    assert numeric.ui.numeric_edits["##number"].committed
    numeric.event("add_key_event", imgui.Key.enter, False)


def test_vector_aggregates_actual_axis_drag_signals(numeric):
    numeric.vector = True
    numeric.frame()
    x, y = numeric.point("##vectorY")
    numeric.event("add_mouse_pos_event", x, y)
    numeric.event("add_mouse_button_event", 0, True)
    assert numeric.ui.numeric_edits["##vector"].activated
    numeric.event("add_mouse_pos_event", x + 25 * numeric.ui.s, y)
    assert numeric.changed and numeric.values[1] > 2.0
    assert numeric.values[0] == 1.0 and numeric.values[2] == 3.0
    assert numeric.ui.numeric_edits["##vector"].active
    numeric.event("add_mouse_button_event", 0, False)
    assert numeric.ui.numeric_edits["##vector"].deactivated
    assert numeric.ui.numeric_edits["##vector"].committed


def test_axis_badge_has_native_press_release_and_configured_reset(numeric):
    numeric.reset_value = 0.25
    numeric.frame()
    numeric.event("add_mouse_pos_event", *numeric.point("##number-reset"))
    numeric.event("add_mouse_button_event", 0, True)
    edit = numeric.ui.numeric_edits["##number"]
    assert edit.activated and edit.active and not edit.reset
    assert numeric.value == 1.0
    numeric.event("add_mouse_button_event", 0, False)
    edit = numeric.ui.numeric_edits["##number"]
    assert edit.reset and edit.deactivated and edit.committed and not edit.active
    assert numeric.changed and numeric.value == 0.25


def test_disabled_axis_badge_preserves_value(numeric):
    numeric.disabled = True
    numeric.frame()
    numeric.event("add_mouse_pos_event", *numeric.point("##number-reset"))
    numeric.event("add_mouse_button_event", 0, True)
    numeric.event("add_mouse_button_event", 0, False)
    assert numeric.value == 1.0
    assert not numeric.changed and not numeric.ui.numeric_edits["##number"].reset


def test_axis_badge_has_distinct_hover_and_press_colors(numeric, monkeypatch):
    from examples.ui_design.native import widgets

    colors = []
    painter = widgets.draw_joined_field_frame

    def observe(*args, **kwargs):
        colors.append(kwargs["badge_color"])
        return painter(*args, **kwargs)

    monkeypatch.setattr(widgets, "draw_joined_field_frame", observe)
    numeric.frame()
    base = colors[-1]
    numeric.event("add_mouse_pos_event", *numeric.point("##number-reset"))
    hovered = colors[-1]
    numeric.event("add_mouse_button_event", 0, True)
    pressed = colors[-1]
    assert base != hovered and hovered != pressed
    numeric.event("add_mouse_button_event", 0, False)


def test_vector_badge_reset_uses_per_axis_defaults(numeric):
    numeric.vector = True
    numeric.reset_values = (1.0, 1.0, 1.0)
    numeric.frame()
    numeric.event("add_mouse_pos_event", *numeric.point("##vectorY-reset"))
    numeric.event("add_mouse_button_event", 0, True)
    numeric.event("add_mouse_button_event", 0, False)
    assert numeric.values == [1.0, 1.0, 3.0]
    assert numeric.changed and numeric.ui.numeric_edits["##vector"].reset


@pytest.mark.parametrize("language", ("en", "zh_CN"))
@pytest.mark.parametrize(
    "width,columns", ((180, 1), (228, 1), (280, 1), (296, 3), (320, 3), (520, 3), (700, 3))
)
@pytest.mark.parametrize(
    "values,format,shown",
    (
        ((0.0041472, -90.0, 1.728), "%.7f", ("0.0041472", "-90", "1.728")),
        ((0.2, 0.000123, -0.000123), "%.7f", ("0.2", "0.000123", "-0.000123")),
        ((0.0533713, 0.00288568, 0.000123), "%.6g", ("0.0533713", "0.00288568", "0.000123")),
    ),
)
def test_vector_layout_keeps_common_values_visible_in_equal_native_fields(
    numeric, monkeypatch, language, width, columns, values, format, shown
):
    numeric.vector, numeric.vector_width = True, width
    numeric.values, numeric.format = list(values), format
    painted = []
    original = ImguiDraw2D.text

    def text(draw, position, color, value, **kwargs):
        painted.append(value)
        return original(draw, position, color, value, **kwargs)

    monkeypatch.setattr(ImguiDraw2D, "text", text)
    numeric.frame(3)
    ui, s = numeric.ui, numeric.ui.s
    layout = ui.vector_layout(width * s)
    frames = [ui.hits["##vector" + axis + "-frame"] for axis in "XYZ"]
    assert layout.columns == columns
    assert len({round(field[2], 4) for field in frames}) == 1
    assert all(field[3] == pytest.approx(28 * s, abs=1) for field in frames)
    assert all(value in painted for value in shown)
    assert all(not value.endswith("…") for value in painted)
    assert ui.translate("Position") in painted
    for x, y, w, h in frames:
        assert 40 * s <= x and x + w <= (40 + width) * s + 0.5
        assert 50 * s <= y and y + h <= 50 * s + layout.height + 0.5
    if columns == 1:
        assert len({field[0] for field in frames}) == 1
        assert all(
            frames[index + 1][1] - frames[index][1] - frames[index][3]
            == pytest.approx(8 * s, abs=1)
            for index in range(2)
        )
    else:
        assert len({field[1] for field in frames}) == 1
    assert numeric.values == list(values) and not numeric.changed


@pytest.mark.parametrize("language", ("en", "zh_CN"))
def test_stacked_xyz_fields_keep_real_drag_and_badge_reset(numeric, language):
    numeric.vector, numeric.vector_width = True, 180
    numeric.values, numeric.format = [0.0041472, 1.728, 0.000123], "%.7f"
    numeric.frame(3)
    fields = [numeric.ui.hits["##vector" + axis + "-frame"] for axis in "XYZ"]
    x, y = numeric.point("##vectorY")
    numeric.event("add_mouse_pos_event", x, y)
    numeric.event("add_mouse_button_event", 0, True)
    numeric.event("add_mouse_pos_event", x + 25 * numeric.ui.s, y)
    assert numeric.values[1] > 1.728 and numeric.ui.numeric_edits["##vector"].active
    assert numeric.values[0] == 0.0041472 and numeric.values[2] == 0.000123
    assert fields == [numeric.ui.hits["##vector" + axis + "-frame"] for axis in "XYZ"]
    numeric.event("add_mouse_button_event", 0, False)
    assert numeric.ui.numeric_edits["##vector"].committed
    numeric.event("add_mouse_pos_event", *numeric.point("##vectorZ-reset"))
    numeric.event("add_mouse_button_event", 0, True)
    numeric.event("add_mouse_button_event", 0, False)
    assert numeric.values[2] == 0.0 and numeric.ui.numeric_edits["##vector"].reset


@pytest.mark.parametrize("language", ("en", "zh_CN"))
@pytest.mark.parametrize("axis", ("X", "R", "A", None))
def test_long_numeric_display_uses_padded_clipped_ellipsis_and_full_hover(
    numeric, monkeypatch, language, axis
):
    numeric.axis, numeric.width, numeric.value = axis, 82, -1234567890123.456
    numeric.format = "%.6f" + (" 毫米" if language == "zh_CN" else " mm")
    painted, hints = [], []
    original_text, original_hint = ImguiDraw2D.text, numeric.ui.tooltip

    def text(draw, position, color, value, **kwargs):
        draw_list = imgui.get_window_draw_list()
        clip_lo, clip_hi = draw_list.get_clip_rect_min(), draw_list.get_clip_rect_max()
        start = len(draw_list.vtx_buffer)
        original_text(draw, position, color, value, **kwargs)
        if value not in ("X", "R", "A"):
            painted.append(
                (
                    value,
                    position,
                    (clip_lo.x, clip_lo.y, clip_hi.x, clip_hi.y),
                    tuple(
                        (draw_list.vtx_buffer[index].pos.x, draw_list.vtx_buffer[index].pos.y)
                        for index in range(start, len(draw_list.vtx_buffer))
                    ),
                )
            )

    def tooltip(value):
        hints.append(value)
        original_hint(value)

    monkeypatch.setattr(ImguiDraw2D, "text", text)
    monkeypatch.setattr(numeric.ui, "tooltip", tooltip)
    numeric.frame(3)
    shown, position, clip, vertices = painted[-1]
    x, y, width, height = numeric.ui.hits["##number"]
    s = numeric.ui.s
    assert shown.endswith("…") and shown != numeric.format % numeric.value
    assert clip[0] >= x + 8 * s - 0.5
    assert clip[2] <= x + width - 8 * s + 0.5
    assert position[0] == pytest.approx(clip[0], abs=0.5)
    assert vertices
    for vx, vy in vertices:
        assert clip[0] - 0.5 <= vx <= clip[2] + 0.5
        assert y <= vy <= y + height
    if axis:
        badge_x, _, badge_width, _ = numeric.ui.hits["##number-reset"]
        assert clip[0] >= badge_x + badge_width + 8 * s - 0.5
    before = numeric.value
    numeric.event("add_mouse_pos_event", *numeric.point())
    assert hints[-1] == str(before)
    assert numeric.value == before and not numeric.changed


@pytest.mark.parametrize("language", ("en", "zh_CN"))
@pytest.mark.parametrize("axis", ("X", None))
@pytest.mark.parametrize("value", (0.0, 12.34))
def test_short_numeric_display_is_centered_inside_padded_content(
    numeric, monkeypatch, language, axis, value
):
    numeric.axis, numeric.value = axis, value
    painted = []
    original = ImguiDraw2D.text

    def text(draw, position, color, shown, **kwargs):
        draw_list = imgui.get_window_draw_list()
        clip_lo, clip_hi = draw_list.get_clip_rect_min(), draw_list.get_clip_rect_max()
        original(draw, position, color, shown, **kwargs)
        if shown != "X":
            painted.append(
                (shown, position[0], imgui.calc_text_size(shown).x, clip_lo.x, clip_hi.x)
            )

    monkeypatch.setattr(ImguiDraw2D, "text", text)
    numeric.frame(3)
    shown, text_x, text_width, clip_lo, clip_hi = painted[-1]
    x, _, width, _ = numeric.ui.hits["##number"]
    assert shown == f"{value:g}"
    assert clip_lo == pytest.approx(x + 8 * numeric.ui.s, abs=0.5)
    assert clip_hi == pytest.approx(x + width - 8 * numeric.ui.s, abs=0.5)
    assert text_x + text_width / 2 == pytest.approx((clip_lo + clip_hi) / 2, abs=0.5)
    assert text_x > clip_lo and text_x + text_width < clip_hi


@pytest.mark.parametrize("language", ("en", "zh_CN"))
@pytest.mark.parametrize("axis", ("X", None))
def test_disabled_ctrl_click_preserves_padded_idle_display(numeric, monkeypatch, language, axis):
    numeric.axis, numeric.width, numeric.value = axis, 72, -1234567890123.456
    numeric.disabled = True
    painted = []
    original = ImguiDraw2D.text

    def text(draw, position, color, shown, **kwargs):
        if shown != "X":
            painted.append(shown)
        return original(draw, position, color, shown, **kwargs)

    monkeypatch.setattr(ImguiDraw2D, "text", text)
    numeric.frame(3)
    numeric.event("add_key_event", imgui.Key.mod_ctrl, True)
    numeric.event("add_mouse_pos_event", *numeric.point())
    painted.clear()
    numeric.event("add_mouse_button_event", 0, True)
    assert painted and painted[-1].endswith("…")
    assert not numeric.ui.numeric_edits["##number"].active
    assert numeric.value == -1234567890123.456
    numeric.event("add_mouse_button_event", 0, False)
    numeric.event("add_key_event", imgui.Key.mod_ctrl, False)


@pytest.mark.parametrize("language", ("en", "zh_CN"))
def test_narrow_text_editor_keeps_native_text_cursor_and_horizontal_scroll(
    numeric, monkeypatch, language
):
    numeric.width = 72
    input_frames = []
    original = imgui.drag_float

    def drag(identifier, *args, **kwargs):
        alpha = imgui.get_style().color_(imgui.Col_.text).w
        result = original(identifier, *args, **kwargs)
        if imgui.internal.temp_input_is_active(imgui.get_item_id()):
            input_frames.append(alpha)
        return result

    monkeypatch.setattr(imgui, "drag_float", drag)
    numeric.frame()
    numeric.text_input("1234567890.125")
    numeric.frame(3)
    assert numeric.io.want_text_input
    assert input_frames and all(alpha > 0 for alpha in input_frames)
    assert imgui.get_current_context().input_text_state.scroll.x > 0
    assert numeric.value == 1.0
    numeric.event("add_key_event", imgui.Key.escape, True)
    assert numeric.ui.numeric_edits["##number"].canceled
    numeric.event("add_key_event", imgui.Key.escape, False)
    assert numeric.value == 1.0


@pytest.mark.parametrize(
    "value,format,expected",
    (
        (0.0, "%.5f", "0"),
        (12.34, "%.5f", "12.34"),
        (0.004147, "%.6f", "0.004147"),
        (1000.0, "%.2f", "1000"),
        (12.34, "%.5f kg", "12.34000 kg"),
        (0.0, "mass: %.5f", "mass: 0.00000"),
    ),
)
def test_idle_fixed_numbers_drop_only_fractional_zeroes(
    numeric, monkeypatch, value, format, expected
):
    numeric.value, numeric.format, numeric.width = value, format, 230
    painted = []
    original = ImguiDraw2D.text

    def text(draw, position, color, value, **kwargs):
        painted.append(value)
        return original(draw, position, color, value, **kwargs)

    monkeypatch.setattr(ImguiDraw2D, "text", text)
    numeric.frame(3)
    assert expected in painted
    assert numeric.value == value and not numeric.changed


def test_zero_stays_readable_in_a_narrow_axis_field(numeric, monkeypatch):
    numeric.value, numeric.format, numeric.width = 0.0, "%.5f", 52
    painted = []
    original = ImguiDraw2D.text

    def text(draw, position, color, value, **kwargs):
        painted.append(value)
        return original(draw, position, color, value, **kwargs)

    monkeypatch.setattr(ImguiDraw2D, "text", text)
    numeric.frame(3)
    assert painted[-1] == "0"
    assert numeric.value == 0.0
