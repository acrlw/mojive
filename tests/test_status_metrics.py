"""Status typography keeps numeric ink inside its measured production columns."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
from imgui_bundle import imgui

from mojive.ui.viewport_widgets.status import format_simulation_steps
from mojive.ui.viewport_widgets.status_metrics import (
    METRIC_GROUP_GAP,
    METRIC_LABEL_GAP,
    METRIC_TEXT_INSET,
    StatusReadoutStyle,
    _status_performance_layout,
    _StatusNumbers,
)
from mojive.ui.workspace.widgets import UI


@pytest.mark.parametrize(
    "count,caption",
    (
        (999, "999"),
        (1000, "1k"),
        (999_949, "999.9k"),
        (999_999, "1M"),
        (999_999_999, "1B"),
        (10**12, "1T"),
        (2**63 - 1, "9.2E"),
    ),
)
def test_step_units_promote_before_a_fourth_integer_digit(count, caption):
    assert format_simulation_steps(count) == caption


@pytest.mark.parametrize("family", ("system", "portable"))
@pytest.mark.parametrize("size", (13, 14, 21.5, 27))
@pytest.mark.parametrize("width", (0, 1, 18, 48))
@pytest.mark.parametrize("display_format", ("seconds", "clock"))
def test_actual_font_advances_fit_the_number_and_all_unit_columns(
    family, size, width, display_format
):
    context = imgui.create_context()
    io = imgui.get_io()
    io.display_size = (1600, 1000)
    io.delta_time = 1 / 60
    io.set_ini_filename(None)
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    ui = UI(SimpleNamespace(style_scale=1, font_report=SimpleNamespace(cjk_path=None)))
    ui.prepare_fonts(family)
    painted = []

    class FontDraw:
        def text_size(self, text):
            return tuple(imgui.calc_text_size(text))

        def text_ink_bounds(self, text):
            return None

        def text(self, position, color, text, **kwargs):
            painted.append((position[0], position[0] + self.text_size(text)[0], text))

    try:
        imgui.new_frame()
        imgui.push_font(ui.body, size)
        draw = FontDraw()
        # Exercise a wider SI prefix and a time value beyond the ordinary domain.
        for caption, fps, hz, seconds in (
            ("Time 0.000 s", 51.1, 0, 0),
            ("Time 123456d 23:59:59", 1e6, 1e9, None),
            ("Steps 12.3k", 999.9, 99, None),
            ("Time 0 s", 59.8, 100, 3661.125),
            ("Time 0 s", 59.8, 100, 1e308),
            ("Time 0 s", 59.8, 100, 1e-308),
        ):
            layout = _status_performance_layout(
                draw,
                1500,
                1,
                "OpenGL",
                0.002,
                fps,
                metric_text=caption,
                metric_mode="steps" if caption.startswith("Steps") else "time",
                step=12345,
                show_physics=True,
                physics_hz=hz,
                delta_value_width=width,
                physics_value_width=width,
                steps_value_width=width,
                time_value_width=width,
                render_value_width=width,
                sim_time=seconds,
                time_format=display_format,
            )
            for field in layout.fields:
                painted.clear()
                _StatusNumbers(draw).paint(field, 20, (1, 1, 1, 1))
                assert painted
                assert all(
                    field.x <= left <= right <= field.x + field.width + 1e-6
                    for left, right, _ in painted
                )
                if field.key != "backend" and field.unit:
                    numeric_end = field.value_x + (
                        field.value_text_width if field.compact else field.value_width
                    )
                    assert painted[-1][2] == field.unit
                    assert painted[-1][0] - numeric_end == pytest.approx(field.unit_gap)
                    assert field.value_x - field.content_x - field.label_width == pytest.approx(
                        field.label_gap
                    )
                if caption.startswith("Time 123456") and field.key == "metric":
                    assert any(text == "…" for _, _, text in painted)
        imgui.pop_font()
        imgui.render()
    finally:
        imgui.destroy_context(context)


class MetricDraw:
    def __init__(self):
        self.painted = []

    def text_size(self, text):
        return sum({"1": 4, ".": 3, "…": 10}.get(char, 8) for char in text), 14

    def text_ink_bounds(self, text):
        return None

    def text(self, position, color, text, **kwargs):
        self.painted.append((text, position))


def metric_layout(draw, **options):
    return {
        field.key: field
        for field in _status_performance_layout(
            draw,
            1600,
            1,
            "OpenGL",
            options.pop("dt", 0.002),
            options.pop("fps", 59),
            show_physics=True,
            physics_hz=options.pop("physics_hz", 100),
            **options,
        ).fields
    }


def test_internal_gaps_are_compact_and_independent_from_the_number_budget():
    draw = MetricDraw()
    fields = metric_layout(draw)
    compact = metric_layout(draw, label_gap=0, unit_gap=0)
    for key in ("delta", "physics", "render"):
        field, edited = fields[key], compact[key]
        assert field.value_x - field.content_x - draw.text_size(field.label)[0] == 4
        assert field.unit_x - field.value_x - field.value_text_width == 2
        assert edited.value_x - edited.content_x == draw.text_size(edited.label)[0]
        assert edited.unit_x == edited.value_x + edited.value_text_width
        assert field.width - edited.width == 6
        assert field.value_width == edited.value_width
    assert fields["delta"].value == "2.0"


@pytest.mark.parametrize("hz,fps", ((0, 51.1), (60, 59.0), (1e6, 1e6)))
def test_auto_units_follow_displayed_numbers_without_hidden_prefix_space(hz, fps):
    draw = MetricDraw()
    fields = metric_layout(draw, physics_hz=hz, fps=fps)
    for key in ("delta", "physics", "render"):
        field = fields[key]
        draw.painted.clear()
        _StatusNumbers(draw).paint(field, 20, (1, 1, 1, 1))
        unit, position = draw.painted[-1]
        assert unit == field.unit
        assert position[0] == pytest.approx(field.value_x + field.value_text_width + 2)
        assert field.value_width == field.value_text_width <= field.format_width
        assert position[0] + draw.text_size(unit)[0] == pytest.approx(
            field.x + field.width - METRIC_TEXT_INSET
        )


@pytest.mark.parametrize("separator_width", (0, 1, 6, 12))
def test_divider_sides_and_unit_right_inset_are_independent(separator_width):
    draw = MetricDraw()
    measure = _StatusNumbers(draw).width
    styles = {
        "physics": StatusReadoutStyle(draw, measure, 0, 2, 2, right_gap=7, left_gap=3),
        "render": StatusReadoutStyle(draw, measure, 0, 2, 2, right_gap=11, left_gap=5),
    }

    def layout(**changes):
        customized = styles | changes
        result = _status_performance_layout(
            draw,
            1600,
            1,
            "OpenGL",
            0.002,
            59.0,
            physics_hz=0,
            show_physics=True,
            metric_styles=customized,
            separator_width=separator_width,
        )
        physics, render = result.fields[-2:]
        divider = result.dividers[-1]
        assert divider - separator_width / 2 - (physics.x + physics.width) == pytest.approx(
            customized["physics"].right_gap
        )
        assert render.x - divider - separator_width / 2 == pytest.approx(
            customized["render"].left_gap
        )
        assert 1600 - (render.x + render.width) == pytest.approx(customized["render"].right_gap)
        assert physics.unit_x - physics.value_x - physics.value_text_width == 2
        assert physics.x + physics.width - physics.unit_x - physics.unit_width == (
            pytest.approx(customized["physics"].padding_right)
        )
        return physics, render, divider

    physics, render, divider = layout()
    edited_physics, edited_render, edited_divider = layout(
        physics=replace(styles["physics"], right_gap=17)
    )
    assert edited_render == render
    assert edited_divider == divider
    assert edited_physics.unit_x == physics.unit_x - 10
    _, edited_render, edited_divider = layout(render=replace(styles["render"], left_gap=15))
    assert edited_render.x == render.x
    assert edited_divider == divider - 10
    padded, edited_render, edited_divider = layout(
        physics=replace(styles["physics"], padding_right=9)
    )
    assert edited_render == render
    assert edited_divider == divider
    assert padded.unit_x == physics.unit_x - 9


@pytest.mark.parametrize("width", (0, 1, 18, 28, 48))
def test_auto_phrase_padding_and_authored_decimal_anchors_stay_stable(width):
    draw = MetricDraw()
    seen = {}
    for dt, hz in (
        (0.002, 0),
        (1 / 60, 0.01),
        (0.0511, 0.5),
        (0.0999, 10),
        (0.1234, 100),
        (0.9995, 999),
        (1.0, 12345),
    ):
        fields = metric_layout(
            draw, dt=dt, physics_hz=hz, delta_value_width=width, physics_value_width=width
        )
        for key in ("delta", "physics", "render"):
            field = fields[key]
            if not field.compact:
                anchors = (field.x, field.width, field.value_x, field.value_width, field.unit_x)
                assert seen.setdefault(key, anchors) == anchors
            draw.painted.clear()
            _StatusNumbers(draw).paint(field, 20, (1, 1, 1, 1))
            if field.compact:
                assert field.unit_x - field.value_x - field.value_text_width == 2
                assert field.value_width == field.value_text_width <= field.format_width
                assert field.x + field.width - field.unit_x - field.unit_width == 2
            elif "." in field.value:
                decimal = next(position[0] for text, position in draw.painted if text == ".")
                assert seen.setdefault((key, "decimal"), decimal) == decimal


def test_narrow_slots_promote_complete_values_or_show_one_overflow_mark():
    draw = MetricDraw()
    narrow = metric_layout(draw, delta_value_width=16, physics_value_width=16)
    assert (narrow["physics"].value, narrow["physics"].unit) == (".1", "kHz")
    tiny = metric_layout(draw, delta_value_width=1, physics_value_width=1)
    assert tiny["physics"].value == "…"
    assert tiny["physics"].exact == "100.0 Hz"
    assert tiny["delta"].exact == "0.002 s"
    assert tiny["physics"].value_width >= draw.text_size("…")[0]
    assert tiny["render"] == narrow["render"]


def test_small_physics_rates_and_short_delta_values_use_the_available_space():
    draw = MetricDraw()
    fields = metric_layout(draw, physics_hz=0.1, delta_value_width=20)
    assert fields["delta"].value == "2.0"
    assert fields["physics"].value == ".10"
    assert fields["physics"].unit == "Hz"
    assert fields["physics"].exact == "0.1 Hz"


def test_default_physics_budget_keeps_three_digit_rates_in_hz():
    draw = MetricDraw()
    for hz in (0, 1, 9, 10, 60, 99, 100, 480, 999):
        field = metric_layout(draw, physics_hz=hz)["physics"]
        assert field.format_width == 3 * draw.text_size("9")[0]
        assert field.value_width == _StatusNumbers(draw).width(str(hz))
        assert field.value == str(hz)
        assert field.unit == "Hz"


@pytest.mark.parametrize("width", (1, 18, 28, 48))
def test_steps_and_render_widths_preserve_bounds_and_anchors(width):
    draw, seen = MetricDraw(), {}
    for count, fps in (
        (0, 0),
        (100, 9.1),
        (101, 19.1),
        (999, 99.1),
        (1000, 51.1),
        (12345, 59.0),
        (999000, 99.1),
        (999999, 999.9),
    ):
        fields = metric_layout(
            draw,
            metric_mode="steps",
            metric_text=f"Steps {format_simulation_steps(count)}",
            step=count,
            fps=fps,
            steps_value_width=width,
            render_value_width=width,
        )
        for key in ("metric", "render"):
            field = fields[key]
            anchors = (field.x, field.width, field.value_x, field.value_width, field.unit_x)
            assert seen.setdefault(key, anchors) == anchors
            draw.painted.clear()
            _StatusNumbers(draw).paint(field, 20, (1, 1, 1, 1))
            ink = [
                (position[0], position[0] + draw.text_size(text)[0])
                for text, position in draw.painted[1:]
            ]
            assert all(
                field.value_x <= left <= right <= field.x + field.width for left, right in ink
            )
            if "." in field.value:
                point = next(position[0] for text, position in draw.painted if text == ".")
                assert seen.setdefault((key, "point"), point) == point


def test_steps_time_and_render_widths_apply_independently():
    draw = MetricDraw()
    options = {"metric_mode": "steps", "metric_text": "Steps 12.3k", "step": 12345}
    baseline = metric_layout(draw, **options)
    edited = metric_layout(draw, **options, steps_value_width=20, time_value_width=1)
    assert edited["metric"].value_width == 20
    assert edited["render"] == baseline["render"]
    assert edited["physics"] == baseline["physics"]
    time_options = {"metric_mode": "time", "metric_text": "Time 1:23.456"}
    timed = metric_layout(draw, **time_options)
    assert metric_layout(draw, **time_options, steps_value_width=1) == timed
    narrow = metric_layout(draw, **time_options, time_value_width=18)
    assert narrow["metric"].value == "…"
    assert narrow["metric"].value_width == 18
    assert narrow["render"] == timed["render"]
    assert narrow["physics"] == timed["physics"]
    assert metric_layout(draw, **options, steps_value_width=0) == baseline


@pytest.mark.parametrize("display_format", ("seconds", "clock"))
@pytest.mark.parametrize("width", (0, 1, 18, 48, 120))
def test_elapsed_time_uses_static_bounds_even_when_its_format_compacts(display_format, width):
    draw, seen = MetricDraw(), {}
    baseline = None
    for seconds in (0, 59.999, 60, 999.999, 1000, 3599, 3600, 86400, 359999, 360000, 1e9, 1e308):
        fields = metric_layout(
            draw,
            metric_text="Time 0 s",
            sim_time=seconds,
            time_format=display_format,
            time_value_width=width,
        )
        time = fields["metric"]
        anchors = (time.x, time.width, time.value_x, time.value_width)
        if width:
            anchors += (time.unit_x,)
            assert seen.setdefault("anchors", anchors) == anchors
        else:
            assert time.unit_x - time.value_x - time.value_text_width == (2 if time.unit else 0)
            assert time.x + time.width - time.unit_x - time.unit_width == 2
            assert time.value_width == time.value_text_width <= time.format_width
        if baseline is None:
            baseline = fields
        for key in ("delta", "physics", "render"):
            assert fields[key] == baseline[key]
        painter = _StatusNumbers(draw)
        assert painter.width(time.value) <= time.value_width
        if time.value != "…":
            if ":" in time.value:
                assert time.unit == "" and len(time.value) == 8
            else:
                assert time.unit == "s"
                float(time.value)
        if width == 0 and seconds == 1e308:
            assert "e" in time.value
        draw.painted.clear()
        painter.paint(time, 20, (1, 1, 1, 1))
        assert all(
            time.x <= position[0] <= position[0] + draw.text_size(text)[0] <= time.x + time.width
            for text, position in draw.painted
        )


def test_default_seconds_pack_the_phrase_and_keep_their_fractional_anchor():
    draw, seen = MetricDraw(), {}
    for seconds in (0, 1.125, 9.125, 10.125, 99.125, 100.125, 999.125):
        time = metric_layout(draw, metric_text="Time 0 s", sim_time=seconds)["metric"]
        painter = _StatusNumbers(draw)
        assert time.format_width == painter.width("999.999")
        assert time.value_width == painter.width(f"{seconds:.3f}") <= time.format_width
        assert time.format_width < painter.width("999d 99:99:99")
        assert time.value == f"{seconds:.3f}"
        assert time.decimals == 3 and time.unit == "s"
        draw.painted.clear()
        painter.paint(time, 20, (1, 1, 1, 1))
        decimal = next(position[0] for text, position in draw.painted if text == ".")
        assert seen.setdefault("decimal", decimal) == decimal
        assert seen.setdefault("unit", time.unit_x) == time.unit_x
        assert time.unit_x - time.value_x - time.value_text_width == 2
        assert time.x + time.width - time.unit_x - time.unit_width == 2


def test_clock_mode_uses_hours_minutes_seconds_and_preserves_other_metric_geometry():
    draw = MetricDraw()
    seconds = metric_layout(draw, metric_text="Time 0 s", sim_time=3661.125)
    clock = metric_layout(draw, metric_text="Time 0 s", sim_time=3661.125, time_format="clock")
    assert clock["metric"].value == "01:01:01"
    assert clock["metric"].value_width == _StatusNumbers(draw).width("99:59:59")
    assert clock["metric"].unit == ""
    for key in ("delta", "physics", "render"):
        assert clock[key] == seconds[key]


@pytest.mark.parametrize("scale", (1, 2))
def test_metric_defaults_keep_divider_insets_when_one_group_changes_color(tmp_path, scale):
    from copy import deepcopy

    from mojive.design.native.geometry import part_scope
    from mojive.design.native.presentation import DesignPresentation
    from mojive.design.runtime import DesignRuntime
    from mojive.design.store import SpecificationStore
    from mojive.ui.imgui_draw import ImguiDraw2D
    from mojive.ui.workspace.status_components import StatusGroups

    context = imgui.create_context()
    io = imgui.get_io()
    io.set_ini_filename(None)
    io.display_size = (1800 * scale, 500 * scale)
    io.delta_time = 1 / 60
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    ui = UI(
        SimpleNamespace(style_scale=scale, font_report=SimpleNamespace(cjk_path=None)),
        portable_fonts=True,
    )
    runtime = DesignRuntime(SpecificationStore(tmp_path / "status.json"))
    ui.design = DesignPresentation(ui, runtime)

    def layout():
        with part_scope("statusbar"):
            groups = StatusGroups(ui)
            draw = ImguiDraw2D()
            return _status_performance_layout(
                groups.text_draw(draw, "status_metrics"),
                1600 * scale,
                scale,
                "OpenGL",
                0.002,
                59.0,
                physics_hz=480,
                show_physics=True,
                text_budget=groups.text_budget,
                metric_styles=groups.metric_styles(draw, "steps"),
                group_gap=groups.gap("status_metrics"),
                **groups.metric_options(),
            )

    try:
        imgui.new_frame()
        imgui.push_font(ui.body, 14 * scale)
        original = layout()
        spec = deepcopy(runtime.spec)
        spec["parts"]["status_physics"] = {"textColor": "#FFFFFF"}
        runtime.update_spec(spec)
        assert layout() == original
        physics = next(field for field in original.fields if field.key == "physics")
        assert (physics.value, physics.unit) == ("480", "Hz")
        for index, field in enumerate(original.fields):
            assert field.content_x - field.x == pytest.approx(2 * scale)
            if field.key != "backend":
                assert field.value_x - field.content_x - field.label_width == pytest.approx(
                    METRIC_LABEL_GAP * scale
                )
                assert field.unit_x - field.value_x - field.value_text_width == pytest.approx(
                    2 * scale
                )
            if index < len(original.dividers):
                assert original.dividers[index] - scale / 2 - field.x - field.width == (
                    pytest.approx(METRIC_GROUP_GAP * scale)
                )
        imgui.pop_font()
        imgui.render()
    finally:
        imgui.destroy_context(context)
