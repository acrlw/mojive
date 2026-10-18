"""Status phrases share body typography and visible divider padding at every UI scale."""

from copy import deepcopy
from dataclasses import replace
from math import floor
from types import SimpleNamespace

import pytest
from imgui_bundle import imgui

from mojive.design.native.geometry import part_scope
from mojive.design.native.presentation import DesignPresentation
from mojive.design.runtime import DesignRuntime
from mojive.design.specification import DEFAULT
from mojive.design.store import SpecificationStore
from mojive.ui.imgui_draw import ink_box
from mojive.ui.viewport_widgets.model import DEFAULT_VIEWPORT_LABELS
from mojive.ui.viewport_widgets.status_metrics import _status_performance_layout
from mojive.ui.workspace.status_components import StatusGroups
from mojive.ui.workspace.typography import TextRole, Typography
from mojive.ui.workspace.widgets import UI


class AtlasText:
    def __init__(self):
        self.painted = []

    def text_size(self, text):
        return tuple(imgui.calc_text_size(text))

    def text_ink_bounds(self, text):
        return ink_box(imgui.get_font(), imgui.get_font_size(), text)

    def text(self, position, color, text, **options):
        self.painted.append((text, imgui.get_font().font_id, imgui.get_font_size(), position))


@pytest.mark.parametrize("scale", (1, 1.2, 1.25, 2.25))
@pytest.mark.parametrize("language", ("en", "zh_CN"))
def test_real_body_font_phrases_have_equal_divider_padding_and_no_rate_jitter(
    tmp_path, scale, language
):
    context = imgui.create_context()
    io = imgui.get_io()
    io.display_size = (2200 * scale, 1000 * scale)
    io.delta_time = 1 / 60
    io.set_ini_filename(None)
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    ui = UI(
        SimpleNamespace(style_scale=scale, font_report=SimpleNamespace(cjk_path=None)),
        portable_fonts=True,
    )
    # Numeric-input authoring must not silently change telemetry's selected face.
    ui.typography = Typography(numeric=TextRole("mono", 22))
    runtime = DesignRuntime(SpecificationStore(tmp_path / "status.json"))
    ui.design = DesignPresentation(ui, runtime)
    labels = DEFAULT_VIEWPORT_LABELS
    if language == "zh_CN":
        labels = replace(labels, physics="物理", render="渲染", time="时间")
    try:
        imgui.new_frame()
        with ui.design.family_scope("chrome"), part_scope("statusbar"):
            groups, raw = StatusGroups(ui), AtlasText()
            draw = groups.text_draw(raw, "status_metrics")
            stable = None
            for fps in (51.1, 58.7, 99.9):
                layout = _status_performance_layout(
                    draw,
                    2100 * scale,
                    scale,
                    "OpenGL",
                    0.002,
                    fps,
                    metric_text=labels.time + " 12.125 s",
                    sim_time=12.125,
                    physics_hz=480,
                    show_physics=True,
                    labels=labels,
                    text_budget=groups.text_budget,
                    group_gap=groups.gap("status_metrics"),
                    **groups.metric_options(),
                )
                geometry = tuple((f.key, f.x, f.width, f.unit_x) for f in layout.fields)
                if stable is None:
                    stable = geometry
                assert geometry == stable
                for index, field in enumerate(layout.fields):
                    raw.painted.clear()
                    groups.number_painter(draw).paint(field, 30 * scale, (1, 1, 1, 1))
                    assert {font for _, font, _, _ in raw.painted} == {ui.body.font_id}
                    assert {size for _, _, size, _ in raw.painted} == {floor(12 * scale + 0.5)}
                    if field.key == "backend":
                        phrase_right = field.value_x + field.value_text_width
                    else:
                        assert field.value_width == field.value_text_width <= field.format_width
                        assert (
                            field.unit_x - field.value_x - field.value_text_width
                            == pytest.approx(2 * scale)
                        )
                        phrase_right = field.unit_x + field.unit_width
                    assert field.x + field.width - phrase_right == pytest.approx(2 * scale)
                    if index:
                        divider = layout.dividers[index - 1]
                        assert field.content_x - divider - scale / 2 == pytest.approx(8 * scale)
                    if index < len(layout.dividers):
                        divider = layout.dividers[index]
                        assert divider - scale / 2 - phrase_right == pytest.approx(8 * scale)
            chinese = draw.text_ink_bounds("时间")
            digits = draw.text_ink_bounds("0123456789")
            assert chinese is not None and digits is not None
            # MiSans digits stay close to its Chinese em; a mixed JetBrains face
            # previously produced only 8.5px digit ink against 13px Chinese at1.2.
            assert (digits[3] - digits[1]) / (chinese[3] - chinese[1]) >= 0.8
        imgui.render()
    finally:
        imgui.destroy_context(context)


def test_explicit_metric_font_and_fixed_numeric_columns_remain_independent(tmp_path):
    context = imgui.create_context()
    io = imgui.get_io()
    io.display_size = (2200, 1000)
    io.delta_time = 1 / 60
    io.set_ini_filename(None)
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    ui = UI(
        SimpleNamespace(style_scale=1.2, font_report=SimpleNamespace(cjk_path=None)),
        portable_fonts=True,
    )
    runtime = DesignRuntime(SpecificationStore(tmp_path / "status-authoring.json"))
    spec = deepcopy(DEFAULT)
    spec["parts"]["status_physics"] = {"fontFamily": "mono", "fontSize": 17, "valueWidth": 48}
    runtime.update_spec(spec)
    ui.design = DesignPresentation(ui, runtime)
    try:
        imgui.new_frame()
        with ui.design.family_scope("chrome"), part_scope("statusbar"):
            groups, raw = StatusGroups(ui), AtlasText()
            styles = groups.metric_styles(raw, "time")
            draw = groups.text_draw(raw, "status_metrics")
            previous = None
            for hz in (0, 480, 12000):
                layout = _status_performance_layout(
                    draw,
                    2100,
                    1.2,
                    "OpenGL",
                    0.002,
                    58.7,
                    physics_hz=hz,
                    show_physics=True,
                    metric_styles=styles,
                )
                physics = next(f for f in layout.fields if f.key == "physics")
                bounds = (
                    physics.x,
                    physics.width,
                    physics.value_x,
                    physics.value_width,
                    physics.unit_x,
                )
                if previous is None:
                    previous = bounds
                assert bounds == previous
                assert physics.value_width == pytest.approx(48 * 1.2)
                raw.painted.clear()
                groups.number_painter(styles["physics"].draw).paint(physics, 30, (1, 1, 1, 1))
                assert {font for _, font, _, _ in raw.painted} == {ui.mono.font_id}
                assert {size for _, _, size, _ in raw.painted} == {20}
        imgui.render()
    finally:
        imgui.destroy_context(context)
