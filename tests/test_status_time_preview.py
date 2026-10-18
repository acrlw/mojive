"""The guide's status time uses production formatting and exact pointer actions."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
from examples.ui_design.native.guide_panels import telemetry
from imgui_bundle import imgui

from mojive.design.native.geometry import Rect
from mojive.design.native.presentation import DesignPresentation
from mojive.design.runtime import DesignRuntime
from mojive.design.store import SpecificationStore
from mojive.ui.theme import THEME
from mojive.ui.viewport_widgets.status_metrics import _StatusNumbers
from mojive.ui.workspace import status_tooltip
from mojive.ui.workspace.widgets import UI


@pytest.fixture(params=(1, 2))
def time_preview(request, tmp_path, monkeypatch):
    context = imgui.create_context()
    io = imgui.get_io()
    io.set_ini_filename(None)
    scale = request.param
    io.display_size, io.delta_time = (1200 * scale, 300 * scale), 1 / 60
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    ui = UI(
        SimpleNamespace(style_scale=scale, font_report=SimpleNamespace(cjk_path=None)),
        portable_fonts=True,
    )
    runtime = DesignRuntime(SpecificationStore(tmp_path / "status.json"))
    ui.design = DesignPresentation(ui, runtime)
    viewer = SimpleNamespace(
        ui=ui,
        s=scale,
        theme=THEME,
        clock=0,
        values={"metric-mode": "time", "status-readout": "time", "playback-time": 65.125},
        tr=lambda text, **values: text.format(**values),
        box=lambda *args: None,
        rect=lambda x, y, w, h: Rect(x * scale, y * scale, w * scale, h * scale),
        hits={},
        fields={},
        tips=[],
        copied=[],
    )

    def record(key):
        lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
        viewer.hits[key] = Rect(lo.x, lo.y, hi.x - lo.x, hi.y - lo.y)

    viewer.record = record
    paint = _StatusNumbers.paint

    def record_field(painter, field, *args):
        viewer.fields[field.key] = field
        return paint(painter, field, *args)

    monkeypatch.setattr(_StatusNumbers, "paint", record_field)
    monkeypatch.setattr(imgui, "set_clipboard_text", viewer.copied.append)
    monkeypatch.setattr(
        status_tooltip, "status_tooltip", lambda _ui, *lines, **kwargs: viewer.tips.append(lines)
    )

    def frame(count=1):
        for _ in range(count):
            viewer.hits.clear()
            viewer.fields.clear()
            viewer.tips.clear()
            imgui.new_frame()
            imgui.set_next_window_pos((0, 0))
            imgui.set_next_window_size(io.display_size)
            imgui.begin("Status preview", flags=imgui.WindowFlags_.no_title_bar)
            telemetry(viewer, 0, 20, 1160)
            imgui.end()
            imgui.render()

    def click(button):
        rect = viewer.hits["guide-metric/metric"]
        io.add_mouse_pos_event(rect.x + rect.width / 2, rect.y + rect.height / 2)
        frame(2)
        io.add_mouse_button_event(button, True)
        frame()
        io.add_mouse_button_event(button, False)
        frame()

    viewer.frame, viewer.click, viewer.io = frame, click, io
    try:
        frame(3)
        yield viewer
    finally:
        imgui.destroy_context(context)


def test_guide_time_format_packs_auto_phrases_with_a_static_format_cap(time_preview):
    v = time_preview
    assert v.fields["metric"].value == "65.125"
    seconds = v.hits["guide-metric/metric"]
    format_width = v.fields["metric"].format_width
    widths = set()
    for value in (0, 65.125, 999.999, 1e9, 1e308):
        v.values["playback-time"] = value
        v.frame()
        field, rect = v.fields["metric"], v.hits["guide-metric/metric"]
        assert field.compact
        assert field.value_width == field.value_text_width <= field.format_width == format_width
        assert rect.right == pytest.approx(seconds.right)
        assert field.value_x == pytest.approx(field.content_x + field.label_width + field.label_gap)
        assert field.unit_x == pytest.approx(field.value_x + field.value_width + field.unit_gap)
        assert field.unit_x + field.unit_width == pytest.approx(rect.right - 2 * v.s)
        assert rect.x <= field.value_x < field.unit_x + field.unit_width <= rect.right
        if value >= 1e9:
            assert "e" in field.value and field.unit == "s"
        widths.add(rect.width)
    assert len(widths) > 1
    spec = deepcopy(v.ui.design.runtime.spec)
    spec["parts"]["status_time"] = {"displayFormat": "clock", "valueWidth": 100}
    v.ui.design.runtime.update_spec(spec)
    v.values["playback-time"] = 3661.125
    v.frame()
    assert v.fields["metric"].value == "01:01:01"
    authored = v.hits["guide-metric/metric"]
    for value in (3661.125, 61.125, 1e9):
        v.values["playback-time"] = value
        v.frame()
        field = v.fields["metric"]
        assert not field.compact
        assert field.value_width == 100 * v.s
        assert field.value_text_width <= field.value_width
        assert field.unit_x + field.unit_width <= authored.right
        assert v.hits["guide-metric/metric"] == authored


def test_guide_metric_hover_copy_and_switch_use_the_full_exact_value(time_preview):
    v = time_preview
    v.values["playback-time"] = 1234567890.1234567
    v.frame()
    assert "e" in v.fields["metric"].value
    v.click(1)
    assert v.copied == ["1234567890.1234567 s"]
    assert not v.tips
    v.frame(30)
    assert v.tips == [
        ("1234567890.1234567 s", "Click to show steps", "Right-click to copy exact value")
    ]
    v.click(0)
    assert v.values["metric-mode"] == v.values["status-readout"] == "steps"
    v.frame()
    assert v.status_layout.metric_exact == "12500"
    v.click(1)
    assert v.copied[-1] == "12500"
