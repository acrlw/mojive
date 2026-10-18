"""Live sensor cells retain component identity and geometry as values change."""

import ast
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from imgui_bundle import imgui

from mojive.adapters.base import SensorInfo
from mojive.design.groups import group_scope
from mojive.design.native.geometry import Rect
from mojive.design.native.presentation import DesignPresentation
from mojive.design.runtime import DesignRuntime
from mojive.design.store import SpecificationStore
from mojive.ui.imgui_draw import ImguiDraw2D
from mojive.ui.panels.sensors import SensorsPanel, sensor_component_label
from mojive.ui.workspace.panel_host import draw_content
from mojive.ui.workspace.sensors_panel import WorkspaceSensors
from mojive.ui.workspace.widgets import UI


@pytest.mark.parametrize("kind", ("mjSENS_FRAMEPOS", "accelerometer", "force", "gyro"))
def test_cartesian_sensor_axes_are_derived_from_metadata(kind):
    assert [sensor_component_label(kind, 3, i) for i in range(3)] == list("XYZ")


def test_arbitrary_channels_keep_zero_based_indices_instead_of_invented_axes():
    assert [sensor_component_label("user", 3, i) for i in range(3)] == ["0", "1", "2"]
    assert sensor_component_label("user", 14, 13) == "13"


@pytest.fixture(params=(1.0, 2.0))
def readouts(request, tmp_path, monkeypatch):
    context = imgui.create_context()
    io = imgui.get_io()
    io.set_ini_filename(None)
    scale = request.param
    io.display_size, io.delta_time = (900 * scale, 900 * scale), 1 / 60
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    ui = UI(
        SimpleNamespace(style_scale=scale, font_report=SimpleNamespace(cjk_path=None)),
        portable_fonts=True,
    )
    ui.design = DesignPresentation(ui, DesignRuntime(SpecificationStore(tmp_path / "spec.json")))
    workspace = SimpleNamespace(ui=ui, s=scale, widths={})
    panel = WorkspaceSensors(workspace)
    workspace.app = SimpleNamespace(
        panels=SimpleNamespace(get=lambda _: panel), apply_document_edits=lambda: None
    )
    workspace.panel_header = lambda _name, _x, y, _w: y
    for name in (
        "scene_panel",
        "inspector",
        "control_panel",
        "assets_panel",
        "pose_panel",
        "camera_panel",
        "layers_panel",
        "timeline",
    ):
        setattr(workspace, name, None)
    ui.design.runtime.observe_components = True
    opened = []
    plot = SimpleNamespace(focus_sensor=lambda index: opened.append(index))
    ctx = SimpleNamespace(
        tr=str, panels=SimpleNamespace(load=lambda name: plot, open_panel=opened.append)
    )
    painted = []
    original = ui.component_text

    def record(part, rect, text, *args, **options):
        painted.append((rect if isinstance(rect, Rect) else Rect(*rect), str(text), options))
        return original(part, rect, text, *args, **options)

    monkeypatch.setattr(ui, "component_text", record)

    def frame(
        values, width=280, *, kind="user", actions=False, whole=False, missing=False, hosted=False
    ):
        ui.hits.clear()
        painted.clear()
        ui.design.runtime.begin_frame()
        ui.design.begin_frame()
        imgui.new_frame()
        imgui.set_next_window_pos((0, 0))
        imgui.set_next_window_size((width * scale, 760 * scale))
        imgui.push_style_var(imgui.StyleVar_.window_min_size, (1, 1))
        imgui.begin("Sensor", flags=imgui.WindowFlags_.no_title_bar)
        with group_scope("Sensors"), ui.component_scope("sensor"):
            sensor = SensorInfo(7, "joint_position", kind, 0, len(values))
            if whole:
                ctx.session = SimpleNamespace(
                    sensor_infos=[sensor],
                    frame=SimpleNamespace(sensors=None if missing else values),
                )
                if hosted:
                    p, size = imgui.get_cursor_screen_pos(), imgui.get_content_region_avail()
                    draw_content(workspace, "Sensors", ctx, Rect(p.x, p.y, size.x, size.y))
                else:
                    panel.draw(ctx)
            else:
                if actions:
                    panel._actions(ctx, values)
                panel._values(ctx, sensor, values)
        imgui.end()
        imgui.pop_style_var()
        imgui.render()
        return tuple(painted), dict(ui.hits)

    try:
        yield ui, panel, frame, io, opened
    finally:
        imgui.destroy_context(context)


@pytest.mark.parametrize("dimension,kind", ((1, "scalar"), (3, "framepos"), (14, "user")))
@pytest.mark.parametrize("width", (150, 280, 600))
def test_readout_positions_and_component_labels_do_not_follow_numeric_magnitude(
    readouts, dimension, kind, width
):
    ui, _, frame, _, _ = readouts
    values = np.resize(np.array([1.123456789012345, -2e20, 3e-18, np.inf]), dimension)
    first_text, first_hits = frame(np.zeros(dimension), width, kind=kind)
    changed_text, changed_hits = frame(values, width, kind=kind)
    assert first_hits == changed_hits
    assert [p[0] for p in first_text] == [p[0] for p in changed_text]
    assert len(changed_hits) == dimension
    for rect, _, _ in changed_text:
        assert 0 <= rect.x <= rect.right <= width * ui.s
        assert rect.width >= 0
    if dimension > 1:
        labels = [
            text for _, text, options in changed_text if options["font_field"] == "detailSize"
        ]
        assert labels == [sensor_component_label(kind, dimension, i) for i in range(dimension)]


def test_large_sensor_draws_only_visible_components(readouts):
    _, _, frame, _, _ = readouts
    _, hits = frame(np.arange(10_000))
    assert 0 < len(hits) < 100
    assert "sensor-value-7-0" in hits
    assert "sensor-value-7-9999" not in hits


@pytest.mark.parametrize("width", (220, 280, 384, 860))
def test_complete_scalar_panel_keeps_metadata_actions_and_value_together(readouts, width):
    ui, _, frame, _, _ = readouts
    painted, hits = frame(np.array([1.2345]), width, kind="jointpos", whole=True)
    selector, metadata, heading, value = (
        Rect(*hits[key])
        for key in ("##sensor", "sensor-metadata", "sensor-readout-heading", "sensor-value-7-0")
    )
    copy, plot = (Rect(*hits[key]) for key in ("sensor-copy", "sensor-plot"))
    assert metadata.y - selector.bottom == pytest.approx(8 * ui.s)
    assert heading.y - metadata.bottom == pytest.approx(12 * ui.s)
    assert value.y - plot.bottom == pytest.approx(8 * ui.s)
    assert copy.y == plot.y == heading.y
    assert copy.x >= heading.right and plot.right == pytest.approx(value.right)
    assert value.height == pytest.approx(64 * ui.s)
    assert selector.width <= 640 * ui.s
    assert value.width == pytest.approx(selector.width)
    assert [text for _, text, _ in painted].count("Readings") == 1
    assert "jointpos · dimension 1" in [text for _, text, _ in painted]
    value_options = next(options for _, _, options in painted if options.get("mono"))
    assert value_options["align"] == "center"


@pytest.mark.parametrize("width,columns", ((220, 1), (280, 2), (384, 3)))
def test_index_grid_reflows_without_omitting_components(readouts, width, columns):
    ui, _, frame, _, _ = readouts
    _, hits = frame(np.arange(14), width, whole=True)
    frames = [Rect(*hits[f"sensor-value-7-{i}"]) for i in range(14)]
    assert len({rect.x for rect in frames}) == columns
    assert all(rect.right <= width * ui.s for rect in frames)
    assert frames[columns].y - frames[0].bottom == pytest.approx(8 * ui.s)


@pytest.mark.parametrize("width,columns", ((150, 1), (280, 3), (384, 3)))
def test_cartesian_readouts_keep_all_three_axes_in_one_grid_row_when_they_fit(
    readouts, width, columns
):
    _, _, frame, _, _ = readouts
    _, hits = frame(np.array([1.0, 2.0, 3.0]), width, kind="framepos", whole=True)
    frames = [Rect(*hits[f"sensor-value-7-{i}"]) for i in range(3)]
    assert len({rect.x for rect in frames}) == columns


def test_readout_component_overrides_adjust_the_real_values_without_changing_actions(readouts):
    ui, _, frame, _, _ = readouts
    values = np.arange(3, dtype=float)
    _, before = frame(values, 384, kind="framepos", whole=True)
    spec = deepcopy(ui.design.runtime.spec)
    spec["parts"]["sensors_sensor"] = {
        "height": 96,
        "valueSize": 24,
        "detailSize": 16,
        "columnGap": 12,
        "paddingLeft": 16,
    }
    ui.design.runtime.update_spec(spec)
    painted, after = frame(values, 384, kind="framepos", whole=True)
    assert after["sensor-value-7-0"][3] == pytest.approx(96 * ui.s)
    assert before["sensor-copy"][2:] == after["sensor-copy"][2:]
    frame_rect = Rect(*after["sensor-value-7-0"])
    axis = next(rect for rect, text, _ in painted if text == "X")
    assert axis.x - frame_rect.x == pytest.approx(16 * ui.s)


def test_owned_actions_measure_paint_and_hit_the_same_independent_button_geometry(readouts):
    ui, _, frame, _, _ = readouts
    values = np.ones(1)
    _, before = frame(values, 384, whole=True)
    spec = deepcopy(ui.design.runtime.spec)
    spec["parts"].update(
        sensors_button_row={
            "columnGap": 16,
            "paddingLeft": 10,
            "paddingRight": 12,
            "paddingTop": 6,
            "paddingBottom": 4,
        },
        sensors_button_row_icon_button={"width": 42, "widthMode": "fixed", "height": 44},
    )
    ui.design.runtime.update_spec(spec)
    _, after = frame(values, 384, whole=True)
    copy, plot = (Rect(*after[key]) for key in ("sensor-copy", "sensor-plot"))
    assert copy.width == plot.width == pytest.approx(42 * ui.s)
    assert copy.height == plot.height == pytest.approx(44 * ui.s)
    assert plot.x - copy.right == pytest.approx(16 * ui.s)
    assert after["##sensor"] == before["##sensor"]
    assert after["sensor-metadata"] == before["sensor-metadata"]
    row = next(
        item for item in ui.design.runtime.observed if item["key"] == "sensor-readout-actions"
    )
    assert "sensors_button_row" in row["parts"]
    row_rect = Rect(*row["hit"])
    assert copy.x - row_rect.x == pytest.approx(10 * ui.s)
    assert row_rect.right - plot.right == pytest.approx(12 * ui.s)
    assert copy.y - row_rect.y == pytest.approx(6 * ui.s)
    assert after["sensor-value-7-0"][1] - plot.bottom == pytest.approx(12 * ui.s)
    for key in ("sensor-copy", "sensor-plot"):
        button = next(item for item in ui.design.primitives.observed if item["key"] == key)
        assert "sensors_button_row_icon_button" in button["parts"]
        assert button["hit"] == after[key]


def test_panel_body_insets_move_all_regions_inside_the_actual_host_boundary(readouts):
    ui, _, frame, _, _ = readouts
    values = np.ones(1)
    _, before = frame(values, 384, whole=True, hosted=True)
    spec = deepcopy(ui.design.runtime.spec)
    spec["parts"]["sensors_panel_body"] = {"paddingLeft": 32, "paddingRight": 24, "paddingTop": 20}
    ui.design.runtime.update_spec(spec)
    _, after = frame(values, 384, whole=True, hosted=True)
    body = next(item for item in ui.design.runtime.observed if item["key"] == "panel-body-Sensors")
    content = Rect(*body["content"])
    for key in ("##sensor", "sensor-metadata", "sensor-readout-heading", "sensor-value-7-0"):
        rect = Rect(*after[key])
        assert rect.x == pytest.approx(content.x)
        assert rect.y >= content.y
        assert rect.right <= content.right + 0.01
        assert after[key] != before[key]
    plot = Rect(*after["sensor-plot"])
    assert plot.right == pytest.approx(content.right)
    assert after["##sensor"][1] == pytest.approx(content.y)
    assert after["sensor-value-7-0"][2] == pytest.approx(content.width)


def test_cell_padding_is_applied_once_without_moving_other_regions(readouts):
    ui, _, frame, _, _ = readouts
    values = np.arange(3.0)
    _, before = frame(values, 384, kind="framepos", whole=True, hosted=True)
    spec = deepcopy(ui.design.runtime.spec)
    spec["parts"]["sensors_sensor"] = {"paddingLeft": 20}
    ui.design.runtime.update_spec(spec)
    painted, after = frame(values, 384, kind="framepos", whole=True, hosted=True)
    assert after == before
    axis = next(rect for rect, text, _ in painted if text == "X")
    assert axis.x - after["sensor-value-7-0"][0] == pytest.approx(20 * ui.s)


def test_grid_row_gap_column_gap_and_margins_are_independent_from_header_actions(readouts):
    ui, _, frame, _, _ = readouts
    values = np.arange(14.0)
    _, before = frame(values, 600, whole=True)
    spec = deepcopy(ui.design.runtime.spec)
    spec["parts"]["sensors_sensor"] = {
        "rowGap": 20,
        "columnGap": 24,
        "marginLeft": 10,
        "marginRight": 12,
        "marginTop": 6,
        "marginBottom": 8,
    }
    ui.design.runtime.update_spec(spec)
    _, after = frame(values, 600, whole=True)
    first, second, next_row = (Rect(*after[f"sensor-value-7-{i}"]) for i in (0, 1, 3))
    assert first.x - before["sensor-value-7-0"][0] == pytest.approx(10 * ui.s)
    assert first.y - before["sensor-value-7-0"][1] == pytest.approx(6 * ui.s)
    assert second.x - first.right == pytest.approx(24 * ui.s)
    assert next_row.y - first.bottom == pytest.approx(20 * ui.s)
    for key in ("##sensor", "sensor-metadata", "sensor-copy", "sensor-plot"):
        assert after[key] == before[key]


def test_missing_frame_retains_selection_metadata_and_disabled_copy(readouts, monkeypatch):
    _, _, frame, io, _ = readouts
    copied = []
    monkeypatch.setattr(imgui, "set_clipboard_text", copied.append)
    _, hits = frame(np.ones(3), whole=True, missing=True)
    assert "##sensor" in hits and "sensor-metadata" in hits
    assert not any(key.startswith("sensor-value-") for key in hits)
    x, y, width, height = hits["sensor-copy"]
    io.add_mouse_pos_event(x + width / 2, y + height / 2)
    frame(np.ones(3), whole=True, missing=True)
    io.add_mouse_button_event(0, True)
    frame(np.ones(3), whole=True, missing=True)
    io.add_mouse_button_event(0, False)
    frame(np.ones(3), whole=True, missing=True)
    assert not copied


@pytest.mark.parametrize("dimension", (1, 14))
def test_native_glyph_submission_uses_the_measured_value_and_index_sizes(
    readouts, monkeypatch, dimension
):
    ui, _, frame, _, _ = readouts
    submitted = []
    original = ImguiDraw2D.text

    def record(draw, pos, color, text, **options):
        submitted.append((text, imgui.get_font_size(), imgui.get_font().font_id))
        return original(draw, pos, color, text, **options)

    monkeypatch.setattr(ImguiDraw2D, "text", record)
    frame(np.zeros(dimension), 384, whole=True)
    values = [
        text
        for text, size, font in submitted
        if size == pytest.approx(18 * ui.s) and font == ui.mono.font_id
    ]
    assert values == ["0"] * dimension
    if dimension > 1:
        indices = [
            text
            for text, size, font in submitted
            if size == pytest.approx(12 * ui.s) and font == ui.mono.font_id
        ]
        assert indices == [str(index) for index in range(dimension)]


def test_copy_retains_full_precision_and_open_plot_uses_the_selected_sensor(readouts, monkeypatch):
    ui, panel, frame, io, opened = readouts
    values = np.linspace(0.123456789012345, 14.987654321098765, 14)
    copied = []
    monkeypatch.setattr(imgui, "set_clipboard_text", copied.append)
    spec = deepcopy(ui.design.runtime.spec)
    spec["parts"].update(
        sensors_button_row={"columnGap": 16, "paddingTop": 6},
        sensors_button_row_icon_button={"height": 44},
    )
    ui.design.runtime.update_spec(spec)
    panel.sensor_index = 4
    frame(values, actions=True)

    def click(key):
        _, hits = frame(values, actions=True)
        x, y, w, h = hits[key]
        io.add_mouse_pos_event(x + w / 2, y + h / 2)
        frame(values, actions=True)
        io.add_mouse_button_event(0, True)
        frame(values, actions=True)
        io.add_mouse_button_event(0, False)
        frame(values, actions=True)

    click("sensor-copy")
    assert ast.literal_eval(copied[-1]) == values.tolist()
    click("sensor-plot")
    assert opened == [4, "Plot"]


def test_returning_from_scrolled_narrow_sensor_remeasures_plot_target_after_resize(readouts):
    from mojive.ui.panels.plot import PlotPanel

    ui, panel, _frame, io, _opened = readouts
    scale, workspace = ui.s, panel.workspace
    plot, opened = PlotPanel(), []
    ctx = SimpleNamespace(
        tr=str,
        panels=SimpleNamespace(load=lambda _name: plot, open_panel=opened.append),
        session=SimpleNamespace(
            sensor_infos=(
                SensorInfo(7, "components", "user", 0, 14),
                SensorInfo(8, "selected_xyz", "framepos", 14, 3),
            ),
            frame=SimpleNamespace(sensors=np.arange(17.0)),
        ),
    )

    def frame(width=384, *, sensors=True):
        ui.hits.clear()
        ui.design.runtime.begin_frame()
        ui.design.begin_frame()
        imgui.new_frame()
        imgui.set_next_window_pos((0, 0))
        imgui.set_next_window_size((width * scale, 760 * scale))
        imgui.begin("Sensor return" if sensors else "Curves", flags=imgui.WindowFlags_.no_title_bar)
        if sensors:
            with group_scope("Sensors"):
                position, size = imgui.get_cursor_screen_pos(), imgui.get_content_region_avail()
                draw_content(
                    workspace, "Sensors", ctx, Rect(position.x, position.y, size.x, size.y)
                )
        imgui.end()
        imgui.render()

    for _ in range(3):
        frame(220)
    child = next(
        window
        for window in imgui.get_current_context().windows
        if window.active and "Sensor return/Panel content" in window.name
    )
    assert child.scrollbar_y
    imgui.internal.set_scroll_y(child, child.scroll_max.y)
    for _ in range(3):
        frame(220)
    imgui.internal.set_scroll_y(child, 0)
    for _ in range(3):
        frame(280, sensors=False)

    panel.sensor_index = 1
    frame()
    stale = ui.hits["sensor-plot"]
    assert child.scrollbar_y
    io.add_mouse_pos_event(stale[0] + stale[2] / 2, stale[1] + stale[3] / 2)
    frame()
    current = ui.hits["sensor-plot"]
    assert not child.scrollbar_y
    assert current[0] - stale[0] == pytest.approx(imgui.get_style().scrollbar_size)
    assert io.mouse_pos.x < current[0]
    io.add_mouse_button_event(0, True)
    frame()
    io.add_mouse_button_event(0, False)
    frame()
    assert opened == [] and plot.source == "joint"

    for _ in range(4):
        frame()
    stable = ui.hits["sensor-plot"]
    io.add_mouse_pos_event(stable[0] + stable[2] / 2, stable[1] + stable[3] / 2)
    frame()
    assert ui.hits["sensor-plot"] == stable
    io.add_mouse_button_event(0, True)
    frame()
    io.add_mouse_button_event(0, False)
    frame()
    assert opened == ["Plot"]
    assert plot.source == "sensor" and plot.sensor_index == panel.sensor_index == 1


def test_incomplete_sensor_frame_is_reported_instead_of_mislabeling_a_short_slice(monkeypatch):
    panel = SensorsPanel()
    recorded = []
    panel.presentation = SimpleNamespace(
        combo=lambda *args: (False, 0),
    )
    monkeypatch.setattr(imgui, "set_next_item_width", lambda *args: None)
    monkeypatch.setattr(imgui, "text_disabled", recorded.append)
    monkeypatch.setattr(imgui, "text_wrapped", recorded.append)
    monkeypatch.setattr(panel, "_actions", lambda ctx, value: recorded.append(value))
    monkeypatch.setattr(panel, "_values", lambda *args: pytest.fail("Incomplete values were drawn"))
    ctx = SimpleNamespace(
        tr=str,
        session=SimpleNamespace(
            sensor_infos=[SensorInfo(0, "incomplete", "user", 0, 14)],
            frame=SimpleNamespace(sensors=np.arange(6)),
        ),
    )
    panel.draw(ctx)
    assert "user · dimension 14" in recorded
    assert "not produced this frame" in recorded
    assert None in recorded


@pytest.mark.parametrize("readouts", (1, 1.1, 1.2, 1.25, 2.25), indirect=True)
@pytest.mark.parametrize("width", (220, 384))
def test_hosted_sensor_tail_uses_its_native_child_viewport_width(readouts, width):
    from mojive.ui.workspace.menu_geometry import native_epsilon

    _ui, _panel, frame, _io, _opened = readouts
    values = np.arange(14, dtype=float)
    for _ in range(6):
        _painted, hits = frame(values, width, whole=True, hosted=True)
    child = next(
        window for window in imgui.get_current_context().windows if "/Panel content" in window.name
    )
    first_width = hits["sensor-value-7-0"][2]
    for _ in range(6):
        imgui.internal.set_scroll_y(child, child.scroll_max.y)
        _painted, hits = frame(values, width, whole=True, hosted=True)
    tail = hits["sensor-value-7-13"]
    epsilon = native_epsilon(*tail)
    assert tail[2] == pytest.approx(first_width, abs=epsilon)
    assert tail[0] >= child.inner_clip_rect.min.x - epsilon
    assert tail[0] + tail[2] <= child.inner_clip_rect.max.x + epsilon
    assert tail[1] >= child.inner_clip_rect.min.y - epsilon
    assert tail[1] + tail[3] <= child.inner_clip_rect.max.y + epsilon
    for key, hit in hits.items():
        if key.startswith("sensor-value-"):
            assert hit[0] >= child.inner_clip_rect.min.x - epsilon
            assert hit[0] + hit[2] <= child.inner_clip_rect.max.x + epsilon
