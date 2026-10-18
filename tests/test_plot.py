"""Telemetry must draw only valid samples from the current scene."""

from types import SimpleNamespace

import numpy as np
import pytest
from imgui_bundle import imgui

from mojive.adapters.base import JointInfo, SceneFrame, SensorInfo
from mojive.design.groups import group_scope
from mojive.design.native.presentation import DesignPresentation
from mojive.design.runtime import DesignRuntime
from mojive.design.store import SpecificationStore
from mojive.ui.localization import Language, Localizer
from mojive.ui.panels import plot
from mojive.ui.panels.plot import PlotPanel, Ring, joint_channels, sample_window
from mojive.ui.theme import THEME
from mojive.ui.workspace.property_style import NativePropertyPresentation, property_controls
from mojive.ui.workspace.widgets import UI


def test_partial_and_wrapped_plot_buffers_exclude_unwritten_samples(monkeypatch):
    observed = []
    monkeypatch.setattr(
        plot.imgui,
        "plot_lines",
        lambda _, values, **kw: observed.append(np.roll(values, -kw["values_offset"]).copy()),
    )
    panel = PlotPanel()
    ring = Ring(4)
    for value in (5, 6, 7, 8, 9):
        ring.push(value)
        panel._curve("qpos", ring, 90, "joint")
    np.testing.assert_array_equal(observed[0], [5])
    np.testing.assert_array_equal(observed[2], [5, 6, 7])
    np.testing.assert_array_equal(observed[-1], [6, 7, 8, 9])
    assert np.shares_memory(ring.values, ring.data)
    ring.clear()
    panel._curve("qpos", ring, 90, "joint")
    assert observed[-1].size == 0


@pytest.mark.parametrize("source", ("joint", "sensor"))
def test_plot_clears_samples_when_structure_replaces_the_tracked_channel(monkeypatch, source):
    panel = PlotPanel()
    panel.source = source
    monkeypatch.setattr(panel, "_toolbar", lambda ctx: None)
    monkeypatch.setattr(panel, "_draw_joint", lambda ctx: None)
    monkeypatch.setattr(panel, "_draw_sensor", lambda ctx: None)
    ctx = SimpleNamespace(session=SimpleNamespace(structure_generation=1), tr=str, theme=None)
    panel.draw(ctx)
    rings = (panel._angle, panel._velocity, panel._contact, panel._sensor)
    for ring in rings:
        ring.push(42)
    panel.draw(ctx)
    assert all(ring.filled == 1 for ring in rings)
    ctx.session.structure_generation += 1
    panel.draw(ctx)
    assert all(ring.filled == 0 for ring in rings)


def test_missing_joint_addresses_do_not_sample_the_last_state_element():
    panel = PlotPanel()
    frame = SimpleNamespace(qpos=np.array([42]), qvel=np.array([84]))
    joint = SimpleNamespace(qpos_adr=-1, qvel_adr=-1)
    panel._sample(SimpleNamespace(session=SimpleNamespace(frame=frame, joints=[joint])))
    assert panel._angle.filled == panel._velocity.filled == 0


def test_repeated_and_paused_frames_replace_current_sample_without_extending_time():
    panel = PlotPanel()
    frame = SceneFrame(time=0.25, step=10, paused=True, qpos=np.array([1.0]), qvel=np.array([2.0]))
    ctx = SimpleNamespace(
        session=SimpleNamespace(frame=frame, joints=[SimpleNamespace(qpos_adr=0, qvel_adr=0)])
    )
    for _ in range(10):
        panel._sample(ctx)
    assert panel._angle.filled == panel._velocity.filled == 1
    assert panel._angle.duration == 0
    frame.qpos[0] = 3.0
    panel._sample(ctx)
    assert panel._angle.values.tolist() == [3.0]
    frame.time, frame.step = 0.30, 11
    panel._sample(ctx)
    assert panel._angle.values.tolist() == [3.0, 3.0]
    assert panel._angle.duration == pytest.approx(0.05)


def test_sample_window_uses_real_irregular_timestamps_and_clears_on_replay_rewind():
    ring = Ring(3)
    for step, time in enumerate((0.0, 0.1, 0.4, 0.45)):
        ring.push(step, time=time, step=step)
    assert ring.duration == pytest.approx(0.35)
    np.testing.assert_array_equal(np.roll(ring.values, -ring.offset), [1, 2, 3])
    ring.push(9, time=0.2, step=2)
    assert ring.values.tolist() == [9]
    assert ring.duration == 0
    ring.push(10, time=0.2, step=3)
    assert ring.values.tolist() == [9, 10]
    assert ring.duration == 0


def test_plot_scale_does_not_exaggerate_tiny_changes_or_shrink_when_old_extrema_expire():
    ring = Ring(3)
    for value in (10.0, 10.000002):
        ring.push(value)
    low, high = ring.span()
    assert high - low >= 0.2
    ring.push(20)
    wide = ring.span()
    for _ in range(4):
        ring.push(10)
    assert ring.span() == wide
    ring.clear()
    ring.push(0)
    assert ring.span()[1] - ring.span()[0] >= 0.001


@pytest.mark.parametrize("value", (float("nan"), float("inf"), -float("inf")))
def test_nonfinite_samples_do_not_destroy_the_existing_plot_scale(value):
    ring = Ring()
    ring.push(1, time=1, step=1)
    bounds = ring.span()
    ring.push(value, time=2, step=2)
    assert ring.values.tolist() == [1]
    assert ring.span() == bounds


def test_free_and_ball_components_use_state_semantics_instead_of_calling_all_qpos_angles():
    assert joint_channels("free")[:3] == tuple(("Position", axis, "m") for axis in "XYZ")
    assert joint_channels("free")[3:] == joint_channels("ball")
    assert joint_channels("ball")[0] == ("Quaternion", "W", "")
    assert joint_channels("free", velocity=True)[3:] == joint_channels("ball", velocity=True)
    assert joint_channels("hinge")[0] == ("Angle", "", "rad")
    assert joint_channels("slide")[0] == ("Displacement", "", "m")
    panel = PlotPanel()
    panel.position_component, panel.velocity_component = 5, 4
    frame = SceneFrame(qpos=np.arange(10), qvel=np.arange(10) + 20)
    joint = JointInfo(0, "free", "free", False, (0, 0), 2, 1, 6)
    ctx = SimpleNamespace(session=SimpleNamespace(frame=frame, joints=[joint]))
    panel._sample(ctx)
    assert panel._current == {"qpos": 7, "qvel": 25}


def test_unavailable_contacts_are_not_sampled_as_zero_force():
    panel = PlotPanel()
    panel.show_angle = panel.show_velocity = False
    panel.show_contact = True
    frame = SceneFrame()
    ctx = SimpleNamespace(session=SimpleNamespace(frame=frame, joints=[SimpleNamespace()]))
    panel._sample(ctx)
    assert panel._current["contact"] is None
    assert panel._contact.filled == 0
    frame.contacts = np.zeros((0, 7))
    panel._sample(ctx)
    assert panel._current["contact"] == 0
    assert panel._contact.values.tolist() == [0]


@pytest.fixture(params=(1.0, 2.0))
def plotted_panel(request, tmp_path, monkeypatch):
    context = imgui.create_context()
    io = imgui.get_io()
    io.set_ini_filename(None)
    io.display_size, io.delta_time = (900 * request.param, 900 * request.param), 1 / 60
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    ui = UI(
        SimpleNamespace(style_scale=request.param, font_report=SimpleNamespace(cjk_path=None)),
        portable_fonts=True,
    )
    ui.design = DesignPresentation(ui, DesignRuntime(SpecificationStore(tmp_path / "spec.json")))
    panel = PlotPanel()
    panel.presentation = NativePropertyPresentation(ui, prefix="plot")
    session = SimpleNamespace(
        frame=SceneFrame(
            time=0,
            step=0,
            qpos=np.array([0.25, 0, 0, 1, 0, 0, 0]),
            qvel=np.zeros(6),
            sensors=np.arange(14),
        ),
        joints=[JointInfo(0, "floating_base", "free", False, (0, 0), 0, 0, 6)],
        sensor_infos=[SensorInfo(0, "sensor_values", "user", 0, 14)],
        paused=True,
        structure_generation=0,
    )
    ctx = SimpleNamespace(session=session, style_scale=ui.s, theme=THEME)
    text, plots = [], []
    panel.ui = ui
    graph = panel._graph
    combo = panel._combo
    draw_text = plot.ImguiDraw2D.text
    wrapped, disabled, unformatted, lines = (
        imgui.text_wrapped,
        imgui.text_disabled,
        imgui.text_unformatted,
        imgui.plot_lines,
    )

    def record_text(original, value):
        text.append(value)
        original(value)

    def record_lines(key, values, **kwargs):
        lines(key, values, **kwargs)
        minimum, maximum = imgui.get_item_rect_min(), imgui.get_item_rect_max()
        plots.append((key, minimum.x, maximum.x, minimum.y, maximum.y, kwargs))

    monkeypatch.setattr(imgui, "text_wrapped", lambda value: record_text(wrapped, value))
    monkeypatch.setattr(imgui, "text_disabled", lambda value: record_text(disabled, value))
    monkeypatch.setattr(imgui, "text_unformatted", lambda value: record_text(unformatted, value))
    monkeypatch.setattr(imgui, "plot_lines", record_lines)

    def record_graph(key, ring, ctx, x, y, width, height, window, unit=""):
        graph(key, ring, ctx, x, y, width, height, window, unit)
        plots.append((key, x, x + width, y, y + height, {"window": window, "unit": unit}))

    def record_combo(key, index, labels):
        changed, selected = combo(key, index, labels)
        if labels:
            text.append(labels[selected])
        return changed, selected

    def record_draw_text(draw, pos, color, value, **kwargs):
        text.append(value)
        return draw_text(draw, pos, color, value, **kwargs)

    monkeypatch.setattr(panel, "_graph", record_graph)
    monkeypatch.setattr(panel, "_combo", record_combo)
    monkeypatch.setattr(plot.ImguiDraw2D, "text", record_draw_text)

    def frame(width=300, language="en", height=850):
        text.clear()
        plots.clear()
        ui.hits.clear()
        ui.design.begin_frame()
        ui.design.runtime.observed.clear()
        ui.translate = ctx.tr = Localizer(Language(language)).text
        imgui.new_frame()
        imgui.push_style_var(imgui.StyleVar_.window_min_size, (1, 1))
        imgui.set_next_window_pos((0, 0))
        imgui.set_next_window_size((width * ui.s, height * ui.s))
        imgui.begin("Plot", flags=imgui.WindowFlags_.no_title_bar)
        try:
            with group_scope("Plot"), property_controls(ui):
                panel.draw(ctx)
        finally:
            imgui.end()
            imgui.pop_style_var()
            imgui.render()
        return tuple(text), tuple(plots)

    def click(key, width=700, language="en", height=850):
        x, y, w, h = ui.hits[key]
        io.add_mouse_pos_event(x + w / 2, y + h / 2)
        frame(width, language, height)
        io.add_mouse_button_event(0, True)
        frame(width, language, height)
        io.add_mouse_button_event(0, False)
        frame(width, language, height)
        return frame(width, language, height)

    frame.click = click

    try:
        yield panel, session, ctx, frame
    finally:
        imgui.destroy_context(context)


@pytest.mark.parametrize("width", (200, 600))
@pytest.mark.parametrize("language", ("en", "zh_CN"))
def test_production_plot_exposes_current_state_units_and_real_window(
    plotted_panel, width, language
):
    panel, session, ctx, frame = plotted_panel
    frame(width, language)
    text, plots = frame(width, language)
    assert any(ctx.tr("Paused") in value for value in text)
    assert f"{ctx.tr('Position')} X" in text
    assert "0.25" in text
    assert "m" in text
    assert f"{ctx.tr('Linear velocity')} X" in text
    assert "0" in text
    assert "m/s" in text
    assert any(ctx.tr("sample") in value and "0 s" in value for value in text)
    assert panel._angle.filled == 1
    assert all(0 <= left < right <= width * ctx.style_scale for _, left, right, *_ in plots), plots
    session.paused = False
    session.frame.time, session.frame.step = 0.125, 5
    text, _ = frame(width, language)
    assert any(ctx.tr("Live") in value for value in text)
    assert any("0.125 s" in value for value in text)
    session.frame.qpos = session.frame.qvel = None
    text, _ = frame(width, language)
    assert any(ctx.tr("No data") in value for value in text)
    assert any(ctx.tr("Unavailable") in value for value in text)
    assert panel._angle.filled == 2


def test_sensor_plot_uses_requested_component_without_inventing_a_unit(plotted_panel):
    panel, session, _ctx, frame = plotted_panel
    panel.focus_sensor(0, 13)
    text, _ = frame()
    assert "sensor_values" in text
    assert "Component 13" in text
    assert "13" in text
    assert panel._sensor.values.tolist() == [13]
    session.frame.sensors[13] = 42
    frame()
    assert panel._sensor.values.tolist() == [42]


@pytest.mark.parametrize("language", ("en", "zh_CN"))
@pytest.mark.parametrize("width", (160, 200, 600))
def test_live_magnitude_does_not_move_plot_rectangles(plotted_panel, language, width):
    panel, session, _, frame = plotted_panel
    frame(width, language)
    _, before = frame(width, language)
    session.frame.qpos[:] = 1.23456789e30
    session.frame.qvel[:] = -1.23456789e-30
    session.frame.time, session.frame.step = 1234567, 20
    _, after = frame(width, language)
    assert [entry[:5] for entry in before] == [entry[:5] for entry in after]
    assert panel._current["qpos"] == 1.23456789e30


def test_visible_channels_share_actual_time_coordinates_and_reuse_vertices(plotted_panel):
    panel, session, _ctx, frame = plotted_panel
    frame(700)
    for index, time in enumerate((0.1, 0.4), 1):
        session.frame.time, session.frame.step = time, index
        session.frame.qpos[0] = index
        frame(700)
    _, graphs = frame(700)
    assert len(graphs) == 2
    assert graphs[0][3:5] == graphs[1][3:5]
    assert all(graph[-1]["window"] == (0.0, 0.4) for graph in graphs)
    points = panel._points["qpos"]
    assert (points[1].x - points[0].x) / (points[2].x - points[0].x) == pytest.approx(0.25)
    allocated = tuple(id(point) for point in points)
    frame(700)
    assert tuple(id(point) for point in panel._points["qpos"]) == allocated
    assert panel._angle.filled == 3


def test_short_bottom_panel_keeps_both_plots_visible_and_single_sample_is_a_point(plotted_panel):
    panel, _session, ctx, frame = plotted_panel
    frame(700, height=230)
    text, graphs = frame(700, height=230)
    assert len(graphs) == 2
    assert graphs[0][3:5] == graphs[1][3:5]
    assert max(graph[4] for graph in graphs) <= 230 * ctx.style_scale
    assert all(graph[4] - graph[3] >= 80 * ctx.style_scale for graph in graphs)
    assert len(panel._points["qpos"]) == len(panel._points["qvel"]) == 1
    assert any("1 sample" in value for value in text)
    assert "qpos" not in text and "qvel" not in text


def test_partial_channel_histories_share_domain_without_fabricating_early_samples():
    a, b = Ring(), Ring()
    for time in (0, 0.1, 0.4):
        a.push(time, time=time)
    b.push(9, time=0.4)
    assert sample_window((a, b)) == (0, 0.4)
    assert b.filled == 1


def test_channel_selection_without_data_has_a_clear_empty_state(plotted_panel):
    panel, _session, _ctx, frame = plotted_panel
    panel.show_angle = panel.show_velocity = panel.show_contact = False
    text, graphs = frame(700)
    assert not graphs
    assert "Select channels to display." in text


def test_scene_without_physics_channels_keeps_source_selection_available(plotted_panel):
    panel, session, _ctx, frame = plotted_panel
    session.joints = []
    text, graphs = frame(300)
    assert "no joints to plot" in text
    assert not graphs
    assert "plot-source" in panel.ui.hits
    panel.focus_sensor(0, 13)
    text, graphs = frame(300)
    assert len(graphs) == 1
    assert panel._current["sensor"] == 13


def test_real_popup_actions_change_channels_components_and_sensor_source(plotted_panel):
    panel, session, _ctx, frame = plotted_panel
    frame(700)
    frame(700)
    frame.click("##plot-qpos-component")
    frame.click("##plot-qpos-component/choice/2")
    assert panel.position_component == 2
    assert panel._current["qpos"] == session.frame.qpos[2]
    assert panel._angle.filled == 1
    frame.click("plot-channels")
    frame.click("plot-channels/choice/1")
    assert not panel.show_velocity
    assert not panel.frame_needs().qvel
    frame.click("plot-source-1")
    assert panel.source == "sensor"
    frame.click("##plot-sensor-component")
    frame.click("##plot-sensor-component/choice/13")
    assert panel.sensor_component == 13
    assert panel._sensor.values.tolist() == [13]
    assert panel.frame_needs().sensors


def test_custom_select_height_does_not_overlap_current_value_and_graph(plotted_panel):
    from copy import deepcopy

    panel, _session, ctx, frame = plotted_panel
    spec = deepcopy(panel.ui.design.runtime.spec)
    spec["parts"]["select"] = {"height": 52, "marginTop": 4, "marginBottom": 6}
    panel.ui.design.runtime.update_spec(spec)
    frame(700)
    _, graphs = frame(700)
    for graph, key in zip(graphs, ("qpos", "qvel"), strict=True):
        _x, y, _width, height = panel.ui.hits[f"##plot-{key}-component"]
        assert graph[3] >= y + height + 6 * ctx.style_scale


def test_plot_current_ink_matches_body_and_axes_remain_secondary(plotted_panel, monkeypatch):
    panel, _session, ctx, frame = plotted_panel
    current, axes = [], []
    text, painted = imgui.text_unformatted, plot.ImguiDraw2D.text

    def numeric(value):
        if value == "0.25":
            font, size = imgui.get_font(), imgui.get_font_size()
            glyph = font.get_font_baked(size, 1).find_glyph(ord("0"))
            current.append((font.font_id, size, glyph.y1 - glyph.y0))
        text(value)

    def axis(draw, position, color, value, **options):
        if value == "m":
            axes.append(imgui.get_font_size())
        return painted(draw, position, color, value, **options)

    monkeypatch.setattr(imgui, "text_unformatted", numeric)
    monkeypatch.setattr(plot.ImguiDraw2D, "text", axis)
    frame(700)
    assert current[0][0] == panel.ui.numeric_font.font_id
    assert current[0][1] == pytest.approx(18 * ctx.style_scale)
    assert axes == pytest.approx([12 * ctx.style_scale])
    body = panel.ui.body.get_font_baked(14 * ctx.style_scale, 1).find_glyph(ord("0"))
    assert 0.90 <= current[0][2] / (body.y1 - body.y0) <= 1.05


def test_source_navigation_keeps_saved_segment_and_button_groups_independent(plotted_panel):
    from copy import deepcopy

    panel, _session, ctx, frame = plotted_panel
    ui, runtime = panel.ui, panel.ui.design.runtime
    runtime.observe_components = True
    spec = deepcopy(runtime.spec)
    spec["parts"]["plot_segment"] = {
        "paddingLeft": 7,
        "paddingRight": 9,
        "paddingTop": 4,
        "paddingBottom": 6,
        "marginLeft": 5,
        "marginBottom": 8,
    }
    spec["parts"]["plot_segment_button"] = {
        "height": 38,
        "fontSize": 18,
        "textColor": "#f4b860",
        "radius": 9,
    }
    runtime.update_spec(spec)
    frame(700)
    frame(700)
    rows = [*runtime.observed, *ui.design.primitives.observed]
    assert any("plot_segment" in row["parts"] for row in rows)
    buttons = [row for row in rows if row["key"] in ("plot-source-0", "plot-source-1")]
    assert len(buttons) == 2
    assert all("plot_segment_button" in row["parts"] for row in buttons)
    assert all(row["fieldValues"]["fontSize"] == 18 for row in buttons)
    assert ui.hits["plot-source-0"][3] == pytest.approx(38 * ctx.style_scale)
    assert ui.hits["##joint"][3] == pytest.approx(28 * ctx.style_scale)
    source = ui.hits["plot-source"]
    assert ui.hits["plot-source-0"][0] - source[0] == pytest.approx(12 * ctx.style_scale)
    frame.click("plot-source-1")
    assert panel.source == "sensor"
    frame.click("plot-source-0")
    assert panel.source == "joint"
