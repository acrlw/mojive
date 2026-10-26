"""Production and Guide reserve one compact, measured status row at fractional scales."""

from copy import deepcopy
from math import ceil, floor
from types import SimpleNamespace

import pytest
from examples.ui_design.native.guide_panels import telemetry
from imgui_bundle import imgui

from mojive import RecordingInfo
from mojive.design.native.geometry import Rect, part_scope
from mojive.design.native.presentation import DesignPresentation
from mojive.design.native.specimens import ComponentSpecimens
from mojive.design.runtime import DesignRuntime
from mojive.design.store import SpecificationStore
from mojive.ui.app.status import _Status
from mojive.ui.imgui_draw import ImguiDraw2D
from mojive.ui.theme import THEME
from mojive.ui.viewport_widgets.model import DEFAULT_VIEWPORT_LABELS, ToolHint
from mojive.ui.workspace.status_components import StatusBadge, resolve_status_bar_layout
from mojive.ui.workspace.widgets import UI


@pytest.fixture(params=(1, 1.2, 1.25))
def status_surface(request, tmp_path, monkeypatch):
    context = imgui.create_context()
    io = imgui.get_io()
    io.set_ini_filename(None)
    io.display_size, io.delta_time = (1800 * request.param, 700 * request.param), 1 / 60
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    ui = UI(
        SimpleNamespace(style_scale=request.param, font_report=SimpleNamespace(cjk_path=None)),
        portable_fonts=True,
    )
    runtime = DesignRuntime(SpecificationStore(tmp_path / "status.json"))
    runtime.observe_components = True
    ui.design = DesignPresentation(ui, runtime)
    app = SimpleNamespace(
        window=SimpleNamespace(style_scale=ui.s, painter=ImguiDraw2D, pixels_to_points=float),
        workspace=SimpleNamespace(ui=ui),
        session=SimpleNamespace(
            frame=SimpleNamespace(physics_hz=480),
            adapter=SimpleNamespace(caps=SimpleNamespace(simulation=True)),
        ),
        design=None,
        recording=RecordingInfo(),
        localizer=SimpleNamespace(text=str),
        backend=SimpleNamespace(caps=SimpleNamespace(name="opengl")),
        theme=THEME,
        _status_metric_mode="time",
        _frame_rate=SimpleNamespace(value=59.8),
        passive_mode=False,
        external_status="",
        _viewport_labels=DEFAULT_VIEWPORT_LABELS,
        _status_tool_hints=lambda **_kwargs: (
            ToolHint("key", "Esc", "Cancel"),
            ToolHint("mouse", "left", "Orbit"),
        ),
    )
    guide = SimpleNamespace(
        ui=ui,
        s=ui.s,
        theme=THEME,
        clock=0,
        values={},
        tr=lambda text, **values: text.format(**values),
        box=lambda *_args: None,
        rect=lambda x, y, w, h: Rect(x * ui.s, y * ui.s, w * ui.s, h * ui.s),
        record=lambda _key: None,
    )
    painted, lines = [], []
    original_line = ImguiDraw2D.line

    def line(draw, a, b, color, width, **options):
        lines.append((a, b, width))
        return original_line(draw, a, b, color, width, **options)

    monkeypatch.setattr(ImguiDraw2D, "line", line)
    original = ImguiDraw2D.text

    def text(draw, position, color, value, **options):
        painted.append((value, imgui.get_font_size()))
        return original(draw, position, color, value, **options)

    monkeypatch.setattr(ImguiDraw2D, "text", text)

    def frame():
        painted.clear()
        lines.clear()
        imgui.new_frame()
        ui.design.begin_frame()
        _Status._paint_application_status_bar(app, loading=True)
        status_window = imgui.internal.find_window_by_name("Status###application_status")
        native_height = status_window.size.y
        imgui.set_next_window_pos((0, 0))
        imgui.set_next_window_size((1600 * ui.s, 300 * ui.s))
        imgui.begin("Guide status")
        try:
            end = telemetry(guide, 20, 40, 1500)
            specimen = ComponentSpecimens(runtime)
            measured = specimen.measure(ui, "statusbar", 1500 * ui.s)[1]
            reference = specimen.paint(
                ui, "statusbar", Rect(20 * ui.s, 150 * ui.s, 1500 * ui.s, measured)
            )
            resolved = resolve_status_bar_layout(ui)
        finally:
            imgui.end()
            imgui.render()
        records = {row["key"]: row for row in runtime.observed}
        return SimpleNamespace(
            native_height=native_height,
            lines=tuple(lines),
            badges=[r for r in runtime.observed if str(r["key"]).startswith("status-badge-")],
            end=end,
            measured=measured,
            reference=reference,
            resolved=resolved,
            production=records["application-statusbar"],
            guide=records["guide-statusbar"],
            canvas=records["guide-status-canvas"],
        )

    try:
        yield SimpleNamespace(ui=ui, app=app, runtime=runtime, frame=frame, painted=painted)
    finally:
        imgui.destroy_context(context)


def test_compact_status_geometry_matches_production_and_both_guide_references(status_surface):
    state, scale = status_surface, status_surface.ui.s
    for _ in range(3):
        row = state.frame()
    expected = ceil(28 * scale)
    assert row.native_height == row.resolved.height == row.measured == expected
    assert row.production["reservedHeight"] == row.guide["reservedHeight"] == expected
    assert row.production["hit"][3] == row.guide["hit"][3] == expected
    assert row.reference["frame"][3] == expected
    assert row.reference["content"][3] == row.guide["content"][3]
    assert row.production["content"][3] == pytest.approx(expected - 4 * scale)
    assert row.guide["content"][3] == pytest.approx(expected - 4 * scale)
    assert row.production["padding"] == pytest.approx((10 * scale, 2 * scale) * 2)
    assert row.guide["padding"] == row.production["padding"]
    assert {size for text, size in state.painted if text == "OpenGL"} == {floor(12 * scale + 0.5)}
    canvas, bar = Rect(*row.canvas["hit"]), Rect(*row.guide["outer"])
    assert (bar.x - canvas.x, bar.y - canvas.y) == pytest.approx((8 * scale,) * 2)
    assert (canvas.right - bar.right, canvas.bottom - bar.bottom) == pytest.approx((8 * scale,) * 2)
    assert row.canvas["radius"] == row.canvas["innerRadius"] + 8
    assert row.end == pytest.approx(40 + (expected + 16 * scale) / scale)


@pytest.mark.parametrize(
    "parts",
    (
        {"statusbar": {"fontSize": 28}},
        {"status_hints_badge": {"fontSize": 28, "paddingTop": 5, "paddingBottom": 7}},
        {"status_metrics": {"fontSize": 26}},
        {"statusbar": {"paddingTop": 13, "paddingBottom": 17}},
        {"statusbar": {"height": 80}},
    ),
)
def test_larger_authored_type_keycaps_and_padding_expand_without_clipping(status_surface, parts):
    state, scale = status_surface, status_surface.ui.s
    spec = deepcopy(state.runtime.spec)
    for part, values in parts.items():
        spec["parts"].setdefault(part, {}).update(values)
    state.runtime.update_spec(spec)
    for _ in range(3):
        row = state.frame()
    layout = row.resolved
    assert layout.height > ceil(28 * scale)
    assert row.native_height == row.measured == layout.height
    assert row.production["hit"][3] == row.guide["hit"][3] == layout.frame_height
    assert row.reference["frame"][3] == layout.frame_height
    assert layout.content_height >= layout.minimum_content_height
    with part_scope("statusbar"):
        badge_height = StatusBadge(state.ui, keycap=True).size("Esc")[1]
    assert layout.content_height >= badge_height


def test_fractional_margins_and_gap_are_reserved_once_in_both_hosts(status_surface):
    state, scale = status_surface, status_surface.ui.s
    spec = deepcopy(state.runtime.spec)
    spec["parts"]["statusbar"].update(marginTop=3, marginBottom=5)
    spec["parts"]["application"].update(rowGap=7)
    state.runtime.update_spec(spec)
    row = state.frame()
    layout = row.resolved
    assert layout.height == ceil((28 + 3 + 5 + 7) * scale)
    assert layout.frame_height + 15 * scale == pytest.approx(layout.height)
    assert row.native_height == row.measured == layout.height
    assert row.production["hit"][3] == row.guide["hit"][3] == layout.frame_height


def test_external_clock_hint_uses_quiet_workspace_help_after_bar_font_is_restored(
    status_surface, monkeypatch
):
    state, tips = status_surface, []
    state.app.passive_mode = True
    original = imgui.text_unformatted

    def text(value):
        tips.append((value, imgui.get_font().font_id, imgui.get_font_size()))
        return original(value)

    def hovered(*_args, **_kwargs):
        return imgui.get_current_context().current_window.name == "Status###application_status"

    def raw_hint(_text):
        pytest.fail("Workspace status help must not inherit the caller font")

    monkeypatch.setattr(imgui, "text_unformatted", text)
    monkeypatch.setattr(imgui, "is_item_hovered", hovered)
    monkeypatch.setattr(imgui, "set_item_tooltip", raw_hint)
    for _ in range(3):
        state.frame()
    assert len(tips) == 3
    assert tips[-1] == (
        "Simulation is driven by the external application. Recording only captures images.",
        state.ui.body.font_id,
        floor(12 * state.ui.s + 0.5),
    )


def test_status_divider_uses_outer_surface_and_keycaps_center_in_the_visible_row(status_surface):
    for _ in range(3):
        row = status_surface.frame()
    frame = Rect(*row.production["hit"])
    horizontal = [(a, b, width) for a, b, width in row.lines if a[1] == b[1] and b[0] - a[0] > 500]
    assert any(
        a == pytest.approx((frame.x, frame.y + width / 2))
        and b == pytest.approx((frame.right, frame.y + width / 2))
        for a, b, width in horizontal
    )
    production_badges = [Rect(*r["hit"]) for r in row.badges if r["hit"][1] >= frame.y]
    assert production_badges
    for badge in production_badges:
        assert badge.y - frame.y == pytest.approx(frame.bottom - badge.bottom)


def test_saved_status_profile_without_pointer_override_uses_the_reviewed_canvas(status_surface):
    from mojive.ui.workspace.status_components import StatusGroups

    state, scale = status_surface, status_surface.ui.s
    spec = deepcopy(state.runtime.spec)
    spec["parts"].pop("status_hints", None)
    state.runtime.update_spec(spec)
    row = state.frame()
    groups = StatusGroups(state.ui)
    assert groups.icon_size("status_hints", 22 * scale) == 22 * scale
    assert state.runtime.resolved_part("status_hints", "iconSize") == 22
    assert row.native_height == ceil(28 * scale)
