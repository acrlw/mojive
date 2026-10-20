"""Workspace help tags isolate typography and reuse production status keycaps."""

from copy import deepcopy
from math import ceil, floor
from types import SimpleNamespace

import pytest
from imgui_bundle import imgui

from mojive.design.native.geometry import CURRENT_NATIVE_PARTS, part_scope
from mojive.design.native.presentation import DesignPresentation
from mojive.design.runtime import DesignRuntime
from mojive.design.store import SpecificationStore
from mojive.ui.app.gizmo_input import _GizmoInput
from mojive.ui.panels import copyable_name_item
from mojive.ui.panels.assets import AssetsPanel
from mojive.ui.panels.camera import CameraPanel
from mojive.ui.panels.help import HelpPanel
from mojive.ui.panels.info import InfoPanel
from mojive.ui.panels.inspector.fields import _property_tooltip
from mojive.ui.panels.plot import PlotPanel
from mojive.ui.panels.presentation import property_tooltip
from mojive.ui.replay_controls import ReplayControls
from mojive.ui.theme import THEME
from mojive.ui.workspace.inspector_style import NativeInspectorPresentation
from mojive.ui.workspace.menu_geometry import native_epsilon, native_unit
from mojive.ui.workspace.output_panel import WorkspaceOutput
from mojive.ui.workspace.property_style import NativePropertyPresentation
from mojive.ui.workspace.settings_panel import WorkspaceSettings
from mojive.ui.workspace.status_components import StatusBadge
from mojive.ui.workspace.tooltips import ShortcutHint
from mojive.ui.workspace.widgets import UI


@pytest.fixture(params=(1, 1.2, 1.25, 2.25))
def tooltip_frame(request, tmp_path, monkeypatch):
    context = imgui.create_context()
    io = imgui.get_io()
    io.set_ini_filename(None)
    io.display_size = (1200 * request.param, 700 * request.param)
    io.delta_time = 1 / 60
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    ui = UI(
        SimpleNamespace(style_scale=request.param, font_report=SimpleNamespace(cjk_path=None)),
        portable_fonts=True,
    )
    runtime = DesignRuntime(SpecificationStore(tmp_path / "tooltip.json"))
    runtime.observe_components = True
    ui.design = DesignPresentation(ui, runtime)
    rows, badges, translations = [], [], []
    original_text, original_badge = imgui.text_unformatted, StatusBadge.draw

    def text(value):
        rows.append(
            (
                value,
                imgui.get_font().font_id,
                imgui.get_font_size(),
                tuple(imgui.get_style_color_vec4(imgui.Col_.text)),
            )
        )
        return original_text(value)

    def badge(self, x, center_y, label, theme, **options):
        measured = self.size(label)
        badges.append(
            (
                label,
                (options.get("width", measured[0]), options.get("height", measured[1])),
                x,
                center_y,
            )
        )
        return original_badge(self, x, center_y, label, theme, **options)

    def translate(value):
        translations.append(value)
        return {"Rotate the selection": "旋转选中内容"}.get(value, "TRANSLATED " + value)

    def frame(paint):
        rows.clear()
        badges.clear()
        translations.clear()
        runtime.observed.clear()
        imgui.new_frame()
        imgui.set_next_window_pos((0, 0))
        imgui.set_next_window_size(io.display_size)
        imgui.begin("Workspace tooltip caller")
        try:
            return paint()
        finally:
            imgui.end()
            imgui.render()

    monkeypatch.setattr(imgui, "text_unformatted", text)
    monkeypatch.setattr(StatusBadge, "draw", badge)
    monkeypatch.setattr(ui, "translate", translate)
    try:
        yield SimpleNamespace(
            ui=ui, rows=rows, badges=badges, translations=translations, frame=frame, runtime=runtime
        )
    finally:
        imgui.destroy_context(context)


def native_state():
    context, style = imgui.get_current_context(), imgui.get_style()
    return (
        len(context.style_var_stack),
        len(context.color_stack),
        len(context.current_window_stack),
        context.current_window.id_,
        tuple(style.window_padding),
        tuple(style.window_min_size),
        tuple(style.item_spacing),
        style.window_rounding,
        imgui.get_font().font_id,
        imgui.get_font_size(),
        tuple(style.color_(imgui.Col_.text)),
        CURRENT_NATIVE_PARTS.get(),
        context.current_window.dc.text_wrap_pos,
        len(context.current_window.dc.text_wrap_pos_stack),
    )


@pytest.mark.parametrize("caller", ("heading", "numeric"))
@pytest.mark.parametrize("shortcut", ("E", "Shift + Ctrl + K", "Unbound"))
def test_help_action_font_and_binding_are_isolated_from_the_caller(tooltip_frame, caller, shortcut):
    state, ui = tooltip_frame, tooltip_frame.ui
    font, size = (ui.bold, 36) if caller == "heading" else (ui.mono, 14)

    def paint():
        with part_scope("section_header" if caller == "heading" else "input_number"):
            imgui.push_font(font, size * ui.s)
            imgui.push_style_var(imgui.StyleVar_.window_padding, (0, 0))
            imgui.push_style_var(imgui.StyleVar_.window_min_size, (300, 200))
            imgui.push_style_color(imgui.Col_.text, (1, 0, 0, 1))
            try:
                before = native_state()
                ui.tooltip(ShortcutHint("Rotate the selection", shortcut))
                assert native_state() == before
            finally:
                imgui.pop_style_color()
                imgui.pop_style_var(2)
                imgui.pop_font()

    for _ in range(3):
        state.frame(paint)
    assert state.rows[0][0] == "旋转选中内容"
    assert state.rows[0][1] == ui.body.font_id
    assert state.rows[0][2] == floor(12 * ui.s + 0.5)
    assert state.rows[0][3] != (1, 0, 0, 1)
    assert state.translations == ["Rotate the selection"]
    assert [row[0] for row in state.badges] == ([] if shortcut == "Unbound" else [shortcut])
    record = next(row for row in state.runtime.observed if row.get("kind") == "tooltip")
    assert record["text"] == "旋转选中内容"
    assert record["shortcut"] == ("" if shortcut == "Unbound" else shortcut)


@pytest.mark.parametrize("shortcut", ("E", "Shift + Ctrl + K"))
def test_help_keycap_keeps_status_style_with_local_native_frame_quantization(
    tooltip_frame, shortcut
):
    state, ui = tooltip_frame, tooltip_frame.ui
    expected = []

    def paint():
        with part_scope("statusbar"):
            expected.append(StatusBadge(ui, keycap=True).size(shortcut))
        with part_scope("section_header"):
            imgui.push_font(ui.bold, 36 * ui.s)
            try:
                ui.tooltip(ShortcutHint("Rotate the selection", shortcut))
            finally:
                imgui.pop_font()

    for _ in range(3):
        state.frame(paint)
    assert len(state.badges) == 1
    assert state.badges[0][1] == tuple(ceil(value) for value in expected[-1])


@pytest.mark.parametrize("shortcut", ("E", "Shift + Ctrl + K"))
@pytest.mark.parametrize("authored", (False, True))
def test_shortcut_tag_actual_first_window_and_painted_badge_are_concentric(
    tooltip_frame, monkeypatch, shortcut, authored
):
    state, ui, paints = tooltip_frame, tooltip_frame.ui, []
    if authored:
        spec = deepcopy(state.runtime.spec)
        spec["parts"]["badge"] = {
            "radius": 5.5,
            "marginLeft": 4,
            "marginTop": 3,
            "marginRight": 2,
            "marginBottom": 1,
        }
        state.runtime.update_spec(spec)
    rect = ui.rect

    def capture(x, y, w, h, color, r=6, **options):
        paints.append(((x, y, w, h), r * ui.s))
        return rect(x, y, w, h, color, r, **options)

    monkeypatch.setattr(ui, "rect", capture)
    state.frame(lambda: ui.tooltip(ShortcutHint("Rotate the selection", shortcut)))
    record = next(row for row in state.runtime.observed if row.get("kind") == "tooltip")
    badge = next(row for row in state.runtime.observed if row["key"].startswith("status-badge-"))
    x, y, w, h = record["hit"]
    bx, by, bw, bh = badge["hit"]
    inset = native_unit(6 * ui.s)
    assert record["nativePadding"] == [inset, inset]
    epsilon = native_epsilon(x, y, w, h, bx, by, bw, bh)
    assert abs(by - y - inset) <= epsilon
    assert abs(x + w - bx - bw - inset) <= epsilon
    assert abs(y + h - by - bh - inset) <= epsilon
    assert (w, h) == tuple(value + 2 * inset for value in record["contentSize"])
    radii = [radius for frame, radius in paints if frame == tuple(badge["hit"])]
    assert len(radii) == 1
    assert abs(record["nativeRadius"] - radii[0] - inset) <= native_epsilon(
        record["nativeRadius"], radii[0]
    )
    assert tuple(imgui.ImVec4(bx, by, bw, bh)) == record["badgeRect"]


def test_wrapped_shortcut_help_uses_a_compact_row_and_independent_corners(tooltip_frame):
    state, ui = tooltip_frame, tooltip_frame.ui
    action = "Rotate the selection around its local body frame while preserving the origin"

    def paint():
        with ui.corner_scope(0, "badge"):
            ui.tooltip(ShortcutHint(action, "Shift + E"), translate=False)

    state.frame(paint)
    record = next(row for row in state.runtime.observed if row.get("kind") == "tooltip")
    x, y, w, h = record["hit"]
    bx, by, bw, bh = record["badgeRect"]
    inset = native_unit(6 * ui.s)
    assert record["badgeRadius"] > 0
    assert record["contentSize"][1] > bh
    assert record["contentSize"][0] <= ceil(260 * ui.s) + native_unit(8 * ui.s) + bw
    epsilon = native_epsilon(x, y, w, h, bx, by, bw, bh)
    assert abs(by - y - inset) <= epsilon
    assert abs(x + w - bx - bw - inset) <= epsilon
    assert y + h - by - bh >= inset
    assert (w, h) == tuple(value + 2 * inset for value in record["contentSize"])


def test_plain_help_keeps_its_own_radius_without_a_keycap(tooltip_frame):
    state, ui = tooltip_frame, tooltip_frame.ui
    state.frame(lambda: ui.tooltip("Rotate the selection"))
    record = next(row for row in state.runtime.observed if row.get("kind") == "tooltip")
    assert record["nativeRadius"] == pytest.approx(6 * ui.s)
    assert record["badgeRect"] is None and record["badgeRadius"] is None
    assert not state.badges


def test_translation_can_be_disabled_without_changing_the_shortcut(tooltip_frame):
    state, ui = tooltip_frame, tooltip_frame.ui
    for _ in range(3):
        state.frame(
            lambda: ui.tooltip(ShortcutHint("Rotate the selection", "Shift + E"), translate=False)
        )
    assert state.translations == []
    assert state.rows[0][0] == "Rotate the selection"
    assert state.badges[0][0] == "Shift + E"


@pytest.mark.parametrize("structured", (False, True))
def test_help_tag_restores_native_stacks_when_text_paint_fails(
    tooltip_frame, monkeypatch, structured
):
    state, ui = tooltip_frame, tooltip_frame.ui

    def fail(_value):
        raise ValueError("help text paint failed")

    def paint():
        before = native_state()
        with pytest.raises(ValueError, match="help text paint failed"):
            ui.tooltip(
                ShortcutHint("Rotate the selection", "E") if structured else "Rotate the selection"
            )
        assert native_state() == before
        tooltip = imgui.internal.find_window_by_name("##Tooltip_00")
        assert not tooltip.dc.text_wrap_pos_stack

    monkeypatch.setattr(imgui, "text_unformatted", fail)
    state.frame(paint)


def test_suppressed_help_tag_restores_its_callers_stacks(tooltip_frame, monkeypatch):
    state, ui = tooltip_frame, tooltip_frame.ui
    monkeypatch.setattr(imgui, "begin_tooltip", lambda: False)

    def paint():
        imgui.set_next_window_size((137, 83))
        pending = imgui.get_current_context().next_window_data
        previous_size = pending.has_flags, tuple(pending.size_val), pending.size_cond
        before = native_state()
        ui.tooltip(ShortcutHint("Rotate the selection", "E"))
        assert native_state() == before
        assert (pending.has_flags, tuple(pending.size_val), pending.size_cond) == previous_size

    state.frame(paint)
    assert not state.rows and not state.badges


@pytest.mark.parametrize("explicit_hint", (False, True))
@pytest.mark.parametrize("hovered", (False, True))
def test_icon_segment_hints_use_the_quiet_presenter_once(
    tooltip_frame, monkeypatch, explicit_hint, hovered
):
    state, ui = tooltip_frame, tooltip_frame.ui
    monkeypatch.setattr(imgui, "is_item_hovered", lambda *_args, **_kwargs: hovered)

    def raw_tooltip(_text):
        pytest.fail("Workspace segments must use their tooltip presenter")

    monkeypatch.setattr(imgui, "set_item_tooltip", raw_tooltip)

    def paint():
        imgui.push_font(ui.bold, 36 * ui.s)
        try:
            before = native_state()
            ui.segmented(
                "rotation-mode",
                20 * ui.s,
                20 * ui.s,
                140,
                ("Rotation mode",),
                0,
                icons=("rotate",),
                show_labels=False,
                tooltips=(ShortcutHint("Rotate the selection", "E"),) if explicit_hint else None,
            )
            assert native_state() == before
        finally:
            imgui.pop_font()

    for _ in range(3):
        state.frame(paint)
    if not hovered:
        assert not state.rows and not state.badges
        return
    assert len(state.rows) == 1
    assert state.rows[0][0] == ("旋转选中内容" if explicit_hint else "TRANSLATED Rotation mode")
    assert state.rows[0][1] == ui.body.font_id
    assert state.rows[0][2] == floor(12 * ui.s + 0.5)
    assert [row[0] for row in state.badges] == (["E"] if explicit_hint else [])
    assert "TRANSLATED Rotation mode" not in state.translations


@pytest.mark.parametrize(
    "surface", ("settings", "camera", "inspector", "output", "assets", "plot", "info", "help")
)
@pytest.mark.parametrize("hovered", (False, True))
@pytest.mark.parametrize("immediate", (False, True))
def test_registered_panel_hint_delegation_owns_its_font_and_hover_gate(
    tooltip_frame, monkeypatch, surface, hovered, immediate
):
    state, ui = tooltip_frame, tooltip_frame.ui
    monkeypatch.setattr(imgui, "is_item_hovered", lambda *_args, **_kwargs: hovered)
    if surface == "settings":
        panel = WorkspaceSettings.__new__(WorkspaceSettings)
        panel.fields = NativePropertyPresentation(ui)
        hint = panel._tooltip
    elif surface == "camera":
        panel = CameraPanel.__new__(CameraPanel)
        panel.presentation = NativePropertyPresentation(ui)
        hint = panel._tooltip
    elif surface == "inspector":
        ctx = SimpleNamespace(inspector_presentation=NativeInspectorPresentation(ui))

        def hint(text, **options):
            _property_tooltip(ctx, text, **options)
    elif surface == "output":
        panel = WorkspaceOutput.__new__(WorkspaceOutput)
        panel.workspace = SimpleNamespace(ui=ui)
        hint = panel._tooltip
    else:
        panel_type = {
            "assets": AssetsPanel,
            "plot": PlotPanel,
            "info": InfoPanel,
            "help": HelpPanel,
        }[surface]
        panel = panel_type.__new__(panel_type)
        panel.presentation = NativePropertyPresentation(ui)
        hint = panel._tooltip

    def paint():
        imgui.push_font(ui.bold, 36 * ui.s)
        try:
            before = native_state()
            hint("已本地化操作", hovered=immediate)
            assert native_state() == before
        finally:
            imgui.pop_font()

    for _ in range(3):
        state.frame(paint)
    assert state.translations == []
    assert len(state.rows) == int(hovered or immediate)
    if state.rows:
        assert state.rows[0][0] == "已本地化操作"
        assert state.rows[0][1] == ui.body.font_id
        assert state.rows[0][2] == floor(12 * ui.s + 0.5)


def test_optional_panel_presenter_retains_standalone_native_hints(monkeypatch):
    import mojive.ui.panels.presentation as presentation

    hints = []
    monkeypatch.setattr(
        presentation,
        "imgui",
        SimpleNamespace(
            set_tooltip=lambda text: hints.append(("immediate", text)),
            set_item_tooltip=lambda text: hints.append(("item", text)),
        ),
    )
    property_tooltip(None, "Native item")
    property_tooltip(SimpleNamespace(), "Native immediate", hovered=True)
    assert hints == [("item", "Native item"), ("immediate", "Native immediate")]


def test_precise_gizmo_clipped_title_uses_the_workspace_help_font(tooltip_frame, monkeypatch):
    state, ui = tooltip_frame, tooltip_frame.ui
    title = "A deliberately long object and axis title that exceeds the precise editor width"
    app = SimpleNamespace(
        _precise_gizmo_edit=SimpleNamespace(
            action="Translate", label=title, unit="m", absolute_value=None
        ),
        _open_precise_gizmo_popup=True,
        _viewport_rect=(0, 0, 1000 * ui.s, 600 * ui.s),
        _precise_gizmo_angle_unit="degrees",
        _precise_gizmo_absolute=False,
        _precise_gizmo_value=0.0,
        _precise_gizmo_error="",
        _floating_ui=ui,
        localizer=SimpleNamespace(text=str),
        window=SimpleNamespace(style_scale=ui.s),
        workspace=SimpleNamespace(ui=ui),
        theme=THEME,
    )
    monkeypatch.setattr(imgui, "is_item_hovered", lambda *_args, **_kwargs: True)

    def raw_tooltip(_text):
        pytest.fail("Workspace precise editor must use its tooltip presenter")

    monkeypatch.setattr(imgui, "set_item_tooltip", raw_tooltip)

    def paint():
        imgui.push_font(ui.bold, 36 * ui.s)
        try:
            _GizmoInput._draw_precise_gizmo_popup(app)
        finally:
            imgui.pop_font()

    for _ in range(3):
        state.frame(paint)
    assert state.translations == []
    assert len(state.rows) == 1
    assert state.rows[0][0] == "Translate " + title
    assert state.rows[0][1] == ui.body.font_id
    assert state.rows[0][2] == floor(12 * ui.s + 0.5)


@pytest.mark.parametrize("mode", ("presenter", "callback", "caller-owned"))
def test_clipped_copyable_name_keeps_copy_action_and_one_owner_for_help(
    tooltip_frame, monkeypatch, mode
):
    import mojive.ui.panels as panels

    state, ui = tooltip_frame, tooltip_frame.ui
    name = "A deliberately clipped joint name"
    ctx = SimpleNamespace(
        inspector_presentation=NativeInspectorPresentation(ui),
        input_bindings=None,
        tr=str,
        status_hints=(),
    )
    copied = []
    monkeypatch.setattr(imgui, "is_item_hovered", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(imgui, "set_clipboard_text", copied.append)
    monkeypatch.setattr(panels, "pointer_pressed", lambda *_args: True)
    options = (
        {"tooltip": lambda text: ui.tooltip(text, translate=False)}
        if mode == "callback"
        else {"tooltip": False}
        if mode == "caller-owned"
        else {}
    )

    def paint():
        imgui.push_font(ui.bold, 36 * ui.s)
        try:
            imgui.text(name)
            assert copyable_name_item(ctx, name, 1, **options)
        finally:
            imgui.pop_font()

    for _ in range(3):
        state.frame(paint)
    assert copied == [name] * 3
    assert state.translations == []
    assert len(state.rows) == int(mode != "caller-owned")
    if state.rows:
        assert state.rows[0][0] == name
        assert state.rows[0][1] == ui.body.font_id
        assert state.rows[0][2] == floor(12 * ui.s + 0.5)


def test_replay_sync_hint_uses_its_workspace_callback(tooltip_frame, monkeypatch):
    state, ui = tooltip_frame, tooltip_frame.ui
    controls = ReplayControls()
    ctx = SimpleNamespace(tr=str, style_scale=ui.s, submit=lambda _command: None)
    sync = SimpleNamespace(
        pending=False, max_frames=512, start_step=17, received_bytes=2048, error=""
    )
    monkeypatch.setattr(imgui, "is_item_hovered", lambda *_args, **_kwargs: True)

    def raw_tooltip(_text):
        pytest.fail("Workspace replay must use its tooltip callback")

    monkeypatch.setattr(imgui, "set_tooltip", raw_tooltip)

    def paint():
        imgui.push_font(ui.bold, 36 * ui.s)
        try:
            controls._draw_sync(ctx, sync, tooltip=lambda text: ui.tooltip(text, translate=False))
        finally:
            imgui.pop_font()

    for _ in range(3):
        state.frame(paint)
    assert state.translations == []
    assert len(state.rows) == 1
    assert state.rows[0][0] == "Manual sync; playback stays local\nSource sample: 17  |  2.0 KiB"
    assert state.rows[0][1] == ui.body.font_id
    assert state.rows[0][2] == floor(12 * ui.s + 0.5)
