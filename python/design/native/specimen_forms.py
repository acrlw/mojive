"""Editable form references composed from the production Timeline controls."""

from types import SimpleNamespace

from imgui_bundle import imgui

from mojive.config import RecordingConfig
from mojive.design.native.geometry import Box, Rect, part_scope
from mojive.ui.panels import PanelContext
from mojive.ui.workspace.form_popup import measure_form_popup
from mojive.ui.workspace.property_style import toggle_row_height
from mojive.ui.workspace.timeline_panel import NativeTimelineToolbar
from mojive.ui.workspace.window_chrome import surface_ink


@part_scope("form_popup", inherit=False)
def form_reference(owner, ui):
    """Keep sample settings local while reusing the live popup's measurement."""
    key = "form_popup"
    if key not in owner.objects:
        session = SimpleNamespace(state_take_pause_at_end=True)
        ctx = PanelContext(session, None)

        def set_pause(value):
            owner.values[key + "/pause"] = session.state_take_pause_at_end = value

        def set_recording(value, *, persist=True):
            owner.values[key + "/recording"] = ctx.recording_config = value

        ctx.set_take_pause_at_end, ctx.set_recording_config = set_pause, set_recording
        owner.objects[key] = NativeTimelineToolbar(SimpleNamespace(ui=ui, s=ui.s)), ctx
    toolbar, ctx = owner.objects[key]
    ctx.style_scale, ctx.translate = ui.s, ui.translate
    ctx.session.state_take_pause_at_end = owner.values.get(key + "/pause", True)
    ctx.recording_config = owner.values.setdefault(key + "/recording", RecordingConfig())
    labels = tuple(
        ctx.tr(label) for label in ("Pause at last frame", "Start delay (s)", "End hold (s)")
    )
    number_width, number_height = toolbar._numeric_size("%.1f")
    layout = measure_form_popup(
        ui,
        labels,
        field_width=max(96 * ui.s, number_width * ui.s, ui.numeric_minimum()),
        row_height=max(number_height * ui.s, toggle_row_height(ui)),
        group_starts=(1,),
    )
    return toolbar, ctx, layout


@part_scope("form_popup", inherit=False)
def paint_form_reference(owner, ui, bounds):
    """Show the measured popup inline so catalogue navigation stays available."""
    toolbar, ctx, layout = form_reference(owner, ui)
    frame = Rect(bounds.x, bounds.y, layout.width, layout.height)
    content = frame.inset(*layout.padding)
    box = Box(frame, frame, content, layout.padding, (0, 0, 0, 0))
    ui.rect(
        *frame.tuple(),
        surface_ink(ui, "form_popup", "popup"),
        r=layout.radius / ui.s,
        color_resolved=True,
    )
    origin = imgui.ImVec2(frame.x, frame.y)
    imgui.push_id("reference-form_popup")
    try:
        with ui.corner_scope(layout.inner_radius, ("input_number", "axis_input")):
            toolbar._draw_pause_at_end(ctx, layout.row(origin, 0))
            toolbar._draw_recording_fields(ctx, layout, origin)
    finally:
        imgui.pop_id()
    rows = [layout.row(origin, index).tuple() for index in range(3)]
    ui.record_component_box(
        "form_popup",
        box,
        key="reference-form_popup",
        fields=rows,
    )
    return box, rows
