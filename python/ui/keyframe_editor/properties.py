"""Selected keyframe property fields and edit feedback."""

from __future__ import annotations

from imgui_bundle import imgui

from mojive import commands as cmd
from mojive.adapters.base import KeyframeInfo
from mojive.ui.panels import PanelContext, button_row_layout, button_width

from .controller import TimelineEditor, _set_keyframe_command


def draw_selected(editor: TimelineEditor, ctx: PanelContext, editable: bool) -> None:
    if editor.edit_lane != "model" or editor.selected_id < 0 or len(editor.selected_keyframes) > 1:
        draw_error(editor, ctx)
        return
    generation = ctx.session.structure_generation
    if editor.selection_generation != generation:
        editor.selection_generation = generation
        editor.properties = ctx.session.keyframe_properties(editor.selected_id)
        if editor.properties is not None:
            editor.name = editor.properties.name
            editor.time = editor.properties.time
    properties = editor.properties
    if properties is None:
        draw_error(editor, ctx)
        return

    imgui.separator()
    imgui.text_disabled(ctx.tr("selected snapshot"))
    scale = ctx.style_scale
    available = imgui.get_content_region_avail().x
    gap = imgui.get_style().item_spacing.x
    time_width = min(available, 100 * scale)
    fields_inline = available >= 220 * scale
    name_width = min(280 * scale, available - time_width - gap if fields_inline else available)
    imgui.set_next_item_width(max(1, name_width))
    _changed, editor.name = imgui.input_text("##keyframe-name", editor.name)
    imgui.set_item_tooltip(ctx.tr("name"))
    if fields_inline:
        imgui.same_line()
    imgui.set_next_item_width(time_width)
    _changed, editor.time = imgui.input_double("##keyframe-time", editor.time, 0.0, 0.0, "%.9g s")
    imgui.set_item_tooltip(ctx.tr("time"))

    dirty = editor.name.strip() != properties.name or editor.time != properties.time
    if not editable or not dirty or not editor.name.strip():
        imgui.begin_disabled()
    action_labels = (ctx.tr("Apply"), ctx.tr("Load"), ctx.tr("Delete"))
    action_widths = tuple(button_width(label) for label in action_labels)
    actions_inline = (
        fields_inline and available >= name_width + time_width + sum(action_widths) + 5 * gap
    )
    if actions_inline:
        imgui.same_line()
    inline = button_row_layout(
        action_widths,
        imgui.get_content_region_avail().x,
        imgui.get_style().item_spacing.x,
    )
    if imgui.button(action_labels[0]):
        time = float(editor.time)
        ctx.submit_model_edit(
            _set_keyframe_command(properties, editor.name.strip(), time),
            lambda result: editor.snapshot_updated(result, time),
        )
    if not editable or not dirty or not editor.name.strip():
        imgui.end_disabled()
    if inline[1]:
        imgui.same_line()
    if not editable:
        imgui.begin_disabled()
    if imgui.button(action_labels[1]):
        keyframe = KeyframeInfo(
            properties.keyframe_id, properties.name, properties.time, properties.model_id
        )
        editor.load_keyframe(ctx, keyframe)
    if inline[2]:
        imgui.same_line()
    if imgui.button(action_labels[2]):
        ctx.submit_model_edit(
            cmd.RemoveModelKeyframe(properties.keyframe_id), editor.snapshot_removed
        )
    if not editable:
        imgui.end_disabled()
        imgui.set_item_tooltip(ctx.tr("Pause the simulation before editing keyframes"))
    draw_error(editor, ctx)


def draw_error(editor: TimelineEditor, ctx: PanelContext) -> None:
    if editor.error:
        imgui.text_colored(imgui.ImVec4(*ctx.theme.danger), editor.error)
        if imgui.small_button(f"{ctx.tr('Copy error')}##keyframes"):
            imgui.set_clipboard_text(editor.error)
