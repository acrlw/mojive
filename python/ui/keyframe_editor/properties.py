"""Selected keyframe property fields and edit feedback."""

from __future__ import annotations

from typing import Protocol

from imgui_bundle import imgui

from mojive import commands as cmd
from mojive.adapters.base import KeyframeInfo
from mojive.ui.panels import PanelContext, button_row_layout, button_width

from .controller import TimelineEditor, _set_keyframe_command


class SelectedKeyframePresentation(Protocol):
    """Lay out selection details without owning drafts or edit commands."""

    def fields(self, name: str, time: float) -> tuple[str, float]: ...
    def actions(
        self, *, applicable: bool, loadable: bool, deletable: bool
    ) -> tuple[bool, bool, bool]: ...
    def marker(self, name: str, time: float, *, loadable: bool) -> bool: ...
    def error(self, message: str) -> bool: ...


def draw_selected(
    editor: TimelineEditor,
    ctx: PanelContext,
    editable: bool,
    *,
    presentation: SelectedKeyframePresentation | None = None,
) -> None:
    if editor.edit_lane != "model" or editor.selected_id < 0 or len(editor.selected_keyframes) > 1:
        draw_error(editor, ctx, presentation=presentation)
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
        marker = editor.keyframe_by_id.get(editor.selected_id)
        if marker is not None:
            loadable = editor.can_load_keyframes(ctx)
            if presentation is not None:
                load = presentation.marker(marker.name, marker.time, loadable=loadable)
            else:
                imgui.separator()
                imgui.text_disabled(f"{marker.name}  {marker.time:g} {ctx.tr('s')}")
                imgui.begin_disabled(not loadable)
                load = imgui.button(ctx.tr("Load"))
                imgui.end_disabled()
            if load and loadable:
                editor.load_keyframe(ctx, marker)
        draw_error(editor, ctx, presentation=presentation)
        return

    if presentation is None:
        actions_inline = _draw_fields(editor, ctx)
    else:
        editor.name, editor.time = presentation.fields(editor.name, editor.time)
        actions_inline = False
    dirty = editor.name.strip() != properties.name or editor.time != properties.time
    applicable = editable and dirty and bool(editor.name.strip())
    loadable = editor.can_load_keyframes(ctx)
    apply, load, delete = (
        _draw_actions(ctx, applicable, loadable, editable, actions_inline)
        if presentation is None
        else presentation.actions(applicable=applicable, loadable=loadable, deletable=editable)
    )
    if apply and applicable:
        time = float(editor.time)
        ctx.submit_model_edit(
            _set_keyframe_command(properties, editor.name.strip(), time),
            lambda result: editor.snapshot_updated(result, time),
        )
    if load and loadable:
        keyframe = KeyframeInfo(
            properties.keyframe_id, properties.name, properties.time, properties.model_id
        )
        editor.load_keyframe(ctx, keyframe)
    if delete and editable:
        ctx.submit_model_edit(
            cmd.RemoveModelKeyframe(properties.keyframe_id), editor.snapshot_removed
        )
    draw_error(editor, ctx, presentation=presentation)


def _draw_fields(editor: TimelineEditor, ctx: PanelContext) -> bool:
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

    action_labels = (ctx.tr("Apply"), ctx.tr("Load"), ctx.tr("Delete"))
    action_widths = tuple(button_width(label) for label in action_labels)
    return fields_inline and available >= name_width + time_width + sum(action_widths) + 5 * gap


def _draw_actions(ctx, applicable, loadable, deletable, actions_inline):
    action_labels = (ctx.tr("Apply"), ctx.tr("Load"), ctx.tr("Delete"))
    action_widths = tuple(button_width(label) for label in action_labels)
    imgui.begin_disabled(not applicable)
    if actions_inline:
        imgui.same_line()
    inline = button_row_layout(
        action_widths,
        imgui.get_content_region_avail().x,
        imgui.get_style().item_spacing.x,
    )
    apply = imgui.button(action_labels[0])
    imgui.end_disabled()
    if inline[1]:
        imgui.same_line()
    imgui.begin_disabled(not loadable)
    load = imgui.button(action_labels[1])
    imgui.end_disabled()
    if inline[2]:
        imgui.same_line()
    imgui.begin_disabled(not deletable)
    delete = imgui.button(action_labels[2])
    imgui.end_disabled()
    if not deletable:
        imgui.set_item_tooltip(ctx.tr("Pause the simulation before editing keyframes"))
    return apply, load, delete


def draw_error(
    editor: TimelineEditor,
    ctx: PanelContext,
    *,
    presentation: SelectedKeyframePresentation | None = None,
) -> None:
    if editor.error:
        if presentation is None:
            imgui.text_colored(imgui.ImVec4(*ctx.theme.danger), editor.error)
            copy = imgui.small_button(f"{ctx.tr('Copy error')}##keyframes")
        else:
            copy = presentation.error(editor.error)
        if copy:
            imgui.set_clipboard_text(editor.error)
