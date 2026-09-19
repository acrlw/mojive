"""Keyframes panel: compose timeline controls, tracks and selection properties."""

from __future__ import annotations

from imgui_bundle import imgui

from mojive import commands as cmd
from mojive.adapters.base import FrameNeeds
from mojive.ui.controls import IconLabelDrawer
from mojive.ui.icons import draw_icon_label
from mojive.ui.keyframe_editor.controller import TimelineEditor
from mojive.ui.keyframe_editor.controls import timeline_status_hints as timeline_status_hints
from mojive.ui.keyframe_editor.controls import unique_keyframe_name as unique_keyframe_name
from mojive.ui.keyframe_editor.properties import draw_error, draw_selected
from mojive.ui.keyframe_editor.toolbar import TimelineToolbar
from mojive.ui.keyframe_editor.track import draw_dope_sheet
from mojive.ui.replay_controls import ReplayControls
from mojive.ui.timeline import (
    decimated_marker_ids as decimated_marker_ids,
)
from mojive.ui.timeline import (
    fitted_timeline_range as fitted_timeline_range,
)
from mojive.ui.timeline import (
    follow_timeline_range as follow_timeline_range,
)
from mojive.ui.timeline import (
    nearest_take_frame as nearest_take_frame,
)
from mojive.ui.timeline import (
    neighboring_keyframe as neighboring_keyframe,
)
from mojive.ui.timeline import (
    nice_timeline_step as nice_timeline_step,
)
from mojive.ui.timeline import (
    recorded_take_spans as recorded_take_spans,
)
from mojive.ui.timeline import (
    timeline_channel_width as timeline_channel_width,
)
from mojive.ui.timeline import (
    timeline_time_to_x as timeline_time_to_x,
)
from mojive.ui.timeline import (
    timeline_x_to_time as timeline_x_to_time,
)
from mojive.ui.timeline import (
    zoom_timeline_range as zoom_timeline_range,
)

from . import Panel, PanelContext


class KeyframesPanel(Panel):
    """Edit model keyframes, recorded takes and complete scene snapshots."""

    id = "keyframes"
    name = "Keyframes"
    default_open = True
    shortcut = ""
    dock_with = "Output"

    def __init__(self, *, follow_mode_icon_drawer: IconLabelDrawer = draw_icon_label) -> None:
        super().__init__()
        self.editor = TimelineEditor()
        self.toolbar = TimelineToolbar(follow_mode_icon_drawer)
        self._replay = ReplayControls()

    def frame_needs(self) -> FrameNeeds:
        return FrameNeeds.none()

    def toggle_recording(self, ctx: PanelContext):
        return self.editor.toggle_recording(ctx)

    def status_detail(self, translate) -> str:
        return self.editor.status_detail(translate)

    def draw(self, ctx: PanelContext) -> None:
        if ctx.session.replay_info is not None:
            self._replay.draw(ctx)
            return
        editor = self.editor
        models = tuple(ctx.session.scene_models)
        model_ids = tuple(model.model_id for model in models)
        selected = ctx.session.selected_node
        if editor.model_id not in model_ids:
            preferred = selected.model_id if selected is not None else -1
            editor.set_model(
                preferred if preferred in model_ids else model_ids[0] if model_ids else -1
            )
        keyframes, keyframe_by_id = editor.keyframes(ctx)
        take_times = ctx.session.state_take_times
        editable = bool(
            ctx.session.paused
            and not ctx.take_video_active
            and ctx.session.adapter.caps.supports("model.keyframe_edit")
        )
        editor.sync_selection(ctx, keyframe_by_id, take_times)
        self.toolbar.draw_compact_toolbar(editor, ctx, take_times)
        take_times = ctx.session.state_take_times
        editor.sync_selection(ctx, keyframe_by_id, take_times)
        draw_dope_sheet(editor, ctx, models, keyframes, keyframe_by_id, take_times, editable)
        if editor.selected_snapshot >= 0:
            snapshot = next(
                (
                    item
                    for item in ctx.session.scene_snapshots
                    if item.snapshot_id == editor.selected_snapshot
                ),
                None,
            )
            if snapshot is None:
                editor.selected_snapshot = -1
            else:
                imgui.text_disabled(f"{snapshot.name}  {snapshot.time:.3f} {ctx.tr('s')}")
                if imgui.button(ctx.tr("Remove snapshot") + "##remove-scene-snapshot"):
                    ctx.submit(cmd.RemoveSceneSnapshot(snapshot.snapshot_id))
                    editor.selected_snapshot = -1
        draw_selected(editor, ctx, editable)
        draw_error(editor, ctx)
