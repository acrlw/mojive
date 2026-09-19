"""Local preview controls, independent of physics and training clock ownership."""

from imgui_bundle import imgui

from mojive import commands as cmd
from mojive.ui.controls import button_row_layout, button_width


class ReplayControls:
    """Reuse local playback controls for file and manually synchronized rollout clips."""

    def __init__(self):
        self._frame = 0
        self._dragging = False
        self._token = None
        self._window_frames = 128

    def draw(self, ctx):
        """Seek on slider release; ordinary playback and typing need no network traffic."""
        info = ctx.session.replay_info
        if info is None:
            return
        t = ctx.tr
        sync = ctx.session.rollout_sync_info
        token = (
            ctx.session.document_id,
            sync.revision if sync else "",
            sync.window_frames if sync else 0,
        )
        if token != self._token:
            self._token = token
            self._dragging = False
            if sync:
                self._window_frames = sync.window_frames
        actions = (
            (
                t("Play" if info.paused else "Pause") + "##replay_play",
                cmd.SetReplayPlayback(paused=not info.paused),
                True,
            ),
            (
                t("Previous frame") + "##replay_previous",
                cmd.SeekReplay(info.frame_index - 1),
                info.frame_index > 0,
            ),
            (
                t("Next frame") + "##replay_next",
                cmd.SeekReplay(info.frame_index + 1),
                info.frame_index < info.frame_count - 1,
            ),
            (t("Restart") + "##replay_restart", cmd.SeekReplay(0), True),
        )
        inline = button_row_layout(
            tuple(button_width(label) for label, _, _ in actions),
            imgui.get_content_region_avail().x,
            imgui.get_style().item_spacing.x,
        )
        for same_line, (label, command, enabled) in zip(inline, actions, strict=True):
            if same_line:
                imgui.same_line()
            imgui.begin_disabled(not enabled)
            if imgui.button(label):
                ctx.submit(command)
            imgui.end_disabled()
        if not self._dragging:
            self._frame = info.frame_index
        imgui.set_next_item_width(-1)
        _, self._frame = imgui.slider_int("##replay_frame", self._frame, 0, info.frame_count - 1)
        self._dragging = imgui.is_item_active()
        if imgui.is_item_deactivated_after_edit():
            ctx.submit(cmd.SeekReplay(self._frame))
        imgui.text(
            f"{t('Replay frame')}: {info.frame_index + 1} / {info.frame_count}  |  {info.frame_index / info.hz:.3f} s  |  {info.hz:g} Hz"
        )
        if sync is not None:
            self._draw_sync(ctx, sync)
        changed, loop = imgui.checkbox(t("Loop") + "##replay_loop", info.loop)
        if changed:
            ctx.submit(cmd.SetReplayPlayback(loop=loop))
        imgui.same_line()
        imgui.set_next_item_width(150 * ctx.style_scale)
        speeds = (0.25, 0.5, 1.0, 2.0, 4.0)
        if imgui.begin_combo("##replay_speed", f"{info.speed:g}x"):
            for speed in speeds:
                if imgui.selectable(t("{speed:g}×").format(speed=speed), speed == info.speed)[0]:
                    ctx.submit(cmd.SetReplayPlayback(speed=speed))
            imgui.end_combo()
        if info.error:
            imgui.text_wrapped(info.error)

    def _draw_sync(self, ctx, sync):
        t = ctx.tr
        available = imgui.get_content_region_avail().x
        valid = 2 <= self._window_frames <= sync.max_frames
        imgui.begin_disabled(sync.pending or not valid)
        label = t("Downloading rollout..." if sync.pending else "Sync latest rollout")
        if imgui.button(label):
            ctx.submit(cmd.SyncRollout(frame_count=self._window_frames))
        imgui.end_disabled()
        if imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled):
            imgui.set_tooltip(
                f"{t('Manual sync; playback stays local')}\n"
                f"{t('Source sample')}: {sync.start_step}  |  {sync.received_bytes / 1024:.1f} KiB"
            )
        field_width = 110 * ctx.style_scale
        field_label = t("Window frames")
        if (
            button_width(label)
            + field_width
            + imgui.calc_text_size(field_label).x
            + imgui.get_style().item_spacing.x * 3
            <= available
        ):
            imgui.same_line()
        imgui.set_next_item_width(min(field_width, available))
        imgui.begin_disabled(sync.pending)
        _, self._window_frames = imgui.input_int(
            field_label + "##rollout_frames", self._window_frames
        )
        imgui.end_disabled()
        if not valid:
            imgui.text_disabled(f"2–{sync.max_frames}")
        if sync.error:
            imgui.text_wrapped(sync.error)
