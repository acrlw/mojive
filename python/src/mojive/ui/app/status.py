"""App: status."""

from __future__ import annotations

import time

from imgui_bundle import imgui

from mojive.capture import RecordingPhase
from mojive.ui.draw2d import ImguiDraw2D, fit_text
from mojive.ui.files import message_path, reveal_path
from mojive.ui.pointer_bindings import PointerAction
from mojive.ui.viewport_widgets import (
    draw_status,
)

from .support import APPLICATION_STATUS_HEIGHT_PT, _model_filters, _simulation_timestep


class _Status:
    """Private status methods of ViewerApp; state belongs to its owner."""

    def _draw_collapsed_output(self) -> None:
        panel = self.panels.get("Output")
        if panel is None or not panel.open or not panel.collapsed:
            return
        scale = self.window.style_scale
        flags = (
            imgui.WindowFlags_.no_decoration
            | imgui.WindowFlags_.no_docking
            | imgui.WindowFlags_.no_move
            | imgui.WindowFlags_.no_saved_settings
        )
        imgui.push_style_var(imgui.StyleVar_.window_padding, (10 * scale, 8 * scale))
        visible = imgui.internal.begin_viewport_side_bar(
            "##output-summary",
            imgui.get_main_viewport(),
            imgui.Dir.down,
            imgui.get_frame_height() + 16 * scale,
            flags,
        )
        if visible:
            panel.draw_collapsed(self._panel_context())
        imgui.end()
        imgui.pop_style_var()

    def _draw_application_status_bar(self, *, loading: bool = False) -> None:
        """Draw persistent selection, simulation, backend, and frame-rate status."""

        viewport = imgui.get_main_viewport()
        scale = self.window.style_scale
        # One shared text baseline aligns every status group. A small
        # vertical gutter separates the single-line groups without inflating
        # their internal geometry.
        height = APPLICATION_STATUS_HEIGHT_PT * scale
        flags = (
            imgui.WindowFlags_.no_decoration.value
            | imgui.WindowFlags_.no_docking.value
            | imgui.WindowFlags_.no_move.value
            | imgui.WindowFlags_.no_saved_settings.value
            | imgui.WindowFlags_.no_scrollbar.value
            | imgui.WindowFlags_.no_nav_focus.value
            | imgui.WindowFlags_.no_bring_to_front_on_focus.value
        )
        imgui.push_style_var(imgui.StyleVar_.window_padding, imgui.ImVec2(0.0, 0.0))
        imgui.push_style_var(imgui.StyleVar_.window_border_size, 0.0)
        visible = imgui.internal.begin_viewport_side_bar(
            "Status###application_status",
            viewport,
            imgui.Dir.down,
            height,
            flags,
        )
        if visible:
            origin = imgui.get_window_pos()
            size = imgui.get_window_size()
            if loading:
                state = "static"
                sim_time = 0.0
                sim_step = 0
            else:
                caps = self.session.adapter.caps
                state = (
                    "static"
                    if not caps.simulation
                    else "paused"
                    if self.session.paused and not self.session.state_take_playing
                    else "running"
                )
                sim_time = float(self.session.frame.time)
                sim_step = int(self.session.frame.step)
            recording = self.recording
            status_layout = draw_status(
                ImguiDraw2D(),
                (origin.x, origin.y),
                size.x,
                size.y,
                self.theme,
                scale,
                selected="",
                state=state,
                sim_time=sim_time,
                step=sim_step,
                metric_mode=self._status_metric_mode,
                backend=(
                    "OpenGL" if self.backend.caps.name == "opengl" else str(self.backend.caps.name)
                ),
                dt=_simulation_timestep(self.session.adapter, loading=loading),
                fps=self._frame_rate.value,
                physics_hz=self.session.frame.physics_hz,
                show_physics=self.session.adapter.caps.simulation,
                recording_phase=recording.phase.value,
                recording_duration=recording.duration,
                countdown_remaining=recording.countdown_remaining,
                recording_surface=recording.surface.value,
                tool_hints=self._status_tool_hints(loading=loading),
                labels=self._viewport_labels,
                pixel_size=self.window.pixels_to_points(1.0),
            )
            if status_layout.metric_rect is not None and not loading:
                x0, y0, x1, y1 = status_layout.metric_rect
                imgui.set_cursor_screen_pos(imgui.ImVec2(x0, y0))
                if imgui.invisible_button(
                    "##status_simulation_metric",
                    imgui.ImVec2(x1 - x0, y1 - y0),
                ):
                    self._toggle_status_metric()
                hovered = imgui.is_item_hovered()
                if hovered and self.input_bindings.pointer_match(
                    PointerAction.STATUS_COPY, self.input_bindings.pointer_frame(), press=True
                ):
                    imgui.set_clipboard_text(status_layout.metric_exact)
                if hovered:
                    switch = (
                        self._viewport_labels.show_steps
                        if self._status_metric_mode == "time"
                        else self._viewport_labels.show_time
                    )
                    imgui.set_tooltip(f"{switch} · {self._viewport_labels.copy_exact}")
            if status_layout.recording_pause_rect is not None and not loading:
                x0, y0, x1, y1 = status_layout.recording_pause_rect
                imgui.set_cursor_screen_pos(imgui.ImVec2(x0, y0))
                if imgui.invisible_button(
                    "##status_recording_pause", imgui.ImVec2(x1 - x0, y1 - y0)
                ):
                    if recording.phase is RecordingPhase.PAUSED:
                        self.resume_recording()
                    else:
                        self.pause_recording()
                if imgui.is_item_hovered():
                    imgui.set_tooltip(
                        self.localizer.text(
                            "Resume Recording"
                            if recording.phase is RecordingPhase.PAUSED
                            else "Pause Recording"
                        )
                    )
            if status_layout.recording_stop_rect is not None and not loading:
                x0, y0, x1, y1 = status_layout.recording_stop_rect
                imgui.set_cursor_screen_pos(imgui.ImVec2(x0, y0))
                if imgui.invisible_button(
                    "##status_recording_stop", imgui.ImVec2(x1 - x0, y1 - y0)
                ):
                    self.stop_recording()
                if imgui.is_item_hovered():
                    imgui.set_tooltip(
                        self.localizer.text(
                            "Cancel Recording"
                            if recording.phase is RecordingPhase.COUNTDOWN
                            else "Stop Recording"
                        )
                    )
        imgui.end()
        imgui.pop_style_var(2)

    def _draw_model_drop_overlay(self, overlay: ImguiDraw2D) -> None:
        empty = not self._has_scene_content()
        notice = self._model_drop_notice if time.monotonic() < self._model_drop_notice_until else ""
        caps = self.session.adapter.caps
        dragging = self.window.file_drag_active and (bool(_model_filters(caps)) or caps.scene_files)
        if not empty and not notice and not dragging:
            return
        empty_hint = (
            "Drop a .mojive.json scene here\nFile > Open Scene...  ·  Entity > Create"
            if caps.scene_files
            else (
                "Drop a supported model here\nFile > Open Model..."
                if _model_filters(caps)
                else "Waiting for scene data"
            )
        )
        if dragging:
            message = (
                "Release to add model(s)"
                if caps.model_composition and self.session.scene_models
                else "Release to open model"
            )
        else:
            message = notice or empty_hint
        self._draw_center_notice(overlay, message, border=dragging)

    def _sync_session_status(self) -> None:
        revision = int(getattr(self.session, "message_revision", 0))
        if revision == self._seen_message_revision:
            return
        self._seen_message_revision = revision
        self.output.publish(
            self.session.last_message,
            level=getattr(self.session, "last_message_level", "info"),
            duration=self.viewport_overlays.status_duration,
            copy_text=getattr(self.session, "last_message_copy_text", None),
        )

    def _draw_viewport_status(self, overlay: ImguiDraw2D) -> None:
        self._status_path_bounds = None
        message = self.output.active_status()
        if message is None:
            return
        x, y, width, height = self._viewport_rect
        pad = 14.0 * self.window.style_scale
        # File reveal is available only over the status text with the shortcut held.
        text = fit_text(overlay, self.localizer.text(message.text), max(0.0, width - 2 * pad))
        position = (x + pad, y + height - pad - imgui.get_text_line_height())
        size = imgui.calc_text_size(text)
        if getattr(self, "_status_path_sequence", None) != message.sequence:
            self._status_path_sequence = message.sequence
            self._status_path = message_path(message.text, message.copy_text)
        path = self._status_path
        if path is not None:
            self._status_path_bounds = (*position, position[0] + size.x, position[1] + size.y)
        if path is not None and imgui.is_mouse_hovering_rect(
            position, (position[0] + size.x, position[1] + size.y)
        ):
            imgui.set_tooltip(self.localizer.text("Ctrl+click to reveal file") + "\n" + str(path))
            io = imgui.get_io()
            if (io.key_ctrl or io.key_super) and imgui.is_mouse_clicked(0):
                reveal_path(path)
        overlay.text(
            position,
            (*self.theme.text[:3], 0.55),
            text,
        )

    def _draw_center_notice(
        self,
        overlay: ImguiDraw2D,
        message: str,
        *,
        border: bool = False,
    ) -> None:
        lines = message.splitlines()
        sizes = [overlay.text_size(line) for line in lines]
        scale = self.window.style_scale
        pad_x, pad_y = 18.0 * scale, 12.0 * scale
        width = max(size[0] for size in sizes) + 2.0 * pad_x
        height = sum(size[1] for size in sizes) + 2.0 * pad_y + (len(lines) - 1) * 3.0 * scale
        x, y, w, h = self._viewport_rect
        left = x + (w - width) * 0.5
        top = y + (h - height) * 0.5
        if border:
            overlay.rect(
                (x + 3.0 * scale, y + 3.0 * scale),
                (x + w - 3.0 * scale, y + h - 3.0 * scale),
                (0.95, 0.68, 0.24, 0.95),
                2.0 * scale,
                rounding=8.0 * scale,
            )
        overlay.rect_filled(
            (left, top),
            (left + width, top + height),
            (0.08, 0.09, 0.11, 0.88),
            rounding=7.0 * scale,
        )
        cursor_y = top + pad_y
        for line, size in zip(lines, sizes, strict=True):
            overlay.text(
                (left + (width - size[0]) * 0.5, cursor_y),
                (0.93, 0.94, 0.95, 1.0),
                line,
            )
            cursor_y += size[1] + 3.0 * scale
