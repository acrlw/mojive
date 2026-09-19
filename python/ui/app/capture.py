"""App: capture."""

from __future__ import annotations

import math
import time
from concurrent.futures import Future
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

import numpy as np
from imgui_bundle import imgui

from mojive import commands as cmd
from mojive.capture import CaptureSurface, RecordingInfo, RecordingPhase
from mojive.config import (
    RecordingConfig,
)
from mojive.ui.take_video import TakeVideo

from .support import log


class _Capture:
    """Private capture methods of ViewerApp; state belongs to its owner."""

    @property
    def recording(self) -> RecordingInfo:
        """Return the current interactive recording state."""

        return RecordingInfo(
            error=getattr(self, "_recording_error", ""),
            phase=getattr(self, "_viewport_recording_phase", RecordingPhase.IDLE),
            surface=getattr(self, "_viewport_recording_surface", CaptureSurface.SCENE),
            path=getattr(self, "_viewport_recording_path", None),
            fps=getattr(self, "_viewport_recording_fps", 60.0),
            frames=getattr(self, "_viewport_recording_frames", 0),
            duration=getattr(self, "_viewport_recording_duration", 0.0),
            countdown_remaining=(
                max(0.0, self._recording_deadline - time.monotonic())
                if getattr(self, "_viewport_recording_phase", None) is RecordingPhase.COUNTDOWN
                else 0.0
            ),
        )

    def set_recording_config(self, value: RecordingConfig, *, persist: bool = True) -> None:
        """Update interactive recording defaults without changing display pacing."""
        self.recording_config = RecordingConfig.from_mapping(asdict(value))
        if persist:
            self.localizer.set_preferences({"recording": asdict(self.recording_config)})

    def set_take_pause_at_end(self, enabled: bool, *, persist: bool = True) -> None:
        self.session.submit(cmd.SetStateTakePauseAtEnd(enabled))
        if persist:
            self.localizer.set_preferences({"take_pause_at_end": bool(enabled)})

    @staticmethod
    def _capture_output(surface: CaptureSurface, suffix: str) -> Path:
        stem = {
            CaptureSurface.SCENE: "scene",
            CaptureSurface.VIEWPORT: "viewport-ui",
            CaptureSurface.WINDOW: "window",
        }[surface]
        return (
            Path("output") / f"{stem}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:12]}{suffix}"
        )

    def request_capture(
        self,
        output: str | Path | None = None,
        *,
        surface: CaptureSurface | str = CaptureSurface.SCENE,
    ) -> Path:
        """Capture one future presented frame; pure scene output is the default."""
        if getattr(self, "_released", False):
            raise RuntimeError("The viewer is closed")
        surface = CaptureSurface(surface)
        path = Path(output) if output is not None else self._capture_output(surface, ".png")
        self._capture_requests.append((path, surface))
        return path

    def request_capture_async(
        self, output=None, *, surface=CaptureSurface.VIEWPORT, memory: bool = False, out=None
    ) -> Future:
        """Return a presented capture as a file or an owned in-memory image."""
        if memory and output is not None:
            raise ValueError("An in-memory capture cannot also specify an output path")
        if out is not None and not memory:
            raise ValueError("A capture destination requires memory=True")
        if out is not None and (
            out.dtype != np.uint8 or not out.flags.writeable or not out.flags.c_contiguous
        ):
            raise ValueError("out must be a writable C-contiguous uint8 array")
        surface = CaptureSurface(surface)
        path = (
            None
            if memory
            else (Path(output) if output is not None else self._capture_output(surface, ".png"))
        )
        future = Future()
        if self._released:
            future.set_exception(RuntimeError("The viewer is closed"))
        else:
            self._capture_tasks.append((path, surface, future, out))
        return future

    def start_recording(
        self,
        output: str | Path | None = None,
        *,
        surface: CaptureSurface | str | None = None,
        fps: float | None = None,
        countdown: float | None = None,
    ) -> Path:
        """Start an interactive recording of the scene, viewport UI, or full window."""
        if getattr(self, "_released", False):
            raise RuntimeError("The viewer is closed")
        if self.recording.active:
            raise RuntimeError("a viewer recording is already active")
        config = getattr(self, "recording_config", RecordingConfig())
        surface = CaptureSurface(config.surface if surface is None else surface)
        fps = float(config.fps if fps is None else fps)
        countdown = float(config.countdown if countdown is None else countdown)
        if not np.isfinite(fps) or fps <= 0.0:
            raise ValueError("frame rate must be finite and positive")
        if not np.isfinite(countdown) or countdown < 0.0:
            raise ValueError("countdown must be finite and nonnegative")
        if surface is CaptureSurface.SCENE:
            target = getattr(self.backend, "target", None)
            if target is None or not hasattr(target, "read_color"):
                raise RuntimeError("scene recording is unavailable for this backend")
        path = Path(output) if output is not None else self._capture_output(surface, ".mp4")
        self._viewport_recorder = None
        self._recording_error = ""
        self._viewport_recording_path = path
        self._viewport_recording_surface = surface
        self._viewport_recording_fps = fps
        self._viewport_recording_frames = 0
        self._viewport_recording_duration = 0.0
        self._viewport_record_elapsed = 0.0
        self._recording_deadline = time.monotonic() + countdown
        self._viewport_recording_phase = RecordingPhase.COUNTDOWN
        self._recording_run_simulation = (
            config.run_simulation and self.session.adapter.caps.clock_control
        )
        self._active_recording_config = config
        return path

    def _advance_recording_countdown(self) -> None:
        self._poll_recording()
        if self._viewport_recording_phase is not RecordingPhase.COUNTDOWN:
            return
        if (
            time.monotonic() < self._recording_deadline
            or getattr(self, "_popup_owned_frame", False)
            or getattr(self, "_model_load_future", None) is not None
        ):
            return
        # Transition before UI construction so neither the countdown nor the
        # menu frame can become the first encoded image, including a zero delay.
        self._viewport_recording_phase = RecordingPhase.RECORDING
        self._viewport_record_elapsed = 1.0 / self._viewport_recording_fps
        self._recording_first_frame = True

    def start_take_video(
        self,
        output: str | Path | None = None,
        *,
        surface: CaptureSurface | str | None = None,
        fps: float | None = None,
        countdown: float | None = None,
        end_hold: float | None = None,
    ) -> Path:
        """Record the whole take once, waiting at the first and final poses."""
        session = self.session
        if session.state_take_recording or not session.state_take_times:
            raise ValueError("Stop recording a simulation take before recording its video")
        config = self.recording_config
        end_hold = float(config.end_hold if end_hold is None else end_hold)
        if not math.isfinite(end_hold) or end_hold < 0:
            raise ValueError("end hold must be finite and nonnegative")
        path = self.start_recording(output, surface=surface, fps=fps, countdown=countdown)
        self._recording_run_simulation = False
        result = session.submit(cmd.SeekStateTake(0))
        if not result.ok:
            self.stop_recording(report=False)
            raise RuntimeError(result.message)
        self._take_video = TakeVideo(
            len(session.state_take_times),
            session.structure_generation,
            math.ceil(end_hold * self._viewport_recording_fps),
            take_id=session.active_state_take_id,
        )
        return path

    def _validate_take_video(self) -> bool:
        take = getattr(self, "_take_video", None)
        if take is None:
            return False
        session = self.session
        if (
            session.structure_generation != take.structure_generation
            or session.active_state_take_id != take.take_id
            or len(session.state_take_times) != take.frame_count
            or session.state_take_recording
            or not session.paused
            or session.state_take_cursor != take.cursor
            or session.state_take_playing != take.playing
        ):
            # A transport command or scene edit ends the export before an
            # unrelated pose can be encoded into the same video.
            self.stop_recording()
            return False
        return True

    def _prepare_take_video(self, dt: float) -> float:
        if not self._validate_take_video():
            return dt
        take = self._take_video
        take.frame_due = False
        if self._viewport_recording_phase is not RecordingPhase.RECORDING:
            return 0.0
        if not take.started:
            result = self.session.submit(cmd.PlayStateTake(loop=False, pause_at_end=True))
            if not result.ok:
                self.stop_recording(report=False)
                self.session.report_message(result.message, level="error")
                return 0.0
        return take.prepare_frame(dt, self._viewport_recording_fps)

    def _draw_recording_countdown(self) -> None:
        if self._viewport_recording_phase is not RecordingPhase.COUNTDOWN:
            return
        recording = self.recording
        x, y, width, _height = self._viewport_rect
        scale = self.window.style_scale
        capsule = self._playback_widget_rect
        center_x = (capsule[0] + capsule[2]) * 0.5 if capsule else x + width * 0.5
        top = capsule[3] + 8 * scale if capsule else y + 12 * scale
        imgui.push_style_var(imgui.StyleVar_.window_padding, (10 * scale, 8 * scale))
        imgui.push_style_var(imgui.StyleVar_.item_spacing, (10 * scale, 6 * scale))
        imgui.set_next_window_pos(
            (min(x + width - 85 * scale, max(x + 85 * scale, center_x)), top),
            imgui.Cond_.always,
            (0.5, 0),
        )
        flags = (
            imgui.WindowFlags_.always_auto_resize
            | imgui.WindowFlags_.no_title_bar
            | imgui.WindowFlags_.no_saved_settings
            | imgui.WindowFlags_.no_move
            | imgui.WindowFlags_.no_docking
            | imgui.WindowFlags_.no_focus_on_appearing
        )
        imgui.begin("##recording_countdown", None, flags)
        remaining = math.ceil(recording.countdown_remaining)
        imgui.text(self.localizer.text("Recording starts in"))
        if remaining:
            imgui.push_font(None, imgui.get_font_size() * 1.6)
            imgui.text(f"{remaining} {self.localizer.text('s')}")
            imgui.pop_font()
        else:
            imgui.text(self.localizer.text("Close menus to begin"))
        imgui.same_line()
        if imgui.button(self.localizer.text("Cancel")):
            self.stop_recording()
        imgui.end()
        imgui.pop_style_var(2)

    def pause_recording(self) -> bool:
        """Pause an active interactive recording without finalizing its video."""

        if self._viewport_recording_phase is not RecordingPhase.RECORDING:
            return False
        self._viewport_recording_phase = RecordingPhase.PAUSED
        self._viewport_record_elapsed = 0.0
        take = getattr(self, "_take_video", None)
        if take is not None:
            self.session.submit(cmd.PauseStateTake())
            take.playing = False
            take.frame_due = False
            take.elapsed = 0.0
        return True

    def resume_recording(self) -> bool:
        """Resume a paused interactive recording."""

        if self._viewport_recording_phase is not RecordingPhase.PAUSED:
            return False
        self._viewport_recording_phase = RecordingPhase.RECORDING
        self._viewport_record_elapsed = 1.0 / self._viewport_recording_fps
        take = getattr(self, "_take_video", None)
        if take is not None and take.started and take.tail_frames is None:
            result = self.session.submit(cmd.PlayStateTake(loop=False, pause_at_end=True))
            if not result.ok:
                self.stop_recording(report=False)
                self.session.report_message(result.message, level="error")
                return False
            take.playing = self.session.state_take_playing
        return True

    def stop_recording(self, *, report: bool = True, raise_on_error: bool = False) -> Path | None:
        """Finalize the active interactive recording and return its destination."""

        if not self.recording.active:
            if raise_on_error and self.recording.error:
                raise RuntimeError(self.recording.error)
            return None
        recorder = self._viewport_recorder
        path = self._viewport_recording_path
        frames = self._viewport_recording_frames
        self._viewport_recorder = None
        self._viewport_recording_path = None
        self._viewport_record_elapsed = 0.0
        self._viewport_recording_phase = RecordingPhase.IDLE
        self._recording_run_simulation = False
        take, self._take_video = getattr(self, "_take_video", None), None
        if take is not None and self.session.state_take_playing:
            self.session.submit(cmd.PauseStateTake())
        if recorder is None and frames == 0:
            if report:
                self.session.report_message(self.localizer.text("Recording canceled"), level="info")
            return None
        try:
            recorder.close()
        except Exception as exc:
            self._recording_error = str(exc)
            if report:
                self.session.report_message(
                    f"{self.localizer.text('Recording failed')}: {exc}", level="error"
                )
            if raise_on_error:
                raise
            return path
        if report and path is not None:
            destination = path.resolve()
            # Interactive export receipts must reach the caller's terminal even
            # when an embedding application has left runtime logging disabled.
            print(f"Saved video to {destination}", flush=True)
            self.session.report_message(
                f"{self.localizer.text('Saved video to')} {destination}",
                level="success",
                duration=8.0,
                copy_text=str(destination),
            )
            self._copy_saved_capture(destination, "file")
        return path

    def _request_recording_stop(self) -> Path | None:
        """Finalize UI recordings in the background; keep public stop synchronous."""
        from mojive.capture.video_queue import BufferedVideoRecorder

        recorder = self._viewport_recorder
        if recorder is None:
            return self.stop_recording()
        if self._viewport_recording_phase is RecordingPhase.FINALIZING:
            return self._viewport_recording_path
        if not isinstance(recorder, BufferedVideoRecorder):
            recorder = self._viewport_recorder = BufferedVideoRecorder(recorder)
        self._viewport_recording_phase = RecordingPhase.FINALIZING
        self._recording_run_simulation = False
        take, self._take_video = getattr(self, "_take_video", None), None
        if take is not None and self.session.state_take_playing:
            self.session.submit(cmd.PauseStateTake())
        recorder.begin_close()
        return self._viewport_recording_path

    def _poll_recording(self) -> None:
        from mojive.capture.video_queue import BufferedVideoRecorder

        recorder = getattr(self, "_viewport_recorder", None)
        if not isinstance(recorder, BufferedVideoRecorder):
            return
        if self._viewport_recording_phase is RecordingPhase.FINALIZING:
            if recorder.poll_close():
                self.stop_recording()
            return
        try:
            recorder.check()
        except Exception:
            # close() preserves encoder and finalization errors in one receipt.
            self._request_recording_stop()
        else:
            self._start_recording_simulation()

    def _start_recording_simulation(self) -> None:
        if getattr(self, "_model_load_future", None) is not None or getattr(
            self, "_model_load_queue", ()
        ):
            return
        if (
            not getattr(self, "_recording_run_simulation", False)
            or self._viewport_recording_phase is not RecordingPhase.RECORDING
        ):
            return
        recorder = self._viewport_recorder
        if recorder is None or recorder.written_frames == 0:
            return
        # Acknowledging the initial pose preserves start ordering without making
        # the viewer wait for FFmpeg to consume its first pipe buffer.
        self._recording_run_simulation = False
        result = self.session.submit(cmd.Play())
        if not result.ok:
            self.stop_recording(report=False)
            self.session.report_message(result.message, level="error")

    def _copy_saved_capture(self, path: Path, kind: str) -> None:
        clipboard = getattr(self, "_capture_clipboard", None)
        if clipboard is not None and self.recording_config.copy_to_clipboard:
            clipboard.submit(path, kind)

    def _poll_capture_clipboard(self) -> None:
        clipboard = getattr(self, "_capture_clipboard", None)
        result = clipboard.poll() if clipboard is not None else None
        if result is None:
            return
        saved = "Saved capture to" if result.kind == "image" else "Saved video to"
        copied = (
            "Image copied to clipboard" if result.kind == "image" else "File copied to clipboard"
        )
        detail = (
            f"{self.localizer.text('Clipboard copy failed')}: {result.error}"
            if result.error
            else self.localizer.text(copied)
        )
        self.session.report_message(
            f"{self.localizer.text(saved)} {result.path} · {detail}",
            level="warning" if result.error else "success",
            duration=8.0,
            copy_text=str(result.path),
        )

    def _needs_presented_readback(self, dt: float) -> bool:
        if any(
            surface is not CaptureSurface.SCENE
            for _, surface in getattr(self, "_capture_requests", [])
        ):
            return True
        if any(
            surface is not CaptureSurface.SCENE
            for _, surface, _, _ in getattr(self, "_capture_tasks", [])
        ):
            return True
        take = getattr(self, "_take_video", None)
        if take is not None:
            return take.frame_due and self._viewport_recording_surface is not CaptureSurface.SCENE
        return (
            self._viewport_recording_phase is RecordingPhase.RECORDING
            and self._viewport_recording_surface is not CaptureSurface.SCENE
            and self._viewport_record_elapsed + max(0.0, min(float(dt), 0.1))
            >= 1.0 / self._viewport_recording_fps
        )

    def _present_frame(self, dt: float) -> None:
        presented = self.window.end_frame(readback=self._needs_presented_readback(dt))
        self._finish_capture_and_recording(presented, dt)
        self._poll_capture_clipboard()
        self._finish_model_load_frame()

    def _finish_model_load_frame(self) -> None:
        completion = self._model_load_completion
        if completion is None:
            return
        self._model_load_completion = None
        now = time.monotonic()
        log.info(
            "{} {} in {:.3f}s (source {:.3f}s, resources {:.3f}s, first frame submitted {:.3f}s)",
            completion.message or "Loaded",
            completion.job.path,
            now - completion.started,
            completion.prepared - completion.started,
            completion.resources_ready - completion.prepared,
            now - completion.resources_ready,
        )

    def _surface_image(
        self, surface: CaptureSurface, presented: np.ndarray | None, *, out=None
    ) -> np.ndarray:
        if surface is CaptureSurface.SCENE:
            x, y, width, height = self.window.points_to_pixels(self._viewport_rect)
            size = getattr(self, "_fixed_render_size", None) or (
                max(1, round(x + width) - round(x)),
                max(1, round(y + height) - round(y)),
            )
            return self._scene_capture.read(
                self.backend, self.session, self._camera_view(), size=size, out=out
            )
        else:
            if presented is None:
                raise RuntimeError("window readback did not produce an image")
            image = np.asarray(presented)[::-1, :, :3]
            if surface is CaptureSurface.VIEWPORT:
                x, y, width, height = self.window.points_to_pixels(self._viewport_rect)
                x0 = max(0, round(x))
                y0 = max(0, round(y))
                x1 = min(image.shape[1], round(x + width))
                y1 = min(image.shape[0], round(y + height))
                if x1 <= x0 or y1 <= y0:
                    raise RuntimeError("the viewport is outside the presented window")
                image = image[y0:y1, x0:x1]
        if out is None:
            return np.ascontiguousarray(image)
        if out.shape != image.shape or out.dtype != image.dtype or not out.flags.writeable:
            raise ValueError(f"out must be a writable {image.dtype} array with shape {image.shape}")
        np.copyto(out, image)
        return out

    def _finish_capture_and_recording(self, presented: np.ndarray | None, dt: float) -> None:
        images: dict[CaptureSurface, np.ndarray] = {}

        def image_for(surface: CaptureSurface) -> np.ndarray:
            if surface not in images:
                images[surface] = self._surface_image(surface, presented)
            return images[surface]

        requests, self._capture_requests = getattr(self, "_capture_requests", []), []
        for path, surface in requests:
            try:
                from PIL import Image

                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(image_for(surface), "RGB").save(path)
            except Exception as exc:
                self.session.report_message(
                    f"{self.localizer.text('Capture failed')}: {exc}", level="error"
                )
            else:
                self.session.report_message(
                    f"{self.localizer.text('Saved capture to')} {path}",
                    level="success",
                    copy_text=str(path.resolve()),
                )
                self._copy_saved_capture(path, "image")
        tasks, self._capture_tasks = getattr(self, "_capture_tasks", []), []
        for path, surface, future, out in tasks:
            if not future.set_running_or_notify_cancel():
                continue
            try:
                from PIL import Image

                image = (
                    self._surface_image(surface, presented, out=out)
                    if out is not None
                    else image_for(surface)
                )
                if path is not None:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(image, "RGB").save(path)
                    self._copy_saved_capture(path, "image")
                future.set_result(
                    {
                        **(
                            {"path": str(path.resolve())}
                            if path is not None
                            else {"image": image if out is not None else image.copy()}
                        ),
                        "scope": surface.value,
                        "mode": "rgb",
                        "orientation": "top_left",
                        "shape": list(image.shape),
                        "dtype": str(image.dtype),
                        "document": {
                            "id": self.session.document_id,
                            "revision": self.session.document_revision,
                        },
                        "structure_generation": self.session.structure_generation,
                        "step": self.session.frame.step,
                        "time": self.session.frame.time,
                    }
                )
            except Exception as exc:
                future.set_exception(exc)
        if self._viewport_recording_phase is not RecordingPhase.RECORDING:
            return
        take = getattr(self, "_take_video", None)
        if take is not None and not self._validate_take_video():
            return
        period = 1.0 / self._viewport_recording_fps
        if take is not None:
            count = int(take.frame_due)
        elif getattr(self, "_recording_first_frame", False):
            self._recording_first_frame = False
            count = min(3, int(self._viewport_record_elapsed / period))
        else:
            self._viewport_record_elapsed += max(0.0, min(float(dt), 0.1))
            count = min(3, int(self._viewport_record_elapsed / period))
        if count <= 0:
            return
        try:
            image = image_for(self._viewport_recording_surface)
            if self._viewport_recorder is None:
                from mojive.capture.recording import VideoRecorder

                config = self._active_recording_config
                h264 = self._viewport_recording_path.suffix.lower() != ".wmv"
                self._viewport_recorder = VideoRecorder(
                    self._viewport_recording_path,
                    (int(image.shape[1]), int(image.shape[0])),
                    fps=self._viewport_recording_fps,
                    pixel_format=config.pixel_format,
                    crf=config.crf
                    if config.rate_control == "quality" and (h264 or config.crf != 25)
                    else None,
                    bitrate=round(config.bitrate_mbps * 1_000_000)
                    if config.rate_control == "bitrate"
                    else None,
                    preset=config.encoder_preset
                    if h264 or config.encoder_preset != "medium"
                    else None,
                )
                if take is None:
                    from mojive.capture.video_queue import BufferedVideoRecorder

                    self._viewport_recorder = BufferedVideoRecorder(self._viewport_recorder)
            elif tuple(self._viewport_recorder.size) != (image.shape[1], image.shape[0]):
                raise RuntimeError("capture size changed while recording")
            for _ in range(count):
                self._viewport_recorder.append(image)
            self._viewport_recording_frames += count
            self._viewport_recording_duration = (
                self._viewport_recording_frames / self._viewport_recording_fps
            )
        except Exception as exc:
            self.stop_recording(report=False)
            self._recording_error = str(exc) + (
                f"; finalization: {self._recording_error}" if self._recording_error else ""
            )
            self.session.report_message(
                f"{self.localizer.text('Recording stopped')}: {exc}", level="error"
            )
            return
        if take is not None:
            take.frame_due = False
            if take.captured(at_end=take.cursor == take.frame_count - 1):
                self.stop_recording()
        else:
            self._viewport_record_elapsed -= count * period
            self._start_recording_simulation()

    def _toggle_viewport_recording(self) -> None:
        if self.recording.active:
            self._request_recording_stop()
            return
        try:
            self.start_recording()
        except Exception as exc:
            self.session.report_message(
                f"{self.localizer.text('Recording failed')}: {exc}", level="error"
            )

    def _record_viewport_frame(self, dt: float) -> None:
        """Compatibility hook for integrations that manually pump raw scene frames."""

        self._advance_recording_countdown()
        self._finish_capture_and_recording(None, dt)

    def _stop_viewport_recording(self, *, report: bool = True) -> None:
        self.stop_recording(report=report)

    def _toggle_state_take_recording(self) -> None:
        self.panels.load("keyframes").toggle_recording(self._panel_context())

    def _toggle_playback(self, *, source: str | None = None) -> None:
        replay = self.session.replay_info
        if replay is not None and getattr(self, "_take_video", None) is None:
            self.session.submit(cmd.SetReplayPlayback(paused=not replay.paused))
            return
        if getattr(self, "_take_video", None) is not None:
            if self.recording.phase is RecordingPhase.PAUSED:
                self.resume_recording()
            else:
                self.pause_recording()
        elif self.session.state_take_recording:
            self._toggle_state_take_recording()
        elif self.session.state_take_playing or not self.session.paused:
            self.session.submit(cmd.Pause())
        elif (source or self.session.playback_source) == "take" and self.session.state_take_times:
            self.session.submit(cmd.PlayStateTake())
        else:
            self.session.submit(cmd.Play())

    def _reset_playback(self) -> None:
        if self.session.replay_info is not None:
            self.session.submit(cmd.SeekReplay(0))
            return
        if getattr(self, "_take_video", None) is not None:
            self.stop_recording()
        if self.session.state_take_recording:
            self._toggle_state_take_recording()
            if self.session.state_take_recording:
                return
        if (self.session.state_take_playing or not self.session.paused) and not self.session.submit(
            cmd.Pause()
        ).ok:
            return
        self.session.submit(cmd.Reset())
