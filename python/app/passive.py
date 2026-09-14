"""Passive MuJoCo viewing with independent physics and display ownership."""

from __future__ import annotations

import math
import multiprocessing as mp
import operator
import threading
import time
import traceback
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import replace
from pathlib import Path
from queue import Empty
from typing import TYPE_CHECKING

import numpy as np

from mojive.capture import CaptureSurface, RecordingInfo
from mojive.capture.shared_image import SharedImage
from mojive.config import CameraTrackingConfig, LayoutConfig, RecordingConfig, ViewerConfig
from mojive.session.rates import StepRate

from .passive_input import PassiveAction, PassiveEvent

if TYPE_CHECKING:
    from mojive.ui import ToolHint


class _Mailbox:
    """One bounded latest-state slot; no render work runs under its lock."""

    def __init__(self, context, state_size, nu, nbody):
        self.lock = context.Lock()
        self.state = context.RawArray("d", state_size)
        self.ctrl = context.RawArray("d", nu)
        self.ctrl_changed = context.RawArray("B", nu)
        self.force = context.RawArray("d", nbody * 6)
        self.force_changed = context.RawArray("B", nbody)
        self.sequence = context.RawValue("q", 0)
        self.step = context.RawValue("q", 0)
        self.physics_hz = context.RawValue("d", math.nan)
        self.rendered = context.RawValue("q", 0)
        self.displayed_sequence = context.RawValue("q", 0)
        self.displayed_time = context.RawValue("d", 0)

    def arrays(self):
        return (
            np.frombuffer(self.state, np.float64),
            np.frombuffer(self.ctrl, np.float64),
            np.frombuffer(self.ctrl_changed, np.uint8),
            np.frombuffer(self.force, np.float64).reshape(-1, 6),
            np.frombuffer(self.force_changed, np.uint8),
        )


class PassiveViewer:
    """A caller-owned MuJoCo simulation with a separately scheduled viewer process.

    Create through :func:`launch_passive`. ``sync()`` exchanges only the latest
    integration state and UI input, at most ``max_fps`` times per second. It never
    renders or advances physics. The window remains responsive between calls.
    ``model`` and ``data`` retain their caller-side identity; the display process
    owns private copies. Model structure and parameters are fixed at launch.
    """

    def __init__(self, model, data, *, max_fps, timeout, viewer_options, control_writeback=True):
        import mujoco

        if not isinstance(model, mujoco.MjModel):
            raise TypeError("model must be a mujoco.MjModel")
        if not isinstance(data, mujoco.MjData):
            raise TypeError("data must be a mujoco.MjData")
        if data.model is not model:
            raise ValueError("data was created for a different MuJoCo model")
        if not math.isfinite(max_fps) or max_fps <= 0:
            raise ValueError("max_fps must be finite and positive")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        if not isinstance(control_writeback, bool):
            raise TypeError("control_writeback must be a bool")
        self._model, self._data = model, data
        self._max_fps = float(max_fps)
        self._timeout = float(timeout)
        self._lock = threading.RLock()
        self._closed = False
        self._next_sync = 0.0
        self._step = 0
        self._step_known = False
        self._step_rate = StepRate()
        self._state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
        context = mp.get_context("spawn")
        self._mailbox = _Mailbox(
            context, mujoco.mj_stateSize(model, self._state_spec), model.nu, model.nbody
        )
        self._state, self._ctrl, self._ctrl_changed, self._force, self._force_changed = (
            self._mailbox.arrays()
        )
        self._applied_force = np.zeros(model.nbody, dtype=bool)
        self._stop = context.Event()
        self._connection, child = context.Pipe()
        self._completion, completion_sender = context.Pipe(duplex=False)
        self._events = context.Queue(maxsize=64)
        self._process = context.Process(
            target=_run_viewer,
            args=(
                model,
                self._mailbox,
                self._stop,
                child,
                completion_sender,
                self._events,
                self.max_fps,
                control_writeback,
                viewer_options,
            ),
            name="mojive-passive",
            daemon=True,
        )
        mujoco.mj_getState(model, data, self._state, self._state_spec)
        self._mailbox.sequence.value = 1
        try:
            self._process.start()
            child.close()
            completion_sender.close()
            self._receive()
        except BaseException:
            child.close()
            completion_sender.close()
            self.close()
            raise

    @property
    def model(self):
        """Return the exact MuJoCo model supplied by the caller."""
        return self._model

    @property
    def data(self):
        """Return the exact MuJoCo data supplied by the caller."""
        return self._data

    @property
    def max_fps(self) -> float:
        """Return the configured display and state-publication frequency limit."""
        return self._max_fps

    def lock(self):
        """Serialize caller threads accessing this handle and its physics state."""
        return self._lock

    def is_running(self) -> bool:
        """Return false after window close, worker failure, or explicit close."""
        return not self._closed and not self._stop.is_set() and self._process.is_alive()

    def sync(self, *, step: int | None = None) -> None:
        """Publish current state and apply UI input without waiting for a rendered frame.

        ``step`` optionally supplies an episode step counter for physics-rate
        measurement and capture metadata. Without it, the rate stays unknown.
        MuJoCo does not expose that counter; its default is zero.
        Control slider edits are applied once. Perturbation owns ``xfrc_applied``
        only on the dragged body, including one clear when the drag finishes.
        """
        with self._lock:
            if not self.is_running():
                return
            now = time.monotonic()
            self._step_known = step is not None
            if step is not None:
                step = operator.index(step)
                if step < 0:
                    raise ValueError("step must be nonnegative")
                self._step = step
                self._step_rate.update(step, float(self.data.time), now=now)
            if now < self._next_sync:
                return
            if self._publish():
                self._next_sync = now + 1.0 / self.max_fps

    def _publish(self):
        import mujoco

        # A failed display process must not leave a physics loop waiting on its lock.
        if not self._mailbox.lock.acquire(timeout=0.01):
            return False
        try:
            np.copyto(self.data.ctrl, self._ctrl, where=self._ctrl_changed.astype(bool))
            self._ctrl_changed.fill(0)
            changed = self._force_changed != 0
            np.copyto(self.data.xfrc_applied, self._force, where=changed[:, None])
            np.equal(self._force_changed, 1, out=self._applied_force)
            self._force_changed[self._force_changed == 2] = 0
            mujoco.mj_getState(self.model, self.data, self._state, self._state_spec)
            self._mailbox.sequence.value += 1
            self._mailbox.step.value = self._step
            rate = (
                self._step_rate.update(self._step, float(self.data.time))
                if self._step_known
                else None
            )
            self._mailbox.physics_hz.value = rate if rate is not None else math.nan
            return True
        finally:
            self._mailbox.lock.release()

    @property
    def stats(self) -> dict:
        """Read display progress independently of the caller's physics rate."""
        with self._lock:
            return self._request("stats")

    def set_camera(self, view) -> None:
        """Adopt a backend-neutral camera view in the display process."""
        self._request("camera", view)

    def track_body(self, body: str | int | None) -> None:
        """Follow a MuJoCo body by name or index in the display process; None stops."""
        self._request("track_body", body)

    def configure_tracking(self, value: CameraTrackingConfig, *, persist: bool = False) -> None:
        """Set a CameraTrackingConfig without changing the caller's physics cadence."""
        self._request("configure_tracking", (value, persist))

    @property
    def recording(self) -> RecordingInfo:
        """Read video progress; this does not report or change the physics clock."""
        return self._request("recording")

    def configure_recording(self, value: RecordingConfig, *, persist: bool = False) -> None:
        """Set video defaults; run_simulation has no effect on caller-owned physics."""
        self._request("configure_recording", (value, persist))

    def start_recording(
        self,
        output: str | Path | None = None,
        *,
        surface: CaptureSurface | str | None = None,
        fps: float | None = None,
        countdown: float | None = None,
    ) -> Path:
        """Schedule live video in the display process without pausing the caller.

        Frames follow display wall time, not every physics step. A zero countdown
        starts on the next display frame. Stop explicitly to confirm file finalization.
        """
        with self._lock:
            if not self.is_running():
                raise RuntimeError("The passive viewer is closed")
            if not self._publish():
                raise RuntimeError("The display state mailbox is unavailable")
            return self._request(
                "start_recording",
                {
                    "output": output,
                    "surface": surface,
                    "fps": fps,
                    "countdown": countdown,
                },
            )

    def pause_recording(self) -> bool:
        """Pause video writing only; policy inference and physics keep their owner."""
        return self._request("pause_recording")

    def resume_recording(self) -> bool:
        """Resume video writing without changing the external simulation."""
        return self._request("resume_recording")

    def stop_recording(self) -> Path | None:
        """Wait for encoder finalization and return the saved path, or raise on failure."""
        return self._request("stop_recording")

    def configure_actions(
        self,
        actions: tuple[PassiveAction, ...],
        *,
        hints: tuple[ToolHint, ...] | None = None,
        hint_surface: str = "status",
    ) -> None:
        """Bind caller-owned actions and optional replacement hints.

        By default each key appears in the status bar. Supply ``hints`` to group
        related inputs; ``hint_surface="scene"`` uses the viewport hint capsule.
        An empty tuple hides action hints without disabling their inputs.
        """
        self._request(
            "configure_actions",
            (tuple(actions), None if hints is None else tuple(hints), hint_surface),
        )

    def configure_tool_hints(self, hints: Sequence[ToolHint], *, surface: str = "status") -> None:
        """Replace action hints without rebinding inputs or waiting for pending actions.

        Use surface="scene" for viewport capsules or "status" for inline hints.
        An empty sequence hides action hints, not the actions themselves. This
        sends data to the display process; it never executes caller draw callbacks.
        """
        self._request("configure_tool_hints", (tuple(hints), surface))

    def poll_events(self) -> tuple[PassiveEvent, ...]:
        """Drain available requests without waiting for the display process.

        Handle them on the simulation owner's thread at an inference/step boundary,
        then acknowledge each event. Repeated requests for a pending action coalesce.
        """
        with self._lock:
            if self._closed:
                return ()
            events = []
            for _ in range(64):
                try:
                    events.append(self._events.get_nowait())
                except Empty:
                    break
            return tuple(events)

    def acknowledge_event(self, event: PassiveEvent, *, error: str | None = None) -> None:
        """Confirm a handled request, or report its failure without changing physics."""
        self._request("acknowledge_event", (event, error))

    def set_status(self, text: str, *, paused: bool | None = None) -> None:
        """Publish caller-reported activity and optional pause state in the status bar.

        This updates presentation only. It never pauses physics or infers policy
        health from publication rate. Empty text uses the localized Running/Paused
        label when paused is known, or only Passive when it is unknown.
        """
        if not isinstance(text, str) or (paused is not None and not isinstance(paused, bool)):
            raise TypeError("status requires text and an optional bool paused value")
        self._request("set_status", (text, paused))

    def capture_array(self, *, surface: CaptureSurface | str = CaptureSurface.SCENE) -> np.ndarray:
        """Publish state and wait for one owned RGB capture, without disk I/O."""
        with self._lock:
            if not self.is_running():
                raise RuntimeError("The passive viewer is closed")
            if not self._publish():
                raise RuntimeError("The display state mailbox is unavailable")
            return self._request("capture", CaptureSurface(surface).value)

    def capture(self, output: str | Path, *, surface=CaptureSurface.SCENE) -> Path:
        """Save an explicitly requested RGB capture in the caller's process."""
        from PIL import Image

        image = self.capture_array(surface=surface)
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image).save(output)
        return output

    def capture_into(self, buffer: SharedImage, *, surface=CaptureSurface.SCENE) -> dict:
        """Capture into caller-owned shared memory, returning only frame metadata.

        Read ``buffer.array`` after completion and finish before the next write.
        The buffer must match the selected surface's current RGB dimensions.
        On timeout, discard the buffer because completion is unknown.
        """
        with self._lock:
            if not self.is_running():
                raise RuntimeError("The passive viewer is closed")
            if not self._publish():
                raise RuntimeError("The display state mailbox is unavailable")
            return self._request(
                "capture_into",
                {
                    "surface": CaptureSurface(surface).value,
                    "buffer": buffer.descriptor,
                },
            )

    def start_rpc(self, socket_path=None) -> Path:
        """Attach local JSON-RPC to the passive display; return the socket path."""
        return Path(self._request("start_rpc", socket_path))

    def stop_rpc(self) -> None:
        """Stop the display's local JSON-RPC server."""
        self._request("stop_rpc")

    def _request(self, operation, payload=None):
        with self._lock:
            if not self.is_running():
                raise RuntimeError("The passive viewer is closed")
            try:
                self._connection.send((operation, payload))
                return self._receive(
                    timeout=max(self._timeout, 35.0) if operation == "stop_recording" else None
                )
            except (EOFError, BrokenPipeError, OSError) as exc:
                raise RuntimeError("The passive viewer disconnected") from exc

    def _receive(self, *, timeout=None):
        deadline = time.monotonic() + (self._timeout if timeout is None else timeout)
        while not self._connection.poll(min(0.1, max(0.0, deadline - time.monotonic()))):
            if self._stop.is_set() or not self._process.is_alive():
                self.close()
                raise RuntimeError(f"The passive viewer exited (code {self._process.exitcode})")
            if time.monotonic() >= deadline:
                self.close()
                raise TimeoutError("The passive viewer did not respond before its timeout")
        response = self._connection.recv()
        if "error" in response:
            raise RuntimeError(response["error"])
        return response.get("result")

    def close(self) -> None:
        """Finalize any video and reap the process; repeated calls are harmless.

        Allow the encoder's 30-second finalization deadline before forced cleanup.
        A timeout or finalization failure is reported instead of claiming a saved video.
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._stop.set()
            try:
                if self._process.pid is not None:
                    deadline = time.monotonic() + 35.0
                    # Drain completion before joining: an error traceback may
                    # exceed the pipe buffer and block the sender until read.
                    while (
                        self._process.is_alive()
                        and not self._completion.poll(0.05)
                        and time.monotonic() < deadline
                    ):
                        pass
                    acknowledged = False
                    error = None
                    if self._completion.poll():
                        try:
                            error = self._completion.recv()
                            acknowledged = True
                        except EOFError:
                            pass
                    self._process.join(timeout=max(0.0, deadline - time.monotonic()))
                    if self._process.is_alive():
                        self._process.terminate()
                        self._process.join(timeout=5.0)
                        raise TimeoutError(
                            "The passive viewer did not finish shutdown; video completion is unknown"
                        )
                    if error:
                        raise RuntimeError(error)
                    if not acknowledged or self._process.exitcode != 0:
                        raise RuntimeError(
                            f"The passive viewer exited without successful shutdown acknowledgement "
                            f"(code {self._process.exitcode})"
                        )
            finally:
                self._connection.close()
                self._completion.close()
                self._events.close()
                # Clear only forces actually applied to the caller, even if the
                # worker died while holding the state mailbox lock.
                self.data.xfrc_applied[self._applied_force] = 0.0

    def __enter__(self) -> PassiveViewer:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def launch_passive(
    model,
    data,
    *,
    max_fps: float = 60.0,
    timeout: float = 30.0,
    renderer: str | None = None,
    width: int = 1600,
    height: int = 1000,
    samples: int = 4,
    title: str = "Mojive",
    show_window: bool = True,
    config: ViewerConfig | None = None,
    control_writeback: bool = True,
) -> PassiveViewer:
    """Launch an independently scheduled window for an existing MuJoCo simulation.

    Call from an importable Python script inside ``if __name__ == '__main__':``.
    The spawned process owns GLFW on its main thread on every platform; it does
    not share the physics loop's GIL. Call ``sync()`` after stepping or resetting.
    ``max_fps`` limits display and snapshot publication, never physics stepping.
    Use :func:`mojive.build` for synchronous embedding on an existing UI thread.
    Set ``control_writeback=False`` when a policy exclusively owns actuator controls.
    Inspection, camera, recording and physical mouse perturbation remain available.
    """
    config = config or ViewerConfig(layout=LayoutConfig(persistence=False))
    config = replace(
        config,
        interactions=replace(config.interactions, gizmo=False, playback_shortcuts=False),
    )
    return PassiveViewer(
        model,
        data,
        max_fps=max_fps,
        timeout=timeout,
        control_writeback=control_writeback,
        viewer_options={
            "renderer": renderer,
            "width": width,
            "height": height,
            "samples": samples,
            "title": title,
            "show_window": show_window,
            "config": config,
            "vsync": False,
        },
    )


def _run_viewer(
    model, mailbox, stop, connection, completion, events, max_fps, control_writeback, options
):
    import mujoco

    from mojive.adapters.mujoco import MuJoCoAdapter
    from mojive.app.composition import build_from_adapter
    from mojive.app.passive_input import PassiveInput

    state, ctrl, ctrl_changed, force, force_changed = mailbox.arrays()

    class DisplayAdapter(MuJoCoAdapter):
        published_step = 0
        physics_hz = None
        received_at = 0.0

        def frame(self, needs):
            frame = super().frame(needs)
            frame.step = self.published_step
            frame.physics_hz = (
                0.0
                if self.physics_hz is not None and time.monotonic() - self.received_at > 0.5
                else self.physics_hz
            )
            return frame

        def reset(self):
            raise RuntimeError("Reset belongs to the external physics caller")

        def set_ctrl(self, index, value):
            if not super().set_ctrl(index, value):
                return False
            with mailbox.lock:
                ctrl[index] = self.data.ctrl[index]
                ctrl_changed[index] = 1
            return True

        def set_ctrl_vector(self, values):
            if not super().set_ctrl_vector(values):
                return False
            with mailbox.lock:
                np.copyto(ctrl, self.data.ctrl)
                ctrl_changed.fill(1)
            return True

        def apply_perturb(self, node_id, target_position, target_rotation, mode):
            if not super().apply_perturb(node_id, target_position, target_rotation, mode):
                return False
            with mailbox.lock:
                force[self._perturb_body] = self.data.xfrc_applied[self._perturb_body]
                force_changed[self._perturb_body] = 1
            return True

        def clear_perturb(self):
            with mailbox.lock:
                active = force_changed == 1
                force[active] = 0.0
                force_changed[active] = 2
            super().clear_perturb()

    viewer = None
    worker_error = None
    try:
        adapter = DisplayAdapter(external_clock=True)
        adapter.load_model(model)
        adapter.caps = replace(
            adapter.caps,
            asset_loading=False,
            write_pose=False,
            write_qpos=False,
            state_snapshots=False,
            write_ctrl=control_writeback,
            keyframes=False,
            equality_constraints=False,
            reload=False,
            topology_editing=False,
            model_properties=False,
            model_assets=False,
        )
        local_state = np.empty_like(state)
        sequence = -1

        def receive_state():
            nonlocal sequence
            with mailbox.lock:
                if sequence == mailbox.sequence.value:
                    return
                np.copyto(local_state, state)
                sequence = mailbox.sequence.value
                adapter.published_step = mailbox.step.value
                rate = mailbox.physics_hz.value
                adapter.physics_hz = None if math.isnan(rate) else rate
                adapter.received_at = time.monotonic()
            mujoco.mj_setState(
                model, adapter.data, local_state, mujoco.mjtState.mjSTATE_INTEGRATION
            )
            mujoco.mj_forward(model, adapter.data)

        def draw(surface=None, out=None):
            receive_state()
            if out is not None:
                image = viewer.capture_into(out, surface=surface)
            else:
                image = (
                    viewer.capture_array(surface=surface) if surface is not None else viewer.sync()
                )
            with mailbox.lock:
                mailbox.rendered.value += 1
                mailbox.displayed_sequence.value = sequence
                mailbox.displayed_time.value = float(adapter.data.time)
            return image

        receive_state()
        viewer = build_from_adapter(adapter, **options)
        viewer.app.passive_mode = True
        passive_input = PassiveInput(viewer, events)
        viewer.set_input_handler(passive_input)
        draw()
        connection.send({"result": None})
        period = 1.0 / max_fps
        due = time.monotonic() + period
        parent = mp.parent_process()
        while not stop.is_set() and parent.is_alive() and viewer.is_running():
            pending = connection.poll(min(0.1, max(0.0, due - time.monotonic())))
            if pending:
                operation, payload = connection.recv()
                try:
                    result = None
                    if operation == "capture":
                        result = draw(payload)
                    elif operation == "capture_into":
                        with SharedImage.attach(payload["buffer"]) as shared:
                            draw(payload["surface"], shared.array)
                            result = {
                                "transport": "shared_memory",
                                "buffer": shared.descriptor,
                                "scope": payload["surface"],
                                "mode": "rgb",
                                "orientation": "top_left",
                                "step": adapter.published_step,
                                "time": float(adapter.data.time),
                            }
                    elif operation == "camera":
                        viewer.set_camera(payload)
                    elif operation == "track_body":
                        viewer.track_body(payload)
                    elif operation == "configure_tracking":
                        viewer.configure_tracking(payload[0], persist=payload[1])
                    elif operation == "configure_recording":
                        viewer.configure_recording(payload[0], persist=payload[1])
                    elif operation == "recording":
                        result = viewer.recording
                    elif operation == "start_recording":
                        receive_state()
                        result = viewer.start_recording(**payload)
                    elif operation == "pause_recording":
                        result = viewer.pause_recording()
                    elif operation == "resume_recording":
                        result = viewer.resume_recording()
                    elif operation == "stop_recording":
                        result = viewer.stop_recording()
                    elif operation == "configure_actions":
                        passive_input.configure(
                            payload[0], hints=payload[1], hint_surface=payload[2]
                        )
                    elif operation == "configure_tool_hints":
                        passive_input.configure_hints(payload[0], surface=payload[1])
                    elif operation == "acknowledge_event":
                        passive_input.acknowledge(payload[0], error=payload[1])
                    elif operation == "set_status":
                        viewer.app.external_status, viewer.app.external_paused = payload
                    elif operation == "stats":
                        with mailbox.lock:
                            result = {
                                "rendered_frames": mailbox.rendered.value,
                                "published_frames": mailbox.sequence.value,
                                "displayed_sequence": mailbox.displayed_sequence.value,
                                "displayed_time": mailbox.displayed_time.value,
                                "physics_hz": viewer.session.frame.physics_hz,
                                "max_fps": max_fps,
                            }
                    elif operation == "start_rpc":
                        result = str(viewer.start_rpc(payload).socket_path)
                    elif operation == "stop_rpc":
                        viewer.stop_rpc()
                    else:
                        raise ValueError(f"Unknown passive operation: {operation}")
                    if operation in ("capture", "capture_into"):
                        # A capture already presented a display frame. Start the
                        # next interval here instead of immediately redrawing it.
                        due = time.monotonic() + period
                    connection.send({"result": result})
                except Exception as exc:
                    connection.send({"error": str(exc)})
            now = time.monotonic()
            if now >= due:
                draw()
                due = max(due + period, time.monotonic())
    except KeyboardInterrupt:
        pass
    except BaseException:
        worker_error = traceback.format_exc()
    finally:
        stop.set()
        error = worker_error
        try:
            if viewer is not None:
                viewer.stop_recording()
        except Exception:
            error = (
                f"{error}\nDuring video finalization:\n" if error else ""
            ) + traceback.format_exc()
        finally:
            if viewer is not None:
                viewer.release()
            with suppress(OSError):
                completion.send(error)
            completion.close()
            events.cancel_join_thread()
            events.close()
            connection.close()
