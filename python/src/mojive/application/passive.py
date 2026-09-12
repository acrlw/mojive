"""Passive MuJoCo viewing with independent physics and display ownership."""

from __future__ import annotations

import math
import multiprocessing as mp
import operator
import threading
import time
import traceback
from contextlib import suppress
from dataclasses import replace
from pathlib import Path

import numpy as np

from mojive.capture import CaptureSurface
from mojive.capture.shared_image import SharedImage
from mojive.config import CameraTrackingConfig, LayoutConfig, ViewerConfig
from mojive.session.rates import StepRate


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

    def __init__(self, model, data, *, max_fps, timeout, viewer_options):
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
        self._process = context.Process(
            target=_run_viewer,
            args=(model, self._mailbox, self._stop, child, self.max_fps, viewer_options),
            name="mojive-passive",
            daemon=True,
        )
        mujoco.mj_getState(model, data, self._state, self._state_spec)
        self._mailbox.sequence.value = 1
        try:
            self._process.start()
            child.close()
            self._receive()
        except BaseException:
            child.close()
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
                return self._receive()
            except (EOFError, BrokenPipeError, OSError) as exc:
                raise RuntimeError("The passive viewer disconnected") from exc

    def _receive(self):
        deadline = time.monotonic() + self._timeout
        while not self._connection.poll(min(0.1, max(0.0, deadline - time.monotonic()))):
            if not self._process.is_alive():
                raise RuntimeError(f"The passive viewer exited (code {self._process.exitcode})")
            if time.monotonic() >= deadline:
                self.close()
                raise TimeoutError("The passive viewer did not respond before its timeout")
        response = self._connection.recv()
        if "error" in response:
            raise RuntimeError(response["error"])
        return response.get("result")

    def close(self) -> None:
        """Stop display work and reap its process; safe to call more than once."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._stop.set()
            if self._process.pid is not None:
                self._process.join(timeout=5.0)
                if self._process.is_alive():
                    self._process.terminate()
                    self._process.join(timeout=5.0)
            self._connection.close()
            # Only clear forces actually applied to the caller. The worker may
            # have exited while holding the mailbox lock.
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
) -> PassiveViewer:
    """Launch an independently scheduled window for an existing MuJoCo simulation.

    Call from an importable Python script inside ``if __name__ == '__main__':``.
    The spawned process owns GLFW on its main thread on every platform; it does
    not share the physics loop's GIL. Call ``sync()`` after stepping or resetting.
    ``max_fps`` limits display and snapshot publication, never physics stepping.
    Use :func:`mojive.build` for synchronous embedding on an existing UI thread.
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


def _run_viewer(model, mailbox, stop, connection, max_fps, options):
    import mujoco

    from mojive.adapters.mujoco import MuJoCoAdapter
    from mojive.application.composition import build_from_adapter

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
    try:
        adapter = DisplayAdapter(external_clock=True)
        adapter.load_model(model)
        adapter.caps = replace(
            adapter.caps,
            asset_loading=False,
            write_pose=False,
            write_qpos=False,
            state_snapshots=False,
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
    except BaseException:
        with suppress(OSError, EOFError):
            connection.send({"error": traceback.format_exc()})
    finally:
        stop.set()
        if viewer is not None:
            viewer.release()
        connection.close()
