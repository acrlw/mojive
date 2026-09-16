"""MuJoCo-style passive handle with a separately scheduled UI thread."""

from __future__ import annotations

import operator
import threading
import time
from contextlib import contextmanager
from dataclasses import replace

import mujoco

from mojive.adapters.mujoco.updates import CONTROL_FIELDS

from .panels import install_key_callback
from .state import StateExchange, _Field
from .visuals import Presentation, draw_user_geometries, snapshot_geometries

_UI_OWNER = threading.Lock()


@contextmanager
def _ui_owner():
    # ImGui and GLFW have process-wide state; simultaneous UI loops are not safe.
    if not _UI_OWNER.acquire(blocking=False):
        raise RuntimeError("Only one mojive.viewer window may run in a process at a time")
    try:
        yield
    finally:
        _UI_OWNER.release()


class Handle:
    """Synchronize a caller-owned model/data pair with the full Mojive UI.

    ``sync(state_only=False)`` transfers model edits, integration state and UI
    input. ``state_only=True`` leaves caller model edits unpublished. Rendering
    accesses private copies, so caller data is accessed only during sync.
    """

    def __init__(self, model, data, *, key_callback, options, timeout=30.0):
        self._exchange = StateExchange(model, data)
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._error = None
        self._viewer = None
        self._user_scn = mujoco.MjvScene(model, maxgeom=10000)
        self._display_scn = mujoco.MjvScene(self._exchange.display_model, maxgeom=0)
        self._flags = _Field(
            self._user_scn, self._display_scn, "flags", self._user_scn.flags.copy()
        )
        self._geometries = ()
        self._geometry_pending = False
        self._resources_pending = False
        self._resource_waiters: list[threading.Event] = []
        self._viewport = mujoco.MjrRect(0, 0, 0, 0)
        self._thread = threading.Thread(
            target=self._run, args=(key_callback, options), name="mojive-viewer", daemon=True
        )
        self._owner = _ui_owner()
        self._owner.__enter__()
        try:
            self._thread.start()
        except BaseException:
            self._owner.__exit__(None, None, None)
            raise
        if not self._ready.wait(timeout):
            self._stop.set()
            raise TimeoutError("The Mojive viewer did not finish startup")
        if self._error is not None:
            raise RuntimeError("The Mojive viewer could not start") from self._error

    @property
    def m(self):
        """The caller's original MjModel, or None after closing."""
        return self._exchange.model if self.is_running() else None

    @property
    def d(self):
        """The caller's original MjData, or None after closing."""
        return self._exchange.data if self.is_running() else None

    @property
    def cam(self):
        """Mutable MjvCamera exchanged on sync."""
        return self._exchange.cam

    @property
    def opt(self):
        """Mutable MjvOption exchanged on sync."""
        return self._exchange.opt

    @property
    def perturb(self):
        """Mutable MjvPerturb exchanged on sync."""
        return self._exchange.perturb

    @property
    def pert(self):
        """Alias matching the spelling used in MuJoCo's viewer documentation."""
        return self.perturb

    @property
    def user_scn(self):
        """MjvScene containing user geometry and render flags published on sync."""
        return self._user_scn

    @property
    def viewport(self):
        """Current bottom-left framebuffer pixel rectangle, or None after closing."""
        return self._viewport if self.is_running() else None

    def lock(self):
        """Return the context manager protecting viewer exchanges and UI edits."""
        return self._lock

    def is_running(self) -> bool:
        """Return whether the viewer thread is still running."""
        return not self._stop.is_set() and self._thread.is_alive()

    def sync(self, state_only: bool = False) -> None:
        """Exchange simulation/UI state; optionally leave caller model edits unpublished."""
        with self._lock:
            if not self.is_running():
                return
            if not isinstance(state_only, bool):
                raise TypeError("state_only must be a bool")
            geometries = snapshot_geometries(self.user_scn)
            self._exchange.sync(state_only)
            self._exchange.options_changed |= self._flags.sync(True)
            self._geometries = geometries
            self._geometry_pending = True

    def _update_resource(self, category, index):
        with self._lock:
            if not self.is_running():
                return
            index = operator.index(index)
            count = int(getattr(self._exchange.model, "n" + category))
            if not 0 <= index < count:
                raise ValueError(f"{category} ID {index} is outside [0, {count})")
            self._exchange.sync_resource(category, index)
            self._resources_pending = True
            completed = threading.Event()
            self._resource_waiters.append(completed)
        deadline = time.monotonic() + 30.0
        while not completed.wait(0.05):
            if not self.is_running():
                raise RuntimeError("The viewer closed before the resource upload completed")
            if time.monotonic() >= deadline:
                raise TimeoutError("The resource upload did not complete before its deadline")

    def update_hfield(self, hfieldid: int):
        """Upload one heightfield's samples; call outside lock()."""
        self._update_resource("hfield", hfieldid)

    def update_mesh(self, meshid: int):
        """Upload one mesh's vertices, normals, UVs and faces; call outside lock()."""
        self._update_resource("mesh", meshid)

    def update_texture(self, texid: int):
        """Upload one texture's pixels; call outside lock()."""
        self._update_resource("tex", texid)

    def set_figures(self, viewports_figures):
        raise NotImplementedError(
            "MjvFigure overlays are not supported; use Mojive Plot or Canvas2D"
        )

    def clear_figures(self):
        return None

    def set_texts(self, texts):
        raise NotImplementedError("Mjr text overlays are not supported; use Mojive Canvas2D")

    def clear_texts(self):
        return None

    def set_images(self, viewports_images):
        raise NotImplementedError("Mjr image overlays are not supported; use Mojive Canvas2D")

    def clear_images(self):
        return None

    def close(self):
        """Close the window and wait for UI resource cleanup."""
        self._stop.set()
        if threading.current_thread() is not self._thread:
            self._thread.join(timeout=35.0)
            if self._thread.is_alive():
                raise TimeoutError("The Mojive UI thread did not finish shutdown")
        if self._error is not None:
            raise RuntimeError("The Mojive viewer failed") from self._error

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def _run(self, key_callback, options):
        from mojive.adapters.mujoco import MuJoCoAdapter
        from mojive.app.composition import build_from_adapter

        viewer = None
        try:
            exchange = self._exchange
            adapter = MuJoCoAdapter(external_clock=True)
            adapter.load_model(exchange.display_model, exchange.display_data)
            # Recompiling/replacing the model would invalidate the caller-owned identities.
            adapter.caps = replace(
                adapter.caps,
                asset_loading=False,
                reload=False,
                topology_editing=False,
                model_composition=False,
            )
            adapter._perturb = exchange.display_perturb
            viewer = build_from_adapter(adapter, paused=True, **options)
            self._viewer = viewer
            viewer.app.passive_mode = True
            viewer.app._startup()
            install_key_callback(viewer, key_callback)
            presentation = Presentation(exchange, self._display_scn)
            self._ready.set()
            while not self._stop.is_set():
                started = time.monotonic()
                if not self._lock.acquire(timeout=0.05):
                    continue
                try:
                    if not viewer.is_running():
                        break
                    if self._resources_pending:
                        adapter.refresh_model_visuals(force=True)
                        self._resources_pending = False
                    if exchange.pending_model_fields:
                        adapter.refresh_model_fields(exchange.pending_model_fields)
                        if not exchange.pending_model_fields.isdisjoint(CONTROL_FIELDS):
                            viewer.session.refresh_control_metadata()
                        exchange.pending_model_fields.clear()
                    presentation.before_frame(viewer)
                    if self._geometry_pending:
                        draw_user_geometries(viewer.backend.debug, self._geometries)
                        self._geometry_pending = False
                    viewer.sync()
                    presentation.after_frame(viewer)
                    for completed in self._resource_waiters:
                        completed.set()
                    self._resource_waiters.clear()
                    x, y, width, height = viewer.window.points_to_pixels(viewer.app._viewport_rect)
                    self._viewport.left = round(x)
                    self._viewport.bottom = round(viewer.window.size_pixels[1] - y - height)
                    self._viewport.width, self._viewport.height = round(width), round(height)
                finally:
                    self._lock.release()
                self._stop.wait(max(0.0, 1.0 / 60.0 - (time.monotonic() - started)))
        except BaseException as exc:
            self._error = exc
        finally:
            self._stop.set()
            try:
                if viewer is not None:
                    viewer.release()
            except BaseException as exc:
                if self._error is None:
                    self._error = exc
            self._owner.__exit__(None, None, None)
            self._ready.set()
