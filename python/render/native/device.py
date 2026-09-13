"""Lazy native loading and reference-counted process device ownership."""

from __future__ import annotations

import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

from ...log import get_logger
from ...native import native_module as native_module

_lock = threading.RLock()
_shared = None


class NativeDevice:
    """One GPU owner shared by independent renderers and windows in this process."""

    def __init__(self, wayland=None):
        self.api = native_module()
        if not self.api.has_renderer:
            raise RuntimeError("The native extension was built without a renderer")
        build = os.environ.get("MOJIVE_NATIVE_BUILD")
        shaders = os.environ.get("MOJIVE_NATIVE_SHADER_DIR")
        if not shaders:
            shaders = (
                str(Path(build) / "shaders")
                if build
                else str(Path(self.api.__file__).parent / "shaders")
            )
        if wayland is None:
            wayland = (
                sys.platform.startswith("linux")
                and bool(os.environ.get("WAYLAND_DISPLAY"))
                and (os.environ.get("XDG_SESSION_TYPE") != "x11" or not os.environ.get("DISPLAY"))
            )
        self.wayland = bool(wayland)
        self.runtime = self.api.RenderRuntime(shaders, wayland=self.wayland)
        from .programs import ShaderReload

        self.shaders = ShaderReload(self.runtime, shaders)
        from .resources import ResourceCache

        self.resources = ResourceCache(self.api)
        self.users = 0
        self.readback_slots = threading.BoundedSemaphore(8)
        self._readback_thread = threading.local()
        self.readbacks = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="mojive-readback",
            initializer=self._start_readback_worker,
        )
        self.textures = {}
        self._next_texture = 1
        self._log_cursor = 0

    def acquire_readback_slot(self, *, blocking=True):
        # Completion callbacks must not wait for cleanup queued behind themselves.
        blocking = blocking and not self.in_readback_worker()
        if not self.readback_slots.acquire(blocking=blocking):
            raise RuntimeError("Readback queue is full; wait for a result before submitting more")

    @contextmanager
    def readback_slot(self):
        self.acquire_readback_slot()
        try:
            yield
        finally:
            self.readback_slots.release()

    def _start_readback_worker(self):
        self._readback_thread.active = True

    def drain_logs(self):
        """Mirror owned native records on the caller thread without a worker callback."""
        batch = self.runtime.log.read(self._log_cursor)
        self._log_cursor = batch.next
        for record in batch.records:
            get_logger(record.component).bind(
                native_runtime_id=record.runtime_id,
                native_thread_id=record.thread_id,
                native_timestamp_ns=record.timestamp_ns,
                native_sequence=record.sequence,
                origin="native",
            ).log(record.level.name, record.message)
        if batch.missed:
            get_logger("render.runtime").warning(
                "Native log subscriber missed {} records", batch.missed
            )

    def register_texture(self, texture):
        key = self._next_texture
        self._next_texture += 1
        self.textures[key] = texture
        return key

    def unregister_texture(self, key):
        self.textures.pop(key, None)

    def in_readback_worker(self):
        return bool(getattr(self._readback_thread, "active", False))

    def release(self):
        global _shared
        with _lock:
            self.users -= 1
            if self.users == 0:
                try:
                    # Scene owners have drained their readbacks. Do not join user
                    # completion callbacks while holding the device registry lock.
                    self.shaders.close()
                    self.readbacks.shutdown(wait=False)
                    self.runtime.close()
                    self.drain_logs()
                finally:
                    self.resources.clear()
                    _shared = None


def acquire_device(*, wayland=None):
    global _shared
    with _lock:
        if _shared is None:
            _shared = NativeDevice(wayland)
        elif wayland is not None and _shared.wayland != wayland:
            raise RuntimeError("Native windows in one process must use the same window system")
        _shared.users += 1
        return _shared
