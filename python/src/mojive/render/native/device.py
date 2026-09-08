"""Lazy native loading and reference-counted process device ownership."""

from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ...log import get_logger

_lock = threading.RLock()
_shared = None


def native_module():
    """Load a packaged extension or an explicitly selected development build."""
    if "mojive._native" in sys.modules:
        return sys.modules["mojive._native"]
    build = os.environ.get("MOJIVE_NATIVE_BUILD")
    if not build:
        try:
            return importlib.import_module("mojive._native")
        except ImportError as exc:
            raise RuntimeError("Build the native backend with make native-viewer first") from exc
    directory = Path(build).resolve() / "python/mojive"
    for suffix in importlib.machinery.EXTENSION_SUFFIXES:
        path = directory / ("_native" + suffix)
        if path.is_file():
            spec = importlib.util.spec_from_file_location("mojive._native", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            sys.modules[spec.name] = module
            return module
    raise RuntimeError(
        f"No compatible native extension in {directory}; run make native-python-build"
    )


class NativeDevice:
    """One GPU owner shared by independent renderers and windows in this process."""

    def __init__(self):
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
        self.runtime = self.api.RenderRuntime(shaders)
        self.users = 0
        self._readback_thread = threading.local()
        self.readbacks = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="mojive-readback",
            initializer=self._start_readback_worker,
        )
        self.textures = {}
        self._next_texture = 1
        self._log_cursor = 0

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
                    self.readbacks.shutdown(wait=False)
                    self.runtime.close()
                    self.drain_logs()
                finally:
                    _shared = None


def acquire_device():
    global _shared
    with _lock:
        if _shared is None:
            _shared = NativeDevice()
        _shared.users += 1
        return _shared
