"""Opt-in native shader rebuilding and transactional program reload."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from pathlib import Path

from ...log import get_logger

log = get_logger("render.native.shaders")


class ShaderReload:
    """Watch development sources or packaged binaries without restarting scenes."""

    def __init__(self, runtime, directory):
        self.runtime = runtime
        self.directory = Path(directory).resolve()
        self.build = self.directory.parent
        self.sources = None
        cache = self.build / "CMakeCache.txt"
        if self.directory.name == "shaders" and cache.is_file():
            for line in cache.read_text().splitlines():
                if line.startswith("CMAKE_HOME_DIRECTORY:INTERNAL="):
                    self.sources = Path(line.split("=", 1)[1]) / "shaders"
        self._state = None
        self._next_check = 0.0
        self.error = ""
        self._build_future = None
        self._executor = None
        self._stop = threading.Event()
        self._lock = threading.RLock()

    def _files(self):
        files = list(self.directory.glob("*.bin"))
        if self.sources is not None:
            files.extend(self.sources.glob("*.sc"))
            files.extend(self.sources.glob("*.sh"))
        return tuple(
            sorted((str(path), path.stat().st_mtime_ns, path.stat().st_size) for path in files)
        )

    def enable(self):
        if self._state is None:
            self._state = self._files()

    @staticmethod
    def _source_state(state):
        return tuple(item for item in state if not item[0].endswith(".bin"))

    @staticmethod
    def _stop_process(process, *, force=False):
        if os.name == "posix":
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL if force else signal.SIGTERM)
        elif force:
            process.kill()
        else:
            process.terminate()

    def _compile(self):
        local_cmake = Path(sys.executable).parent / (
            "cmake.exe" if sys.platform == "win32" else "cmake"
        )
        cmake = str(local_cmake) if local_cmake.is_file() else shutil.which("cmake")
        error = ""
        try:
            if cmake is None:
                raise RuntimeError("Native shader reload requires CMake")
            deadline = time.monotonic() + 120
            with subprocess.Popen(
                [
                    cmake,
                    "--build",
                    str(self.build),
                    "--target",
                    "mojive_probe_shaders",
                    "--parallel",
                    "2",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            ) as process:
                while True:
                    try:
                        stdout, stderr = process.communicate(timeout=0.1)
                        if process.returncode:
                            raise RuntimeError((stdout + stderr)[-4000:])
                        break
                    except subprocess.TimeoutExpired:
                        if self._stop.is_set() or time.monotonic() >= deadline:
                            self._stop_process(process)
                            try:
                                process.communicate(timeout=2)
                            except subprocess.TimeoutExpired:
                                self._stop_process(process, force=True)
                                process.communicate()
                            raise RuntimeError(
                                "Native shader rebuild canceled or timed out"
                            ) from None
        except (OSError, RuntimeError) as exc:
            error = str(exc)
        return self._files(), error

    def _reload(self, error=""):
        try:
            if error:
                raise RuntimeError(error)
            self.runtime.reload_shaders()
        except (OSError, RuntimeError) as exc:
            self.error = str(exc)
            log.warning("Shader reload failed; retaining previous programs: {}", self.error)
        else:
            self.error = ""

    def check(self):
        with self._lock:
            if self._stop.is_set():
                return
            if self._build_future is not None:
                if not self._build_future.done():
                    return
                finished = self._build_future
                self._build_future = None
                try:
                    state, error = finished.result()
                    changed = self._source_state(state) != self._source_state(self._state)
                    if changed:
                        # A newer edit arrived during compilation. Compile that
                        # generation before publishing any partial binary set.
                        self._next_check = 0
                        return
                    self._state = state
                    self._reload(error)
                except OSError as exc:
                    self._reload(str(exc))
                return
            now = time.monotonic()
            if now < self._next_check:
                return
            self._next_check = now + 0.25
            try:
                state = self._files()
            except OSError as exc:
                self._reload(str(exc))
                return
            if state == self._state:
                return
            self._state = state
            if self.sources is None:
                self._reload()
            else:
                if self._executor is None:
                    self._executor = ThreadPoolExecutor(
                        max_workers=1, thread_name_prefix="mojive-shaders"
                    )
                self._build_future = self._executor.submit(self._compile)

    def close(self):
        """Stop owned compilation before releasing the native runtime."""
        with self._lock:
            self._stop.set()
            executor = self._executor
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
