"""Opt-in native shader rebuilding and transactional program reload."""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
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

    def check(self):
        now = time.monotonic()
        if now < self._next_check:
            return
        self._next_check = now + 0.25
        try:
            state = self._files()
            if state == self._state:
                return
            self._state = state
            if self.sources is not None:
                local_cmake = Path(sys.executable).parent / (
                    "cmake.exe" if sys.platform == "win32" else "cmake"
                )
                cmake = str(local_cmake) if local_cmake.is_file() else shutil.which("cmake")
                if cmake is None:
                    raise RuntimeError("Native shader reload requires CMake")
                built = subprocess.run(
                    [
                        cmake,
                        "--build",
                        str(self.build),
                        "--target",
                        "mojive_probe_shaders",
                        "--parallel",
                        "2",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=120,
                    check=False,
                )
                # A failed build can still write some binaries. Keep the current
                # programs and wait for another source edit before trying again.
                self._state = self._files()
                if built.returncode:
                    raise RuntimeError((built.stdout + built.stderr)[-4000:])
            self.runtime.reload_shaders()
        except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
            self.error = str(exc)
            log.warning("Shader reload failed; retaining previous programs: {}", self.error)
        else:
            self.error = ""
