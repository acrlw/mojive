"""Opt-in packaging of an explicitly built native extension and its shaders."""

from __future__ import annotations

import importlib.machinery
import os
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class NativeWheelHook(BuildHookInterface):
    def initialize(self, version, build_data):
        selected = os.environ.get("MOJIVE_NATIVE_WHEEL_BUILD")
        if not selected:
            return
        if self.target_name != "wheel" or version != "standard":
            raise RuntimeError("Native artifacts require a standard wheel build")
        root = Path(selected).resolve()
        candidates = [
            root / "python/mojive" / ("_native" + suffix)
            for suffix in importlib.machinery.EXTENSION_SUFFIXES
        ]
        binaries = [path for path in candidates if path.is_file()]
        if len(binaries) != 1:
            raise RuntimeError(f"Expected one extension matching the build interpreter in {root}")
        shaders = root / "shaders"
        expected = {
            source.stem + ".bin" for source in (Path(self.root) / "cpp/shaders").glob("[vf]s_*.sc")
        }
        if not expected or any(not (shaders / name).is_file() for name in expected):
            raise RuntimeError("Native shaders are incomplete; run make native-python-build")
        build_data["pure_python"] = False
        build_data["infer_tag"] = True
        includes = build_data["force_include"]
        includes[str(binaries[0])] = "mojive/" + binaries[0].name
        for name in expected:
            includes[str(shaders / name)] = "mojive/shaders/" + name
        for name in ("bgfx", "bx", "bimg", "glm", "spdlog", "nanobind", "robinMap"):
            directory = Path(self.root) / "thirdParty" / name
            for license in ("LICENSE", "LICENSE.txt", "LICENSE.md", "copying.txt"):
                path = directory / license
                if path.is_file():
                    includes[str(path)] = "mojive/nativeLicenses/" + name + ".txt"
                    break
        includes[str(Path(self.root) / "thirdParty/bgfx/3rdparty/meshoptimizer/LICENSE.md")] = (
            "mojive/nativeLicenses/meshoptimizer.txt"
        )
        includes[str(Path(self.root) / "thirdParty/stbImageResizeLicense.txt")] = (
            "mojive/nativeLicenses/stbImageResize.txt"
        )
