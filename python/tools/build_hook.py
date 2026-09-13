"""Map the flat Python source tree and package optional native build products."""

from __future__ import annotations

import importlib.machinery
import os
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class PackageBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        if self.target_name == "wheel" and version == "editable":
            self._prepare_editable()
        selected = os.environ.get("MOJIVE_NATIVE_WHEEL_BUILD")
        if not selected:
            return
        if self.target_name != "wheel" or version != "standard":
            raise RuntimeError("Native artifacts require a standard wheel build")
        root = Path(selected).resolve()
        binary = self._native_binary(root)
        shaders = root / "shaders"
        expected = {
            source.stem + ".bin" for source in (Path(self.root) / "cpp/shaders").glob("[vcf]s_*.sc")
        }
        if not expected or any(not (shaders / name).is_file() for name in expected):
            raise RuntimeError("Native shaders are incomplete; run make native-python-build")
        build_data["pure_python"] = False
        build_data["infer_tag"] = True
        includes = build_data["force_include"]
        includes[str(binary)] = "mojive/" + binary.name
        for name in expected:
            includes[str(shaders / name)] = "mojive/shaders/" + name
        for name in ("bgfx", "bx", "bimg", "glm", "spdlog", "nanobind", "robin-map", "libtess2"):
            directory = Path(self.root) / "3rdparty" / name
            for license in ("LICENSE", "LICENSE.txt", "LICENSE.md", "copying.txt"):
                path = directory / license
                if path.is_file():
                    includes[str(path)] = "mojive/native_licenses/" + name + ".txt"
                    break
        includes[str(Path(self.root) / "3rdparty/bgfx/3rdparty/meshoptimizer/LICENSE.md")] = (
            "mojive/native_licenses/meshoptimizer.txt"
        )
        includes[str(Path(self.root) / "3rdparty/stbImageResizeLicense.txt")] = (
            "mojive/native_licenses/stbImageResize.txt"
        )

    def _prepare_editable(self):
        from editables import EditableProject

        selected = os.environ.get("MOJIVE_NATIVE_EDITABLE_BUILD")
        binary = self._native_binary(Path(selected).resolve()) if selected else None
        # The generated module loads lazily and sets the real package search path.
        # There is no runtime dependency on editables or eager Mojive import.
        project = EditableProject("mojive", self.root)
        project.map_method = "self_replace"
        project.map("mojive", "python")
        destination = Path(self.root) / "output/editable"
        destination.mkdir(parents=True, exist_ok=True)
        for name, content in project.files():
            if name == "mojive.py" and binary is not None:
                content += f"\nsys.modules[__name__].__path__.append({str(binary.parent)!r})\n"
            (destination / name).write_text(content, encoding="utf-8")

    @staticmethod
    def _native_binary(root: Path) -> Path:
        candidates = [
            root / "python/mojive" / ("_native" + suffix)
            for suffix in importlib.machinery.EXTENSION_SUFFIXES
        ]
        binaries = [path for path in candidates if path.is_file()]
        if len(binaries) != 1:
            raise RuntimeError(f"Expected one extension matching the build interpreter in {root}")
        return binaries[0]
