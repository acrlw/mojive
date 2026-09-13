"""Load optional native infrastructure without initializing a graphics device."""

from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import os
import sys
import threading
from pathlib import Path

_lock = threading.RLock()


def native_shader_directory(module) -> Path:
    """Resolve shaders beside a wheel or inside the loaded editable build."""
    override = os.environ.get("MOJIVE_NATIVE_SHADER_DIR")
    if override:
        return Path(override)
    build = os.environ.get("MOJIVE_NATIVE_BUILD")
    if build:
        return Path(build) / "shaders"
    package = Path(module.__file__).resolve().parent
    # Editable installs load <build>/python/mojive/_native directly. Prove
    # that layout with its CMake cache; never search unrelated build outputs.
    if package.parent.name == "python" and (package.parent.parent / "CMakeCache.txt").is_file():
        return package.parent.parent / "shaders"
    return package / "shaders"


def native_module():
    """Load the installed extension or an explicitly selected development build."""
    with _lock:
        if "mojive._native" in sys.modules:
            return sys.modules["mojive._native"]
        build = os.environ.get("MOJIVE_NATIVE_BUILD")
        if not build:
            try:
                return importlib.import_module("mojive._native")
            except ImportError as exc:
                raise RuntimeError(
                    "Native infrastructure is unavailable; run make native-editable "
                    "or select an existing build with MOJIVE_NATIVE_BUILD"
                ) from exc
        directory = Path(build).resolve() / "python/mojive"
        for suffix in importlib.machinery.EXTENSION_SUFFIXES:
            path = directory / ("_native" + suffix)
            if path.is_file():
                spec = importlib.util.spec_from_file_location("mojive._native", path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                sys.modules[spec.name] = module
                return module
        raise RuntimeError(f"No compatible native extension in {directory}; run make cpp-python")
