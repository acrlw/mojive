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
