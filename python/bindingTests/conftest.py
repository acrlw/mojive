"""Load only the explicitly selected private build for native acceptance."""

import importlib.machinery
import importlib.util
import os
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def native():
    explicit = os.environ.get("MOJIVE_NATIVE_TEST_MODULE")
    if explicit:
        path = Path(explicit)
    else:
        root = Path(os.environ["MOJIVE_NATIVE_TEST_BUILD"]) / "python" / "mojive"
        paths = [root / f"_native{suffix}" for suffix in importlib.machinery.EXTENSION_SUFFIXES]
        matches = [path for path in paths if path.is_file()]
        if len(matches) != 1:
            pytest.fail(f"Expected one native extension in {root}, found {matches}")
        path = matches[0]
    os.environ["MOJIVE_NATIVE_TEST_MODULE"] = str(path.resolve())
    spec = importlib.util.spec_from_file_location("mojive._native", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
