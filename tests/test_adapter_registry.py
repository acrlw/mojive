"""Registration and selection for external scene adapters."""

from __future__ import annotations

import pytest

from mojive import register_adapter, unregister_adapter
from mojive.adapters.base import SceneAdapterBase
from mojive.adapters.registry import make_adapter
from mojive.backends import available_backends, backend_info


class CustomAdapter(SceneAdapterBase):
    def __init__(self):
        self.loaded = None
        self.released = False

    def load(self, path):
        self.loaded = path

    def release(self):
        self.released = True


def test_registered_factory_supports_hyphenated_names_and_discovery(tmp_path):
    created = []

    def factory():
        adapter = CustomAdapter()
        created.append(adapter)
        return adapter

    register_adapter("custom-engine", factory, label="Custom engine")
    try:
        path = tmp_path / "model.custom"
        path.write_text("example")
        adapter = make_adapter("custom-engine", path)
        assert created == [adapter]
        assert adapter.loaded == path
        info = backend_info("custom-engine")
        assert info.available and info.physics == "Custom engine"
        assert info in available_backends()
        with pytest.raises(ValueError, match="already registered"):
            register_adapter("custom-engine", factory)
    finally:
        unregister_adapter("custom-engine")
    assert not backend_info("custom-engine").available


def test_adapter_factory_releases_resources_when_loading_fails(tmp_path):
    adapter = CustomAdapter()
    register_adapter("broken", lambda: adapter)
    try:
        with pytest.raises(FileNotFoundError):
            make_adapter("broken", tmp_path / "missing.asset")
        assert adapter.released
    finally:
        unregister_adapter("broken")


@pytest.mark.parametrize("name", ["mujoco", "toy", "mujoco-classic"])
def test_built_in_names_cannot_be_replaced(name):
    with pytest.raises(ValueError, match="reserved"):
        register_adapter(name, CustomAdapter)
