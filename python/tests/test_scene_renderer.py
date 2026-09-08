"""Scene provider consumption without requiring the complete editor protocol."""

from __future__ import annotations

import numpy as np
import pytest

from mojive import SceneProvider, SceneRenderer
from mojive.adapters.base import SceneFrame, SceneSource
from mojive.render.backend import NullBackend, RenderProduct


class Provider:
    def __init__(self):
        self.structure_revision = 0
        self.source_reads = 0

    def frame(self, needs):
        assert needs.poses and needs.deformables
        return SceneFrame()

    def scene_source(self):
        self.source_reads += 1
        return SceneSource()


@pytest.fixture
def backend(monkeypatch):
    backend = NullBackend()
    monkeypatch.setattr(backend, "set_shadow_quality", lambda value: True)
    monkeypatch.setattr("mojive.scene_renderer._select_backend", lambda *args: (None, backend))
    return backend


def test_provider_requires_only_three_stream_members_and_uploads_on_revision(backend):
    provider = Provider()
    assert isinstance(provider, SceneProvider)
    with SceneRenderer(width=8, height=6) as renderer:
        renderer.update_from(provider)
        renderer.update_from(provider)
        assert provider.source_reads == 1
        provider.structure_revision += 1
        renderer.update_from(provider)
        assert provider.source_reads == 2
        other = Provider()
        renderer.update_from(other)
        assert other.source_reads == 1


def test_invalid_render_products_and_output_buffers_fail_before_render(backend):
    with SceneRenderer(width=8, height=6) as renderer:
        with pytest.raises(ValueError, match="exactly one"):
            renderer.render(product=RenderProduct.COLOR | RenderProduct.OBJECT_ID)
        with pytest.raises(ValueError, match="out must"):
            renderer.render(out=np.zeros((6, 8, 3), np.float32))
        readonly = np.zeros((6, 8, 3), np.uint8)
        readonly.flags.writeable = False
        with pytest.raises(ValueError, match="out must"):
            renderer.render(out=readonly)


@pytest.mark.parametrize("width, height", [(0, 8), (8, -1)])
def test_invalid_dimensions_are_rejected_before_context_creation(width, height):
    with pytest.raises(ValueError, match="positive"):
        SceneRenderer(width=width, height=height)
