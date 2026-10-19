"""Render-target allocation failures must not invalidate the presented target."""

from types import SimpleNamespace

import pytest

from mojive.render.opengl import targets
from mojive.render.opengl.backend import OpenGLBackend


class Resource:
    def __init__(self, kind):
        self.kind = kind
        self.releases = 0

    def release(self):
        self.releases += 1


class Context:
    def __init__(self):
        self.resources = []
        self.fail_after = None
        self.screen = SimpleNamespace(use=lambda: None)

    def allocate(self, kind):
        if len(self.resources) == self.fail_after:
            raise RuntimeError("Allocation failed")
        resource = Resource(kind)
        self.resources.append(resource)
        return resource

    def texture(self, *args, **kwargs):
        return self.allocate("texture")

    def depth_texture(self, *args, **kwargs):
        return self.allocate("depth")

    def framebuffer(self, *args, **kwargs):
        return self.allocate("framebuffer")

    def program(self, *args, **kwargs):
        return self.allocate("program")

    def vertex_array(self, *args, **kwargs):
        return self.allocate("vertex_array")


@pytest.fixture
def backend(monkeypatch):
    context = Context()
    probes = []

    def probe(ctx, samples):
        probes.append((ctx, samples))
        return targets.IdLayout.SHARED

    monkeypatch.setattr(targets, "probe_id_layout", probe)
    monkeypatch.setattr(targets.G, "native", lambda: SimpleNamespace())
    backend = OpenGLBackend.__new__(OpenGLBackend)
    backend.target = targets.RenderTarget(context, 80, 60, 4)
    backend.caps = object()
    yield backend, context, probes
    backend.target.release()
    assert all(resource.releases == 1 for resource in context.resources)


def test_resize_retains_capabilities_and_size_independent_resources(backend):
    backend, context, probes = backend
    caps = backend.caps
    program, vao = backend.target._clear_prog, backend.target._clear_vao
    for width, height in ((160, 90), (161, 91), (80, 60), (0, -1)):
        old = backend.target
        backend.resize(width, height)
        assert backend.target.size == (max(1, width), max(1, height))
        assert backend.target is not old
        assert old.resolve_tex.releases == 1
        assert backend.caps is caps
        assert backend.target._clear_prog is program
        assert backend.target._clear_vao is vao
        assert program.releases == vao.releases == 0
        assert backend.target._rgb_stage is None
        assert backend.target._metric_stage is None
        assert backend.target._segmentation_stage is None

    count = len(context.resources)
    retained = backend.target
    backend.resize(0, -1)
    assert backend.target is retained
    assert len(context.resources) == count
    assert len(probes) == 1
    assert [r.kind for r in context.resources].count("program") == 1
    assert [r.kind for r in context.resources].count("vertex_array") == 1


@pytest.mark.parametrize("allocated_before_failure", (0, 1, 5, 12))
def test_failed_resize_releases_partial_allocations_and_preserves_previous_target(
    backend, allocated_before_failure
):
    backend, context, probes = backend
    previous = backend.target
    caps = backend.caps
    count = len(context.resources)
    context.fail_after = count + allocated_before_failure

    with pytest.raises(RuntimeError, match="Allocation failed"):
        backend.resize(160, 90)

    assert backend.target is previous
    assert previous.size == (80, 60)
    assert backend.caps is caps
    assert all(r.releases == 0 for r in context.resources[:count])
    assert all(r.releases == 1 for r in context.resources[count:])
    assert previous._owns_clear_resources
    assert len(probes) == 1

    context.fail_after = None
    backend.resize(160, 90)
    assert backend.target.size == (160, 90)
    assert backend.target._clear_prog is previous._clear_prog
    assert backend.target._clear_prog.releases == 0


def test_initial_shader_failure_releases_attachments(monkeypatch):
    context = Context()
    monkeypatch.setattr(targets.G, "native", lambda: SimpleNamespace())

    def fail_program(*args, **kwargs):
        raise RuntimeError("Shader failed")

    monkeypatch.setattr(context, "program", fail_program)
    with pytest.raises(RuntimeError, match="Shader failed"):
        targets.RenderTarget(context, 80, 60, 4, targets.IdLayout.SHARED)
    assert context.resources
    assert all(r.releases == 1 for r in context.resources)
