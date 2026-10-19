"""Native renderer ownership and failure cleanup without a GPU context."""

from types import SimpleNamespace

import pytest

from mojive.ui import opengl_imgui, window


@pytest.fixture
def renderer_api(monkeypatch):
    events = []
    state = SimpleNamespace(context=None, gl=None, failure=None)

    def event(name, data=None):
        assert state.gl == state.context.name
        events.append((name, state.context.name, data))

    def init(version):
        event("init", version)
        if state.failure == "init":
            return False
        io = state.context.io
        io.backend_renderer_user_data = object()
        io.backend_flags = 0 if state.failure == "textures" else 1
        if state.failure == "partial":
            raise RuntimeError("partial initialization")
        return True

    def create():
        event("create")
        return state.failure != "resources"

    def shutdown():
        event("shutdown")
        state.context.io.backend_renderer_user_data = None

    api = SimpleNamespace(
        get_current_context=lambda: state.context,
        set_current_context=lambda context: setattr(state, "context", context),
        get_io=lambda: state.context.io,
        BackendFlags_=SimpleNamespace(renderer_has_textures=1),
        backends=SimpleNamespace(
            opengl3_init=init,
            opengl3_create_device_objects=create,
            opengl3_new_frame=lambda: event("new_frame"),
            opengl3_render_draw_data=lambda data: event("render", data),
            opengl3_shutdown=shutdown,
        ),
    )
    monkeypatch.setattr(opengl_imgui, "imgui", api)

    def context(name):
        result = SimpleNamespace(
            name=name, io=SimpleNamespace(backend_renderer_user_data=None, backend_flags=0)
        )
        state.context = result

        def make_current():
            state.gl = name

        return result, make_current

    return state, events, context


def test_two_renderers_submit_and_shutdown_in_their_own_context(renderer_api):
    state, events, context = renderer_api
    first, activate_first = context("first")
    renderer_first = opengl_imgui.OpenGLImGuiRenderer(activate_first)
    second, activate_second = context("second")
    renderer_second = opengl_imgui.OpenGLImGuiRenderer(activate_second)
    data = object()
    renderer_first.new_frame()
    renderer_first.render(data)
    renderer_second.new_frame()
    renderer_first.shutdown()
    renderer_first.shutdown()
    assert first.io.backend_renderer_user_data is None
    assert second.io.backend_renderer_user_data is not None
    renderer_second.render(data)
    renderer_second.shutdown()
    assert events == [
        ("init", "first", "#version 330"),
        ("create", "first", None),
        ("init", "second", "#version 330"),
        ("create", "second", None),
        ("new_frame", "first", None),
        ("render", "first", data),
        ("new_frame", "second", None),
        ("shutdown", "first", None),
        ("render", "second", data),
        ("shutdown", "second", None),
    ]
    assert state.context is second


@pytest.mark.parametrize("failure", ("init", "partial", "textures", "resources"))
def test_failed_initialization_releases_only_allocated_renderer_data(renderer_api, failure):
    state, events, context = renderer_api
    owner, activate = context("owner")
    state.failure = failure
    with pytest.raises(RuntimeError):
        opengl_imgui.OpenGLImGuiRenderer(activate)
    assert owner.io.backend_renderer_user_data is None
    assert sum(name == "shutdown" for name, *_ in events) == (failure != "init")


def test_existing_renderer_is_not_overwritten_or_shutdown(renderer_api):
    _state, events, context = renderer_api
    owner, activate = context("owner")
    existing = owner.io.backend_renderer_user_data = object()
    with pytest.raises(RuntimeError, match="already has a renderer"):
        opengl_imgui.OpenGLImGuiRenderer(activate)
    assert owner.io.backend_renderer_user_data is existing
    assert events == []


@pytest.mark.parametrize("has_renderer", (False, True))
def test_frame_prepares_renderer_before_imgui_but_keeps_native_window_optional(
    monkeypatch, has_renderer
):
    events = []
    owner = window.Window.__new__(window.Window)
    owner._imgui_context = SimpleNamespace(open_popup_stack=[])
    owner.make_current = lambda: events.append("context")
    owner._refresh_scales = lambda: None
    owner._sync_style_scale = lambda: None
    owner._input = SimpleNamespace(process_inputs=lambda: events.append("input"))
    owner._layout_reset_pending = False
    if has_renderer:
        owner._renderer_new_frame = lambda: events.append("renderer")
    monkeypatch.setattr(window, "glfw", SimpleNamespace(poll_events=lambda: events.append("poll")))
    monkeypatch.setattr(
        window,
        "imgui",
        SimpleNamespace(
            set_current_context=lambda _context: None,
            new_frame=lambda: events.append("imgui"),
        ),
    )
    owner.begin_frame()
    assert events == ["context", "poll", "input", *(["renderer"] if has_renderer else []), "imgui"]


@pytest.mark.parametrize("other_windows", (0, 1))
def test_partial_window_initialization_releases_resources_without_terminating_peers(
    monkeypatch, other_windows
):
    events = []
    monkeypatch.setattr(window, "_live_windows", other_windows)
    monkeypatch.setattr(
        window.native_drop, "uninstall", lambda token: events.append(("drop", token))
    )
    monkeypatch.setattr(
        window, "imgui", SimpleNamespace(destroy_context=lambda context: events.append(context))
    )
    monkeypatch.setattr(
        window,
        "glfw",
        SimpleNamespace(
            destroy_window=lambda handle: events.append(handle),
            terminate=lambda: events.append("terminate"),
        ),
    )

    def initialize(owner, _config):
        owner._window = "window"
        owner._imgui_context = "imgui"
        owner._native_drop_token = 7
        owner._counted_window = True
        window._live_windows += 1

        def shutdown():
            events.append("renderer")
            raise RuntimeError("cleanup failure")

        owner._impl = SimpleNamespace(shutdown=shutdown)
        raise ValueError("initialization failure")

    monkeypatch.setattr(window.Window, "_initialize", initialize)
    with pytest.raises(ValueError, match="initialization failure"):
        window.Window()
    assert events == [("drop", 7), "renderer", "imgui", "window"] + (
        [] if other_windows else ["terminate"]
    )
    assert window._live_windows == other_windows
