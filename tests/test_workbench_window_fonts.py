"""Detached design references prepare their own context's requested font faces."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
from imgui_bundle import imgui

from mojive.design.editor import SpecificationEditor
from mojive.design.native.window import WorkbenchWindow
from mojive.design.native.workbench import NativeWorkbench
from mojive.design.runtime import DesignRuntime
from mojive.design.specification import DEFAULT
from mojive.design.store import SpecificationStore, read_spec


@pytest.fixture
def detached(tmp_path, monkeypatch):
    context = imgui.create_context()
    io = imgui.get_io()
    io.set_ini_filename(None)
    io.delta_time = 1 / 60
    io.display_size = (760, 900)
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures

    def begin_frame():
        imgui.set_current_context(context)
        imgui.new_frame()

    def end_frame(*, readback=False):
        imgui.render()

    native = SimpleNamespace(
        style_scale=1.0,
        font_report=SimpleNamespace(cjk_path=None),
        begin_frame=begin_frame,
        end_frame=end_frame,
        close=lambda: None,
    )
    monkeypatch.setattr("mojive.design.native.window.create_window", lambda *args: native)
    store = SpecificationStore(tmp_path / "design.json")
    store.write(DEFAULT, save=True)
    runtime = DesignRuntime(store)
    editor = SpecificationEditor(store.saved_path, store.live_path, read_spec)
    bench = NativeWorkbench(editor, runtime, runtime, width=760)
    bench.page = "设计规范"
    bench.part = "button"
    with WorkbenchWindow(bench, visible=False) as window:
        try:
            yield window, store, runtime
        finally:
            editor.close()
            imgui.destroy_context(context)


@pytest.mark.parametrize("family", ("system", "portable"))
def test_detached_specimens_load_explicit_and_global_families_in_their_own_context(
    detached, family
):
    window, store, runtime = detached
    spec = deepcopy(DEFAULT)
    spec["tokens"]["font_family"] = family
    spec["parts"]["button"] = {"fontFamily": "portable" if family == "system" else "system"}
    store.write(spec)
    runtime.poll(force=True)
    window.frame()
    ui = window.surface.ui
    assert {"system", "portable"} <= ui._font_families.keys()
    assert ui.body is ui._font_families[family][1][0]
    assert "button" in window.workbench.specimens.last
    faces = tuple(font.font_id for font in imgui.get_io().fonts.fonts)
    for _ in range(5):
        window.frame()
    assert tuple(font.font_id for font in imgui.get_io().fonts.fonts) == faces


def test_detached_window_initially_loads_only_its_configured_family(detached):
    window, _store, _runtime = detached
    ui = window.surface.ui
    assert ui._font_family == "portable"
    assert set(ui._font_families) == {"portable"}
    assert [source.label for source, _ratio in ui.font_sources[:2]] == [
        "MiSans Regular",
        "MiSans Semibold",
    ]
    window.frame()
    assert set(ui._font_families) == {"portable"}


def test_a_font_family_changed_during_the_editor_frame_can_be_painted_immediately(
    detached, monkeypatch
):
    window, _store, _runtime = detached
    bench = window.workbench
    contents = bench.draw_contents

    def edit_then_draw(surface):
        bench.change(("parts", "button", "fontFamily"), "portable")
        contents(surface)

    monkeypatch.setattr(bench, "draw_contents", edit_then_draw)
    window.frame()
    assert "portable" in window.surface.ui._font_families
    assert "button" in bench.specimens.last
