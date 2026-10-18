"""Interactive workspace launch options use native docking without a renderer."""

from types import SimpleNamespace

import pytest
from imgui_bundle import imgui

from mojive.tools.native_workspace import float_panel, main


@pytest.mark.parametrize("scale", (1.0, 1.2, 1.25))
@pytest.mark.parametrize("grouped", (False, True))
@pytest.mark.parametrize("translated", (False, True))
def test_interactive_float_keeps_native_panel_or_dock_group_visible(scale, grouped, translated):
    context = imgui.create_context()
    try:
        io = imgui.get_io()
        io.set_ini_filename(None)
        io.display_size = (1200, 800)
        io.delta_time = 1 / 60
        io.config_flags |= imgui.ConfigFlags_.docking_enable
        io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
        io.fonts.add_font_default()
        shell = SimpleNamespace(open={}, focus_next=None)
        shell.activate = lambda name: shell.open.update({name: True})
        initialized = False

        def sync():
            nonlocal initialized
            imgui.new_frame()
            dock_id = imgui.dock_space_over_viewport()
            if not initialized:
                for name in ("Timeline", "Output"):
                    imgui.internal.dock_builder_dock_window(name, dock_id)
                imgui.internal.dock_builder_finish(dock_id)
                initialized = True
            for name in ("Timeline", "Output"):
                title = "Translated###" + name if translated else name
                if shell.focus_next == name:
                    imgui.set_next_window_focus()
                    shell.focus_next = None
                imgui.begin(title)
                imgui.text(name)
                imgui.end()
            imgui.render()

        viewer = SimpleNamespace(
            app=SimpleNamespace(workspace=shell),
            window=SimpleNamespace(style_scale=scale),
            sync=sync,
        )
        float_panel(viewer, "Timeline", with_panel="Output" if grouped else None)
        timeline = imgui.internal.find_window_by_name("Timeline")
        output = imgui.internal.find_window_by_name("Output")
        assert timeline.active and not timeline.collapsed
        if grouped:
            assert timeline.dock_id == output.dock_id != 0
            root = timeline.dock_node
            assert not root.is_dock_space()
            assert root.host_window is not None
            position, size = root.pos, root.size
        else:
            assert timeline.dock_id == 0
            assert output.dock_id != 0
            position, size = timeline.pos, timeline.size
        viewport = imgui.get_main_viewport()
        assert viewport.work_pos.x < position.x
        assert viewport.work_pos.y < position.y
        assert position.x + size.x < viewport.work_pos.x + viewport.work_size.x
        assert position.y + size.y < viewport.work_pos.y + viewport.work_size.y
    finally:
        imgui.destroy_context(context)


@pytest.mark.parametrize(
    "arguments,error",
    (
        (("--capture", "--float-panel", "Timeline"), "requires interactive mode"),
        (("--float-with", "Output"), "requires --float-panel"),
        (("--float-panel", "Timeline", "--float-with", "Timeline"), "different panel"),
        (("--float-panel", "Timeline", "--float-with", "Settings"), "standalone window"),
    ),
)
def test_invalid_float_options_fail_before_opening_native_window(arguments, error, capsys):
    with pytest.raises(SystemExit) as raised:
        main(arguments)
    assert raised.value.code == 2
    assert error in capsys.readouterr().err
