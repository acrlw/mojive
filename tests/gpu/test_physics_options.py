"""Exercise environment physics controls through real menu, pointer and text input."""

import os
from pathlib import Path

import numpy as np
import pytest
from imgui_bundle import imgui

from mojive import build
from mojive import commands as cmd
from mojive.adapters.base import NodeType
from mojive.tools.ui_runtime import (
    _click,
    _item_center,
    _open_main_menu,
    _park_cursor,
    _save,
    _save_window_crop,
)

pytestmark = [pytest.mark.gpu, pytest.mark.physics]


def _text(viewer, key, value):
    _click(viewer, _item_center(viewer, "input_text", f"##physics-{key}"))
    io = imgui.get_io()
    modifier = imgui.Key.mod_super if io.config_mac_osx_behaviors else imgui.Key.mod_ctrl
    io.add_key_event(modifier, True)
    io.add_key_event(imgui.Key.a, True)
    viewer.sync()
    io.add_key_event(imgui.Key.a, False)
    io.add_key_event(modifier, False)
    io.add_input_characters_utf8(value)
    viewer.sync()


def _enter(viewer):
    io = imgui.get_io()
    io.add_key_event(imgui.Key.enter, True)
    viewer.sync()
    io.add_key_event(imgui.Key.enter, False)
    viewer.sync()


def _groups(viewer, monkeypatch, *groups):
    """Expose each section without depending on persisted window scroll position."""
    native = imgui.collapsing_header
    labels = {viewer.app.localizer.text(group) for group in groups}

    def header(label, *args, **kwargs):
        imgui.set_next_item_open(label in labels, imgui.Cond_.always)
        return native(label, *args, **kwargs)

    with monkeypatch.context() as context:
        context.setattr(imgui, "collapsing_header", header)
        viewer.sync()
    for _ in range(3):
        viewer.sync()


@pytest.mark.parametrize("scale", [1, 1.5])
@pytest.mark.parametrize("language", ["en", "zh_CN"])
def test_menu_and_live_physics_controls(tmp_path, monkeypatch, scale, language):
    monkeypatch.setenv("MOJIVE_SETTINGS", str(tmp_path / "settings.json"))
    monkeypatch.setenv("MOJIVE_UI_SCALE", str(scale))
    model = tmp_path / "physics.xml"
    model.write_text("""<mujoco><option timestep=".002" integrator="implicitfast"/>
      <worldbody><geom type="plane" size="2 2 .1"/>
      <body name="box" pos="0 0 1"><freejoint/>
      <geom type="box" size=".15 .15 .15" rgba=".2 .55 .8 1"/></body></worldbody>
    </mujoco>""")
    with build(
        model,
        paused=True,
        vsync=False,
        width=round(1440 * scale),
        height=round(1120 * scale),
        show_window=False,
    ) as viewer:
        viewer.app.set_language(language)
        for _ in range(8):
            viewer.sync()
        tr = viewer.app.localizer.text
        _open_main_menu(viewer, tr("Window"))
        _click(viewer, _item_center(viewer, "menu_item", tr("Physics Options...")))
        for _ in range(3):
            viewer.sync()
        assert viewer.session.selected_node.type is NodeType.ENVIRONMENT
        adapter = getattr(viewer.session.adapter, "primary", viewer.session.adapter)
        _click(viewer, _item_center(viewer, "begin_combo", "##physics-integrator"))
        _click(viewer, _item_center(viewer, "selectable", "Euler"))
        assert int(adapter.model.opt.integrator) == 0
        _text(viewer, "timestep", "5e-3")
        assert adapter.model.opt.timestep == 0.002
        _enter(viewer)
        assert adapter.model.opt.timestep == 0.005
        assert viewer.session.submit(cmd.Undo())
        viewer.sync()
        assert adapter.model.opt.timestep == 0.002
        assert viewer.session.submit(cmd.Redo())
        viewer.sync()
        assert adapter.model.opt.timestep == 0.005
        _text(viewer, "timestep", "-1")
        _enter(viewer)
        assert adapter.model.opt.timestep == 0.005
        assert viewer.app.panels.get("Inspector")._physics_error
        _text(viewer, "timestep", "0.005")
        _enter(viewer)
        assert not viewer.app.panels.get("Inspector")._physics_error
        _text(viewer, "timestep", "0.002")
        _enter(viewer)
        assert adapter.model.opt.timestep == 0.002
        assert not viewer.app.panels.get("Inspector")._physics_error

        output = os.environ.get("MOJIVE_PHYSICS_OPTIONS_CAPTURE")
        directory = Path(output) / f"{language}-{scale:g}" if output else None

        def capture(name):
            if directory:
                directory.mkdir(parents=True, exist_ok=True)
                _park_cursor(viewer)
                for _ in range(3):
                    viewer.sync()
                _save_window_crop(viewer, "Inspector", directory / f"{name}.png")

        capture("algorithms")
        if directory:
            _save(viewer, directory / "window.png")
        _groups(viewer, monkeypatch, "Physical parameters")
        _text(viewer, "gravity", "0 0 -3")
        # Leaving a field commits exactly once, just like Enter.
        _click(viewer, _item_center(viewer, "input_text", "##physics-wind"))
        np.testing.assert_array_equal(adapter.model.opt.gravity, [0, 0, -3])
        capture("physical")
        _groups(viewer, monkeypatch, "Disable flags")
        _click(viewer, _item_center(viewer, "checkbox", "##physics-disableflags.gravity"))
        assert adapter.model.opt.disableflags & 128
        assert adapter.data.qacc[2] == 0
        capture("disable-flags")
        _groups(viewer, monkeypatch, "Enable flags", "Contact override")
        _click(viewer, _item_center(viewer, "checkbox", "##physics-enableflags.override"))
        assert adapter.model.opt.enableflags & 1
        capture("enable-flags")
