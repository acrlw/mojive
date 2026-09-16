"""Exercise held Control drags through the real viewport and physics command path."""

import os
from pathlib import Path

import numpy as np
import pytest
from imgui_bundle import imgui
from PIL import Image

from mojive import build
from mojive import commands as cmd
from mojive.interaction.input import add_physical_mouse_button_event
from mojive.ui.input_bindings import DEFAULT_INPUT_BINDINGS
from mojive.ui.perturb import project

pytestmark = [pytest.mark.gpu, pytest.mark.physics]


@pytest.mark.parametrize("mode,button", [("translate", 0), ("rotate", 1)])
def test_ctrl_drag_at_fixed_pivot_stays_stable(tmp_path, monkeypatch, mode, button):
    monkeypatch.setenv("MOJIVE_SETTINGS", str(tmp_path / "settings.json"))
    monkeypatch.setenv("MOJIVE_UI_SCALE", "1")
    model = tmp_path / "pivot.xml"
    model.write_text("""
    <mujoco>
      <option timestep=".01" integrator="implicitfast" gravity="0 0 0"/>
      <worldbody>
        <body name="offset">
          <joint type="ball" armature=".0001" damping=".0005"/>
          <geom type="box" pos=".019 .019 0" size=".0087 .0087 .0087"
                mass=".00253704" rgba=".2 .55 .8 1"/>
        </body>
      </worldbody>
    </mujoco>
    """)
    with build(model, paused=True, vsync=False, width=1280, height=800, show_window=False) as v:
        v.app.input_bindings = DEFAULT_INPUT_BINDINGS
        v.set_gizmo_mode("translate")
        node = next(n for n in v.session.nodes if n.name == "offset")
        assert v.session.submit(cmd.SelectNode(node.node_id))
        for _ in range(12):
            v.sync()
        initial = v.session.adapter.capture_state().qpos.copy()
        cam = v.app._camera_view()
        rect = v.app._viewport_rect
        x, y, _ = project(cam, [[0.019, 0.019, 0]], rect)[0]
        io = imgui.get_io()
        io.add_mouse_pos_event(float(x), float(y))
        for _ in range(2):
            v.sync()
        origin, direction = v.app._cursor_ray((io.mouse_pos.x, io.mouse_pos.y))
        hit, distance = v.session.query(cmd.Pick(origin, direction))
        assert hit == node.object_id
        expected_point = origin + distance * direction
        modifiers = (imgui.Key.mod_ctrl, imgui.Key.left_ctrl)
        for modifier in modifiers:
            io.add_key_event(modifier, True)
        v.sync()
        add_physical_mouse_button_event(io, button, True)
        v.sync()
        assert v.session.perturb.active and v.session.perturb.mode == mode, v.app._input_state()
        if mode == "translate":
            np.testing.assert_allclose(v.session.perturb.grab_point, expected_point, atol=1e-7)
        for frame in range(100):
            io.add_mouse_pos_event(
                float(x + 60 * np.sin(frame / 20)), float(y + 30 * np.sin(frame / 15))
            )
            assert v.session.submit(cmd.Step())
            v.sync()
            assert v.session.perturb.active
        assert v.session.frame.time == pytest.approx(1)
        state = v.session.adapter.capture_state()
        assert np.isfinite(state.qpos).all()
        assert np.linalg.norm(state.qpos - initial) > 0.01
        np.testing.assert_allclose(v.app._camera_view().eye, cam.eye)
        output = os.environ.get("MOJIVE_PERTURB_CAPTURE")
        if output:
            directory = Path(output)
            directory.mkdir(parents=True, exist_ok=True)
            Image.fromarray(v.capture_array(surface="window")).save(directory / f"{mode}.png")
        add_physical_mouse_button_event(io, button, False)
        for modifier in modifiers:
            io.add_key_event(modifier, False)
        for _ in range(3):
            v.sync()
        assert not v.session.perturb.active
