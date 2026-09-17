"""Visible contact styles, shared capture state, and real settings controls."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from imgui_bundle import imgui
from PIL import Image

from mojive import ContactStyle, SceneRenderer, ViewerConfig, build
from mojive.adapters.base import FrameNeeds
from mojive.adapters.mujoco import MuJoCoAdapter
from mojive.interaction.gizmo import project
from mojive.render.backend import RenderFlag
from mojive.render.debugdraw import PrimitiveType
from mojive.tools.keyframe_timeline import show_settings
from mojive.tools.ui_runtime import (
    _click,
    _item_center,
    _item_rect,
    _park_cursor,
    _save_window_crop,
)
from mojive.types import CameraView
from mojive.ui.camera_preview import CameraPreview
from mojive.ui.scene_capture import SceneCapture

pytestmark = pytest.mark.gpu


@pytest.fixture
def contact_model(tmp_path):
    path = tmp_path / "contact.xml"
    path.write_text("""<mujoco><visual><scale contactwidth=".15" contactheight=".05"/>
      <rgba contactpoint="1 1 0 1"/></visual><worldbody>
      <geom type="plane" size="2 2 .1" rgba=".15 .18 .22 1"/>
      <body pos="0 0 .149"><freejoint/><geom type="box" size=".35 .25 .15" rgba=".95 .65 .1 1"/></body>
      </worldbody></mujoco>""")
    return path


def test_contact_shapes_colors_capture_and_preview(contact_model, backend_name):
    adapter = MuJoCoAdapter(contact_model)
    source = adapter.scene_source()
    frame = adapter.frame(FrameNeeds(poses=True, contacts=True))
    assert len(frame.contacts) == 4
    camera = CameraView(eye=np.array((1.1, -1.7, 0.85)), target=np.array((0, 0, 0.06)))
    capture, preview = SceneCapture(), CameraPreview()
    output = Path("output/contact-style") / backend_name
    output.mkdir(parents=True, exist_ok=True)
    session = SimpleNamespace(source=source, frame=frame, structure_generation=1)
    try:
        with SceneRenderer(source, width=960, height=640, camera=camera) as view:
            try:
                view.set_geometry_view("collision")
                view.update(frame)
                before = view.render().copy()
                view.set_flag(RenderFlag.CONTACTPOINT, True)
                images = {}
                for shape in ("cylinder", "sphere", "point"):
                    view.set_contact_style(ContactStyle(shape=shape, scale=1.5))
                    view.update(frame)
                    images[shape] = view.render().copy()
                    Image.fromarray(images[shape]).save(output / f"{shape}.png")
                    red = (images[shape][..., 0] > 150) & (images[shape][..., 1] < 110)
                    assert np.count_nonzero(red) > 50
                    if shape == "sphere":
                        pixels = project(
                            camera.with_aspect(1.5), frame.contacts[:, :3], (0, 0, 960, 640)
                        )
                        for x, y, *_ in pixels:
                            # A front-facing ball center receives the headlight. Back-face
                            # overdraw or world-space normals produce a dark middle band.
                            assert images[shape][round(y), round(x), 0] > 220
                    with view._current():
                        pixels = capture.read(view._backend, session, camera)
                    np.testing.assert_array_equal(pixels, images[shape])
                    assert capture._backend.get_contact_style() == view._backend.get_contact_style()
                assert np.count_nonzero(images["cylinder"] != images["sphere"]) > 100
                view.set_contact_style(ContactStyle("sphere", (0.1, 0.8, 0.95, 1), 2))
                view.update(frame)
                cyan = view.render().copy()
                assert np.count_nonzero(cyan != images["sphere"]) > 100
                Image.fromarray(cyan).save(output / "sphere-cyan.png")
                with view._current():
                    preview._enabled = True
                    preview.update(view._backend, source, 1, frame, camera, (320, 240))
                    assert preview._backend.get_contact_style() == view._backend.get_contact_style()
                    assert (
                        preview._backend.debug.layer("physics.contact.points").count_of(
                            PrimitiveType.SPHERE
                        )
                        == 4
                    )
                view.set_flag(RenderFlag.CONTACTPOINT, False)
                view.update(frame)
                np.testing.assert_array_equal(view.render(), before)
            finally:
                with view._current():
                    preview.release()
                    capture.release()
    finally:
        adapter.release()


@pytest.mark.parametrize("language,scale", [("en", 1), ("zh_CN", 1.5)])
def test_contact_settings_persist_and_override(
    contact_model, tmp_path, monkeypatch, backend_name, language, scale
):
    settings = tmp_path / "settings.json"
    monkeypatch.setenv("MOJIVE_SETTINGS", str(settings))
    monkeypatch.setenv("MOJIVE_UI_SCALE", str(scale))
    output = Path("output/contact-style") / backend_name
    output.mkdir(parents=True, exist_ok=True)
    with build(
        contact_model,
        paused=True,
        width=round(1280 * scale),
        height=round(1000 * scale),
        vsync=False,
    ) as viewer:
        viewer.app.set_language(language)
        show_settings(viewer, "MuJoCo Visuals")
        # Keep the changed section visible in the production scrollable page.
        _click(
            viewer,
            _item_center(
                viewer,
                "collapsing_header",
                f"{viewer.app.localizer.text('Both appearance')}###geometry_style",
            ),
        )
        _click(viewer, _item_center(viewer, "invisible_button", "##contact_enabled"))
        assert viewer.backend.get_flag(RenderFlag.CONTACTPOINT)
        _click(viewer, _item_center(viewer, "begin_combo", "##contact_shape"))
        _click(viewer, _item_center(viewer, "selectable", "Sphere"))
        assert viewer.contact_style.shape == "sphere"
        low, high = _item_rect(viewer, "slider_float", "##contact_scale")
        _click(viewer, ((low[0] + high[0]) / 2, (low[1] + high[1]) / 2))
        assert 4 < viewer.contact_style.scale < 6
        _click(viewer, _item_center(viewer, "invisible_button", "##contact_model_color"))
        assert viewer.contact_style.use_model_color
        _click(viewer, _item_center(viewer, "invisible_button", "##contact_model_color"))
        for channel, value in enumerate((32, 204, 242)):
            low, high = _item_rect(viewer, "color_edit4", "##contact_color")
            fields_width = (
                high[0] - low[0] - imgui.get_frame_height() - imgui.get_style().item_inner_spacing.x
            )
            point = (low[0] + fields_width * (channel + 0.5) / 4, (low[1] + high[1]) / 2)
            _click(viewer, point)
            _click(viewer, point)
            io = imgui.get_io()
            modifier = imgui.Key.mod_super if io.config_mac_osx_behaviors else imgui.Key.mod_ctrl
            io.add_key_event(modifier, True)
            io.add_key_event(imgui.Key.a, True)
            viewer.sync()
            io.add_key_event(imgui.Key.a, False)
            io.add_key_event(modifier, False)
            io.add_input_characters_utf8(str(value))
            viewer.sync()
            io.add_key_event(imgui.Key.enter, True)
            viewer.sync()
            io.add_key_event(imgui.Key.enter, False)
            viewer.sync()
            assert viewer.contact_style.color[channel] == pytest.approx(value / 255)
        expected = viewer.contact_style
        assert ContactStyle(**json.loads(settings.read_text())["contact_style"]) == expected
        _park_cursor(viewer)
        for _ in range(3):
            viewer.sync()
        _save_window_crop(viewer, "Settings", output / f"settings-{language}.png")
        _click(
            viewer, _item_center(viewer, "button", viewer.app.localizer.text("Reset contact style"))
        )
        assert viewer.contact_style == ContactStyle()
        viewer.configure_contact_style(expected, persist=True)
    with build(contact_model, paused=True, vsync=False) as viewer:
        assert viewer.contact_style == expected
    override = ContactStyle("point", (0.1, 0.8, 0.9, 1))
    with build(
        contact_model, config=ViewerConfig(contact_style=override), paused=True, vsync=False
    ) as viewer:
        assert viewer.contact_style == override
    assert ContactStyle(**json.loads(settings.read_text())["contact_style"]) == expected
