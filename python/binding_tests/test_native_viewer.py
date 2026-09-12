"""Public Python API and full native Viewer acceptance, including real GPU ownership."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mojive import Renderer, RenderProduct, Scene, SceneRenderer
from mojive.adapters.static import StaticSceneAdapter

OUTPUT = Path(os.environ.get("MOJIVE_NATIVE_VIEWER_OUTPUT", "output/native-viewer"))


@pytest.fixture(autouse=True)
def isolated_preferences(monkeypatch, tmp_path):
    monkeypatch.setenv("MOJIVE_IMGUI_INI", str(tmp_path / "imgui.ini"))
    monkeypatch.setenv("MOJIVE_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("MOJIVE_LANGUAGE", "en")
    OUTPUT.mkdir(parents=True, exist_ok=True)


def test_public_products_pose_reload_resize_and_independent_scenes():
    red, green = Scene(), Scene()
    box = red.box(color=(1, 0, 0, 1))
    green.box(color=(0, 1, 0, 1))
    adapter = StaticSceneAdapter(red)
    with SceneRenderer(width=96, height=72, samples=0, renderer="bgfx") as a:
        a.update_from(adapter)
        expected = a.render()
        assert expected[36, 48, 0] > expected[36, 48, 1] + 30
        np.testing.assert_array_equal(expected[0, 0], [0, 0, 0])
        assert a.render(product=RenderProduct.OBJECT_ID)[36, 48] == box.object_id
        assert a.render(product=RenderProduct.METRIC_DEPTH)[36, 48] == pytest.approx(3.5, abs=0.01)
        strided = np.empty((72, 192, 3), np.uint8)[:, ::2]
        assert a.render(out=strided) is strided
        np.testing.assert_array_equal(expected, strided)
        with SceneRenderer(green.source, width=64, height=48, samples=4, renderer="bgfx") as b:
            b.update(green.frame)
            other = b.render()
            assert other[24, 32, 1] > other[24, 32, 0] + 30
            np.testing.assert_array_equal(a.render(), expected)
            box.set_pose((0, 0, 0.7))
            a.update_from(adapter)
            raised = a.render(product=RenderProduct.OBJECT_ID)
            assert np.nonzero(raised == box.object_id)[0].mean() < 30
            box.remove()
            a.update_from(adapter)
            assert not a.render(product=RenderProduct.OBJECT_ID).any()
            np.testing.assert_array_equal(b.render(), other)
            a.resize(37, 29)
            assert a.render().shape == (29, 37, 3)
        a.render()
    with pytest.raises(RuntimeError, match="closed"):
        a.render()
    # The final owner closes the process device; a subsequent one can initialize it again.
    with SceneRenderer(green.source, width=32, height=24, renderer="bgfx") as fresh:
        fresh.update(green.frame)
        assert fresh.render()[12, 16, 1] > 100


def test_mujoco_renderer_async_frame_identity_out_and_close():
    import mujoco

    model = mujoco.MjModel.from_xml_path("assets/joint_gizmo.xml")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    renderer = Renderer(model, width=160, height=120, renderer="bgfx")
    try:
        renderer.update_scene(data)
        expected = renderer.render()
        out = np.empty((120, 320, 3), np.float32)[:, ::2]
        future = renderer.render_async(out=out)
        assert future.result(timeout=15) is out
        np.testing.assert_array_equal(out, expected)
        renderer.enable_depth_rendering()
        depth = renderer.render_async().result(timeout=15)
        assert depth.shape == (120, 160) and depth.dtype == np.float32
        assert np.isfinite(depth).all() and depth.min() > 0
        renderer.enable_segmentation_rendering()
        segment = renderer.render_async().result(timeout=15)
        assert segment.shape == (120, 160, 2) and segment.dtype == np.int32
        assert np.any(segment[..., 0] >= 0)
        Image.fromarray(expected).save(OUTPUT / "renderer.png")
    finally:
        renderer.close()
    np.testing.assert_array_equal(out, expected)


def test_full_viewer_two_windows_resize_selection_and_peer_survival():
    from mojive import commands as cmd
    from mojive.application.composition import build
    from mojive.ui import window as window_module

    baseline = window_module._live_windows
    a = build(
        Path("assets/joint_gizmo.xml"),
        renderer="bgfx",
        width=1100,
        height=760,
        vsync=False,
        show_window=False,
    )
    b = None
    try:
        for _ in range(6):
            a.sync()
        b = build(
            Path("assets/test_scene.xml"),
            renderer="bgfx",
            width=800,
            height=600,
            vsync=False,
            show_window=False,
        )
        for _ in range(4):
            a.sync()
            b.sync()
        image = a.capture_array(surface="window")
        width, height = a.window.size_pixels
        assert image.shape == (height, width, 3) and np.std(image) > 10
        Image.fromarray(image).save(OUTPUT / "viewer.png")
        # A scene target remains valid after a peer is resized, reloaded, and closed.
        a.backend.render()
        expected = a.backend.target.read_rgb()
        window_module.glfw.set_window_size(b.window._window, 620, 480)
        for _ in range(3):
            b.sync()
        # A texture reference from a closed, no-longer-drawn preview is retired next frame.
        image = b.backend.render()
        a.window.viewport_texture_ref(image)
        assert image.payload.id in a.window._viewport_textures
        b.close()
        b = None
        np.testing.assert_array_equal(a.backend.target.read_rgb(), expected)
        result = a.session.submit(cmd.Select(4))
        assert result.ok
        for _ in range(2):
            a.sync()
        assert a.session.selected == 4
        assert image.payload.id not in a.window._viewport_textures
        Image.fromarray(a.capture_array(surface="window")).save(OUTPUT / "selection.png")
        # Recreate while the first device is still alive.
        b = build(
            Path("assets/joint_gizmo.xml"),
            renderer="bgfx",
            width=600,
            height=480,
            vsync=False,
            show_window=False,
        )
        b.sync()
        b.close()
        b = None
        a.sync()
    finally:
        if b is not None:
            b.close()
        a.close()
    assert window_module._live_windows == baseline


def test_failed_window_initialization_releases_platform_and_device(monkeypatch, tmp_path):
    from mojive.application.composition import build
    from mojive.ui import window as window_module

    baseline = window_module._live_windows
    monkeypatch.setenv("MOJIVE_NATIVE_SHADER_DIR", str(tmp_path))
    with pytest.raises(RuntimeError, match="shader"):
        build(Path("assets/joint_gizmo.xml"), renderer="bgfx", show_window=False)
    assert window_module._live_windows == baseline
    monkeypatch.delenv("MOJIVE_NATIVE_SHADER_DIR")
    with build(
        Path("assets/joint_gizmo.xml"),
        renderer="bgfx",
        width=640,
        height=480,
        vsync=False,
        show_window=False,
    ) as viewer:
        viewer.sync()
    assert window_module._live_windows == baseline


def test_hundred_humanoid_viewer_physics_and_rendering():
    from mojive.application.composition import build

    model = Path(os.environ.get("MOJIVE_HUMANOIDS_MODEL", ""))
    if not model.is_file():
        pytest.skip("Set MOJIVE_HUMANOIDS_MODEL to the official MuJoCo 100_humanoids.xml")
    started = time.perf_counter()
    with build(
        model, renderer="bgfx", paused=False, width=1280, height=800, vsync=False, show_window=False
    ) as viewer:
        frames = 0
        deadline = time.perf_counter() + 15
        while frames < 3 or viewer.session.frame.time < 0.05:
            viewer.sync()
            frames += 1
            if time.perf_counter() > deadline:
                pytest.fail(
                    f"Physics did not advance: {frames} frames, simulation time {viewer.session.frame.time}"
                )
        assert viewer.session.frame.step > 0
        rgb = viewer.capture_array(surface="window")
        Image.fromarray(rgb).save(OUTPUT / "humanoids100.png")
        ids = viewer.backend.target.read_ids()
        assert len(np.unique(ids)) > 100
        assert viewer.backend.stats.instances >= 5000
        report = {
            "frames": frames,
            "physics_time": viewer.session.frame.time,
            "physics_steps": viewer.session.frame.step,
            "instances": viewer.backend.stats.instances,
            "visible_ids": len(np.unique(ids)),
            "elapsed_seconds": time.perf_counter() - started,
            "device": viewer.backend.describe(),
            "image": list(rgb.shape),
            "scope": "Full Viewer smoke test; this is not a matched-quality speed comparison",
        }
        (OUTPUT / "acceptance.json").write_text(json.dumps(report, indent=2) + "\n")


@pytest.mark.parametrize("scale", [1.5, 2.5])
def test_shown_scaled_window_and_cjk(monkeypatch, scale):
    from mojive.application.composition import build

    monkeypatch.setenv("MOJIVE_LANGUAGE", "zh")
    monkeypatch.setenv("MOJIVE_UI_SCALE", str(scale))
    with build(
        Path("assets/joint_gizmo.xml"),
        renderer="bgfx",
        width=round(1000 * scale),
        height=round(680 * scale),
        vsync=False,
        show_window=True,
    ) as viewer:
        for _ in range(12):
            viewer.sync()
        rgb = viewer.capture_array(surface="window")
        width, height = viewer.window.size_pixels
        assert rgb.shape == (height, width, 3)
        assert viewer.window.shown
        assert viewer.window.ui_scale == pytest.approx(scale * viewer.window.pixel_scale)
        suffix = "" if scale == 1.5 else f"-{scale}"
        Image.fromarray(rgb).save(OUTPUT / f"scaled-cjk{suffix}.png")
        (OUTPUT / f"display-scale{suffix}.json").write_text(
            json.dumps(
                {
                    "pixels": [width, height],
                    "points": viewer.window.size_points,
                    "framebuffer_scale": viewer.window.pixel_scale,
                    "ui_scale": viewer.window.ui_scale,
                },
                indent=2,
            )
            + "\n"
        )


def test_alternate_capture_and_real_alpha_do_not_replace_viewport():
    from mojive.render.native.backend import NativeBackend

    backend = NativeBackend(64, 48)
    try:
        backend.set_background((0.1, 0.2, 0.3, 0.5))
        backend.render()
        rgba = backend.target.read_color()
        assert rgba.shape == (48, 64, 4)
        np.testing.assert_array_equal(rgba[0, 0], [26, 51, 77, 128])
        backend.capture(OUTPUT / "alternate.png", size=(83, 29))
        assert Image.open(OUTPUT / "alternate.png").size == (83, 29)
        np.testing.assert_array_equal(backend.target.read_color(), rgba)
    finally:
        backend.release()


def test_async_completion_can_close_the_last_renderer():
    from concurrent.futures import Future

    import mujoco

    model = mujoco.MjModel.from_xml_path("assets/joint_gizmo.xml")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    renderer = Renderer(model, width=80, height=60, renderer="bgfx")
    renderer.update_scene(data)
    done = Future()
    future = renderer.render_async()

    def close(_):
        try:
            renderer.close()
            done.set_result(True)
        except Exception as exc:
            done.set_exception(exc)

    future.add_done_callback(close)
    assert future.result(timeout=15).shape == (60, 80, 3)
    assert done.result(timeout=15)
    with SceneRenderer(width=16, height=16, renderer="bgfx") as fresh:
        fresh.render()


def test_viewer_projection_and_colors_match_opengl_after_resize(monkeypatch):
    from mojive import CameraView
    from mojive.application.composition import build
    from mojive.ui import window as window_module

    # Fixed-size projection comparisons need a fixed layout scale. Desktop DPI can
    # otherwise force dock panels to their minimum sizes and alter their proportions.
    monkeypatch.setenv("MOJIVE_UI_SCALE", "1")
    captures = {}
    for backend in ("opengl", "bgfx"):
        with build(
            Path("assets/test_scene.xml"),
            renderer=backend,
            width=1100,
            height=760,
            vsync=False,
            paused=True,
            show_window=False,
        ) as viewer:
            for _ in range(4):
                viewer.sync()
            original = viewer.app.camera.view()
            # The application camera intentionally carries aspect=1; render targets own aspect.
            viewer.app.camera.adopt(
                CameraView(
                    eye=np.array([2.6, 1.4, 0.67], np.float32),
                    target=np.array([0, 0, 0.2], np.float32),
                    near=original.near,
                    far=509,
                )
            )
            for width, height in ((1100, 760), (820, 880), (1100, 760)):
                window_module.glfw.set_window_size(viewer.window._window, width, height)
                for _ in range(4):
                    viewer.sync()
                rgb = viewer.backend.target.read_rgb(flip=True)
                key = (backend, width, height)
                if key in captures:
                    np.testing.assert_array_equal(captures[key], rgb)
                captures[key] = rgb.copy()
                Image.fromarray(rgb).save(OUTPUT / f"projection-{backend}-{width}-{height}.png")
    for width, height in ((1100, 760), (820, 880)):
        reference, native = captures["opengl", width, height], captures["bgfx", width, height]
        error = np.abs(native.astype(float) - reference)
        assert error.mean() < 1
        assert np.percentile(error, 99) <= 3


def _metal_presentation(window):
    import ctypes as c

    objc = c.CDLL("/usr/lib/libobjc.A.dylib")
    objc.sel_registerName.argtypes, objc.sel_registerName.restype = [c.c_char_p], c.c_void_p
    objc.object_getClassName.argtypes, objc.object_getClassName.restype = [c.c_void_p], c.c_char_p

    def send(obj, name, *args, result=c.c_void_p):
        call = c.CFUNCTYPE(result, c.c_void_p, c.c_void_p, *([c.c_ulong] * len(args)))(
            ("objc_msgSend", objc)
        )
        return call(obj, objc.sel_registerName(name.encode()), *args)

    def array(obj, name):
        items = send(obj, name)
        return [
            send(items, "objectAtIndex:", i)
            for i in range(send(items, "count", result=c.c_ulong) if items else 0)
        ]

    states = []

    def layer(value):
        if not value:
            return
        if "MetalLayer" in objc.object_getClassName(value).decode():
            states.append(
                (
                    bool(send(value, "displaySyncEnabled", result=c.c_bool)),
                    int(send(value, "maximumDrawableCount", result=c.c_ulong)),
                )
            )
        for child in array(value, "sublayers"):
            layer(child)

    def view(value):
        layer(send(value, "layer"))
        for child in array(value, "subviews"):
            view(child)

    view(send(window._native_handle()[0], "contentView"))
    assert states, "The window must own a real Metal presentation layer"
    return states


@pytest.mark.skipif(os.sys.platform != "darwin", reason="Inspect the actual macOS display layer")
def test_vsync_controls_gpu_presentation_and_survives_peer_close():
    from mojive.application.composition import build

    with build(
        Path("assets/test_scene.xml"),
        renderer="bgfx",
        width=640,
        height=480,
        vsync=False,
        show_window=False,
    ) as a:
        for enabled in (True, False, True):
            a.window.set_vsync(enabled)
            for _ in range(3):
                a.sync()
            assert all(
                value == (enabled, 2 if enabled else 3) for value in _metal_presentation(a.window)
            )
        with build(
            Path("assets/test_scene.xml"),
            renderer="bgfx",
            width=640,
            height=480,
            vsync=False,
            show_window=False,
        ) as b:
            a.sync()
            b.sync()
            # bgfx shares presentation policy: a synchronized peer retains VSync.
            assert all(
                value == (True, 2)
                for window in (a.window, b.window)
                for value in _metal_presentation(window)
            )
        a.sync()
        assert all(value == (True, 2) for value in _metal_presentation(a.window))
        a.window.set_vsync(False)
        for _ in range(3):
            a.sync()
        assert all(value == (False, 3) for value in _metal_presentation(a.window))


@pytest.mark.parametrize(("requested", "effective"), [(0, 1), (3, 2), (8, 8), (24, 8)])
def test_mujoco_sample_requests_use_portable_color_targets(requested, effective):
    import mujoco

    model = mujoco.MjModel.from_xml_path("assets/test_scene.xml")
    model.vis.quality.offsamples = requested
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    with Renderer(model, width=96, height=72, renderer="bgfx") as renderer:
        renderer.update_scene(data)
        rgb = renderer.render()
        assert rgb.shape == (72, 96, 3) and np.std(rgb) > 5
        assert renderer._backend.caps.msaa_samples == effective


def test_zero_alpha_geometry_remains_available_to_semantic_export():
    import mujoco

    model = mujoco.MjModel.from_xml_string("""
        <mujoco><worldbody>
          <geom name="opaque" type="sphere" size="0.4" rgba="0 1 0 1"/>
          <geom name="invisible" type="box" size="0.5 0.1 0.5" pos="0 -1 0"
                rgba="1 0 0 0"/>
        </worldbody></mujoco>
    """)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = 0
    camera.distance, camera.azimuth, camera.elevation = 4, 90, 0
    results = {}
    for backend in ("opengl", "bgfx"):
        with Renderer(model, width=160, height=120, renderer=backend) as renderer:
            renderer.update_scene(data, camera)
            rgb = renderer.render()
            renderer.enable_segmentation_rendering()
            ids = renderer.render()
            assert np.count_nonzero(ids[..., 0] == 1) > 100
            results[backend] = rgb, ids
    np.testing.assert_array_equal(results["opengl"][1], results["bgfx"][1])
    assert np.abs(results["opengl"][0].astype(float) - results["bgfx"][0]).mean() < 1
