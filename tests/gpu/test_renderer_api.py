"""Real OpenGL tests for the public MuJoCo-compatible Renderer API."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from mojive import Renderer, ShadowQuality  # noqa: E402
from mojive.render.backend import (  # noqa: E402
    DebugView,
    FrameMode,
    LabelMode,
    RenderFlag,
    RenderProduct,
    RenderRequest,
)

pytestmark = pytest.mark.gpu


def _model():
    return mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <visual>
            <global offwidth="160" offheight="120"/>
            <quality offsamples="4"/>
          </visual>
          <worldbody>
            <light pos="0 -2 3"/>
            <camera name="fixed" pos="0 -3 1"
                    xyaxes="1 0 0 0 .31622777 .94868330"/>
            <geom type="plane" size="3 3 .1" rgba=".15 .2 .25 1"/>
            <geom pos=".8 0 .35" type="sphere" size=".3" rgba=".2 .7 .9 .35"/>
            <site pos="-.7 0 .4" type="sphere" size=".12" rgba=".2 .9 .4 1"/>
            <body pos="0 0 .5">
              <freejoint/>
              <geom type="box" size=".3 .2 .5" rgba=".8 .2 .1 1"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )


def test_wgpu_rgb_packing_handles_partial_four_pixel_group() -> None:
    from mojive.render.selection import render_backend_name

    if render_backend_name() != "wgpu":
        pytest.skip("WebGPU RGB packing contract")
    model = _model()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    renderer = Renderer(model, height=95, width=127)
    try:
        renderer.update_scene(data, camera="fixed")
        with renderer._gl_current():
            renderer._backend.render(request=RenderRequest.color())
            rgba = renderer._backend.target.read_color(flip=True)
            rgb = renderer._backend.target.read_rgb(flip=True)
        assert np.array_equal(rgb, rgba[..., :3])
        assert np.array_equal(renderer.render_async().result(timeout=10.0), rgb)
    finally:
        renderer.close()


def test_renderer_rgb_camera_out_and_lifecycle():
    model = _model()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    renderer = Renderer(
        model,
        height=96,
        width=128,
        max_geom=32,
        shadow_quality=ShadowQuality.HIGH,
    )

    assert renderer.model is model
    assert renderer.height == 96
    assert renderer.width == 128
    assert isinstance(renderer.scene, mujoco.MjvScene)
    assert renderer.scene.maxgeom == 32
    assert renderer.shadow_quality is ShadowQuality.HIGH
    renderer.set_shadow_quality(ShadowQuality.PERFORMANCE)
    assert renderer.shadow_quality is ShadowQuality.PERFORMANCE

    renderer.update_scene(data)
    assert renderer.scene.ngeom > 0
    image = renderer.render()
    assert image.shape == (96, 128, 3)
    assert image.dtype == np.uint8
    assert image.flags.c_contiguous
    assert np.ptp(image) > 0

    combined_request = RenderRequest(
        RenderProduct.COLOR | RenderProduct.METRIC_DEPTH | RenderProduct.SEGMENTATION
    )
    with renderer._gl_current():
        assert renderer._backend.render(request=combined_request) is not None
        target = renderer._backend.target
        rgba = target.read_color(flip=True)
        assert np.array_equal(target.read_rgb(flip=True), rgba[..., :3])
        assert np.array_equal(target.read_rgb(flip=False), rgba[::-1, ..., :3])
        assert renderer._backend.target.read_metric_depth().shape == image.shape[:2]
        assert renderer._backend.target.read_segmentation().shape == (*image.shape[:2], 2)

    out = np.empty_like(image)
    assert renderer.render(out=out) is out
    assert np.array_equal(out, image)
    cast_out = np.empty(image.shape, np.float32)
    assert renderer.render(out=cast_out) is cast_out
    assert np.array_equal(cast_out, image)
    strided_storage = np.empty((96, 256, 3), np.uint8)
    strided_out = strided_storage[:, ::2]
    assert not strided_out.flags.c_contiguous
    assert renderer.render(out=strided_out) is strided_out
    assert np.array_equal(strided_out, image)
    if renderer._backend.caps.name == "wgpu":
        assert renderer._backend.target._readbacks is None
    async_image = renderer.render_async().result(timeout=10.0)
    assert np.array_equal(async_image, image)
    if renderer._backend.caps.name == "wgpu":
        assert renderer._backend.target._readbacks is not None
    queued = [renderer.render_async() for _ in range(5)]
    assert all(np.array_equal(future.result(timeout=10.0), image) for future in queued)
    async_cast_out = np.empty(image.shape, np.float32)
    assert renderer.render_async(out=async_cast_out).result(timeout=10.0) is async_cast_out
    assert np.array_equal(async_cast_out, image)
    if renderer._backend.caps.name == "wgpu":
        target = renderer._backend.target
        packed_size = ((renderer.width * renderer.height + 3) // 4) * 12
        assert target._rgb_packer._size == packed_size
        assert target._sync_readback._capacity >= packed_size
        assert {slot.capacity for slot in target._readbacks._slots if slot.buffer is not None} == {
            packed_size
        }
    with pytest.raises(ValueError, match=r"out\.shape"):
        renderer.render(out=np.empty((96, 128), np.uint8))

    renderer.update_scene(data, camera="fixed")
    renderer.update_scene(data, camera=0)
    renderer.update_scene(data, camera=mujoco.MjvCamera())
    with pytest.raises(ValueError, match="does not exist"):
        renderer.update_scene(data, camera="missing")
    with pytest.raises(ValueError, match="out of range"):
        renderer.update_scene(data, camera=1)

    renderer.update_scene(data, camera="fixed")
    renderer.enable_depth_rendering()
    depth = renderer.render()
    assert depth.shape == (96, 128)
    assert depth.dtype == np.float32
    assert np.all(np.isfinite(depth))
    assert np.all(depth > 0.0)
    depth_out = np.empty_like(depth)
    assert renderer.render(out=depth_out) is depth_out
    assert np.array_equal(depth_out, depth)
    depth_cast_out = np.empty(depth.shape, np.float64)
    assert renderer.render(out=depth_cast_out) is depth_cast_out
    assert np.array_equal(depth_cast_out, depth)
    assert np.array_equal(renderer.render_async().result(timeout=10.0), depth)

    renderer.enable_segmentation_rendering()
    segmentation = renderer.render()
    assert segmentation.shape == (96, 128, 2)
    assert segmentation.dtype == np.int32
    pairs = {tuple(pair) for pair in segmentation.reshape(-1, 2)}
    assert (-1, -1) in pairs
    assert (0, int(mujoco.mjtObj.mjOBJ_GEOM)) in pairs
    assert (1, int(mujoco.mjtObj.mjOBJ_GEOM)) in pairs
    assert (2, int(mujoco.mjtObj.mjOBJ_GEOM)) in pairs
    assert (0, int(mujoco.mjtObj.mjOBJ_SITE)) in pairs
    segmentation_out = np.empty_like(segmentation)
    assert renderer.render(out=segmentation_out) is segmentation_out
    assert np.array_equal(segmentation_out, segmentation)
    segmentation_cast_out = np.empty(segmentation.shape, np.int64)
    assert renderer.render(out=segmentation_cast_out) is segmentation_cast_out
    assert np.array_equal(segmentation_cast_out, segmentation)
    assert np.array_equal(renderer.render_async().result(timeout=10.0), segmentation)

    renderer.disable_segmentation_rendering()
    assert renderer.render().shape == (96, 128, 3)

    pending_on_close = renderer.render_async()
    renderer.close()
    assert pending_on_close.result(timeout=10.0).shape == (96, 128, 3)
    renderer.close()
    with pytest.raises(RuntimeError, match="after close"):
        renderer.render()


def test_renderer_rejects_data_from_another_model():
    model = _model()
    other = _model()
    with (
        Renderer(model, height=48, width=64) as renderer,
        pytest.raises(ValueError, match="different MuJoCo model"),
    ):
        renderer.update_scene(mujoco.MjData(other))


def test_renderer_wireframe_geometry_shader():
    model = _model()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    with Renderer(model, height=96, width=128) as renderer:
        renderer.update_scene(data)
        shaded = renderer.render()
        renderer.set_render_flag("mjRND_WIREFRAME", True)
        wireframe = renderer.render()

    assert np.ptp(wireframe) > 0
    assert not np.array_equal(wireframe, shaded)


@pytest.mark.parametrize("projection", ["perspective", "orthographic"])
def test_renderer_depth_is_metric_for_perspective_and_orthographic(projection):
    projection_xml = (
        'projection="orthographic" fovy="4"' if projection == "orthographic" else 'fovy="45"'
    )
    model = mujoco.MjModel.from_xml_string(
        f"""
        <mujoco>
          <visual>
            <global offwidth="80" offheight="80"/>
            <quality offsamples="0"/>
          </visual>
          <worldbody>
            <camera name="fixed" pos="0 0 3" {projection_xml}/>
            <geom type="box" size=".5 .5 .5"/>
          </worldbody>
        </mujoco>
        """
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    with Renderer(model, height=80, width=80) as renderer:
        renderer.update_scene(data, camera="fixed")
        renderer.enable_depth_rendering()
        depth = renderer.render()

    assert depth[40, 40] == pytest.approx(2.5, abs=1e-3)
    assert depth[0, 0] > depth[40, 40]


def test_renderer_maps_mjv_options_to_opengl_state():
    model = _model()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    option = mujoco.MjvOption()
    option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = 1
    option.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = 0
    option.label = mujoco.mjtLabel.mjLABEL_BODY
    option.frame = mujoco.mjtFrame.mjFRAME_WORLD
    option.bvh_depth = 3

    with Renderer(model, height=48, width=64) as renderer:
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_WIREFRAME] = 1
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_DEPTH] = 1
        renderer.update_scene(data, camera="fixed", scene_option=option)
        backend = renderer._backend

        assert backend.get_flag(RenderFlag.JOINT)
        assert not backend.get_flag(RenderFlag.TENDON)
        assert backend.get_flag(RenderFlag.WIREFRAME)
        assert backend.get_label_mode() is LabelMode.BODY
        assert backend.get_frame_mode() is FrameMode.WORLD
        assert backend.get_bvh_depth() == 3
        assert backend.get_debug_view() is DebugView.DEPTH

        option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = 1
        renderer.update_scene(data, camera="fixed", scene_option=option)
        dynamic = ~renderer._source.geom_static
        assert np.all(renderer._source.geom_rgba[dynamic, 3] <= 0.3)


def test_renderer_max_geom_limits_mujoco_and_opengl_scenes():
    model = _model()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    with Renderer(model, height=48, width=64, max_geom=1) as renderer:
        renderer.update_scene(data, camera="fixed")

        assert renderer.scene.ngeom == 1
        assert set(renderer._source.geom_source) == {0}


def test_multiple_renderers_with_different_sizes_coexist():
    model = _model()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    with (
        Renderer(model, height=48, width=64) as small,
        Renderer(model, height=72, width=96) as large,
    ):
        small.update_scene(data, camera="fixed")
        large.update_scene(data, camera="fixed")
        assert small.render().shape == (48, 64, 3)
        assert large.render().shape == (72, 96, 3)
        assert small.render().shape == (48, 64, 3)


def test_multi_camera_concurrency_survives_interleaved_render_and_partial_close():
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <visual>
            <global offwidth="128" offheight="96"/>
            <headlight ambient=".3 .3 .3" diffuse=".8 .8 .8"/>
          </visual>
          <worldbody>
            <camera name="front" pos="0 -3 1" mode="targetbody" target="target"/>
            <camera name="side" pos="3 0 1" mode="targetbody" target="target"/>
            <camera name="top" pos=".2 0 4" mode="targetbody" target="target"/>
            <body name="target" pos="0 0 .4">
              <geom pos="-.6 0 0" type="box" size=".4 .15 .2" rgba=".9 .2 .1 1"/>
              <geom pos=".55 .3 .35" type="sphere" size=".3" rgba=".1 .7 .9 1"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    renderers = [Renderer(model, height=72, width=96) for _ in range(3)]
    cameras = ("front", "side", "top")
    outputs = [np.empty((72, 96, 3), np.uint8) for _ in renderers]
    try:
        for _ in range(12):
            for renderer, camera, output in zip(renderers, cameras, outputs, strict=True):
                renderer.update_scene(data, camera=camera)
                assert renderer.render(out=output) is output
        assert not np.array_equal(outputs[0], outputs[1])
        assert not np.array_equal(outputs[1], outputs[2])

        renderers[1].close()
        for index in (0, 2):
            renderers[index].update_scene(data, camera=cameras[index])
            assert np.ptp(renderers[index].render()) > 0
    finally:
        for renderer in renderers:
            renderer.close()


def test_renderer_can_be_repeatedly_created_and_destroyed():
    model = _model()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for _ in range(200):
        with Renderer(model, height=24, width=32) as renderer:
            renderer.update_scene(data, camera="fixed")
            assert renderer.render().shape == (24, 32, 3)


def test_renderer_segmentation_maps_flex_and_skin_objects():
    path = Path(__file__).parents[2] / "assets" / "deformables.xml"
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    with Renderer(model, height=96, width=128) as renderer:
        renderer.update_scene(data)
        renderer.enable_segmentation_rendering()
        pairs = {tuple(pair) for pair in renderer.render().reshape(-1, 2)}

    assert any(object_type == int(mujoco.mjtObj.mjOBJ_FLEX) for _, object_type in pairs)
    assert any(object_type == int(mujoco.mjtObj.mjOBJ_SKIN) for _, object_type in pairs)
