"""CPU contract tests for the public MuJoCo-compatible Renderer API."""

from __future__ import annotations

import inspect

import pytest

mujoco = pytest.importorskip("mujoco")

import mojive  # noqa: E402
import mojive.renderer as renderer_module  # noqa: E402
from mojive.adapters.mujoco_adapter import MuJoCoAdapter  # noqa: E402


def _model(*, width: int = 64, height: int = 48):
    return mujoco.MjModel.from_xml_string(
        f"""
        <mujoco>
          <visual><global offwidth="{width}" offheight="{height}"/></visual>
          <worldbody><geom type="box" size=".1 .1 .1"/></worldbody>
        </mujoco>
        """
    )


def test_renderer_is_exported_with_mujoco_compatible_constructor():
    signature = inspect.signature(mojive.Renderer)

    assert list(signature.parameters) == [
        "model",
        "height",
        "width",
        "max_geom",
        "font_scale",
    ]
    assert signature.parameters["height"].default == 240
    assert signature.parameters["width"].default == 320
    assert signature.parameters["max_geom"].default == 10000
    assert signature.parameters["font_scale"].default == mujoco.mjtFontScale.mjFONTSCALE_150


@pytest.mark.parametrize(
    ("width", "height", "message"),
    [
        (65, 48, "Image width 65 > framebuffer width 64."),
        (64, 49, "Image height 49 > framebuffer height 48."),
    ],
)
def test_renderer_rejects_dimensions_larger_than_mujoco_framebuffer(width, height, message):
    with pytest.raises(ValueError, match=message):
        mojive.Renderer(_model(), width=width, height=height)


def test_adapter_binds_programmatic_model_data():
    model = _model()
    first = mujoco.MjData(model)
    second = mujoco.MjData(model)
    adapter = MuJoCoAdapter()

    adapter.load_model(model, first)
    assert adapter.model is model
    assert adapter.data is first

    adapter.use_data(second)
    assert adapter.data is second

    other_model = _model()
    with pytest.raises(ValueError, match="different MuJoCo model"):
        adapter.use_data(mujoco.MjData(other_model))
    adapter.release()


def test_adapter_applies_mjv_option_visual_groups():
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <geom name="default" type="box" size=".1 .1 .1" group="0"/>
            <geom name="hidden" type="sphere" size=".1" group="5"/>
          </worldbody>
        </mujoco>
        """
    )
    adapter = MuJoCoAdapter()
    adapter.load_model(model)
    try:
        assert set(adapter.scene_source().geom_source) == {0}

        option = mujoco.MjvOption()
        option.geomgroup[:] = 0
        option.geomgroup[5] = 1
        assert adapter.apply_scene_option(option)
        assert set(adapter.scene_source().geom_source) == {1}
        assert not adapter.apply_scene_option(option)
    finally:
        adapter.release()


def test_adapter_refreshes_direct_model_visual_edits():
    model = _model()
    adapter = MuJoCoAdapter()
    adapter.load_model(model)
    try:
        first = adapter.scene_source()
        model.geom_rgba[0] = (0.2, 0.4, 0.8, 0.6)

        assert adapter.refresh_model_visuals()
        second = adapter.scene_source()
        assert second is not first
        assert second.geom_rgba[0] == pytest.approx((0.2, 0.4, 0.8, 0.6))
        assert not adapter.refresh_model_visuals()
    finally:
        adapter.release()


def test_renderer_requests_bvh_data_only_when_visible():
    option = mujoco.MjvOption()

    assert not renderer_module._frame_needs(option).bvh

    option.flags[mujoco.mjtVisFlag.mjVIS_MESHBVH] = 1
    assert renderer_module._frame_needs(option).bvh


def test_renderer_requests_only_visible_optional_frame_data():
    option = mujoco.MjvOption()
    option.flags[:] = 0
    option.label = mujoco.mjtLabel.mjLABEL_NONE
    option.frame = mujoco.mjtFrame.mjFRAME_NONE

    needs = renderer_module._frame_needs(option)
    assert needs.poses
    assert not any(
        (
            needs.contacts,
            needs.tendons,
            needs.actuator,
            needs.deformables,
            needs.diagnostics,
            needs.islands,
            needs.bvh,
        )
    )

    option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = 1
    assert renderer_module._frame_needs(option).contacts
    option.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = 1
    needs = renderer_module._frame_needs(option)
    assert needs.actuator and needs.tendons and needs.diagnostics
    option.flags[mujoco.mjtVisFlag.mjVIS_ISLAND] = 1
    needs = renderer_module._frame_needs(option)
    assert needs.islands and needs.contacts and needs.tendons and needs.deformables


def test_renderer_releases_backend_without_a_separate_graphics_context():
    class Releasable:
        def __init__(self):
            self.releases = 0

        def release(self):
            self.releases += 1

    renderer = renderer_module.Renderer.__new__(renderer_module.Renderer)
    renderer._closed = False
    renderer._context = None
    backend = Releasable()
    adapter = Releasable()
    renderer._backend = backend
    renderer._adapter = adapter

    renderer.close()
    renderer.close()

    assert renderer._backend is None
    assert renderer._adapter is None
    assert backend.releases == 1
    assert adapter.releases == 1


def test_renderer_releases_backend_when_initialization_fails(monkeypatch):
    class BrokenBackend:
        def __init__(self):
            self.releases = 0

        def set_background(self, color):
            del color
            raise RuntimeError("backend setup failed")

        def release(self):
            self.releases += 1

    backend = BrokenBackend()
    monkeypatch.setattr(renderer_module, "_select_backend", lambda *args: (None, backend))

    with pytest.raises(RuntimeError, match="backend setup failed"):
        renderer_module.Renderer(_model(), width=64, height=48)

    assert backend.releases == 1


@pytest.mark.parametrize(
    "method",
    (
        "enable_depth_rendering",
        "disable_depth_rendering",
        "enable_segmentation_rendering",
        "disable_segmentation_rendering",
    ),
)
def test_renderer_output_modes_reject_calls_after_close(method):
    renderer = renderer_module.Renderer.__new__(renderer_module.Renderer)
    renderer._closed = True

    with pytest.raises(RuntimeError, match=f"{method} cannot be called after close"):
        getattr(renderer, method)()
