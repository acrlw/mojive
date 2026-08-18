"""CPU contract tests for the public MuJoCo-compatible Renderer API."""

from __future__ import annotations

import inspect

import pytest

mujoco = pytest.importorskip("mujoco")

import forge_viewer  # noqa: E402
from forge_viewer.adapters.mujoco_adapter import MuJoCoAdapter  # noqa: E402


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
    signature = inspect.signature(forge_viewer.Renderer)

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
        forge_viewer.Renderer(_model(), width=width, height=height)


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
